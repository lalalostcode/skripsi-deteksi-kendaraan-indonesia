"""Shared validation-threshold, test AP, COCO-scale, and efficiency evaluation."""

from __future__ import annotations

import json
import math
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from ultralytics import YOLO

from .config import CLASS_NAMES, ExperimentConfig
from .data import PreparedDataset
from .diagnostics import save_fusion_diagnostics
from .models.factory import NECKS, verify_model_graph


def image_records(dataset: PreparedDataset, split: str) -> list:
    frame = dataset.audit[
        (dataset.audit["split"] == split)
        & (dataset.audit["status"].isin(("ok", "empty_label")))
    ]
    return list(frame.itertuples(index=False))


def read_yolo_boxes(
    label: str | Path, width: int, height: int
) -> list[tuple[int, float, float, float, float]]:
    result = []
    for line in Path(label).read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        cls, xc, yc, bw, bh = map(float, line.split())
        result.append(
            (
                int(cls),
                (xc - bw / 2) * width,
                (yc - bh / 2) * height,
                (xc + bw / 2) * width,
                (yc + bh / 2) * height,
            )
        )
    return result


def predict_records(
    model: YOLO,
    dataset: PreparedDataset,
    config: ExperimentConfig,
    split: str,
    device: str,
) -> dict[str, list[dict]]:
    """Explicit chunks prevent Ultralytics from batching the whole split."""
    paths = [str(Path(row.image).resolve()) for row in image_records(dataset, split)]
    if not paths:
        raise RuntimeError(f"Split {split} tidak berisi citra")
    output: dict[str, list[dict]] = {}
    for start in range(0, len(paths), config.eval_batch):
        chunk = paths[start : start + config.eval_batch]
        source = chunk[0] if len(chunk) == 1 else chunk
        for result in model.predict(
            source=source,
            imgsz=config.imgsz,
            conf=config.prediction_conf,
            iou=config.prediction_iou,
            nms=False,
            device=device,
            batch=len(chunk),
            verbose=False,
            stream=True,
        ):
            boxes = result.boxes
            output[str(Path(result.path).resolve())] = (
                []
                if boxes is None
                else [
                    {
                        "cls": int(cls),
                        "conf": float(score),
                        "xyxy": [float(value) for value in box],
                    }
                    for box, score, cls in zip(
                        boxes.xyxy.cpu().numpy(),
                        boxes.conf.cpu().numpy(),
                        boxes.cls.cpu().numpy(),
                    )
                ]
            )
    if set(output) != set(paths):
        raise RuntimeError(
            f"Prediksi {split}: {len(output)} hasil untuk {len(paths)} citra"
        )
    return output


def iou_one_to_many(box: list[float], boxes: list[tuple[float, ...]]) -> np.ndarray:
    if not boxes:
        return np.empty(0)
    candidate = np.asarray(boxes, dtype=float)
    anchor = np.asarray(box, dtype=float)
    extent = np.maximum(
        0,
        np.minimum(anchor[2:], candidate[:, 2:])
        - np.maximum(anchor[:2], candidate[:, :2]),
    )
    intersection = extent[:, 0] * extent[:, 1]
    union = (
        np.prod(anchor[2:] - anchor[:2])
        + np.prod(candidate[:, 2:] - candidate[:, :2], axis=1)
        - intersection
    )
    return intersection / np.maximum(union, 1e-9)


def match_class_boxes(
    gt_boxes: list[tuple[float, ...]],
    predictions: list[dict],
    threshold: float,
    iou_threshold: float = 0.5,
) -> tuple[int, int, int]:
    matched: set[int] = set()
    tp = fp = 0
    for prediction in sorted(
        (item for item in predictions if item["conf"] >= threshold),
        key=lambda item: item["conf"],
        reverse=True,
    ):
        available = [index for index in range(len(gt_boxes)) if index not in matched]
        ious = iou_one_to_many(
            prediction["xyxy"], [gt_boxes[index] for index in available]
        )
        if len(ious) and ious.max() >= iou_threshold:
            matched.add(available[int(ious.argmax())])
            tp += 1
        else:
            fp += 1
    return tp, fp, len(gt_boxes) - len(matched)


def detection_counts(
    dataset: PreparedDataset,
    split: str,
    predictions: dict[str, list[dict]],
    threshold: float,
) -> np.ndarray:
    counts = np.zeros((len(CLASS_NAMES), 3), dtype=int)
    for record in image_records(dataset, split):
        key = str(Path(record.image).resolve())
        gt = read_yolo_boxes(record.label, record.width, record.height)
        for class_id in range(len(CLASS_NAMES)):
            boxes = [item[1:] for item in gt if item[0] == class_id]
            predicted = [
                item for item in predictions.get(key, []) if item["cls"] == class_id
            ]
            counts[class_id] += match_class_boxes(boxes, predicted, threshold)
    return counts


def counts_to_frame(counts: np.ndarray) -> pd.DataFrame:
    rows = []
    for class_id, (tp, fp, fn) in enumerate(counts):
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        f1 = (
            2 * precision * recall / (precision + recall) if precision + recall else 0.0
        )
        rows.append(
            {
                "class_id": class_id,
                "class_name": CLASS_NAMES[class_id],
                "TP": int(tp),
                "FP": int(fp),
                "FN": int(fn),
                "precision": precision,
                "recall": recall,
                "f1": f1,
                "ground_truth": int(tp + fn),
            }
        )
    return pd.DataFrame(rows)


def summarize_counts(counts: np.ndarray) -> dict:
    tp, fp, fn = counts.sum(axis=0)
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "TP": int(tp),
        "FP": int(fp),
        "FN": int(fn),
    }


def select_threshold(
    dataset: PreparedDataset, predictions: dict[str, list[dict]]
) -> tuple[float, pd.DataFrame]:
    candidates = np.linspace(0.01, 0.99, 99)
    scores = []
    for threshold in candidates:
        frame = counts_to_frame(
            detection_counts(dataset, "val", predictions, float(threshold))
        )
        valid = frame[frame["ground_truth"] > 0]
        scores.append(float(valid["f1"].mean()) if len(valid) else 0.0)
    return float(candidates[int(np.argmax(scores))]), pd.DataFrame(
        {"confidence": candidates, "macro_f1": scores}
    )


def to_letterbox_canvas_xyxy(
    xyxy: tuple | list,
    original_width: int,
    original_height: int,
    target_size: int = 640,
) -> list[float]:
    scale = min(target_size / original_width, target_size / original_height)
    pad_x = (target_size - original_width * scale) / 2
    pad_y = (target_size - original_height * scale) / 2
    x1, y1, x2, y2 = map(float, xyxy)
    return [
        float(np.clip(x1 * scale + pad_x, 0, target_size)),
        float(np.clip(y1 * scale + pad_y, 0, target_size)),
        float(np.clip(x2 * scale + pad_x, 0, target_size)),
        float(np.clip(y2 * scale + pad_y, 0, target_size)),
    ]


def build_coco_eval_canvas(
    dataset: PreparedDataset,
    split: str,
    predictions: dict[str, list[dict]],
    target_size: int = 640,
) -> tuple[dict, list[dict]]:
    """Use the same 640-square scale categories for all four architectures."""
    ground_truth = {
        "info": {},
        "licenses": [],
        "images": [],
        "annotations": [],
        "categories": [
            {"id": i + 1, "name": name} for i, name in enumerate(CLASS_NAMES)
        ],
    }
    results = []
    annotation_id = 1
    for image_id, record in enumerate(image_records(dataset, split), 1):
        ground_truth["images"].append(
            {
                "id": image_id,
                "file_name": Path(record.image).name,
                "width": target_size,
                "height": target_size,
            }
        )
        for cls, x1, y1, x2, y2 in read_yolo_boxes(
            record.label, record.width, record.height
        ):
            tx1, ty1, tx2, ty2 = to_letterbox_canvas_xyxy(
                (x1, y1, x2, y2), record.width, record.height, target_size
            )
            width, height = max(0.0, tx2 - tx1), max(0.0, ty2 - ty1)
            ground_truth["annotations"].append(
                {
                    "id": annotation_id,
                    "image_id": image_id,
                    "category_id": cls + 1,
                    "bbox": [tx1, ty1, width, height],
                    "area": width * height,
                    "iscrowd": 0,
                }
            )
            annotation_id += 1
        for prediction in predictions.get(str(Path(record.image).resolve()), []):
            tx1, ty1, tx2, ty2 = to_letterbox_canvas_xyxy(
                prediction["xyxy"], record.width, record.height, target_size
            )
            results.append(
                {
                    "image_id": image_id,
                    "category_id": prediction["cls"] + 1,
                    "bbox": [tx1, ty1, max(0.0, tx2 - tx1), max(0.0, ty2 - ty1)],
                    "score": prediction["conf"],
                }
            )
    return ground_truth, results


def coco_scale_eval(ground_truth: dict, predictions: list[dict]) -> dict[str, float]:
    counts = {"small": 0, "medium": 0, "large": 0}
    for annotation in ground_truth["annotations"]:
        area = annotation["area"]
        counts["small" if area < 32**2 else "medium" if area < 96**2 else "large"] += 1
    if not predictions:
        return {
            "AP_small": 0.0 if counts["small"] else math.nan,
            "AP_medium": 0.0 if counts["medium"] else math.nan,
            "AP_large": 0.0 if counts["large"] else math.nan,
        }
    from pycocotools.coco import COCO
    from pycocotools.cocoeval import COCOeval

    coco_gt = COCO()
    coco_gt.dataset = ground_truth
    coco_gt.createIndex()
    evaluator = COCOeval(coco_gt, coco_gt.loadRes(predictions), "bbox")
    evaluator.params.maxDets = [1, 10, 100]
    evaluator.evaluate()
    evaluator.accumulate()
    evaluator.summarize()
    values = evaluator.stats[3:6]
    return {
        key: float(value) if counts[size] and value >= 0 else math.nan
        for key, size, value in zip(
            ("AP_small", "AP_medium", "AP_large"), counts, values
        )
    }


def benchmark_model(model: YOLO, config: ExperimentConfig, sample_path: str) -> dict:
    """Comparable Kaggle CUDA timings; CPU users still get AP and parameter count."""
    core = model.model
    parameters = sum(parameter.numel() for parameter in core.parameters())
    try:
        from ultralytics.utils.torch_utils import get_flops

        gflops = float(get_flops(core, imgsz=config.imgsz))
    except Exception:
        gflops = math.nan
    output = {
        "parameters": int(parameters),
        "gflops": gflops,
        "forward_latency_ms_mean": math.nan,
        "forward_latency_ms_std": math.nan,
        "forward_fps": math.nan,
        "pipeline_latency_ms_mean": math.nan,
        "pipeline_latency_ms_std": math.nan,
        "pipeline_fps": math.nan,
        "peak_vram_mb": math.nan,
        "gpu": None,
        "imgsz": config.imgsz,
        "batch": 1,
        "precision": "FP16" if config.amp else "FP32",
        "backend": "PyTorch",
        "benchmark_warmup": config.benchmark_warmup,
        "benchmark_runs": config.benchmark_runs,
        "status": "skipped_no_cuda",
    }
    if not torch.cuda.is_available():
        return output
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    core = core.to("cuda").eval()
    use_half = bool(config.amp)
    core = core.half() if use_half else core.float()
    dummy = torch.zeros(
        1,
        3,
        config.imgsz,
        config.imgsz,
        device="cuda",
        dtype=torch.float16 if use_half else torch.float32,
    )
    forward_times = []
    with torch.inference_mode():
        for _ in range(config.benchmark_warmup):
            core(dummy)
        torch.cuda.synchronize()
        for _ in range(config.benchmark_runs):
            start = time.perf_counter()
            core(dummy)
            torch.cuda.synchronize()
            forward_times.append((time.perf_counter() - start) * 1000)
    for _ in range(config.benchmark_warmup):
        model.predict(
            sample_path,
            imgsz=config.imgsz,
            device=0,
            half=use_half,
            nms=False,
            verbose=False,
        )
    pipeline_times = []
    for _ in range(config.benchmark_runs):
        torch.cuda.synchronize()
        start = time.perf_counter()
        model.predict(
            sample_path,
            imgsz=config.imgsz,
            device=0,
            half=use_half,
            nms=False,
            verbose=False,
        )
        torch.cuda.synchronize()
        pipeline_times.append((time.perf_counter() - start) * 1000)
    forward_mean, pipeline_mean = (
        float(np.mean(forward_times)),
        float(np.mean(pipeline_times)),
    )
    output.update(
        {
            "forward_latency_ms_mean": forward_mean,
            "forward_latency_ms_std": float(np.std(forward_times, ddof=1)),
            "forward_fps": 1000 / forward_mean,
            "pipeline_latency_ms_mean": pipeline_mean,
            "pipeline_latency_ms_std": float(np.std(pipeline_times, ddof=1)),
            "pipeline_fps": 1000 / pipeline_mean,
            "peak_vram_mb": torch.cuda.max_memory_allocated() / 1024**2,
            "gpu": torch.cuda.get_device_name(0),
            "status": "completed",
        }
    )
    return output


def _save(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False)
    frame.to_json(
        path.with_suffix(".json"), orient="records", indent=2, force_ascii=False
    )


def evaluate_checkpoint(
    config: ExperimentConfig, checkpoint: str | Path, dataset: PreparedDataset
) -> dict:
    checkpoint = Path(checkpoint).resolve()
    if not checkpoint.is_file():
        raise FileNotFoundError(checkpoint)
    model = YOLO(str(checkpoint))
    verify_model_graph(model.model, config.architecture)
    if config.architecture != "baseline" and not isinstance(
        model.model.model[11], NECKS[config.architecture]
    ):
        raise RuntimeError("Checkpoint memakai kelas neck yang berbeda")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    eval_dir = checkpoint.parent.parent / "evaluation"
    eval_dir.mkdir(parents=True, exist_ok=True)

    val_predictions = predict_records(model, dataset, config, "val", device)
    threshold, curve = select_threshold(dataset, val_predictions)
    _save(curve, eval_dir / "validation_threshold_curve.csv")
    import matplotlib.pyplot as plt

    figure, axis = plt.subplots(figsize=(8, 4))
    axis.plot(curve["confidence"], curve["macro_f1"])
    axis.axvline(threshold, color="tab:red", linestyle="--")
    axis.set(
        xlabel="Confidence",
        ylabel="Macro F1",
        title="Pemilihan threshold dari validation",
    )
    figure.savefig(
        eval_dir / "validation_macro_f1_threshold.png", dpi=180, bbox_inches="tight"
    )
    plt.close(figure)
    del val_predictions
    test_predictions = predict_records(model, dataset, config, "test", device)
    counts = detection_counts(dataset, "test", test_predictions, threshold)
    per_class_operating = counts_to_frame(counts)
    summary = summarize_counts(counts)
    _save(per_class_operating, eval_dir / "test_operating_point_per_class.csv")

    metrics = model.val(
        data=str(dataset.runtime_yaml),
        split="test",
        imgsz=config.imgsz,
        batch=config.eval_batch,
        conf=config.prediction_conf,
        iou=config.prediction_iou,
        nms=False,
        device=device,
        plots=True,
        project=str(eval_dir),
        name="ultralytics",
        exist_ok=True,
        verbose=False,
    )
    ap50 = np.full(len(CLASS_NAMES), np.nan)
    indices = np.asarray(metrics.box.ap_class_index, dtype=int)
    present_ap50 = np.asarray(metrics.box.ap50, dtype=float)
    if len(indices) == len(present_ap50):
        ap50[indices] = present_ap50
    ap50_95 = np.asarray(metrics.box.maps, dtype=float)
    if len(ap50_95) != len(CLASS_NAMES):
        ap50_95 = np.full(len(CLASS_NAMES), np.nan)
        present = np.asarray(metrics.box.ap, dtype=float)
        if len(indices) == len(present):
            ap50_95[indices] = present
    per_class_ap = pd.DataFrame(
        {
            "class_id": range(len(CLASS_NAMES)),
            "class_name": CLASS_NAMES,
            "AP50": ap50,
            "AP50_95": ap50_95,
        }
    ).merge(
        per_class_operating[["class_id", "ground_truth"]], on="class_id", how="left"
    )
    _save(per_class_ap, eval_dir / "test_per_class_ap.csv")

    coco_gt, coco_predictions = build_coco_eval_canvas(
        dataset, "test", test_predictions, config.imgsz
    )
    scale = coco_scale_eval(coco_gt, coco_predictions)
    (eval_dir / "coco_ground_truth.json").write_text(
        json.dumps(coco_gt), encoding="utf-8"
    )
    (eval_dir / "coco_predictions.json").write_text(
        json.dumps(coco_predictions), encoding="utf-8"
    )
    (eval_dir / "coco_scale_metrics.json").write_text(
        json.dumps(scale, indent=2), encoding="utf-8"
    )
    sample_path = str(image_records(dataset, "test")[0].image)
    save_fusion_diagnostics(model, config, sample_path, eval_dir, device)
    efficiency = benchmark_model(model, config, sample_path)

    overall = {
        "architecture": config.architecture,
        "seed": config.seed,
        "confidence_from_validation": threshold,
        "mAP50_95": float(metrics.box.map),
        "mAP50": float(metrics.box.map50),
        **summary,
        **scale,
        "dataset_fingerprint": dataset.fingerprint,
        "checkpoint": str(checkpoint),
    }
    (eval_dir / "overall_metrics.json").write_text(
        json.dumps(overall, indent=2), encoding="utf-8"
    )
    (eval_dir / "efficiency.json").write_text(
        json.dumps(efficiency, indent=2), encoding="utf-8"
    )
    label = (
        "YOLO26n FPN-PAN"
        if config.architecture == "baseline"
        else f"YOLO26n-{config.architecture.upper()}"
    )
    _save(
        pd.DataFrame(
            [
                {
                    "Model": label,
                    "Seed": config.seed,
                    "mAP50-95": overall["mAP50_95"],
                    "mAP50": overall["mAP50"],
                    "Precision": overall["precision"],
                    "Recall": overall["recall"],
                    "F1-score": overall["f1"],
                }
            ]
        ),
        eval_dir / "tabel_3_5_kinerja_keseluruhan.csv",
    )
    table_36 = (
        per_class_ap.set_index("class_name")["AP50_95"]
        .reindex(CLASS_NAMES)
        .to_frame()
        .T
    )
    table_36.insert(0, "Model", label)
    table_36.insert(1, "Seed", config.seed)
    _save(table_36, eval_dir / "tabel_3_6_ap_per_kelas.csv")
    _save(
        pd.DataFrame(
            [
                {
                    "Model": label,
                    "Seed": config.seed,
                    "AP-S": scale["AP_small"],
                    "AP-M": scale["AP_medium"],
                    "AP-L": scale["AP_large"],
                }
            ]
        ),
        eval_dir / "tabel_3_7_ap_berdasarkan_skala.csv",
    )
    _save(
        pd.DataFrame(
            [
                {
                    "Model": label,
                    "Seed": config.seed,
                    "Parameters": efficiency["parameters"],
                    "GFLOPs": efficiency["gflops"],
                    "Training time (s)": _training_seconds(checkpoint),
                    "Latency (ms)": efficiency["forward_latency_ms_mean"],
                    "FPS": efficiency["forward_fps"],
                }
            ]
        ),
        eval_dir / "tabel_3_8_efisiensi.csv",
    )
    return {"overall": overall, "efficiency": efficiency, "evaluation_dir": eval_dir}


def _training_seconds(checkpoint: Path) -> float:
    metadata = checkpoint.parent.parent / "run_metadata.json"
    if not metadata.exists():
        return math.nan
    return float(
        json.loads(metadata.read_text(encoding="utf-8")).get("duration_seconds")
        or math.nan
    )
