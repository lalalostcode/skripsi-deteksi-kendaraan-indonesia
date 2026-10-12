"""Offline checks only: no train(), optimizer, Roboflow, or W&B calls."""

from __future__ import annotations

import json
import time
import uuid
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from PIL import Image
from ultralytics import YOLO

from skripsi.analysis import summarize_experiments
from skripsi.config import CLASS_NAMES, ExperimentConfig
from skripsi.data import prepare_dataset
from skripsi.diagnostics import save_fusion_diagnostics
from skripsi.evaluation import (
    build_coco_eval_canvas,
    coco_scale_eval,
    match_class_boxes,
    select_threshold,
)
from skripsi.evaluation import (
    evaluate_checkpoint as evaluate_synthetic_checkpoint,
)
from skripsi.models.factory import build_model, build_model_core, detect_input_shapes
from skripsi.pipeline import (
    ModularDetectionTrainer,
    _epoch_metrics,
    _WandbEpochLogger,
    evaluate_checkpoint,
)


@pytest.fixture
def tmp_path():
    # Pytest's Windows 0700 temp dirs are inaccessible in this managed workspace.
    path = Path.cwd() / "outputs" / "test_modular" / uuid.uuid4().hex
    path.mkdir(parents=True)
    return path


@pytest.fixture
def tiny_dataset(tmp_path: Path):
    source = tmp_path / "source-v2"
    colors = {"train": "red", "valid": "green", "test": "blue"}
    labels = {
        "train": "0 0.1 0.1 0.3 0.1 0.2 0.3\n",
        "valid": "",
        "test": "0 0.5 0.5 0.2 0.2\n",
    }
    for split, color in colors.items():
        (source / split / "images").mkdir(parents=True)
        (source / split / "labels").mkdir(parents=True)
        Image.new("RGB", (100, 100), color=color).save(
            source / split / "images" / f"{split}.png"
        )
        (source / split / "labels" / f"{split}.txt").write_text(
            labels[split], encoding="utf-8"
        )
    (source / "data.yaml").write_text(
        "names:\n"
        + "".join(f"- {name}\n" for name in CLASS_NAMES)
        + "nc: 6\nroboflow:\n  version: 2\ntrain: ../train/images\nval: ../valid/images\ntest: ../test/images\n",
        encoding="utf-8",
    )
    config = ExperimentConfig(
        dataset_dir=source, output_root=tmp_path / "out", model_weights="yolo26n.yaml"
    )
    return source, config


def test_working_copy_audit_and_evaluator(tiny_dataset):
    source, config = tiny_dataset
    dataset = prepare_dataset(config, strict_v2=False)
    assert (source / "train" / "labels" / "train.txt").read_text(
        encoding="utf-8"
    ).count(" ") == 6
    converted = dataset.root / "train" / "labels" / "train.txt"
    assert len(converted.read_text(encoding="utf-8").split()) == 5
    assert len(dataset.audit) == 3
    assert set(dataset.audit.status) == {"ok", "empty_label"}
    assert dataset.fingerprint == prepare_dataset(config, strict_v2=False).fingerprint
    conversion_report = config.reports_dir / "polygon_conversion.csv"
    assert len(conversion_report.read_text(encoding="utf-8").splitlines()) == 2

    path = str((dataset.root / "valid" / "images" / "valid.png").resolve())
    threshold, curve = select_threshold(dataset, {path: []})
    assert threshold == pytest.approx(0.01)
    assert len(curve) == 99
    assert match_class_boxes(
        [(10, 10, 30, 30)], [{"xyxy": [10, 10, 30, 30], "conf": 0.9}], 0.5
    ) == (1, 0, 0)
    test_path = str((dataset.root / "test" / "images" / "test.png").resolve())
    gt, predictions = build_coco_eval_canvas(
        dataset,
        "test",
        {test_path: [{"cls": 0, "conf": 0.9, "xyxy": [40, 40, 60, 60]}]},
    )
    assert len(gt["annotations"]) == len(predictions) == 1
    assert gt["annotations"][0]["bbox"] == predictions[0]["bbox"]
    assert gt["images"][0]["width"] == 640


def test_public_evaluation_rejects_nonresearch_dataset(tiny_dataset):
    _source, config = tiny_dataset
    dataset = prepare_dataset(config, strict_v2=False)
    with pytest.raises(RuntimeError, match="snapshot dataset v2"):
        evaluate_checkpoint(config, config.run_dir / "weights" / "best.pt", dataset)


def test_cross_architecture_summary_waits_for_all_seeds(tmp_path: Path):
    root = tmp_path / "research"
    assert summarize_experiments(root)["status"] == "not_run"
    from skripsi.config import EXPECTED_V2_FINGERPRINT, SEEDS

    for architecture_index, architecture in enumerate(
        ("baseline", "bifpn", "asff", "afpn")
    ):
        for seed_index, seed in enumerate(SEEDS):
            evaluation = (
                root
                / "runs"
                / f"yolo26n-{architecture}-train-seed-{seed}"
                / "evaluation"
            )
            evaluation.mkdir(parents=True)
            (evaluation / "overall_metrics.json").write_text(
                json.dumps(
                    {
                        "architecture": architecture,
                        "seed": seed,
                        "dataset_fingerprint": EXPECTED_V2_FINGERPRINT,
                        "mAP50_95": 0.1 + architecture_index * 0.05 + seed_index * 0.01,
                        "mAP50": 0.2,
                        "precision": 0.3,
                        "recall": 0.4,
                        "f1": 0.34,
                    }
                ),
                encoding="utf-8",
            )
            (evaluation / "efficiency.json").write_text(
                json.dumps({"gpu": "T4", "precision": "FP16", "imgsz": 640}),
                encoding="utf-8",
            )
    assert summarize_experiments(root)["status"] == "completed"
    assert (root / "reports" / "hasil_friedman.csv").is_file()


@pytest.mark.parametrize("architecture", ["baseline", "bifpn", "asff", "afpn"])
def test_model_forward_and_checkpoint_roundtrip(architecture: str, tmp_path: Path):
    config = ExperimentConfig(
        architecture=architecture, model_weights="yolo26n.yaml", output_root=tmp_path
    )
    model = build_model_core(config)
    assert model.model[-1].nc == 6
    assert detect_input_shapes(model, 640) == [
        (1, 64, 80, 80),
        (1, 128, 40, 40),
        (1, 256, 20, 20),
    ]
    checkpoint = tmp_path / f"{architecture}.pt"
    torch.save(
        {"model": model, "train_args": {"task": "detect", "imgsz": 640}}, checkpoint
    )
    restored = YOLO(str(checkpoint)).model
    assert type(restored.model[-1]) is type(model.model[-1])
    assert restored.model[-1].nc == 6


def test_new_notebooks_parse_and_do_not_train_by_default():
    notebooks = sorted(Path("notebooks/modular").glob("*.ipynb"))
    assert len(notebooks) == 4
    for path in notebooks:
        notebook = json.loads(path.read_text(encoding="utf-8"))
        cells = [
            "".join(cell["source"])
            for cell in notebook["cells"]
            if cell["cell_type"] == "code"
        ]
        assert any("RUN_TRAINING = False" in source for source in cells)
        assert any("RUN_EVALUATION = False" in source for source in cells)
        assert all(
            cell["execution_count"] is None and not cell["outputs"]
            for cell in notebook["cells"]
            if cell["cell_type"] == "code"
        )


def test_local_notebook_bootstrap(monkeypatch):
    root = Path(__file__).resolve().parents[1]
    monkeypatch.chdir(root)
    notebook = json.loads(
        (root / "notebooks" / "modular" / "03_yolo26n_bifpn.ipynb").read_text(
            encoding="utf-8"
        )
    )
    namespace: dict = {}
    for index in (2, 4):  # Bootstrap and configuration only.
        source = "".join(notebook["cells"][index]["source"])
        exec(
            compile(
                source,
                str(root / "notebooks" / "modular" / "03_yolo26n_bifpn.ipynb"),
                "exec",
            ),
            namespace,
        )
    assert namespace["config"].architecture == "bifpn"
    assert namespace["config"].dataset_version == 2
    assert namespace["config"].git_commit


def test_coco_scale_evaluator_perfect_predictions():
    areas = (16, 64, 128)
    ground_truth = {
        "info": {},
        "licenses": [],
        "images": [
            {"id": index, "file_name": f"{index}.png", "width": 200, "height": 200}
            for index in range(1, 4)
        ],
        "annotations": [
            {
                "id": index,
                "image_id": index,
                "category_id": 1,
                "bbox": [0, 0, width, width],
                "area": width**2,
                "iscrowd": 0,
            }
            for index, width in enumerate(areas, 1)
        ],
        "categories": [{"id": 1, "name": "objek"}],
    }
    predictions = [
        {
            "image_id": item["image_id"],
            "category_id": 1,
            "bbox": item["bbox"],
            "score": 0.99,
        }
        for item in ground_truth["annotations"]
    ]
    scores = coco_scale_eval(ground_truth, predictions)
    assert all(value > 0.99 for value in scores.values())


def test_custom_trainer_keeps_graph_without_training(tmp_path: Path):
    config = ExperimentConfig(
        architecture="asff", model_weights="yolo26n.yaml", output_root=tmp_path
    )
    template = build_model_core(config)
    ModularDetectionTrainer.MODEL_TEMPLATE = template
    ModularDetectionTrainer.ARCHITECTURE = "asff"
    try:
        trainer = object.__new__(ModularDetectionTrainer)
        trainer.data = {"nc": 6, "names": dict(enumerate(CLASS_NAMES))}
        restored = trainer.get_model()
        assert restored is not template
        assert type(restored.model[11]).__name__ == "ASFFNeck"
        assert restored.model[-1].f == [12, 13, 14]
    finally:
        ModularDetectionTrainer.MODEL_TEMPLATE = None
        ModularDetectionTrainer.ARCHITECTURE = None


def test_wandb_epoch_logger_includes_train_val_and_skips_final_eval():
    logged = []
    run = SimpleNamespace(log=logged.append)
    trainer = SimpleNamespace(
        epoch=2,
        train_time_start=time.time() - 10,
        epoch_time=3.5,
        tloss={
            "box_loss": torch.tensor(0.7),
            "cls_loss": torch.tensor(0.4),
            "dfl_loss": torch.tensor(0.2),
        },
        metrics={
            "metrics/precision(B)": 0.8,
            "metrics/recall(B)": 0.75,
            "metrics/mAP50(B)": 0.6,
            "metrics/mAP50-95(B)": 0.45,
            "val/box_loss": 0.5,
            "val/cls_loss": 0.3,
            "val/dfl_loss": 0.15,
            "unsupported": "not a metric",
        },
        lr={"lr/pg0": 0.001, "lr/pg1": 0.002},
        fitness=0.65,
        label_loss_items=lambda losses: {
            f"train/{key}": float(value) for key, value in losses.items()
        },
    )
    expected = _epoch_metrics(trainer)
    logger = _WandbEpochLogger(run)
    logger.on_train_epoch_start(trainer)
    logger.on_fit_epoch_end(trainer)
    assert len(logged) == 1
    assert logged[0].keys() == expected.keys()
    assert all(
        logged[0][key] == value for key, value in expected.items() if key != "time"
    )
    assert expected["epoch"] == 3
    assert expected["time"] >= 10
    assert expected["train/box_loss"] == pytest.approx(0.7)
    assert expected["train/cls_loss"] == pytest.approx(0.4)
    assert expected["train/dfl_loss"] == pytest.approx(0.2)
    assert expected["val/metrics/precision(B)"] == 0.8
    assert expected["val/metrics/recall(B)"] == 0.75
    assert expected["val/metrics/mAP50(B)"] == 0.6
    assert expected["val/metrics/mAP50-95(B)"] == 0.45
    assert expected["val/box_loss"] == 0.5
    assert expected["val/cls_loss"] == 0.3
    assert expected["val/dfl_loss"] == 0.15
    assert expected["val/fitness"] == 0.65
    assert expected["lr/pg0"] == 0.001
    assert expected["lr/pg1"] == 0.002
    assert expected["train/epoch_time"] == 3.5
    assert "val/unsupported" not in expected

    trainer.epoch += 1  # Ultralytics final_eval calls on_fit_epoch_end again.
    logger.on_fit_epoch_end(trainer)
    assert len(logged) == 1


@pytest.mark.slow
def test_checkpoint_evaluation_smoke_without_training(tiny_dataset):
    _source, config = tiny_dataset
    dataset = prepare_dataset(config, strict_v2=False)
    model = build_model_core(config)
    checkpoint = config.run_dir / "weights" / "best.pt"
    checkpoint.parent.mkdir(parents=True)
    torch.save(
        {"model": model, "train_args": {"task": "detect", "imgsz": 640}}, checkpoint
    )
    (config.run_dir / "run_metadata.json").write_text(
        json.dumps(
            {
                "architecture": config.architecture,
                "dataset_fingerprint": dataset.fingerprint,
                "duration_seconds": 0,
            }
        ),
        encoding="utf-8",
    )
    with pytest.raises(RuntimeError, match="snapshot dataset v2"):
        evaluate_checkpoint(config, checkpoint, dataset)
    result = evaluate_synthetic_checkpoint(config, checkpoint, dataset)
    assert (result["evaluation_dir"] / "tabel_3_5_kinerja_keseluruhan.csv").is_file()
    assert (result["evaluation_dir"] / "tabel_3_7_ap_berdasarkan_skala.csv").is_file()
    assert result["overall"]["dataset_fingerprint"] == dataset.fingerprint


@pytest.mark.slow
@pytest.mark.parametrize(
    "architecture,expected_rows", [("bifpn", 9), ("asff", 9), ("afpn", 13)]
)
def test_fusion_diagnostics_without_training(
    tiny_dataset, architecture: str, expected_rows: int
):
    _source, base = tiny_dataset
    config = ExperimentConfig(
        architecture=architecture,
        model_weights="yolo26n.yaml",
        output_root=base.output_root,
    )
    model = build_model(config)
    sample = base.dataset_dir / "test" / "images" / "test.png"
    frame = save_fusion_diagnostics(
        model, config, sample, config.output_root / architecture, "cpu"
    )
    assert frame is not None and len(frame) == expected_rows
    assert (config.output_root / architecture / "fusion_weights.json").exists()
