"""Architecture-specific fusion diagnostics kept outside evaluation logic."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import torch
from ultralytics import YOLO

from .config import ExperimentConfig
from .models.afpn import AFPNNeck
from .models.asff import ASFFNeck
from .models.bifpn import BiFPNNeck


def save_fusion_diagnostics(
    model: YOLO,
    config: ExperimentConfig,
    sample_path: str | Path,
    output_dir: Path,
    device: str,
) -> pd.DataFrame | None:
    """BiFPN global weights or one held-out image's ASFF/AFPN spatial weights."""
    if config.architecture == "baseline":
        return None
    neck = model.model.model[11]
    output_dir.mkdir(parents=True, exist_ok=True)
    if config.architecture == "bifpn":
        if not isinstance(neck, BiFPNNeck):
            raise TypeError(type(neck).__name__)
        modules = {
            "P4_top_down": neck.fuse_p4_td,
            "P3_top_down": neck.fuse_p3_td,
            "P4_bottom_up": neck.fuse_p4_out,
            "P5_bottom_up": neck.fuse_p5_out,
        }
        rows = [
            {"fusion_node": label, "source_index": index, "weight": float(value)}
            for label, module in modules.items()
            for index, value in enumerate(
                module.normalized_weights().detach().float().cpu().tolist()
            )
        ]
        frame = pd.DataFrame(rows)
        frame.to_csv(output_dir / "fusion_weights.csv", index=False)
        (output_dir / "fusion_weights.json").write_text(
            json.dumps(rows, indent=2), encoding="utf-8"
        )
        return frame

    if config.architecture == "asff":
        if not isinstance(neck, ASFFNeck):
            raise TypeError(type(neck).__name__)
        modules = {
            "P3_target": neck.asff_p3.weight_levels,
            "P4_target": neck.asff_p4.weight_levels,
            "P5_target": neck.asff_p5.weight_levels,
        }
        labels = {key: ("P3", "P4", "P5") for key in modules}
    else:
        if not isinstance(neck, AFPNNeck):
            raise TypeError(type(neck).__name__)
        body = neck.body
        modules = {
            "stage1_P3": body.fuse_p3_s1.weight_levels,
            "stage1_P4": body.fuse_p4_s1.weight_levels,
            "stage2_P3": body.fuse_p3_s2.weight_levels,
            "stage2_P4": body.fuse_p4_s2.weight_levels,
            "stage2_P5": body.fuse_p5_s2.weight_levels,
        }
        labels = {
            "stage1_P3": ("P3", "P4"),
            "stage1_P4": ("P3", "P4"),
            "stage2_P3": ("P3_stage1", "P4_stage1", "P5"),
            "stage2_P4": ("P3_stage1", "P4_stage1", "P5"),
            "stage2_P5": ("P3_stage1", "P4_stage1", "P5"),
        }
    captured: dict[str, torch.Tensor] = {}
    handles = []

    def make_hook(label: str):
        def hook(_module, _inputs, logits):
            captured[label] = torch.softmax(logits.detach().float(), dim=1).cpu()

        return hook

    try:
        for label, module in modules.items():
            handles.append(module.register_forward_hook(make_hook(label)))
        model.predict(
            str(sample_path),
            imgsz=config.imgsz,
            conf=config.prediction_conf,
            iou=config.prediction_iou,
            nms=False,
            device=device,
            batch=1,
            verbose=False,
        )
    finally:
        for handle in handles:
            handle.remove()
    if set(captured) != set(modules):
        raise RuntimeError(
            f"Fusion hook hilang: {sorted(set(modules) - set(captured))}"
        )
    rows = []
    summary = {}
    for node, weights in captured.items():
        means = weights.mean(dim=(0, 2, 3)).numpy()
        stds = weights.std(dim=(0, 2, 3), unbiased=False).numpy()
        error = float((weights.sum(dim=1) - 1).abs().max().item())
        summary[node] = {"max_softmax_sum_error": error}
        for source, mean, std in zip(labels[node], means, stds):
            rows.append(
                {
                    "fusion_node": node,
                    "source_scale": source,
                    "mean_weight": float(mean),
                    "std_weight": float(std),
                }
            )
            summary[node][source] = {"mean": float(mean), "std": float(std)}
    frame = pd.DataFrame(rows)
    frame.to_csv(output_dir / "fusion_weights.csv", index=False)
    (output_dir / "fusion_weights.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    return frame
