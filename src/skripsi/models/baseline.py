"""Unmodified YOLO26n FPN-PAN control with the shared six-class head transfer."""

from __future__ import annotations

from copy import deepcopy

from ultralytics import YOLO
from ultralytics.nn.tasks import DetectionModel

from ..config import CLASS_NAMES, ExperimentConfig


def build_base_core(config: ExperimentConfig) -> DetectionModel:
    """Match the notebook baseline's pretrained-to-six-class loading protocol."""
    source = YOLO(config.model_weights).model
    if source.model[-1].f != [16, 19, 22]:
        raise RuntimeError(f"YOLO26n Detect source berubah: {source.model[-1].f}")
    target = DetectionModel(
        cfg=deepcopy(source.yaml), ch=3, nc=len(CLASS_NAMES), verbose=False
    )
    target.names = dict(enumerate(CLASS_NAMES))
    target.load(source, verbose=False)
    return target


def verify_baseline_graph(core: DetectionModel) -> None:
    if core.model[-1].nc != len(CLASS_NAMES) or core.model[-1].f != [16, 19, 22]:
        raise RuntimeError("Graph FPN-PAN baseline berubah")
