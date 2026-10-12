"""Construct the four models with one pretrained six-class head protocol."""

from __future__ import annotations

from copy import deepcopy

import torch
from torch import nn
from ultralytics import YOLO
from ultralytics.nn.tasks import DetectionModel

from ..config import CLASS_NAMES, ExperimentConfig
from .afpn import AFPNNeck, FeatureSelect as AFPNSelect
from .asff import ASFFNeck, FeatureSelect as ASFFSelect
from .baseline import build_base_core, verify_baseline_graph
from .bifpn import BiFPNNeck, FeatureSelect as BiFPNSelect

NECKS = {"bifpn": BiFPNNeck, "asff": ASFFNeck, "afpn": AFPNNeck}
SELECTORS = {"bifpn": BiFPNSelect, "asff": ASFFSelect, "afpn": AFPNSelect}
EXPECTED_BACKBONE_CHANNELS = (128, 128, 256)
EXPECTED_DETECT_CHANNELS = (64, 128, 256)
BACKBONE_SOURCES = (4, 6, 10)


def set_seed(seed: int) -> None:
    import random
    import numpy as np

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _metadata(module: nn.Module, index: int, source: int | list[int]) -> nn.Module:
    module.i = index
    module.f = source
    module.type = module.__class__.__name__
    module.np = sum(p.numel() for p in module.parameters())
    return module


def build_model_core(config: ExperimentConfig) -> DetectionModel:
    """Return a fresh seed-specific six-class YOLO26n core."""
    set_seed(config.seed)
    base = build_base_core(config)
    if config.architecture == "baseline":
        return base

    old = list(base.model)
    neck_args = {
        "bifpn": dict(
            in_channels=EXPECTED_BACKBONE_CHANNELS,
            fusion_channels=128,
            out_channels=EXPECTED_DETECT_CHANNELS,
        ),
        "asff": dict(
            in_channels=EXPECTED_BACKBONE_CHANNELS,
            out_channels=EXPECTED_DETECT_CHANNELS,
            weight_channels=16,
        ),
        "afpn": dict(
            in_channels=EXPECTED_BACKBONE_CHANNELS,
            out_channels=EXPECTED_DETECT_CHANNELS,
            reduction_ratio=4,
            compress_channels=8,
            residual_blocks=3,
        ),
    }
    neck = _metadata(
        NECKS[config.architecture](**neck_args[config.architecture]),
        11,
        list(BACKBONE_SOURCES),
    )
    selector = SELECTORS[config.architecture]
    selects = [_metadata(selector(i), 12 + i, 11) for i in range(3)]
    detect = deepcopy(old[-1])
    detect.i = 15
    detect.f = [12, 13, 14]
    detect.type = detect.__class__.__name__
    detect.np = sum(p.numel() for p in detect.parameters())
    core = deepcopy(base)
    core.model = nn.Sequential(*(deepcopy(old[:11]) + [neck, *selects, detect]))
    core.save = sorted(
        set([i for i in base.save if i <= 10] + [4, 6, 10, 11, 12, 13, 14])
    )
    core.names = deepcopy(base.names)
    core.stride = deepcopy(base.stride)
    verify_model_graph(core, config.architecture)
    return core


def verify_model_graph(core: DetectionModel, architecture: str) -> None:
    if core.model[-1].nc != len(CLASS_NAMES):
        raise RuntimeError("Jumlah kelas detection head berubah")
    if architecture == "baseline":
        verify_baseline_graph(core)
    elif not isinstance(core.model[11], NECKS[architecture]) or core.model[-1].f != [
        12,
        13,
        14,
    ]:
        raise RuntimeError(f"Graph {architecture} tidak sesuai")


def build_model(config: ExperimentConfig) -> YOLO:
    """Public model builder; never starts training or evaluation."""
    core = build_model_core(config)
    wrapper = YOLO(config.model_weights)
    wrapper.model = core
    wrapper.task = "detect"
    return wrapper


def detect_input_shapes(
    core: DetectionModel, imgsz: int = 640, device: str = "cpu"
) -> list[tuple[int, ...]]:
    """One synthetic inference forward; no backward pass or optimizer."""
    model = deepcopy(core).to(device).eval()
    shapes: list[tuple[int, ...]] = []

    def capture(_module, inputs):
        shapes.extend(tuple(feature.shape) for feature in inputs[0])

    handle = model.model[-1].register_forward_pre_hook(capture)
    try:
        with torch.inference_mode():
            model(torch.zeros(1, 3, imgsz, imgsz, device=device))
    finally:
        handle.remove()
        del model
    expected = [(1, 64, 80, 80), (1, 128, 40, 40), (1, 256, 20, 20)]
    if shapes != expected:
        raise RuntimeError(f"Detect input shapes {shapes} != {expected}")
    return shapes
