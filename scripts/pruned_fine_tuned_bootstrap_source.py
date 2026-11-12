"""Bootstrap declarations embedded in the fine-tuning notebooks."""
# ruff: noqa: F401, F821, I001 -- Imports and configuration are shared across embedded cells.
import hashlib
import csv as _eda_csv
import json
import json as _eda_json
import math
import logging
import re
import sys
import time
import uuid
from contextlib import contextmanager
from datetime import UTC, datetime
from importlib import metadata
from pathlib import Path as _EdaPath
from subprocess import run as run_command
from types import SimpleNamespace
from unittest.mock import patch


if bool(RAW_ARTIFACT_REF) == bool(RAW_MANIFEST_PATH):
    raise ValueError("Isi tepat satu input raw.")
RUN_DIR = OUTPUT_DIR.expanduser().resolve() / f"{datetime.now(UTC):%Y%m%dT%H%M%SZ}_{uuid.uuid4().hex[:8]}"
RUN_DIR.mkdir(parents=True, exist_ok=False)

PINS = {"ultralytics": "8.4.155", "wandb": "0.28.1"}
PRESERVED_TORCH = {name: metadata.version(name) for name in ("torch", "torchvision")}
needed = [f"{package}=={version}" for package, version in PINS.items()
          if not any(dist.version == version for dist in metadata.distributions(name=package))]
if needed:
    if not INSTALL_DEPENDENCIES:
        raise RuntimeError(f"Instal dependency baseline di kernel ini: {needed}")
    protected = RUN_DIR / "preserve_torch.txt"
    protected.write_text("".join(f"{name}=={version}\n" for name, version in PRESERVED_TORCH.items()), encoding="utf-8")
    run_command([sys.executable, "-m", "pip", "install", "--constraint", str(protected), *needed], check=True)
    if any(metadata.version(name) != version for name, version in PRESERVED_TORCH.items()):
        raise RuntimeError("Versi Torch/torchvision kernel berubah selama instalasi.")

import matplotlib.pyplot as _eda_plt
import numpy as _eda_np
import numpy as np
import pandas as pd
import torch
import ultralytics
import wandb
import yaml
from thop import profile as thop_profile
from matplotlib.colors import ListedColormap as _EdaColors
from ultralytics import YOLO
from ultralytics.models.yolo.detect import DetectionTrainer
from ultralytics.nn.modules import Detect
from ultralytics.nn.modules.block import AAttn, Attention
from ultralytics.nn.tasks import DetectionModel as _RecoveryDetectionModel
from ultralytics.utils.torch_utils import _attention_ops

_recovery_torch = torch
_recovery_hashlib = hashlib
_eda_math = math
if (ultralytics.__version__, wandb.__version__) != tuple(PINS.values()):
    raise RuntimeError("Restart kernel setelah memasang versi dependency baseline.")
if MODE not in {"full_val", "smoke"} or SEED not in {42, 0, 2024, 7, 3407}:
    raise ValueError("Pilih mode full_val/smoke dan satu seed yang tersedia pada baseline.")
if DEVICE != "cpu" and not torch.cuda.is_available():
    raise RuntimeError("Pilih kernel GPU atau gunakan DEVICE='cpu' dalam smoke test lokal.")
if MODE == "full_val" and DEVICE == "cpu":
    raise RuntimeError("Full validation protokol baseline memerlukan GPU.")

RECOVERY_EPOCHS = 1 if MODE == "smoke" else EPOCHS
POLTRADA_V3_ID = "ilhamrafiqin/poltrada-v3-yolo26-clean"
POLTRADA_V3_MOUNT = Path("/kaggle/input/datasets") / POLTRADA_V3_ID
DATASET_ID = globals().get("DATASET_ID", POLTRADA_V3_ID)
BASELINE_REF = "ilhamrafiqin-brawijaya-university/yolom26-kd-indotraffic/02_baseline_yolo26s-q5kp9k9d-checkpoints:v62"
BASELINE_SHA256 = "5bc3ae61f8da1c9f03916922de10146805c27c64a11e46edd766a00d0f6f18fd"

ENVIRONMENT = {"python": sys.version.split()[0], "kernel": sys.executable, "torch": torch.__version__,
               "torchvision": metadata.version("torchvision"), "ultralytics-thop": metadata.version("ultralytics-thop"),
               **PINS, "cuda": torch.version.cuda,
               "gpu": torch.cuda.get_device_name(DEVICE) if DEVICE != "cpu" else None,
               "dataset_root": str(DATASET_ROOT), "output": str(RUN_DIR)}
print(json.dumps(ENVIRONMENT, indent=2, ensure_ascii=False))


def sha256_file(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()
