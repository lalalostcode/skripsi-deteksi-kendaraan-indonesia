"""Small offline checks of the exact notebook cells; CUDA results are separate."""
from __future__ import annotations

import ast
import copy
import hashlib
import importlib.metadata
import json
import math
import shutil
import subprocess
import sys
import tempfile
import time
import types
import uuid
from pathlib import Path

import numpy as np
import pytest
import torch
import yaml
from PIL import Image
from ultralytics import YOLO, settings
from ultralytics.models.yolo.detect import DetectionTrainer
from ultralytics.nn.tasks import DetectionModel
from ultralytics.utils.torch_utils import ModelEMA, unwrap_model

ROOT = Path(__file__).resolve().parents[1]
FOLDER = ROOT / "notebooks/yolo26-kd-indotraffic/notebooks"
NAMES = ["Bus", "Mobil Penumpang", "Pejalan Kaki", "Sepeda Motor", "Truck", "Unmotorized"]


class OfflineArtifact:
    def __init__(self, name, type, metadata):
        self.name, self.type, self.metadata = name, type, metadata
        self.files = {}

    def add_file(self, path, name):
        self.files[name] = Path(path)


class OfflineRun:
    def __init__(self, identifier):
        self.id, self.summary = identifier, {}
        self.uploads, self.logs = [], []

    def log_artifact(self, artifact, aliases):
        def wait():
            # Read only when the SDK would finish its asynchronous upload.
            assert all(path.is_file() for path in artifact.files.values())
            self.uploads.append((artifact.metadata, {k: p.stat().st_size for k, p in artifact.files.items()}))
        return types.SimpleNamespace(wait=wait)

    def log(self, values, step):
        self.logs.append(dict(values))

    def finish(self, exit_code=0):
        self.exit_code = exit_code


def environment(method):
    number = 13 if method == "head" else 14
    path = FOLDER / f"{number}_{method}_yolo26n_teacher_yolo26s.ipynb"
    notebook = json.loads(path.read_bytes())
    cells = {c["id"]: "".join(c["source"]) for c in notebook["cells"]}
    module = types.ModuleType("clean_kaggle_" + method + uuid.uuid4().hex)
    sys.modules[module.__name__] = module
    module.__dict__.update(torch=torch, nn=torch.nn, YOLO=YOLO, DetectionTrainer=DetectionTrainer,
        DetectionModel=DetectionModel, ModelEMA=ModelEMA, unwrap_model=unwrap_model,
        copy=copy, hashlib=hashlib, json=json, math=math, shutil=shutil, tempfile=tempfile,
        time=time, uuid=uuid, Path=Path, metadata=importlib.metadata)
    for name in ("config", "helpers", "model", "numerical", "trainer", "logging", "training"):
        exec(compile(cells[method + "-" + name], str(path), "exec"), module.__dict__)  # noqa: S102 -- Trusted local notebook declarations.
    return module, notebook


@pytest.mark.parametrize("method", ["head", "rm"])
def test_objective_is_preserved_and_default_training_is_off(method):
    module, notebook = environment(method)
    assert module.RUN_FULL_TRAINING is False
    assert module.BATCH == 16 and module.IMGSZ == 640 and module.SEED == 42
    for cell in notebook["cells"]:
        if cell["cell_type"] == "code":
            compile("".join(cell["source"]), cell["id"], "exec")
    result = (module.numerical_verification() if method == "head" else module.rm_numerical_verification())
    assert result
    old_path = ROOT / f"tmp/head-rm-worktree/scripts/head_rm/{method}.py"
    if old_path.is_file():
        old = ast.parse(old_path.read_text(encoding="utf-8"))
        current = ast.parse("\n".join("".join(c["source"]) for c in notebook["cells"] if c["cell_type"] == "code"))
        name = "HEADModel" if method == "head" else "RMModel"
        old_class = next(n for n in old.body if isinstance(n, ast.ClassDef) and n.name == name)
        new_class = next(n for n in current.body if isinstance(n, ast.ClassDef) and n.name == name)
        assert ast.dump(old_class) == ast.dump(new_class)


def test_fused_adamw_amp_overflow_is_skipped_before_counting_real_updates():
    module, _ = environment("head")
    model = torch.nn.Module()
    model.student_model = torch.nn.Linear(2, 1)
    model.teacher_model = torch.nn.Linear(2, 1).requires_grad_(False)
    optimizer = torch.optim.AdamW(model.student_model.parameters(), lr=0.001, fused=True)
    scaler = torch.amp.GradScaler("cpu", init_scale=8)
    trainer = types.SimpleNamespace(model=model, optimizer=optimizer, device=torch.device("cpu"))
    callbacks, context = module.make_callbacks(OfflineRun("amp"), diagnostic=True)
    callbacks["on_train_start"](trainer)
    before = copy.deepcopy(model.student_model.state_dict())
    try:
        for multiplier in (float("inf"), 1.0):
            optimizer.zero_grad()
            loss = model.student_model(torch.ones(1, 2)).sum() * multiplier
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 10)
            scaler.step(optimizer)
            scaler.update()
            if math.isinf(multiplier):
                assert context["optimizer_steps"] == 0
                assert context["amp_skipped_steps"] == 1
                assert all(torch.equal(before[k], v) for k, v in model.student_model.state_dict().items())
        assert context["optimizer_steps"] == 1 and context["student_grad"] > 0
        assert any(not torch.equal(before[k], v) for k, v in model.student_model.state_dict().items())
    finally:
        context["gradient_hook"].remove()


@pytest.mark.parametrize("method", ["head", "rm"])
def test_actual_one_epoch_cpu_trainer_validation_and_plain_checkpoint(method):
    previous_threads = torch.get_num_threads()
    previous_settings = {k: settings[k] for k in ("wandb", "mlflow", "comet")}
    torch.set_num_threads(2)
    dict.update(settings, {k: False for k in previous_settings})
    try:
        directory = ROOT / "outputs/head_rm_clean_cpu" / uuid.uuid4().hex
        directory.mkdir(parents=True)
        for split, count in (("train", 4), ("val", 2)):
            for index in range(count):
                image = directory / "images" / split / f"{index}.png"
                label = directory / "labels" / split / f"{index}.txt"
                image.parent.mkdir(parents=True, exist_ok=True)
                label.parent.mkdir(parents=True, exist_ok=True)
                Image.fromarray(np.random.default_rng(index).integers(0, 255, (64, 64, 3), dtype=np.uint8)).save(image)
                label.write_text("0 .5 .5 .4 .4\n")
        data = directory / "data.yaml"
        data.write_text(yaml.safe_dump({"path": str(directory), "train": "images/train", "val": "images/val", "names": NAMES}))
        teacher = DetectionModel("yolo26s.yaml", nc=6, verbose=False)
        teacher.names = dict(enumerate(NAMES))
        teacher_path = directory / "teacher.pt"
        torch.save({"model": teacher}, teacher_path)
        module, _ = environment(method)
        run = OfflineRun("cpu_" + method)
        module.__dict__.update(MODEL_WEIGHTS="yolo26n.yaml", DATA=str(data), EPOCHS=1,
            BATCH=2, IMGSZ=64, WORKERS=0, DEVICE="cpu", AMP=False, CLOSE_MOSAIC=0,
            SESSION_DIR=directory, TEACHER_PATH=teacher_path,
            TEACHER_INFO={"source": "synthetic CPU fixture"}, DATASET_INFO={"source": "synthetic CPU fixture"},
            wandb=types.SimpleNamespace(init=lambda **kwargs: run, Artifact=OfflineArtifact))
        # Only this tiny four-image fixture uses effective batch 2, so it makes
        # real optimizer steps. The Kaggle protocol retains native nbs=64.
        trainer_type = module.ResearchTrainer
        module.ResearchTrainer = lambda overrides: trainer_type(overrides={**overrides, "nbs": 2})
        result = module.run_training(diagnostic=True)
        assert result["status"] == "PASS" and result["optimizer_steps"] >= 2
        assert run.exit_code == 0 and len(run.uploads) == 1
        assert run.uploads[0][0]["best_epoch"] == 1
        assert set(run.uploads[0][1]) == {"last.pt", "best.pt", "training_state_last.pt", "training_state_best.pt"}
        for key in ("train/det_loss", "train/kd_loss", "train/total_loss", "metrics/mAP50-95(B)"):
            assert key in run.logs[0]
        script = """import sys,torch
from ultralytics import YOLO
torch.set_num_threads(2)
m=YOLO(sys.argv[1]).model.eval()
assert m.yaml['scale']=='n' and not hasattr(m,'teacher_model')
with torch.no_grad():
 y=m(torch.zeros(1,3,64,64))
assert torch.isfinite(y[0]).all()
"""
        process = subprocess.run([sys.executable, "-c", script, result["best_pt"]], capture_output=True, timeout=120, check=False)
        assert process.returncode == 0, process.stderr.decode(errors="replace")
    finally:
        torch.set_num_threads(previous_threads)
        dict.update(settings, previous_settings)
