"""Ordinary recovery tests, including native CPU training after structural pruning."""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest
import torch
from PIL import Image
from ultralytics.nn.tasks import DetectionModel, yaml_model_load

ROOT = Path(__file__).resolve().parents[1]


def load_source(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


training = load_source("lamp_recovery_training", ROOT / "scripts/lamp_recovery_training_source.py")
policy = load_source("lamp_recovery_policy_test", ROOT / "scripts/lamp_v2_policy_source.py")


@pytest.fixture(autouse=True)
def no_network(monkeypatch):
    import requests
    import ultralytics.data.utils

    def forbidden(*_args, **_kwargs):
        raise AssertionError("Recovery tests must not download or contact remote services.")
    monkeypatch.setattr(requests.sessions.Session, "request", forbidden)
    monkeypatch.setattr(ultralytics.data.utils, "check_font", lambda *_args, **_kwargs: None)
    torch.set_num_threads(1)


@pytest.fixture
def dataset(tmp_path):
    names = ["car", "bus", "truck"]
    generator = np.random.default_rng(42)
    for split in ("train", "val"):
        images, labels = tmp_path / split / "images", tmp_path / split / "labels"
        images.mkdir(parents=True)
        labels.mkdir(parents=True)
        for index in range(2):
            Image.fromarray(generator.integers(0, 256, (64, 64, 3), dtype=np.uint8)).save(images / f"{index}.jpg")
            (labels / f"{index}.txt").write_text(f"{index} 0.5 0.5 0.35 0.4\n", encoding="utf-8")
    data = tmp_path / "data.yaml"
    data.write_text(f"path: {tmp_path.as_posix()}\ntrain: train/images\nval: val/images\nnames: [car, bus, truck]\n", encoding="utf-8")
    return data, names


def train_settings():
    return {"epochs": 1, "optimizer": "AdamW", "lr0": .001, "lrf": 1., "batch": 2,
            "imgsz": 64, "device": "cpu", "seed": 42, "workers": 0, "plots": False,
            "warmup_epochs": 0., "nbs": 2, "mosaic": 0., "mixup": 0., "copy_paste": 0.,
            "degrees": 0., "translate": 0., "scale": 0., "fliplr": 0., "flipud": 0.,
            "hsv_h": 0., "hsv_s": 0., "hsv_v": 0., "close_mosaic": 0}


@pytest.mark.parametrize("variant", ["n", "s"])
def test_actual_cpu_native_recovery_keeps_pruned_shapes_and_best_ema_fp32(tmp_path, dataset, variant):
    torch.manual_seed(42)
    data, names = dataset
    core = DetectionModel(yaml_model_load(f"yolo26{variant}.yaml"), nc=len(names), verbose=False).float().eval()
    core.names = dict(enumerate(names))
    initial_parameters = sum(p.numel() for p in core.parameters())
    pruner, audit = policy.build_policy_pruner(core, torch.rand(1, 3, 64, 64), "extended", .3, variant)
    policy.apply_policy_pruning(pruner, audit)
    pruned_parameters = sum(p.numel() for p in core.parameters())
    assert pruned_parameters < initial_parameters
    source = tmp_path / "pruned_fp32.pt"
    torch.save({"model": core, "ema": None, "optimizer": None, "train_args": {}}, source)
    expected_structure = training.recovery_structure_signature(core)
    expected_source = training.recovery_state_sha256(core)
    rows = []
    recovered, report = training.fine_tune_pruned_checkpoint(
        source, data, tmp_path / "recovery", names, train_settings(),
        epoch_callback=lambda _trainer, row: rows.append(row))
    assert type(recovered) is DetectionModel
    assert recovered.criterion is None
    assert report["status"] == "PASS" and report["optimizer_steps"] >= 1
    assert report["epochs_completed"] == report["best_epoch"] == 1
    assert report["initial_state_verified"] and report["parameters_updated"]
    assert report["initial_state_sha256"] == expected_source
    assert report["best_state_sha256"] != expected_source
    assert report["structure_preserved"]
    assert training.recovery_structure_signature(recovered) == expected_structure
    assert sum(p.numel() for p in recovered.parameters()) == pruned_parameters
    assert all(p.dtype == torch.float32 for p in recovered.parameters())
    assert rows == report["epochs"] and all(np.isfinite(value) for value in rows[0]["losses"].values())
    payload = torch.load(report["best_fp32_checkpoint"], weights_only=False)
    assert type(payload["model"]) is DetectionModel
    assert payload["ema"] is payload["optimizer"] is None
    assert training.recovery_state_sha256(payload["model"]) == report["best_state_sha256"]
    assert payload["recovery"]["source_sha256"] == report["source_sha256"]
    with torch.no_grad():
        output = recovered(torch.rand(1, 3, 64, 64))
    assert all(torch.isfinite(value).all() for value in policy._v2_trace_outputs(output))
    stored = json.loads((tmp_path / "recovery_recovery_training_report.json").read_text(encoding="utf-8"))
    assert stored == report


@pytest.mark.parametrize("changed", [{"resume": True}, {"optimizer": "auto"}, {"amp": True},
                                     {"distill_model": "teacher.pt"}, {"freeze": 1},
                                     {"epochs": 0}, {"epochs": True}, {"save": False}, {"val": False}])
def test_recovery_rejects_resume_distillation_amp_freezing_and_invalid_budget(tmp_path, dataset, changed):
    data, names = dataset
    source = tmp_path / "source.pt"
    source.write_bytes(b"not read when invalid options are rejected")
    with pytest.raises(ValueError):
        training.fine_tune_pruned_checkpoint(source, data, tmp_path / "training", names,
                                            {**train_settings(), **changed})
    report = json.loads((tmp_path / "training_recovery_training_report.json").read_text(encoding="utf-8"))
    assert report["status"] == "FAIL" and report["errors"]


def test_integration_callbacks_are_blocked_before_constructor_first_event():
    trainer = object.__new__(training.StructuralRecoveryTrainer)
    seen = []

    def remote(_trainer):
        raise AssertionError("SDK must not initialize an external tracker")
    remote.__module__ = "ultralytics.utils.callbacks.wb"

    def local(_trainer):
        seen.append("local")
    trainer.callbacks = {"on_pretrain_routine_start": [remote, local]}
    trainer.recovery_filtered_callbacks = set()
    trainer.run_callbacks("on_pretrain_routine_start")
    assert seen == ["local"]
    assert trainer.recovery_filtered_callbacks == {remote.__module__}


def test_source_class_order_failure_is_saved_and_reraised(tmp_path, dataset):
    data, names = dataset
    core = DetectionModel(yaml_model_load("yolo26n.yaml"), nc=3, verbose=False).float().eval()
    core.names = dict(enumerate(names))
    source = tmp_path / "source.pt"
    torch.save({"model": core, "ema": None}, source)
    with pytest.raises(ValueError, match="class order"):
        training.fine_tune_pruned_checkpoint(source, data, tmp_path / "training", list(reversed(names)), train_settings())
    assert json.loads((tmp_path / "training_recovery_training_report.json").read_text())["status"] == "FAIL"
