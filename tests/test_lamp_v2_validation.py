"""Offline validation exports: actual SDK class mapping, curves, and confusion counts."""
# ruff: noqa: S102 -- Execute only trusted standalone authoring declarations.
from __future__ import annotations

import csv
import json
import math
import uuid
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from PIL import Image
from ultralytics import YOLO
from ultralytics.utils.metrics import ConfusionMatrix

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "scripts/lamp_v2_validation_source.py"
NAMES = ["synthetic_a", "synthetic_b", "synthetic_c"]


@pytest.fixture(autouse=True)
def no_learning_or_network(monkeypatch):
    def forbidden(*_args, **_kwargs):
        pytest.fail("Validation export tests must not train or access the network")

    import requests
    monkeypatch.setattr(YOLO, "train", forbidden)
    monkeypatch.setattr(torch.Tensor, "backward", forbidden)
    monkeypatch.setattr(torch.autograd, "backward", forbidden)
    monkeypatch.setattr(requests.sessions.Session, "request", forbidden)
    for optimizer in (torch.optim.Optimizer, torch.optim.SGD, torch.optim.Adam, torch.optim.AdamW):
        monkeypatch.setattr(optimizer, "step", forbidden)
    # Font installation is unrelated to numerical exports and may download Arial.
    monkeypatch.setattr("ultralytics.data.utils.check_font", lambda *_args, **_kwargs: None)
    monkeypatch.setenv("WANDB_MODE", "disabled")
    monkeypatch.setenv("WANDB_DISABLED", "true")


@pytest.fixture
def folder(monkeypatch):
    import tempfile

    path = ROOT / "outputs/test_lamp_v2_validation" / uuid.uuid4().hex
    path.mkdir(parents=True)
    monkeypatch.setattr(tempfile, "tempdir", str(path))
    monkeypatch.setenv("TEMP", str(path))
    monkeypatch.setenv("TMP", str(path))
    return path


@pytest.fixture
def scope():
    def write_json(path, payload):
        Path(path).write_text(json.dumps(payload, allow_nan=False), encoding="utf-8")

    def write_csv(path, rows):
        with Path(path).open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)

    def assert_class_match(left, right):
        if isinstance(left, dict):
            left = [left[i] for i in range(len(left))]
        if list(left) != list(right):
            raise ValueError("Class order differs")

    env = {"Path": Path, "YOLO": YOLO, "math": math, "write_json": write_json,
           "write_csv": write_csv, "assert_class_match": assert_class_match,
           "IMGSZ": 64, "BATCH": 1, "WORKERS": 0, "TARGET_DEVICE": torch.device("cpu")}
    exec(compile(SOURCE.read_text(encoding="utf-8"), str(SOURCE), "exec"), env)
    torch.set_num_threads(2)
    return env


def test_protocol_collects_confusion_without_changing_primary_metrics(scope, folder):
    before = scope["validation_arguments"]("data.yaml", folder, "baseline")
    after = scope["validation_arguments"]("data.yaml", folder, "pruned")
    assert {k: v for k, v in before.items() if k != "name"} == {
        k: v for k, v in after.items() if k != "name"}
    assert before["plots"] is True
    assert before["split"] == "val" and before["quantize"] is None
    assert before["nms"] is None and before["augment"] is False
    assert (before["conf"], before["iou"], before["max_det"]) == (0.001, 0.7, 300)


def mock_metric():
    return SimpleNamespace(
        box=SimpleNamespace(map=0.12, map50=0.23, mp=0.4, mr=0.5,
                            ap_class_index=np.array([2, 0]), ap50=np.array([0.2, 0.3]),
                            ap=np.array([0.1, 0.14]), p=np.array([0.3, 0.5]),
                            r=np.array([0.4, 0.6]), f1=np.array([0.34, 0.54]),
                            px=np.array([0, 0.5, 1.0]),
                            f1_curve=np.array([[0, 1, 0], [0, 0.8, 0]])),
        nt_per_class=np.array([3, 0, 1]), curves=["Precision-Recall(B)", "F1-Confidence(B)"],
        curves_results=[
            [np.array([0, 0.5, 1.0]), np.array([[0.2, np.nan, 0.4], [0.3, 0.7, np.inf]]),
             "Recall", "Precision"],
            [np.array([0, 1]), np.zeros((1, 2)), "Confidence", "F1"],
        ],
    )


def test_export_maps_sdk_classes_keeps_nulls_and_distinguishes_thresholds(scope, folder):
    metric = mock_metric()
    matrix = ConfusionMatrix(names=dict(enumerate(NAMES)))
    matrix.matrix[3, 0] = 3
    matrix.matrix[2, 2] = 1

    class Wrapper:
        def __init__(self, *_args, **_kwargs):
            self.names = dict(enumerate(NAMES))
            self.validator = SimpleNamespace(confusion_matrix=matrix, confusion_matrix_conf=0.001)

        def val(self, **kwargs):
            assert kwargs["plots"] is True
            return metric

    scope["YOLO"] = Wrapper
    dataset = {"info": {"names": NAMES}, "class_names": NAMES, "yaml": "not_downloaded.yaml"}
    result = scope["evaluate_checkpoint"]("not_downloaded.pt", dataset, folder, "mock_val")
    assert [row["support"] for row in result["per_class"]] == [3, 0, 1]
    assert result["per_class"][0]["precision"] == 0.5
    assert result["per_class"][2]["precision"] == 0.3
    assert result["per_class"][1]["precision"] is None
    assert result["per_class"][1]["ap50"] is None
    curve = result["curves"]["curves"][0]
    assert curve["class_ids"] == [2, 0] and curve["class_names"] == [NAMES[2], NAMES[0]]
    assert curve["status"] == "PARTIAL" and curve["y"][0][1] is None and curve["y"][1][2] is None
    assert result["curves"]["curves"][1]["status"] == "UNAVAILABLE"
    cm = result["confusion_matrix"]
    assert cm["confidence_threshold"] == 0.001 and cm["matching_iou_threshold"] == 0.45
    assert cm["validation_iou"] == 0.7
    assert np.isfinite(np.asarray(cm["normalized_by_true"])).all()
    assert np.asarray(cm["normalized_by_true"])[:, 1].sum() == 0
    assert result["metric_semantics"]["reporting_confidence"] == 0.5
    for name, path in result["exports"].items():
        assert Path(path).exists()
        if name.endswith(".json"):
            assert json.loads(Path(path).read_text(encoding="utf-8"))


def test_missing_auxiliary_measurements_are_unavailable_not_zero(scope):
    metric = SimpleNamespace(box=SimpleNamespace(ap_class_index=[]))
    rows, indices = scope["_validation_per_class"](metric, NAMES)
    assert indices == [] and all(row["support"] is None for row in rows)
    assert all(row["f1"] is None for row in rows)
    assert scope["_validation_curves"](metric, NAMES, []) == {
        "status": "UNAVAILABLE", "reason": "'types.SimpleNamespace' object has no attribute 'curves_results'", "curves": []}
    cm = scope["_validation_confusion_matrix"](None, NAMES, {"iou": 0.7})
    assert cm["status"] == "UNAVAILABLE" and cm["matrix"] is None


@pytest.mark.parametrize("indices", [[3], [0, 0], [float("nan")], [0.5]])
def test_invalid_sdk_class_indices_fail_instead_of_mislabelling(scope, indices):
    metric = mock_metric()
    metric.box.ap_class_index = indices
    with pytest.raises(ValueError, match="Indeks"):
        scope["_validation_per_class"](metric, NAMES)


def test_nonfinite_primary_metric_propagates_failure(scope, folder):
    metric = mock_metric()
    metric.box.map = float("nan")
    scope["YOLO"] = lambda *_args, **_kwargs: SimpleNamespace(
        names=dict(enumerate(NAMES)), val=lambda **_kwargs: metric)
    dataset = {"info": {"names": NAMES}, "class_names": NAMES, "yaml": "unused.yaml"}
    with pytest.raises(ValueError, match="NaN/Inf"):
        scope["evaluate_checkpoint"]("unused.pt", dataset, folder, "nonfinite")


@pytest.mark.parametrize("scale", ["n", "s"])
def test_actual_cpu_tiny_validation_exports_false_negatives_and_class_support(scope, folder, scale):
    from ultralytics.nn.tasks import DetectionModel, yaml_model_load

    core = DetectionModel(yaml_model_load(f"yolo26{scale}.yaml"), nc=3, verbose=False).float().eval()
    core.names = dict(enumerate(NAMES))
    # Force no detections while leaving one real GT object: plots must still collect its FN.
    with torch.no_grad():
        for branch_name in ("cv3", "one2one_cv3"):
            for classifier in getattr(core.model[-1], branch_name):
                classifier[-1].bias.fill_(-40)
    checkpoint = folder / f"synthetic_{scale}.pt"
    torch.save({"model": core, "ema": None, "optimizer": None, "train_args": {}}, checkpoint)
    images, labels = folder / "images", folder / "labels"
    images.mkdir()
    labels.mkdir()
    Image.new("RGB", (64, 64), "gray").save(images / "sample.jpg")
    (labels / "sample.txt").write_text("0 0.5 0.5 0.25 0.25\n", encoding="utf-8")
    data_yaml = folder / "data.yaml"
    data_yaml.write_text(
        f"path: {folder.as_posix()}\ntrain: images\nval: images\nnames: [synthetic_a, synthetic_b, synthetic_c]\n",
        encoding="utf-8")
    dataset = {"yaml": str(data_yaml), "class_names": NAMES, "info": {"names": NAMES}}
    result = scope["evaluate_checkpoint"](checkpoint, dataset, folder, f"tiny_{scale}_val")
    assert result["map50_95"] == 0 and result["precision"] == 0
    assert result["confusion_matrix"]["status"] == "MEASURED"
    matrix = np.asarray(result["confusion_matrix"]["matrix"])
    assert matrix.shape == (4, 4) and matrix[3, 0] == 1 and matrix.sum() == 1
    assert result["confusion_matrix"]["labels"] == [*NAMES, "background"]
    assert result["confusion_matrix"]["matching_iou_threshold"] == 0.45
    assert result["confusion_matrix"]["confidence_threshold"] == 0.001
    assert [row["support"] for row in result["per_class"]] == [1, 0, 0]
    assert result["per_class"][0]["recall"] == 0
    assert result["per_class"][1]["recall"] is None
    assert result["curves"]["status"] == "MEASURED"
    assert len(result["curves"]["curves"]) == 4
    assert all(row["class_ids"] == [0] for row in result["curves"]["curves"])
    assert (folder / f"tiny_{scale}_val" / "per_class_metrics.csv").is_file()
