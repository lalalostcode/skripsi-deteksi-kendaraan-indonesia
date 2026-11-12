"""Exercise FILKOM MGD's record-only dataset policy on isolated synthetic files."""
# ruff: noqa: S102 -- Execute trusted generated declarations without loading a model.
import ast
import hashlib
import json
from contextlib import contextmanager
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest
import yaml

from scripts.generate_selected_kd_notebooks import build_notebook

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def local_dataset(tmp_path):
    root = tmp_path / "dataset"
    for split in ("train", "valid"):
        image = root / split / "images/sample.jpg"
        label = root / split / "labels/sample.txt"
        image.parent.mkdir(parents=True)
        label.parent.mkdir(parents=True)
        image.write_bytes(split.encode())
        label.write_text("0 0.5 0.5 0.2 0.2\n", encoding="utf-8")
    source = root / "data.yaml"
    source.write_text("train: ../train/images\nval: ../valid/images\nnames: [car]\nnc: 1\n", encoding="utf-8")
    def write_json(path, value):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(value), encoding="utf-8")

    scope = {"Path": Path, "json": json, "hashlib": hashlib, "yaml": yaml,
             "np": np, "contextmanager": contextmanager, "patch": patch,
             "sha256_file": lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest(),
             "write_json": write_json,
             "ordered_names": list, "MODE": "full_val", "SMOKE_DATASET_ROOT": None,
             "DATASET_ROOT": root, "DATA_YAML": source, "DATASET_ID": "filkom/roboflow-poltrada-v3",
             "POLTRADA_V3_ID": "ilhamrafiqin/poltrada-v3-yolo26-clean",
             "POLTRADA_V3_MOUNT": Path("/kaggle/input/datasets/ilhamrafiqin/poltrada-v3-yolo26-clean"),
             "kd_comparison_settings": lambda: {"training": {"epochs": 100}, "teacher_sha256": "locked"}}
    book = build_notebook("filkom", "mgd")
    required = {"_gflops_split_images", "_gflops_fingerprint", "gflops_prepare_dataset", "workflow_dataset_manifest"}
    nodes = [node for cell in book["cells"] if cell["cell_type"] == "code"
             for node in ast.parse("".join(cell["source"])).body
             if isinstance(node, ast.FunctionDef) and node.name in required]
    exec(compile(ast.Module(body=nodes, type_ignores=[]), "local-dataset-helpers", "exec"), scope)
    # Load the original context first; the FILKOM reader wraps it without replacing cache behavior.
    original_cache = next(node for cell in book["cells"] if cell["cell_type"] == "code"
                          for node in ast.parse("".join(cell["source"])).body
                          if isinstance(node, ast.FunctionDef) and node.name == "gflops_disable_label_cache")
    exec(compile(ast.Module(body=[original_cache], type_ignores=[]), "label-cache-context", "exec"), scope)
    exec((ROOT / "scripts/filkom_mgd_labels_source.py").read_text(encoding="utf-8"), scope)
    exec((ROOT / "scripts/filkom_mgd_dataset_source.py").read_text(encoding="utf-8"), scope)
    manifest = {"seed": 0, "validation_status": "PASS", "dataset": {
        "dataset_id": scope["POLTRADA_V3_ID"], "provenance_status": "VERIFIED", "class_names": ["car"],
        "train_fingerprint": "0" * 64, "full_val_fingerprint": "1" * 64}}
    return scope, manifest


@pytest.mark.parametrize("matching", ["neither", "train", "val", "both"])
def test_local_dataset_accepted_with_actual_fingerprints_and_honest_provenance(local_dataset, tmp_path, matching):
    scope, manifest = local_dataset
    root = scope["DATASET_ROOT"].resolve()
    for split, key in (("train", "train_fingerprint"), ("valid", "full_val_fingerprint")):
        if matching == "both" or matching == ("val" if split == "valid" else "train"):
            images = scope["_gflops_split_images"](root, f"{split}/images")
            manifest["dataset"][key] = scope["_gflops_fingerprint"](root, images, "val" if split == "valid" else "train", ["car"])[0]
    before = json.loads(json.dumps(manifest))
    dataset, record, train, descriptor = scope["ft_dataset"](manifest, tmp_path / "prepared")
    assert manifest == before  # Historical raw metadata remains immutable.
    assert dataset["provenance_status"] == record["provenance_status"] == "UNVERIFIED"
    assert record["dataset_id"] == "filkom/roboflow-poltrada-v3"
    assert record["matches_raw_dataset"] is (matching == "both")
    assert record["comparability_to_raw_training"] == ("SAME_DATASET" if matching == "both" else "DIFFERENT_DATASET")
    assert record["raw_dataset_reference"]["val_fingerprint"] == before["dataset"]["full_val_fingerprint"]
    assert record["val_fingerprint"] == dataset["full_val_fingerprint"]
    assert json.loads((descriptor.parent / "dataset_manifest.json").read_text()) == record
    assert len(train) == 1 and record["train_count"] == record["val_count"] == 1
    assert yaml.safe_load(descriptor.read_text())["train"] != dataset["info"]["val"]
    settings = scope["kd_comparison_settings"]()
    assert settings["train_fingerprint"] == record["train_fingerprint"]
    assert settings["val_fingerprint"] == record["val_fingerprint"]
    assert settings["dataset_policy"] == record["fingerprint_policy"]


def test_local_dataset_still_rejects_changed_classes(local_dataset, tmp_path):
    scope, manifest = local_dataset
    manifest["dataset"]["class_names"] = ["bus"]
    with pytest.raises(ValueError, match="Urutan kelas"):
        scope["ft_dataset"](manifest, tmp_path / "prepared")


def test_local_dataset_still_rejects_train_val_content_leakage(local_dataset, tmp_path):
    scope, manifest = local_dataset
    root = scope["DATASET_ROOT"]
    (root / "valid/images/sample.jpg").write_bytes((root / "train/images/sample.jpg").read_bytes())
    with pytest.raises(ValueError, match="Train dan val berbagi konten"):
        scope["ft_dataset"](manifest, tmp_path / "prepared")


def test_missing_local_dataset_does_not_silently_succeed(local_dataset, tmp_path):
    scope, manifest = local_dataset
    scope["DATA_YAML"] = tmp_path / "missing.yaml"
    with pytest.raises(FileNotFoundError, match="DATASET_ROOT dan DATA_YAML"):
        scope["ft_dataset"](manifest, tmp_path / "prepared")


def test_cross_run_signature_changes_with_actual_dataset_content(local_dataset, tmp_path):
    scope, manifest = local_dataset
    scope["ft_dataset"](manifest, tmp_path / "one")
    before = scope["kd_comparison_settings"]()
    (scope["DATASET_ROOT"] / "valid/labels/sample.txt").write_text("0 0.4 0.5 0.2 0.2\n", encoding="utf-8")
    scope["ft_dataset"](manifest, tmp_path / "two")
    after = scope["kd_comparison_settings"]()
    assert before["val_fingerprint"] != after["val_fingerprint"]
    assert before["training"] == after["training"] and before["teacher_sha256"] == after["teacher_sha256"]
