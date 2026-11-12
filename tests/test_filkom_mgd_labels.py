"""Real SDK verification and dataset construction for mixed detection/polygon rows."""
# ruff: noqa: S102 -- Execute the trusted notebook adapter with isolated fixture files.
import ast
import hashlib
import json
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pytest
from PIL import Image
from ultralytics.cfg import get_cfg
from ultralytics.data import dataset as sdk_dataset
from ultralytics.data.dataset import YOLODataset
from ultralytics.data.utils import verify_image_label

from scripts.generate_selected_kd_notebooks import build_notebook

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def adapter(tmp_path):
    book = build_notebook("filkom", "mgd")
    cache = next(node for cell in book["cells"] if cell["cell_type"] == "code"
                 for node in ast.parse("".join(cell["source"])).body
                 if isinstance(node, ast.FunctionDef) and node.name == "gflops_disable_label_cache")

    def write_json(path, value):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        Path(path).write_text(json.dumps(value), encoding="utf-8")

    scope = {"Path": Path, "np": np, "contextmanager": contextmanager, "patch": patch, "RUN_DIR": tmp_path / "run",
             "write_json": write_json, "sha256_file": lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()}
    exec(compile(ast.Module(body=[cache], type_ignores=[]), "cache-context", "exec"), scope)
    exec((ROOT / "scripts/filkom_mgd_labels_source.py").read_text(encoding="utf-8"), scope)
    return scope


def make_label_fixture(tmp_path, text):
    root = tmp_path / "dataset"
    image = root / "train/images/sample.jpg"
    label = root / "train/labels/sample.txt"
    image.parent.mkdir(parents=True)
    label.parent.mkdir(parents=True)
    Image.new("RGB", (32, 32), "white").save(image)
    label.write_text(text, encoding="utf-8")
    return root, image, label


MIXED = "0 0.5 0.5 0.2 0.2\n0 0.1 0.1 0.3 0.1 0.3 0.3 0.1 0.3\n"


def test_mixed_label_retains_image_and_uses_sdk_polygon_to_box_math(adapter, tmp_path):
    root, image, label = make_label_fixture(tmp_path, MIXED)
    before = {p: p.read_bytes() for p in (image, label)}
    args = (str(image), str(label), "", False, 1, 0, 0, False)
    original = verify_image_label(args)
    assert original[0] is None and "labels mix segment and detection rows" in original[-1]
    audit = adapter["filkom_mgd_prepare_label_adapter"](root, [image], tmp_path / "run")
    actual = adapter["filkom_mgd_verify_image_label"](args)
    assert actual[0] == str(image) and actual[8] == 0
    np.testing.assert_array_equal(actual[1][0], np.array([0, .5, .5, .2, .2], dtype=np.float32))
    from ultralytics.utils.ops import segments2boxes
    expected = segments2boxes([np.array([[.1, .1], [.3, .1], [.3, .3], [.1, .3]], dtype=np.float32)])[0]
    np.testing.assert_array_equal(actual[1][1, 1:], expected)
    assert audit["converted_file_count"] == 1 and audit["files"][0]["polygon_rows_converted"] == 1
    assert before == {p: p.read_bytes() for p in before}
    assert not (root / "train/labels.cache").exists()


def test_actual_sdk_dataset_keeps_complete_mixed_split_and_context_restores_global_verifier(adapter, tmp_path):
    root, image, label = make_label_fixture(tmp_path, MIXED)
    before = label.read_bytes()
    adapter["filkom_mgd_prepare_label_adapter"](root, [image], tmp_path / "run")
    original_verifier = sdk_dataset.verify_image_label
    with adapter["gflops_disable_label_cache"]():
        dataset = YOLODataset(img_path=str(image.parent), imgsz=64, batch_size=1, cache=False, augment=False,
                              hyp=get_cfg(), data={"names": {0: "car"}}, task="detect")
        adapter["fine_tuned_check_images"](SimpleNamespace(dataset=dataset), [image])
        assert len(dataset) == 1 and len(dataset.labels[0]["cls"]) == 2
    assert sdk_dataset.verify_image_label is original_verifier
    assert label.read_bytes() == before and not (root / "train/labels.cache").exists()


@pytest.mark.parametrize("text", ["0 0.5 0.5 0.2 0.2\n", "0 0.1 0.1 0.3 0.1 0.3 0.3\n"])
def test_uniform_label_formats_keep_native_verifier_behavior(adapter, tmp_path, text):
    root, image, label = make_label_fixture(tmp_path, text)
    audit = adapter["filkom_mgd_prepare_label_adapter"](root, [image], tmp_path / "run")
    args = (str(image), str(label), "", False, 1, 0, 0, False)
    actual, original = adapter["filkom_mgd_verify_image_label"](args), verify_image_label(args)
    assert audit["converted_file_count"] == 0 and actual[0] == original[0]
    np.testing.assert_array_equal(actual[1], original[1])


@pytest.mark.parametrize("change", ["source", "prepared"])
def test_label_changes_after_preparation_are_rejected(adapter, tmp_path, change):
    root, image, label = make_label_fixture(tmp_path, MIXED)
    adapter["filkom_mgd_prepare_label_adapter"](root, [image], tmp_path / "run")
    path = label if change == "source" else adapter["FILKOM_MGD_LABEL_MAP"][str(label.resolve())]["path"]
    path.write_text("0 0.5 0.5 0.2 0.2\n", encoding="utf-8")
    with pytest.raises(ValueError, match="Label berubah setelah preflight"):
        adapter["filkom_mgd_verify_image_label"]((str(image), str(label), "", False, 1, 0, 0, False))


@pytest.mark.parametrize("observed", [["extra.jpg"], [], ["sample.jpg", "sample.jpg"]])
def test_mismatched_dataloader_still_fails_with_auditable_counts(adapter, tmp_path, observed):
    expected = tmp_path / "sample.jpg"
    loader = SimpleNamespace(dataset=SimpleNamespace(im_files=[str(tmp_path / p) for p in observed], prefix="train"))
    with pytest.raises(ValueError, match="expected=1, observed="):
        adapter["fine_tuned_check_images"](loader, [expected])
    report = json.loads((adapter["RUN_DIR"] / "dataloader_mismatch.json").read_text())
    assert report["expected_count"] == 1 and report["observed_count"] == len(observed)


@pytest.mark.parametrize("installed,tracking", [(False, True), (True, True), (False, False)])
def test_nbformat_dependency_is_added_only_when_tracking_needs_it(installed, tracking):
    source = next("".join(c["source"]) for c in build_notebook("filkom", "mgd")["cells"]
                  if "bootstrap" in c.get("metadata", {}).get("tags", []))
    node = next(n for n in ast.parse(source).body if isinstance(n, ast.If) and 'name=\'nbformat\'' in ast.unparse(n.test))
    scope = {"WANDB_TRACKING": tracking, "metadata": SimpleNamespace(distributions=lambda **kwargs: [object()] if installed else []),
             "needed": []}
    exec(compile(ast.Module(body=[node], type_ignores=[]), "notebook-history-dependency", "exec"), scope)
    assert scope["needed"] == (["nbformat>=5.10"] if tracking and not installed else [])
