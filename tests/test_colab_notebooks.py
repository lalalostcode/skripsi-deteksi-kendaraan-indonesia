"""Offline checks of the exact standalone Colab copies; no Colab/GPU claim."""
# ruff: noqa: S102 -- Execute trusted local notebook cells with explicit mock adapters.
from __future__ import annotations

import ast
import hashlib
import io
import json
import re
import runpy
import stat
import subprocess
import sys
import types
import uuid
import zipfile
from pathlib import Path
from unittest.mock import Mock

import nbformat
import pytest
import test_eda_poltrada_v3 as eda_tests
import test_pkd_notebook as pkd_tests
import torch
import yaml
from PIL import Image
from ultralytics.data.utils import IMG_FORMATS, check_det_dataset

from scripts.notebook_paths import notebook_path

ROOT = Path(__file__).resolve().parents[1]
FOLDER = ROOT / "notebooks/colab"
SOURCE = ROOT / "notebooks/kaggle"
MANIFEST = json.loads((FOLDER / "manifest.json").read_text(encoding="utf-8"))
GENERATOR = runpy.run_path(str(ROOT / "scripts/generate_colab_notebooks.py"))
NAMES = ["Bus", "Mobil Penumpang", "Pejalan Kaki", "Sepeda Motor", "Truck", "Unmotorized"]
DECLARATION = GENERATOR["SOURCE_DATA"]


def source(cell):
    return "".join(cell["source"])


def by_id(notebook, identifier):
    return source(next(c for c in notebook["cells"] if c["id"] == identifier))


def notebook(name):
    return json.loads(notebook_path("colab", Path(name).name).read_text(encoding="utf-8"))


@pytest.fixture
def workdir():
    directory = ROOT / "outputs/test_colab" / uuid.uuid4().hex
    directory.mkdir(parents=True)
    return directory


@pytest.fixture
def runtime():
    env = {}
    exec(compile(by_id(notebook(MANIFEST[0]["colab"]), "colab-runtime"), "<embedded-colab-runtime>", "exec"), env)
    return env


@pytest.fixture
def pkd_environment(workdir):
    yield from pkd_tests.environment.__wrapped__(workdir)


def make_archive(directory, prefix="", *, config=None, extras=None):
    image = io.BytesIO()
    Image.new("RGB", (32, 24), (90, 120, 150)).save(image, "JPEG")
    members = {}
    for split in ("train", "val", "test"):
        members[f"{prefix}{split}/images/image.jpg"] = image.getvalue()
        members[f"{prefix}{split}/labels/image.txt"] = b"0 0.5 0.5 0.25 0.25\n"
    if config is None:
        config = {"path": str(Path(DECLARATION).parent).replace("\\", "/"), "train": "train/images", "val": "val/images",
                  "test": "test/images", "nc": 6, "names": dict(enumerate(NAMES))}
    members[prefix + "data.yaml"] = yaml.safe_dump(config, sort_keys=False).encode()
    members.update(extras or {})
    archive = directory / "dataset.zip"
    with zipfile.ZipFile(archive, "w") as stream:
        for name, value in members.items():
            stream.writestr(name, value)
    return archive, members


def config_env(book, *, colab):
    env = {"COLAB_DATA": "/content/test/runtime/data.yaml", "COLAB_OUTPUT_DIR": Path("/drive/test/output")}
    config = next(source(c) for c in book["cells"] if c["cell_type"] == "code"
                  and ("EXPERIMENT_ID =" in source(c) or "DATASET_NAME =" in source(c)))
    exec(compile(config, "<colab-config>" if colab else "<original-config>", "exec"), env)
    return env


def test_complete_collection_and_readme_links():
    assert len(MANIFEST) == len(list(FOLDER.rglob("*.ipynb"))) == 17
    assert [int(Path(row["colab"]).name.split("_", 1)[0]) for row in MANIFEST] == list(range(17))
    assert len({row["colab"] for row in MANIFEST}) == 17
    text = (FOLDER / "README.md").read_text(encoding="utf-8")
    for target in re.findall(r"\]\(([^)]+)\)", text):
        if not target.startswith("https:"):
            assert (FOLDER / target).resolve().exists(), target


@pytest.mark.parametrize("row", MANIFEST, ids=lambda row: row["colab"])
def test_schema_generation_protocol_and_method_parity(row):
    book = notebook(row["colab"])
    nbformat.validate(nbformat.from_dict(book))
    original_path = SOURCE / row["source"]
    assert hashlib.sha256(original_path.read_bytes()).hexdigest() == row["source_sha256"]
    original = json.loads(original_path.read_text(encoding="utf-8"))
    assert book["metadata"]["accelerator"] == "GPU"
    assert book["metadata"]["colab_source"]["sha256"] == row["source_sha256"]
    env, old_env = config_env(book, colab=True), config_env(original, colab=False)
    for key, value in old_env.items():
        if key.isupper() and key not in {"DATA", "PROJECT", "DATA_YAML", "OUTPUT_DIR"}:
            assert env[key] == value, key
    original_cells = {c["id"]: c for c in original["cells"]}
    for c in book["cells"]:
        if c["cell_type"] != "code":
            continue
        assert c["outputs"] == [] and c["execution_count"] is None
        compile(source(c), row["colab"] + c["id"], "exec")
        assert "kaggle_secrets" not in source(c) and "/kaggle/working" not in source(c)
        if c["id"] not in original_cells:
            continue
        # All model/trainer classes and mathematical functions must remain exact AST matches.
        old_nodes = ast.parse(source(original_cells[c["id"]])).body
        new_nodes = ast.parse(source(c)).body
        critical = [node for node in old_nodes if isinstance(node, ast.ClassDef)
                    or isinstance(node, ast.FunctionDef) and any(part in node.name.lower()
                       for part in ("loss", "normalize", "feature", "projector", "gradient_norm", "teacher_digest"))]
        for old in critical:
            current = next(node for node in new_nodes if isinstance(node, type(old)) and node.name == old.name)
            assert ast.dump(old) == ast.dump(current), old.name
    # Generated outputs are reproducible, including embedded helper code and source mapping.
    workflow = next(item for item in GENERATOR["WORKFLOWS"] if item[0] + ".ipynb" == Path(row["colab"]).name)
    generated, record = GENERATOR["build_notebook"](workflow)
    assert generated == book
    assert record == row


@pytest.mark.parametrize("prefix", ["", "poltrada-v3/"])
@pytest.mark.parametrize("legacy_absolute", [False, True])
def test_zip_relocation_preserves_source_bytes_and_works_with_native_loader(runtime, workdir, prefix, legacy_absolute):
    config = {"train": "../train/images", "val": "val/images", "test": "test/images", "nc": 6, "names": NAMES}
    if legacy_absolute:
        config["path"] = str(Path(DECLARATION).parent).replace("\\", "/")
        config["train"] = config["path"] + "/train/images"
    archive, members = make_archive(workdir, prefix, config=config)
    archive_before = archive.read_bytes()
    data, evidence = runtime["prepare_colab_dataset"](archive, workdir / "local", "", DECLARATION)
    assert archive.read_bytes() == archive_before
    assert evidence["class_names"] == NAMES
    assert evidence["split_image_counts"] == {"train": 1, "val": 1, "test": 1}
    assert evidence["zip_sha256"] == hashlib.sha256(archive_before).hexdigest()
    for name, value in members.items():
        assert (workdir / "local/source" / name).read_bytes() == value
    assert evidence["original_yaml_sha256"] != evidence["runtime_yaml_sha256"]
    checked = check_det_dataset(data, autodownload=False)
    assert list(checked["names"].values()) == NAMES
    assert all(Path(checked[split]).is_dir() for split in ("train", "val", "test"))


def test_txt_split_order_duplicates_and_lists_are_preserved(runtime, workdir):
    config = {"train": ["train.txt"], "val": "val/images", "nc": 6, "names": NAMES}
    archive, _ = make_archive(workdir, config=config, extras={"train.txt": b"./train/images/image.jpg\n./train/images/image.jpg\n"})
    data, evidence = runtime["prepare_colab_dataset"](archive, workdir / "local", "", DECLARATION)
    resolved = yaml.safe_load(Path(data).read_text(encoding="utf-8"))
    entries = Path(resolved["train"][0]).read_text(encoding="utf-8").splitlines()
    assert len(entries) == 2 and entries[0] == entries[1]
    assert evidence["split_image_counts"] == {"train": 2, "val": 1}
    assert (workdir / "local/source/train.txt").read_bytes().startswith(b"./")


def test_yaml_ambiguity_requires_explicit_archive_relative_path(runtime, workdir):
    archive, _ = make_archive(workdir, "nested/", extras={"unrelated.yaml": b"a: b\n"})
    with pytest.raises(ValueError, match="DATASET_YAML_RELATIVE"):
        runtime["prepare_colab_dataset"](archive, workdir / "ambiguous", "", DECLARATION)
    _, evidence = runtime["prepare_colab_dataset"](archive, workdir / "explicit", "nested/data.yaml", DECLARATION)
    assert evidence["split_image_counts"]["train"] == 1
    with pytest.raises(FileNotFoundError, match="inside the ZIP"):
        runtime["prepare_colab_dataset"](archive, workdir / "escape", "../../outside.yaml", DECLARATION)


@pytest.mark.parametrize("member", ["../escape.txt", "/absolute.txt", "C:/absolute.txt", "bad\\..\\escape.txt"])
def test_unsafe_zip_members_are_rejected_before_extraction(runtime, workdir, member):
    archive = workdir / "unsafe.zip"
    with zipfile.ZipFile(archive, "w") as stream:
        stream.writestr("safe.txt", b"safe")
        stream.writestr(member, b"unsafe")
    with pytest.raises(ValueError, match="Unsafe"):
        runtime["colab_extract_zip"](archive, workdir / "source")
    assert not (workdir / "source").exists()


def test_zip_symlinks_and_duplicate_members_are_rejected(runtime, workdir):
    link = zipfile.ZipInfo("link")
    link.external_attr = (stat.S_IFLNK | 0o777) << 16
    archive = workdir / "symlink.zip"
    with zipfile.ZipFile(archive, "w") as stream:
        stream.writestr(link, b"../outside")
    with pytest.raises(ValueError, match="Unsafe"):
        runtime["colab_extract_zip"](archive, workdir / "symlink")
    with zipfile.ZipFile(workdir / "duplicate.zip", "w") as stream:
        stream.writestr("same.txt", b"a")
        with pytest.warns(UserWarning, match="Duplicate"):
            stream.writestr("same.txt", b"b")
    with pytest.raises(ValueError, match="duplicate"):
        runtime["colab_extract_zip"](workdir / "duplicate.zip", workdir / "duplicate")


def test_missing_data_incomplete_extraction_and_invalid_schema_fail(runtime, workdir):
    with pytest.raises(FileNotFoundError, match="Edit DATASET_ZIP"):
        runtime["prepare_colab_dataset"](workdir / "missing.zip", workdir / "local", "", DECLARATION)
    archive, _ = make_archive(workdir)
    (workdir / "partial/source").mkdir(parents=True)
    with pytest.raises(RuntimeError, match="new run directory"):
        runtime["prepare_colab_dataset"](archive, workdir / "partial", "", DECLARATION)
    for label, config, message in [
        ("missing", {"train": "missing/images", "val": "val/images", "names": NAMES}, "missing"),
        ("classes", {"train": "train/images", "val": "val/images", "names": NAMES, "nc": 3}, "disagree"),
        ("split", {"train": "train/images", "names": NAMES}, "declare train, val"),
    ]:
        sub = workdir / label
        sub.mkdir()
        broken, _ = make_archive(sub, config=config)
        with pytest.raises((ValueError, FileNotFoundError), match=message):
            runtime["prepare_colab_dataset"](broken, sub / "local", "", DECLARATION)


def test_dot_root_directory_and_duplicate_yaml_keys(runtime, workdir):
    archive, _ = make_archive(workdir, extras={"./": b""})
    _, evidence = runtime["prepare_colab_dataset"](archive, workdir / "valid", "", DECLARATION)
    assert evidence["split_image_counts"]["train"] == 1
    duplicate_yaml = b"train: train/images\ntrain: wrong/images\nval: val/images\nnames: [Bus]\n"
    archive, _ = make_archive(workdir, extras={"data.yaml": duplicate_yaml})
    with pytest.raises(ValueError, match="Duplicate dataset YAML key"):
        runtime["prepare_colab_dataset"](archive, workdir / "duplicate-yaml", "", DECLARATION)


def test_corrupt_archive_stops_on_crc_failure(runtime, workdir):
    archive, members = make_archive(workdir)
    payload = bytearray(archive.read_bytes())
    image = members["train/images/image.jpg"]
    offset = payload.index(image)
    payload[offset + len(image) // 2] ^= 1
    archive.write_bytes(payload)
    with pytest.raises(zipfile.BadZipFile, match="CRC"):
        runtime["prepare_colab_dataset"](archive, workdir / "corrupt", "", DECLARATION)


def test_colab_eda_loader_accepts_derived_yaml_and_preserves_labels(runtime, workdir, monkeypatch):
    archive, members = make_archive(workdir, "wrapper/")
    data, _ = runtime["prepare_colab_dataset"](archive, workdir / "local", "", DECLARATION)
    book = nbformat.reads(json.dumps(notebook("00_eda_poltrada_v3.ipynb")), as_version=4)
    # Reuse the established EDA definition harness on the actual adapted research cells.
    book.cells = [c for c in book.cells if not c.id.startswith("colab-")]
    monkeypatch.setattr(eda_tests, "read_notebook", lambda: book)
    env = eda_tests.definitions_namespace()
    _config, root, names, images, _labels, _sources = env["load_dataset"](Path(data), None)
    assert names == NAMES and all(len(images[split]) == 1 for split in ("train", "val", "test"))
    assert root == (workdir / "local/source/wrapper").resolve()
    for name, original in members.items():
        assert (workdir / "local/source" / name).read_bytes() == original


@pytest.mark.parametrize("filename", ["00_eda_poltrada_v3.ipynb", "15_head_yolo26s_to_yolo26n.ipynb"])
def test_exact_colab_bootstrap_mount_dependencies_and_persistent_provenance(filename, workdir, monkeypatch):
    archive, _ = make_archive(workdir)
    book = notebook(filename)
    env = {}
    exec(compile(by_id(book, "colab-config"), "<exact-colab-configuration>", "exec"), env)
    env.update(DATASET_ZIP=str(archive), COLAB_LOCAL_DIR=workdir / "vm-local",
               COLAB_OUTPUT_DIR=workdir / "drive-results/run-unique",
               COLAB_DATA=str(workdir / "vm-local/runtime/data.yaml"))
    # Mount adapter only: all copying, extraction, YAML and output writes are real local I/O.
    mount = Mock()
    colab = types.ModuleType("google.colab")
    colab.drive = types.SimpleNamespace(mount=mount)
    monkeypatch.setitem(sys.modules, "google.colab", colab)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)  # Dispatch fixture, never a GPU claim.
    def forbid_install(*_args, **_kwargs):
        pytest.fail("Pinned packages already exist in this offline test; no installation/network allowed.")
    monkeypatch.setattr(subprocess, "check_call", forbid_install)
    monkeypatch.chdir(workdir)  # Restores the real cwd after the embedded mount cell changes it.
    for name in ("WANDB_CACHE_DIR", "WANDB_DIR", "PIP_CACHE_DIR", "TORCH_HOME"):
        monkeypatch.setenv(name, "fixture-before-mount")  # Register restoration before notebook assignments.
    research_config = next(source(c) for c in book["cells"] if c["cell_type"] == "code"
                           and ("EXPERIMENT_ID =" in source(c) or "DATASET_NAME =" in source(c)))
    exec(compile(research_config, "<exact-research-configuration>", "exec"), env)
    for identifier in ("colab-drive", "colab-dependencies", "colab-runtime", "colab-dataset"):
        exec(compile(by_id(book, identifier), "<exact-" + identifier + ">", "exec"), env)
    mount.assert_called_once_with("/content/drive")
    provenance_path = env["COLAB_OUTPUT_DIR"] / "dataset_provenance.json"
    saved = json.loads(provenance_path.read_text(encoding="utf-8"))
    assert saved == env["COLAB_DATA_PROVENANCE"]
    assert Path(saved["runtime_yaml"]).is_relative_to(env["COLAB_LOCAL_DIR"])
    assert Path(env.get("DATA", env.get("DATA_YAML"))).is_file()
    if filename.startswith("00_"):
        assert env["OUTPUT_DIR"] == env["COLAB_OUTPUT_DIR"]
    else:
        assert env["PROJECT"] == str(env["COLAB_OUTPUT_DIR"])


def test_colab_secret_environment_precedence_and_ui_errors(runtime, monkeypatch):
    monkeypatch.delenv("WANDB_API_KEY", raising=False)
    getter = Mock(return_value=" fake-test-key ")
    colab = types.ModuleType("google.colab")
    colab.userdata = types.SimpleNamespace(get=getter)
    monkeypatch.setitem(sys.modules, "google.colab", colab)
    assert runtime["colab_api_key"]() == "fake-test-key"
    assert getter.call_count == 1
    assert runtime["colab_api_key"]() == "fake-test-key" and getter.call_count == 1
    monkeypatch.delenv("WANDB_API_KEY")
    getter.side_effect = RuntimeError("private payload fake-test-key")
    with pytest.raises(RuntimeError, match="enable notebook access") as captured:
        runtime["colab_api_key"]()
    assert "fake-test-key" not in str(captured.value)
    getter.side_effect, getter.return_value = None, ""
    with pytest.raises(RuntimeError, match="empty"):
        runtime["colab_api_key"]()


def test_gpu_error_is_actionable(runtime):
    with pytest.raises(RuntimeError, match="Runtime > Change runtime type > GPU"):
        runtime["colab_require_gpu"](types.SimpleNamespace(cuda=types.SimpleNamespace(is_available=lambda: False)))


def test_membership_counting_uses_the_pinned_loader_image_formats():
    tree = ast.parse(by_id(notebook(MANIFEST[0]["colab"]), "colab-runtime"))
    declared = next(node.value for node in ast.walk(tree) if isinstance(node, ast.Assign)
                    and any(isinstance(target, ast.Name) and target.id == "image_suffixes" for target in node.targets))
    assert ast.literal_eval(declared) == {"." + suffix for suffix in IMG_FORMATS}


def test_actual_colab_teacher_audit_accepts_relocation_but_rejects_wrong_source(pkd_environment, workdir):
    env = pkd_environment
    artifact, producer, _teacher, _lookup, _payload = pkd_tests.teacher_fixture(env, workdir)
    env["SOURCE_DATA_DECLARATION"] = env["DATA"]
    env["DATA"] = "/content/local/runtime/data.yaml"
    env["COLAB_LOCAL_DIR"] = workdir
    exec(compile(by_id(notebook("14_pearson_pkd_yolo26s_to_yolo26n.ipynb"), "pkd-teacher"), "<colab-teacher-audit>", "exec"), env)
    checkpoint, provenance = env["download_teacher"]()
    assert Path(checkpoint).is_file() and provenance["tensor_schema_verified"]
    assert provenance["version"] == artifact.version == "v62"
    producer.config["dataset"] = "wrong-dataset.yaml"
    with pytest.raises(ValueError, match="Dataset teacher berbeda"):
        env["download_teacher"]()


@pytest.mark.parametrize("method,filename", [("head", "15_head_yolo26s_to_yolo26n.ipynb"),
                                             ("rm", "16_rank_mimicking_yolo26s_to_yolo26n.ipynb"),
                                             ("pkd", "14_pearson_pkd_yolo26s_to_yolo26n.ipynb")])
def test_trio_run_all_dispatches_full_training_without_smoke(method, filename):
    book = notebook(filename)
    env = config_env(book, colab=True)
    assert env["RUN_FULL_TRAINING"] is True and env["RUN_SMOKE_TEST"] is False and env["EPOCHS"] == 100
    calls = []
    env.update(json=json, status=lambda *_: None)
    if method == "pkd":
        env.update(run_mode=lambda mode: calls.append(mode) or {"status": "PASS"}, require_full_training=lambda _: None,
                   RESUME_STATUS="RESUME_NOT_VERIFIED")
        suffixes = ("diagnostic-execute", "timing", "full-execute")
    else:
        env["run_training"] = lambda diagnostic: calls.append(diagnostic) or {"status": "PASS"}
        suffixes = ("diagnostic", "full")
    for suffix in suffixes:
        exec(compile(by_id(book, method + "-" + suffix), "<mock-dispatch-only>", "exec"), env)
    assert calls == (["training"] if method == "pkd" else [False])
