"""Mocked notebook provenance checks; no W&B login, download, or real model load."""

# Execute only trusted local notebook code with explicitly mocked network/model objects.
# ruff: noqa: S102

import ast
import base64
import hashlib
import json
import uuid
from pathlib import Path
from types import SimpleNamespace

import pytest

NOTEBOOKS = Path(__file__).resolve().parents[1] / "notebooks/yolo26-kd-indotraffic/notebooks"
REFERENCE = (
    "ilhamrafiqin-brawijaya-university/yolom26-kd-indotraffic/"
    "01_teacher_yolo26m-fux5wdcb-checkpoints:v50"
)
DATA = "/kaggle/input/datasets/ilhamrafiqin/poltrada-v3-yolo26-clean/data.yaml"
NAMES = [f"class-{i}" for i in range(6)]


@pytest.fixture
def tmp_path():
    # Managed Windows environments can deny pytest's default 0700 temp folders.
    directory = Path(__file__).resolve().parents[1] / "outputs/test_csakd_provenance" / uuid.uuid4().hex
    directory.mkdir(parents=True)
    return directory


@pytest.fixture(params=["08_csakd_yolo26s.ipynb", "09_csakd_yolo26n.ipynb"])
def teacher_source(request):
    notebook = json.loads((NOTEBOOKS / request.param).read_text(encoding="utf-8"))
    return "".join(notebook["cells"][8]["source"])


def declarations_function(source):
    tree = ast.parse(source)
    definition = next(
        node for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name == "verify_teacher_declarations"
    )
    namespace = {"DATA": DATA, "Path": Path}
    exec(compile(ast.Module(body=[definition], type_ignores=[]), "<declarations>", "exec"), namespace)
    return namespace[definition.name]


def test_teacher_declarations_match_or_remain_explicitly_missing(teacher_source):
    verify = declarations_function(teacher_source)
    datasets, models = verify({"checkpoint": {"data": DATA, "model": "/weights/yolo26m.pt"}})
    assert datasets == {"checkpoint": DATA}
    assert models == {"checkpoint": "/weights/yolo26m.pt"}
    assert verify({"checkpoint": {}}) == ({}, {})


@pytest.mark.parametrize("bad", [
    {"data": "/different/data.yaml", "model": "yolo26m.pt"},
    {"data": DATA, "model": "yolo26s.pt"},
])
def test_teacher_declared_mismatches_stop_run(teacher_source, bad):
    with pytest.raises(ValueError):
        declarations_function(teacher_source)({"checkpoint": bad})


def provenance_fixture(tmp_path):
    payload = b"mock checkpoint bytes: not a real trained model"
    checkpoint = tmp_path / "best.pt"
    checkpoint.write_bytes(payload)
    entry = SimpleNamespace(size=len(payload), digest=base64.b64encode(hashlib.md5(payload).digest()).decode())
    producer = SimpleNamespace(
        id="fux5wdcb", name="teacher experiment", path=["entity", "project", "fux5wdcb"],
        config={"dataset": DATA, "model": "yolo26m.pt", "epoch": 100},
    )
    artifact = SimpleNamespace(
        version="v50", name="01_teacher_yolo26m-fux5wdcb-checkpoints:v50",
        metadata={"epoch": 7, "dataset": DATA}, digest="mock artifact-level digest",
        manifest=SimpleNamespace(entries={"best.pt": entry}),
        logged_by=lambda: producer,
        download=lambda **kwargs: str(tmp_path),
    )
    requests = []

    def artifact_lookup(reference, **kwargs):
        requests.append((reference, kwargs))
        return artifact

    class MockWandbError(Exception):
        pass

    teacher = SimpleNamespace(
        model=SimpleNamespace(nc=6, names=NAMES, yaml={"scale": "m", "yaml_file": "yolo26m.yaml"}),
        ckpt={"train_args": {"data": DATA, "model": "yolo26m.pt"}},
    )
    namespace = {
        "Path": Path, "DATA": DATA, "EXPECTED_NC": 6, "DIAGNOSTIC_ONLY": False,
        "TEACHER_ARTIFACT": REFERENCE, "TEACHER_RUN_ID": "fux5wdcb", "MODEL_WEIGHTS": "yolo26s.pt",
        "data_info": {"nc": 6, "names": NAMES},
        "class_names": lambda names: [str(value) for value in names],
        "YOLO": lambda path: teacher,
        "wandb": SimpleNamespace(
            Api=lambda: SimpleNamespace(artifact=artifact_lookup),
            errors=SimpleNamespace(Error=MockWandbError),
        ),
    }
    return namespace, artifact, producer, teacher, payload, requests


def test_exact_manifest_version_hash_and_epoch_are_recorded(teacher_source, tmp_path):
    namespace, artifact, _producer, _teacher, payload, requests = provenance_fixture(tmp_path)
    exec(compile(teacher_source, "<mock teacher provenance>", "exec"), namespace)
    provenance = namespace["TEACHER_PROVENANCE"]
    assert requests == [(REFERENCE, {"type": "model"})]
    assert provenance["requested_artifact"] == REFERENCE
    assert provenance["version"] == "v50"
    assert provenance["artifact_metadata"]["epoch"] == 7
    assert provenance["best_pt_sha256"] == hashlib.sha256(payload).hexdigest()
    assert provenance["best_pt_manifest_digest"] == artifact.manifest.entries["best.pt"].digest
    assert provenance["metadata_complete"] is True
    assert provenance["missing_metadata"] == []


@pytest.mark.parametrize("failure", [
    "wrong_version", "stale_file_without_manifest", "size_mismatch", "checksum_mismatch",
    "wrong_producer", "wrong_scale", "wrong_family", "wrong_nc", "wrong_names", "wrong_declared_dataset",
])
def test_teacher_provenance_rejects_concrete_mismatches(teacher_source, tmp_path, failure):
    namespace, artifact, producer, teacher, _payload, _requests = provenance_fixture(tmp_path)
    if failure == "wrong_version":
        artifact.version = "v51"
    elif failure == "stale_file_without_manifest":
        artifact.manifest.entries.clear()  # Local best.pt still exists; it must not be accepted.
    elif failure == "size_mismatch":
        artifact.manifest.entries["best.pt"].size += 1
    elif failure == "checksum_mismatch":
        artifact.manifest.entries["best.pt"].digest = base64.b64encode(b"0" * 16).decode()
    elif failure == "wrong_producer":
        producer.id = "another-run"
    elif failure == "wrong_scale":
        teacher.model.yaml["scale"] = "n"
    elif failure == "wrong_family":
        teacher.model.yaml["yaml_file"] = "yolo11m.yaml"
    elif failure == "wrong_nc":
        teacher.model.nc = 80
    elif failure == "wrong_names":
        teacher.model.names = list(reversed(NAMES))
    elif failure == "wrong_declared_dataset":
        artifact.metadata["dataset"] = "coco8.yaml"
    with pytest.raises(ValueError):
        exec(compile(teacher_source, "<mock teacher provenance>", "exec"), namespace)


def test_unavailable_metadata_warns_without_claiming_verified(teacher_source, tmp_path):
    namespace, artifact, _producer, teacher, payload, _requests = provenance_fixture(tmp_path)
    artifact.logged_by = lambda: None
    artifact.metadata = {}
    teacher.ckpt = {"train_args": {}}
    namespace["DIAGNOSTIC_ONLY"] = True
    with pytest.warns(UserWarning, match="Provenance teacher belum lengkap"):
        exec(compile(teacher_source, "<mock teacher provenance>", "exec"), namespace)
    provenance = namespace["TEACHER_PROVENANCE"]
    assert provenance["metadata_complete"] is False
    assert set(provenance["missing_metadata"]) == {"producer", "declared dataset", "declared model"}
    assert provenance["best_pt_sha256"] == hashlib.sha256(payload).hexdigest()


def test_missing_provenance_blocks_research(teacher_source, tmp_path):
    namespace, artifact, _producer, _teacher, _payload, _requests = provenance_fixture(tmp_path)
    artifact.logged_by = lambda: None
    with (
        pytest.warns(UserWarning, match="Provenance teacher belum lengkap"),
        pytest.raises(ValueError, match="lengkapi bukti source/dataset"),
    ):
        exec(compile(teacher_source, "<mock teacher provenance>", "exec"), namespace)
