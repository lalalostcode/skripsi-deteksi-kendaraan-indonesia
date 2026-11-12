"""Selection, parallel isolation, immutable publication and standalone notebook contracts."""
# ruff: noqa: S102 -- Execute trusted notebook declarations without invoking training.
from __future__ import annotations

import ast
import hashlib
import json
import shutil
import uuid
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace

import nbformat
import pytest

from scripts.generate_pruned_fine_tuned_notebooks import (
    ROOT,
    STUDENT_PINS,
    build_notebook,
)


def declarations(*names):
    source = (ROOT / "scripts/pruned_finetune_selected_source.py").read_text(encoding="utf-8")
    return ast.Module(body=[node for node in ast.parse(source).body if isinstance(node, ast.FunctionDef) and node.name in names],
                      type_ignores=[])


def configuration(platform="filkom", **overrides):
    scope = {}
    for cell in build_notebook(platform)["cells"]:
        tags = cell["metadata"].get("tags", [])
        if {"config", "selection", "protocol"}.intersection(tags):
            exec("".join(cell["source"]), scope)
            if "config" in tags:
                scope.update(overrides)
    exec(compile(declarations("ft_select", "ft_identity", "ft_name", "ft_group"), "selection", "exec"), scope)
    return scope


@pytest.mark.parametrize("platform", ["filkom", "kaggle"])
def test_notebook_generated_with_single_raw_entry_and_baseline_protocol(platform):
    book = build_notebook(platform)
    path = ROOT / "notebooks" / platform / "recovery/02_yolo26s_standard_finetuning.ipynb"
    assert json.loads(path.read_text(encoding="utf-8")) == book
    nbformat.validate(nbformat.from_dict(book))
    for cell in book["cells"]:
        if cell["cell_type"] == "code":
            compile("".join(cell["source"]), str(path), "exec")
            assert cell["execution_count"] is None and cell["outputs"] == []
    scope = configuration(platform)
    assert scope["EPOCHS"] == 100 and scope["RESUME_UPLOAD_INTERVAL"] == 5
    assert scope["ft_name"]() == "yolo26s-lamp-su1.20x-seed42-ft"
    assert scope["ft_group"]() == "yolo26s-lamp-ft-20261107T020459Z_45575a7f"
    assert scope["SEED"] == scope["ft_identity"]()["seed"] == 42


@pytest.mark.parametrize("inputs", [[], list(STUDENT_PINS)[:2], ["untrusted:v0"]])
def test_invalid_input_rejected_before_dependencies_or_model_loading(inputs):
    with pytest.raises(ValueError, match="tepat satu"):
        configuration(STUDENT_INPUTS=inputs)


@pytest.mark.parametrize("name", STUDENT_PINS)
def test_each_input_inherits_its_raw_identity(name):
    scope = configuration(STUDENT_INPUTS=[name])
    identity = scope["ft_identity"]()
    pin = STUDENT_PINS[name]
    assert identity["seed"] == scope["SEED"] == pin["seed"]
    assert identity["raw_sha256"] == pin["raw_sha256"]
    assert identity["baseline_artifact_ref"] == pin["baseline_ref"]
    assert identity["target_speedup"] == pin["target"]
    assert identity["raw_artifact_ref"].endswith(name)
    assert len(pin["manifest_sha256"]) == 64


def test_parallel_choices_have_shared_group_and_distinct_run_names():
    scopes = [configuration(STUDENT_INPUTS=[name]) for name in STUDENT_PINS]
    assert len({scope["ft_name"]() for scope in scopes}) == 7
    assert len({scope["ft_group"]() for scope in scopes}) == 1
    assert configuration(PLATFORM="kaggle")["ft_group"]() == scopes[0]["ft_group"]()
    assert configuration(MODE="smoke")["ft_group"]().endswith("-smoke")


def test_study_settings_match_across_seeds_and_distinguish_training_budget():
    settings = []
    for name in STUDENT_PINS:
        scope = configuration(STUDENT_INPUTS=[name])
        exec(compile(declarations("ft_study_settings"), "study-settings", "exec"), scope)
        settings.append(scope["ft_study_settings"]())
    assert all(value == settings[0] for value in settings)
    scope["EPOCHS"] = 50
    assert scope["ft_study_settings"]() != settings[0]


def test_raw_package_hash_and_seed_verified_before_loading(tmp_path):
    scope = configuration()
    scope.update(Path=Path, json=json, shutil=shutil,
                 sha256_file=lambda path: hashlib.sha256(Path(path).read_bytes()).hexdigest())
    exec(compile(declarations("ft_read", "ft_source"), "raw-package", "exec"), scope)
    folder = tmp_path / "run"
    cache = folder / "raw_artifact"
    cache.mkdir(parents=True)
    name = scope["STUDENT_INPUTS"][0]
    pin = scope["STUDENT_PINS"][name]
    for file in ("pruned_fp32.pt", "baseline_fp32.pt"):
        (cache / file).write_bytes(file.encode())
    pin["raw_sha256"] = scope["sha256_file"](cache / "pruned_fp32.pt")
    pin["baseline_fp32_sha256"] = scope["sha256_file"](cache / "baseline_fp32.pt")
    manifest = {"suite_id": scope["PRUNING_SUITE_ID"], "seed": 42, "baseline_seed": 42, "target_speedup": 1.2,
                "stage": "raw_pruned", "format_version": 2, "model_variant": "s", "structural_status": "PASS",
                "oracle_status": "PASS", "reload_status": "PASS", "audit_files": {},
                "source_baseline": {"requested_artifact": pin["baseline_ref"], "sha256": pin["baseline_source_sha256"]},
                "raw_checkpoint": {"sha256": pin["raw_sha256"]}, "baseline_checkpoint": {"sha256": pin["baseline_fp32_sha256"]}}
    path = cache / "raw_pruned_manifest.json"
    path.write_text(json.dumps(manifest), encoding="utf-8")
    pin["manifest_sha256"] = scope["sha256_file"](path)
    loaded, files = scope["ft_source"](folder)
    assert loaded["seed"] == 42 and len(files) == 3
    backup = tmp_path / "trusted_package"
    shutil.copytree(cache, backup)
    scope["LOCAL_RAW_PACKAGES"] = {name: backup}
    (cache / "pruned_fp32.pt").unlink()
    assert scope["ft_source"](folder)[0]["seed"] == 42
    (cache / "pruned_fp32.pt").write_bytes(b"tampered")
    with pytest.raises(ValueError, match="SHA-256"):
        scope["ft_source"](folder)
    manifest["seed"] = 0
    path.write_text(json.dumps(manifest), encoding="utf-8")
    pin["manifest_sha256"] = scope["sha256_file"](path)
    with pytest.raises(ValueError, match="Identitas"):
        scope["ft_source"](folder)


def test_publisher_recovers_timeout_and_verifies_remote_bytes(tmp_path):
    def digest(path):
        return hashlib.sha256(Path(path).read_bytes()).hexdigest()

    def write_json(path, value):
        Path(path).write_text(json.dumps(value), encoding="utf-8")

    logged = []

    class Artifact:
        def __init__(self, name, type, metadata):
            self.name, self.type, self.metadata = name, type, metadata
            self.files = {}
            self.qualified_name = f"entity/project/{name}:v0"
            self.url = "https://example.invalid/artifact"

        def add_file(self, path, name):
            self.files[name] = Path(path)

        def wait(self, timeout):
            raise TimeoutError("Committed remotely before response")

        def get_path(self, name):
            return SimpleNamespace(download=lambda root: shutil.copyfile(self.files[name], Path(root) / name))

        def download(self, root):
            for name in self.files:
                self.get_path(name).download(root)
            return root

    api = SimpleNamespace(run=lambda _: SimpleNamespace(logged_artifacts=lambda: logged),
                          artifact=lambda ref: next(item for item in logged if item.qualified_name == ref))
    scope = {"Path": Path, "json": json, "TemporaryDirectory": TemporaryDirectory, "WANDB_ENTITY": "entity",
             "WANDB_PROJECT": "project", "sha256_file": digest, "write_json": write_json,
             "workflow_version_ref": str, "wandb": SimpleNamespace(Artifact=Artifact, Api=lambda **_: api)}
    exec(compile(declarations("ft_read", "ft_publish"), "publisher", "exec"), scope)

    def log(artifact, aliases):
        assert aliases == ["latest", "best", "last"]
        logged.append(artifact)
        return artifact

    run = SimpleNamespace(name="yolo26s-lamp-su1.20x-seed0-ft", id="run1", log_artifact=log)
    checkpoint = tmp_path / "best.pt"
    checkpoint.write_bytes(b"trusted checkpoint bytes")
    with pytest.raises(TimeoutError):
        scope["ft_publish"](run, tmp_path, "checkpoints", {"best.pt": checkpoint}, {})
    result = scope["ft_publish"](run, tmp_path, "checkpoints", {"best.pt": checkpoint}, {})
    assert result["status"] == "CONFIRMED" and len(logged) == 1
    assert result["files_sha256"]["best.pt"] == digest(checkpoint)
    checkpoint.write_bytes(b"changed")
    with pytest.raises(ValueError, match="berubah"):
        scope["ft_publish"](run, tmp_path, "checkpoints", {"best.pt": checkpoint}, {})


def test_finished_run_requires_epoch_history_tables_and_media(tmp_path):
    def write_json(path, value):
        Path(path).write_text(json.dumps(value), encoding="utf-8")

    write_json(tmp_path / "wandb_identity.json", {"id": "verified"})
    write_json(tmp_path / "wandb_media.json", {"keys": ["visuals/eda/accuracy"]})
    summary = {"visuals/eda/accuracy": {"_type": "image-file", "path": "media/accuracy.png"}}
    tables = {"comparison/three_stage_metrics": 3, "comparison/three_stage_per_class": 18, "comparison/epoch_history": 1}
    for key, rows in tables.items():
        summary[key] = {"nrows": rows, "path": key + ".json"}
    for stage in ("baseline", "raw_pruned", "recovered"):
        summary.update({f"{stage}/{metric}": 0.0 for metric in ("map50_95", "map50", "precision", "recall", "gflops", "parameters")})
    remote = SimpleNamespace(summary=summary, url="https://example.invalid/run", lastHistoryStep=10,
                             history=lambda **_: [{"epoch": 1, "fitness": 0}],
                             files=lambda **_: [SimpleNamespace(name=path) for path in ["media/accuracy.png", *[k + ".json" for k in tables]]])
    scope = {"Path": Path, "json": json, "write_json": write_json, "WANDB_ENTITY": "entity", "WANDB_PROJECT": "project",
             "wandb": SimpleNamespace(Api=lambda **_: SimpleNamespace(run=lambda _: remote))}
    exec(compile(declarations("ft_read", "ft_epoch_history", "ft_verify_logging"), "logging-verification", "exec"), scope)
    result = {"epochs": 1, "dataset": {"class_names": list(range(6))}}
    assert scope["ft_verify_logging"](result, tmp_path)["status"] == "PASS"
    assert summary["logging_verified"]
    summary["comparison/three_stage_metrics"]["nrows"] = 2
    with pytest.raises(ValueError, match="Tabel"):
        scope["ft_verify_logging"](result, tmp_path)
    summary["comparison/three_stage_metrics"]["nrows"] = 3
    del summary["visuals/eda/accuracy"]
    with pytest.raises(ValueError, match="Media"):
        scope["ft_verify_logging"](result, tmp_path)


def test_history_requests_every_step_through_public_graphql_api():
    calls = []

    def history(**kwargs):
        calls.append(kwargs)
        return [{"epoch": 1, "fitness": 0.0}]

    scope = {}
    exec(compile(declarations("ft_epoch_history"), "history", "exec"), scope)
    rows = scope["ft_epoch_history"](SimpleNamespace(history=history, lastHistoryStep=700))
    assert rows == [{"epoch": 1, "fitness": 0.0}]
    assert calls == [{"keys": ["epoch", "fitness"], "samples": 701, "pandas": False}]


def test_all_eda_csvs_are_interactive_tables_with_student_identity(tmp_path):
    import pandas as pd

    scope = configuration(MODE="smoke")
    folder = tmp_path / "eda" / "pruning"
    folder.mkdir(parents=True)
    csv_path = folder / "lamp_scores.csv"
    csv_path.write_text("layer_name,importance\nmodel.0.conv,0.125\nmodel.1.conv,0.75\n", encoding="utf-8")
    tables = []
    scope.update(pd=pd, wandb=SimpleNamespace(Table=lambda dataframe: dataframe),
                 sha256_file=lambda _: "csv-hash", write_json=lambda *_: None)
    exec(compile(declarations("ft_log_eda_tables"), "eda-tables", "exec"), scope)
    manifest = scope["ft_log_eda_tables"](SimpleNamespace(log=tables.append), {"raw_gflops_speedup": 1.2002}, tmp_path)
    key = "tables/eda/pruning/lamp_scores"
    assert manifest == {key: {"rows": 2, "source": "eda/pruning/lamp_scores.csv", "source_sha256": "csv-hash"}}
    frame = tables[0][key]
    assert frame["seed"].tolist() == [42, 42]
    assert frame["target_speedup"].tolist() == [1.2, 1.2]
    assert frame["actual_gflops_speedup"].tolist() == [1.2002, 1.2002]
    assert frame["importance"].tolist() == [0.125, 0.75]


def test_completed_epoch_snapshot_contains_training_config_for_remote_resume(tmp_path):
    scope = configuration(MODE="smoke")
    state = {"training_finished": True, "epoch_rows": [{"epoch": 1, "fitness": 0.0}]}
    index = {"epoch": 1, "training_finished": True, "relative_path": "checkpoints/resume.pt",
             "best": {"relative_path": "checkpoints/best.pt", "epoch": 1, "fitness": 0.0}}
    for name in ("checkpoints/resume.pt", "checkpoints/best.pt", "latest_state.json", "resume_contract.json",
                 "wandb_identity.json", "epoch_metrics.json", "training_config.json"):
        path = tmp_path / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("{}", encoding="utf-8")
    (tmp_path / "logged_epochs.json").write_text("[1]", encoding="utf-8")
    (tmp_path / "latest_state.json").write_text(json.dumps(index), encoding="utf-8")
    snapshots = []

    def publish(_run, _folder, label, files, metadata, event):
        snapshots.append(files)
        assert label == "resume" and event == "epoch-001"
        assert all(Path(path).is_file() for path in files.values())

    def unexpected_log(_payload):
        raise AssertionError("Already logged epoch was replayed")

    scope.update(Path=Path, json=json, uuid=uuid, recovery_committed_state=lambda *_: (state, index),
                 recovery_epoch_rows=lambda rows: rows, ft_publish=publish)
    exec(compile(declarations("ft_read", "ft_fit"), "snapshot-files", "exec"), scope)
    run = SimpleNamespace(summary={}, log=unexpected_log)
    result = scope["ft_fit"](None, tmp_path / "raw.pt", {"raw_checkpoint": {"sha256": "source"}}, {}, {}, [],
                             tmp_path / "data.yaml", tmp_path, run, {"sha256": "contract"})
    assert result is state and len(snapshots) == 1
    assert snapshots[0]["training_config.json"] == tmp_path / "training_config.json"


def test_group_reader_handles_cpu_gpu_field_omitted_by_wandb():
    import pandas as pd

    scope = configuration(MODE="smoke")
    name = scope["STUDENT_INPUTS"][0]
    pin = scope["STUDENT_PINS"][name]
    summary = {"status": "PASS", "logging_verified": True, "raw_artifact_ref": scope["ft_identity"]()["raw_artifact_ref"],
               "raw_sha256": pin["raw_sha256"], "platform": "filkom", "target_speedup": 1.20,
               "seed": 42, "baseline_map50_95": 0.5, "raw_pruned_map50_95": 0.2, "recovered_map50_95": 0.4}
    run = SimpleNamespace(summary=summary, config={"protocol": "selected-raw-finetune-v2", "study_signature": "settings",
                                                   "platform": "filkom", "environment": {"gpu": None}},
                          created_at="2026-11-07T00:00:00Z", url="https://example.invalid/run")
    displays = []
    scope.update(pd=pd, display=displays.append, workflow_login=lambda: None, workflow_digest=lambda _: "settings",
                 wandb=SimpleNamespace(Api=lambda **_: SimpleNamespace(runs=lambda *args, **kwargs: [run])))
    exec(compile(declarations("ft_study_settings", "ft_group_results"), "group-reader", "exec"), scope)
    rows = scope["ft_group_results"]()
    assert len(rows) == 1 and rows[0]["gpu"] == "CPU"
    assert len(displays) == 1 and displays[0].index[0] == ("filkom", "CPU")
