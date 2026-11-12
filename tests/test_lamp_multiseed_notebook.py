"""Contracts for independent seed audits, resumable publication, and recovery input."""
# ruff: noqa: S102 -- Compile trusted notebook declarations without running training.
from __future__ import annotations

import ast
import csv
import hashlib
import json
import os
import runpy
import shutil
import uuid
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace

import nbformat
import pytest

ROOT = Path(__file__).resolve().parents[1]
GENERATOR = ROOT / "scripts/generate_lamp_multiseed_notebook.py"
BOOK = ROOT / "notebooks/kaggle/pruning/06_lamp_yolo26s_multiseed_wandb.ipynb"
DATASET_ID = "ilhamrafiqin/poltrada-v3-yolo26-clean"


def file_sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


@pytest.fixture(scope="module")
def notebook():
    return json.loads(BOOK.read_text(encoding="utf-8"))


@pytest.fixture
def scope(notebook):
    values = {"os": os, "uuid": uuid, "csv": csv, "json": json, "hashlib": hashlib, "sha256_file": file_sha,
              "safe_error": str, "TARGET_DEVICE": SimpleNamespace(type="cpu"), "ENVIRONMENT": {"gpu": None}}
    for cell in notebook["cells"]:
        text = "".join(cell["source"])
        tags = cell["metadata"].get("tags", [])
        if text.startswith("from datetime import UTC") or {"config", "runtime", "recovery-input"}.intersection(tags):
            exec(compile(text, f"<{BOOK.name}>", "exec"), values)
    values.update(PLATFORM="local", MODE="smoke", WANDB_ENTITY="entity", WANDB_PROJECT="project")
    return values


def test_standalone_notebook_matches_generator_and_locks_research_queue(notebook, scope):
    nbformat.validate(nbformat.read(BOOK, as_version=4))
    assert runpy.run_path(str(GENERATOR))["build_notebook"]() == notebook
    for cell in notebook["cells"]:
        if cell["cell_type"] == "code":
            compile("".join(cell["source"]), BOOK.name, "exec")
            assert cell["outputs"] == [] and cell["execution_count"] is None
    assert scope["PRUNING_QUEUE"] == ((1.10, 42), (1.20, 42), (1.70, 42), (1.20, 0), (1.20, 2024), (1.20, 7), (1.20, 3407))
    assert scope["ms_run_name"](1.20, 42) == "yolo26s-lamp-su1.20x-seed42"
    assert {seed: row["ref"].split("/")[-1] for seed, row in scope["BASELINE_SOURCES"].items()} == {
        42: "02_baseline_yolo26s-q5kp9k9d-checkpoints:v62",
        0: "02_baseline_yolo26s-7f5jmtwb-checkpoints:v71",
        2024: "02_baseline_yolo26s-1o9ggnyw-checkpoints:v41",
        7: "02_baseline_yolo26s-9cq437l3-checkpoints:v49",
        3407: "02_baseline_yolo26s-bmshdzv9-checkpoints:v43",
    }
    assert len({row["sha256"] for row in scope["BASELINE_SOURCES"].values()}) == 5
    config = next("".join(cell["source"]) for cell in notebook["cells"] if "config" in cell["metadata"].get("tags", []))
    assert 'MODE = "full_val"' in config and scope["RESUME_SUITE_DIR"] is None


def test_layout_keeps_original_numbered_headings_and_workflow_positions(notebook):
    template_path = BOOK.with_name("06_lamp_yolo26s_gflops_no_recovery.ipynb")
    template = json.loads(template_path.read_text(encoding="utf-8"))

    def heading_cells(book):
        return [tuple(line for line in "".join(cell["source"]).splitlines() if line.startswith("#"))
                for cell in book["cells"] if cell["cell_type"] == "markdown"]

    assert heading_cells(notebook) == heading_cells(template)
    assert sum(map(len, heading_cells(notebook))) == 24
    sections_by_tag = {}
    for cell in notebook["cells"]:
        if cell["cell_type"] == "markdown":
            section = cell["source"][0].split()[1]
        for tag in cell["metadata"].get("tags", []):
            sections_by_tag.setdefault(tag, []).append(section)
    assert sections_by_tag["config"] == ["02"]
    assert sections_by_tag["run"] == ["10"]
    assert sections_by_tag["display"] == ["11", "11.2", "11.3", "12", "12.2", "12.3", "13", "13.2", "13.3", "13.4", "14"]
    assert sections_by_tag["recovery-input"] == ["14"]


def test_not_run_display_uses_empty_tables_and_separate_history(notebook, scope, tmp_path, monkeypatch, capsys):
    import IPython.display as notebook_display

    displayed = []
    monkeypatch.setattr(notebook_display, "display", displayed.append)
    scope.update(BUNDLE={"status": "NOT_RUN", "models": [], "reason": "CUDA tidak tersedia."}, SUITE_DIR=tmp_path)
    for cell in notebook["cells"]:
        source = "".join(cell["source"])
        if source.startswith('"""Cross-configuration analysis') or "display" in cell["metadata"].get("tags", []):
            exec(compile(source, f"<{BOOK.name} display>", "exec"), scope)

    assert scope["ANALYSIS"]["status"] == "NOT_RUN"
    assert scope["REGISTRY_ROWS"] == scope["EDA_DETAILS"] == []
    history, = [item for item in displayed if isinstance(item, scope["pd"].DataFrame)]
    assert len(history) == 6
    assert set(history["model_variant"]) == {"s"}
    assert set(history["provenance"]) == {"UNVERIFIED"}
    assert "P5-only NOT_RUN" in capsys.readouterr().out
    assert not (tmp_path / "comparison").exists()


def fixture_data():
    return {"dataset_id": DATASET_ID, "fingerprint": "f" * 64, "full_val_fingerprint": "f" * 64,
            "train_fingerprint": "a" * 64, "provenance_status": "VERIFIED", "class_names": ["Bus"],
            "image_count": 3, "split": "val", "seed": 42}


class FakeArtifact:
    def __init__(self, name, type, metadata):
        self.name = name
        self.type = type
        self.metadata = metadata
        self.files = {}
        self.url = "https://example.invalid/model"

    def add_file(self, path, name):
        self.files[name] = Path(path)

    def download(self, root):
        for name in self.files:
            self.get_path(name).download(root)
        return root

    def get_path(self, name):
        def download(root):
            target = Path(root) / name
            target.parent.mkdir(parents=True, exist_ok=True)
            return shutil.copyfile(self.files[name], target)

        return SimpleNamespace(download=download)

    def wait(self, timeout):
        return self


def test_seven_paired_publications_resume_and_load_recovery(scope, tmp_path):
    dataset = fixture_data()
    artifacts, runs, versions = {}, {}, {}
    for seed, pin in scope["BASELINE_SOURCES"].items():
        pin["ref"] = "entity/project/" + pin["ref"].split("/")[-1]
        original = tmp_path / f"original-seed{seed}.pt"
        original.write_text(f"baseline source {seed}", encoding="utf-8")
        pin["sha256"] = file_sha(original)
        artifact = FakeArtifact(pin["ref"].split("/")[-1], "model", {"seed": seed})
        artifact.qualified_name = pin["ref"]
        artifact.add_file(original, "best.pt")
        artifacts[pin["ref"]] = artifact

    @contextmanager
    def init(**config):
        run_id = config["id"]
        assert run_id not in runs and config["resume"] == "never"
        run = runs[run_id] = SimpleNamespace(config=config, baseline=None, summary={}, artifacts=[])

        def use_artifact(ref, type):
            assert type == "model" and ref in artifacts
            run.baseline = ref

        def log_artifact(artifact):
            version = versions.get(artifact.name, 1 if config["config"]["seed"] == 42 else 0)
            versions[artifact.name] = version + 1
            artifact.qualified_name = f"entity/project/{artifact.name}:v{version}"
            server_copy = Path(artifact.download(tmp_path / "server" / artifact.name / f"v{version}"))
            artifact.files = {name: server_copy / name for name in artifact.files}
            artifacts[artifact.qualified_name] = artifact
            run.artifacts.append(artifact)
            return artifact

        run.use_artifact = use_artifact
        run.log_artifact = log_artifact
        yield run

    api = SimpleNamespace(artifact=lambda ref, type: artifacts[ref],
                          run=lambda ref: SimpleNamespace(logged_artifacts=lambda: runs[ref.split("/")[-1]].artifacts))
    scope.update(wandb=SimpleNamespace(Api=lambda **_: api, Artifact=FakeArtifact, init=init), ms_wandb_environment=lambda: None)
    baseline_calls, audits, validations = [], [], []

    def save_checkpoint(core, path, _provenance):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(core, encoding="utf-8")
        return {"path": str(path), "sha256": file_sha(path), "file_bytes": path.stat().st_size}

    def baseline(path, source, variant, _image, folder, _dataset):
        seed = source["baseline_seed"]
        assert file_sha(path) == scope["BASELINE_SOURCES"][seed]["sha256"] and variant == "s"
        baseline_calls.append(seed)
        saved = save_checkpoint(f"baseline FP32 {seed}", folder / "baseline_fp32.pt", source)
        result = {"checkpoint": saved, "validation_parity_status": "PASS", "reserialized": {
            "gflops": 22, "parameters": 22, "validation_status": "PASS",
            "validation": {"map50_95": seed / 10000 + 0.42, "map50": 0.55},
            "latency": {"status": "PASS", "median_ms": 3.0, "p95_ms": 3.5},
        }}
        scope["write_json"](folder / "baseline_report.json", result)
        return result

    def search(folder, path, source, _baseline, seed, targets):
        assert source["baseline_seed"] == seed and file_sha(path) == source["sha256"]
        results = {"targets": [{"target": target, "status": "TARGET_NOT_REACHED" if target == 1.70 else "REACHED",
                                "speed_up": min(target + 0.001, 1.232)} for target in targets]}
        scope["write_json"](folder / "target_search.json", results)
        return results

    def audit(path, source, _baseline, seed, target):
        assert source["baseline_seed"] == seed and file_sha(path) == source["sha256"]
        audits.append((target["target"], seed))
        # Deliberately make all raw checkpoint bytes identical to exercise independent publication.
        key = "a" * 64
        proof = dict.fromkeys(("identity_sha256", "state_sha256", "structure_sha256", "ledger_sha256", "plan_sha256"), key)
        proof.update(structure_signature={}, audit={"target_policy_eligible_channels": 3}, oracle={"status": "PASS"},
                     pruned={"gflops": 22 / target["speed_up"], "parameters": 10,
                             "latency": {"status": "PASS", "median_ms": 2.2, "p95_ms": 2.5}},
                     selected_plan={"root": [0]}, groups=[])
        return key, proof, [], []

    def validate(row, _checkpoint, _dataset, folder):
        if folder.name.endswith("seed7") and folder.name not in validations:
            validations.append(folder.name)
            raise KeyboardInterrupt("Interruption after raw reload")
        validations.append(folder.name)
        row.update(validation_status="PASS", validation={"map50_95": 0.35, "map50": 0.48, "validation_image_count": 3})

    scope.update(
        MODE="full_val", TARGET_DEVICE=SimpleNamespace(type="cuda"),
        torch=SimpleNamespace(cuda=SimpleNamespace(empty_cache=lambda: None)), ms_dataset=lambda _: (dataset, dataset),
        checkpoint_provenance=lambda path, ref, **kwargs: {
            "checkpoint_path": str(path), "requested_artifact": ref, "sha256": file_sha(path), **kwargs},
        gflops_baseline=baseline, ms_search=search, ms_audit=audit, workflow_save_checkpoint=save_checkpoint,
        verify_fresh_reload=lambda *_args, **_kwargs: {}, gflops_reload=lambda _, core: core, model_state_sha256=lambda core: core,
        ms_signature=lambda _: {}, ms_probe=lambda seed: seed, seed_environment=lambda _: None, gflops_validation=validate,
        build_run_eda=lambda *_: {"status": "MEASURED"}, metric_comparison=lambda *_: {},
    )
    suite = tmp_path / "first-suite"
    suite.mkdir()
    execute = scope["ms_execute"]
    partial = execute(suite)
    assert partial["status"] == "PARTIAL" and partial["registry"]["confirmed_artifact_count"] == 5
    assert partial["registry"]["rows"][-2]["reload_status"] == "PASS"
    assert partial["registry"]["rows"][-2]["validation_status"] == "NOT_RUN"
    assert partial["registry"]["rows"][-1]["target_status"] == "NOT_RUN"
    assert len(partial["registry"]["rows"]) == 7

    moved = tmp_path / "relocated-suite"
    suite.rename(moved)
    scope["RESUME_SUITE_DIR"] = moved
    result = execute(moved)
    rows = result["registry"]["rows"]
    assert result["status"] == "PASS" and result["registry"]["confirmed_artifact_count"] == 7
    assert audits == list(scope["PRUNING_QUEUE"]) and baseline_calls == [42, 0, 2024, 7, 3407]
    assert len(validations) == 8
    assert len({row["raw_sha256"] for row in rows}) == 1
    assert len({row["checkpoint_relative_path"] for row in rows}) == 7
    assert len({row["artifact_ref"] for row in rows}) == len({row["wandb_run_id"] for row in rows}) == len(runs) == 7
    assert {run.config["group"] for run in runs.values()} == {"yolo26s-lamp-multiseed-first-suite"}
    assert all(row["recovery_ready"] and row["validation_image_count"] == 3 for row in rows)
    assert rows[2]["target_status"] == "TARGET_NOT_REACHED" and rows[2]["actual_gflops_speedup"] == 1.232
    with (moved / "checkpoint_registry.csv").open(encoding="utf-8", newline="") as stream:
        assert [row["artifact_ref"] for row in csv.DictReader(stream)] == [row["artifact_ref"] for row in rows]
    for row in rows:
        seed = row["pruning_seed"]
        run = runs[row["wandb_run_id"]]
        artifact, = run.artifacts
        assert artifact.qualified_name.endswith(":v1" if seed == 42 else ":v0")
        assert artifact.name == f"yolo26s-lamp-su{row['target_speedup']:.2f}x-seed{seed}-raw"
        assert row["source_baseline_artifact_ref"] == run.baseline == scope["BASELINE_SOURCES"][seed]["ref"]
        assert artifact.metadata["baseline_source_sha256"] == row["source_baseline_sha256"]
        assert run.summary["baseline_map50_95"] == row["baseline_map50_95"] == seed / 10000 + 0.42
        assert run.summary["map50_95"] == row["map50_95"] == 0.35

    lookup = scope["recovery_config_from_registry"]
    registry_path = moved / "checkpoint_registry.json"
    reference, raw_hash, seed, baseline_ref, baseline_hash = lookup(registry_path, seed=0, include_baseline=True)
    scope.update(RAW_ARTIFACT_REF=reference, EXPECTED_RAW_SHA256=raw_hash, SEED=seed, BASELINE_REF=baseline_ref,
                 BASELINE_SHA256=baseline_hash, workflow_login=lambda: None, recovery_read_json=scope["ms_read"])
    recovery = (ROOT / "scripts/pruned_fine_tuned_source.py").read_text(encoding="utf-8")
    loader = next(node for node in ast.parse(recovery).body if isinstance(node, ast.FunctionDef) and node.name == "fine_tuned_source")
    exec(compile(ast.Module(body=[loader], type_ignores=[]), "recovery-input", "exec"), scope)
    loaded, files = scope["fine_tuned_source"](tmp_path / "consumer")
    assert loaded["manifest"]["seed"] == loaded["manifest"]["baseline_seed"] == seed == 0
    assert file_sha(files["pruned_fp32.pt"]) == raw_hash
    assert file_sha(files["baseline_fp32.pt"]) == rows[3]["baseline_fp32_sha256"]

    scope["BASELINE_SOURCES"][0]["sha256"] = "e" * 64
    with pytest.raises(ValueError, match="Kontrak resume berbeda"):
        execute(moved)
    scope["BASELINE_SOURCES"][0]["sha256"] = baseline_hash
    assert len(runs) == 7
    first_ids = set(runs)
    scope["RESUME_SUITE_DIR"] = None
    fresh = tmp_path / "fresh-suite"
    fresh.mkdir()
    second = execute(fresh)
    assert second["status"] == "PASS" and len(runs) == 14
    assert not first_ids.intersection(row["wandb_run_id"] for row in second["registry"]["rows"])
    assert all(a["artifact_ref"] != b["artifact_ref"] for a, b in zip(rows, second["registry"]["rows"], strict=True))


def raw_fixture(scope, tmp_path):
    folder = tmp_path / "yolo26s-seed42/target-1.20x-seed42"
    folder.mkdir(parents=True)
    path = folder / "pruned_fp32.pt"
    path.write_bytes(b"raw FP32 fixture")
    baseline_path = folder.parent / "baseline/baseline_fp32.pt"
    baseline_path.parent.mkdir()
    baseline_path.write_bytes(b"baseline FP32 fixture")
    baseline = {"checkpoint": {"sha256": file_sha(baseline_path)}, "reserialized": {"gflops": 22, "parameters": 22},
                "validation_parity_status": "PASS"}
    source = {"requested_artifact": "entity/project/baseline:v62", "sha256": "b" * 64, "baseline_seed": 42}
    attempt = {"configuration_id": folder.name, "target_speedup": 1.20, "pruning_seed": 42, "recovery_seed": 42,
               "target_status": "REACHED", "actual_gflops_speedup": 1.201, "identity_sha256": "a" * 64}
    candidate = {"saved_pruned": {"sha256": file_sha(path)}, "state_sha256": "e" * 64, "structure_signature": {},
                 "identity_sha256": attempt["identity_sha256"], "provenance": source,
                 "audit_files": {}, "structural_smoke_status": "PASS", "oracle": {"status": "PASS"},
                 "reload_status": "PASS", "validation_status": "NOT_RUN", "eda_status": "NOT_RUN", "errors": [],
                 "pruned": {"gflops": 22 / 1.201, "parameters": 21, "latency": {"status": "NOT_RUN"}}}
    scope.update(IMGSZ=64, IMPLEMENTATION_SHA256="implementation", metric_comparison=lambda *_: {})
    return folder, candidate, attempt, baseline, source, fixture_data()


def test_validation_resume_preserves_raw_and_rejects_changed_sha(scope, tmp_path):
    folder, raw, attempt, baseline, source, data = raw_fixture(scope, tmp_path)
    scope["write_json"](folder / "inventories.json", {"before": [], "after": []})
    validation_calls = []

    def validate(record, *_args):
        validation_calls.append(1)
        if len(validation_calls) == 1:
            record["validation_status"] = "FAIL"
        else:
            record.update(validation_status="PASS", validation={"map50_95": 0.31})

    scope.update(gflops_validation=validate, seed_environment=lambda _: None, build_run_eda=lambda *_: {"status": "MEASURED"})
    complete = scope["ms_complete_raw"]
    for expected in ("FAIL", "PASS", "PASS"):
        complete(folder, raw, attempt, baseline, source, data, "test-suite")
        assert raw["validation_status"] == expected
        assert file_sha(folder / "pruned_fp32.pt") == raw["saved_pruned"]["sha256"]
    assert len(validation_calls) == 2
    (folder / "pruned_fp32.pt").write_bytes(b"tampered")
    complete(folder, raw, attempt, baseline, source, data, "test-suite")
    assert raw["integrity_status"] == "FAIL" and len(validation_calls) == 2


def test_publish_recovers_committed_version_after_wait_timeout(scope, tmp_path):
    folder, raw, attempt, baseline, source, data = raw_fixture(scope, tmp_path)
    scope["MODE"] = "full_val"
    raw["pruned"].update(validation={"map50_95": 0.31, "map50": 0.49},
                         latency={"status": "PASS", "median_ms": 8.2, "p95_ms": 9.1})
    manifest = scope["ms_raw_manifest"](raw, attempt, baseline, source, data, "test-suite")
    manifest_path = folder / "raw_pruned_manifest.json"
    scope["write_json"](manifest_path, manifest)
    logged = []

    class Artifact(FakeArtifact):
        def __init__(self, name, type, metadata):
            super().__init__(name, type, metadata)
            self.qualified_name = f"entity/project/{name}:v8"

        def wait(self, timeout):
            raise TimeoutError("Server committed before wait timeout")

    @contextmanager
    def init(**kwargs):
        assert kwargs["name"] == "yolo26s-lamp-su1.20x-seed42"

        def log(artifact):
            logged.append(artifact)
            return artifact

        yield SimpleNamespace(log_artifact=log, use_artifact=lambda *_, **__: None, summary={})

    api = SimpleNamespace(run=lambda _ref: SimpleNamespace(logged_artifacts=lambda: logged),
                          artifact=lambda ref, type: next(item for item in logged if item.qualified_name == ref))
    scope.update(wandb=SimpleNamespace(Api=lambda **_: api, Artifact=Artifact, init=init), ms_wandb_environment=lambda: None)
    first = scope["ms_publish"](folder, manifest)
    assert first["status"] == "PENDING" and len(logged) == 1
    second = scope["ms_publish"](folder, manifest)
    assert second["status"] == "CONFIRMED" and second["reference"] == "entity/project/yolo26s-lamp-su1.20x-seed42-raw:v8"
    assert second["run_id"] == first["run_id"] and len(logged) == 1
    assert logged[0].metadata["raw_sha256"] == raw["saved_pruned"]["sha256"]
    assert logged[0].metadata["dataset_fingerprint"] == data["fingerprint"]
    assert logged[0].metadata["latency_p95_ms"] == 9.1

    scope["write_json"](tmp_path / "checkpoint_registry.json", {
        "format_version": 1, "mode": "full_val", "workflow": "lamp-yolo26s-multiseed",
        "rows": [{**attempt, "recovery_seed": 3407, "model_variant": "s", "evaluation_mode": "full_val",
                  "dataset_provenance_status": "VERIFIED", "publication_status": "CONFIRMED", "recovery_ready": True,
                  "artifact_ref": second["reference"], "raw_sha256": raw["saved_pruned"]["sha256"]}],
    })
    reference, expected, seed = scope["recovery_config_from_registry"](tmp_path / "checkpoint_registry.json", seed=3407)
    assert seed == 3407 and manifest["seed"] == 42

    assert reference == second["reference"] and expected == raw["saved_pruned"]["sha256"]


def test_recovery_loader_requires_explicit_diagnostic_and_full_verified_artifact(scope, tmp_path):
    path = tmp_path / "checkpoint_registry.json"
    row = {"target_speedup": 1.70, "recovery_seed": 42, "target_status": "TARGET_NOT_REACHED", "model_variant": "s",
           "evaluation_mode": "full_val", "dataset_provenance_status": "VERIFIED", "publication_status": "CONFIRMED",
           "recovery_ready": True, "artifact_ref": "entity/project/yolo26s-lamp-su1.70x-seed42-raw:v3", "raw_sha256": "a" * 64}
    registry = {"format_version": 1, "mode": "full_val", "workflow": "lamp-yolo26s-multiseed", "rows": [row]}
    scope["write_json"](path, registry)
    with pytest.raises(ValueError, match="allow_diagnostic"):
        scope["recovery_config_from_registry"](path, seed=42, target=1.70)
    assert scope["recovery_config_from_registry"](path, seed=42, target=1.70, allow_diagnostic=True)[0] == row["artifact_ref"]
    row["artifact_ref"] = "entity/project/something:latest"
    scope["write_json"](path, registry)
    with pytest.raises(ValueError, match="versi W&B tetap"):
        scope["recovery_config_from_registry"](path, seed=42, target=1.70, allow_diagnostic=True)
