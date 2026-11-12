"""Offline publication order, fixed lineage and retry of a committed epoch."""
# ruff: noqa: S102, C408 -- Run trusted standalone declarations with mock W&B.
from __future__ import annotations

import hashlib
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[1]


def test_raw_publication_gate_and_retry_keep_training_state(tmp_path, monkeypatch):
    events = []
    failed = {"epoch-001": True}
    initializations = []

    class Artifact:
        def __init__(self, name, type, metadata):
            self.name, self.type, self.metadata = name, type, metadata
            self.files = {}

        def add_file(self, path, name):
            self.files[name] = Path(path).read_bytes()

    class Run:
        def __init__(self, kwargs):
            self.name, self.id = kwargs["name"], kwargs["id"]

        def log_artifact(self, artifact, aliases):
            stage = aliases[0]
            events.append(("upload", stage, tuple(aliases)))
            artifact.qualified_name = f"entity/project/{artifact.name}:v0"

            def wait(timeout):
                assert timeout > 0
                if failed.pop(stage, False):
                    raise TimeoutError(stage)
                events.append(("confirmed", stage))

            artifact.wait = wait
            return artifact

        def use_artifact(self, reference, type):
            assert reference.endswith(":v0") and type == "model"
            events.append(("use", reference))

        def log(self, row, *, step):
            events.append(("metrics", step, row["optimizer_steps"]))

        def finish(self):
            pass

    def init(**kwargs):
        initializations.append(kwargs)
        return Run(kwargs)

    monkeypatch.setitem(sys.modules, "wandb", SimpleNamespace(Artifact=Artifact, init=init))
    scope = dict(PLATFORM="filkom", MODE="smoke", RECOVERY_METHOD="finetune", RECOVERY_EPOCHS=2, SEED=42,
                 WANDB_TRACKING=True, WANDB_PROJECT="project", WANDB_ENTITY="entity", NOTEBOOK_VARIANT="n",
                 safe_error=str, sha256_file=lambda path: hashlib.sha256(Path(path).read_bytes()).hexdigest())
    for name in ("lamp_workflow_source", "standard_finetuning_runtime_source"):
        path = ROOT / "scripts" / f"{name}.py"
        exec(compile(path.read_text(encoding="utf-8"), str(path), "exec"), scope)
    scope["workflow_login"] = lambda: None
    files = {}
    for name in ("pruned_fp32.pt", "baseline_fp32.pt", "raw_pruned_manifest.json"):
        files[name] = tmp_path / name
        files[name].write_text(name, encoding="utf-8")
    job = tmp_path / "job"
    source = {"path": files["raw_pruned_manifest.json"], "reference": None}
    failed["raw-pruned"] = True
    with pytest.raises(scope["WorkflowUploadPending"]):
        scope["recovery_raw_reference"](source, files, job, 1.02)
    assert not any(event[0] == "use" for event in events)
    raw_ref = scope["recovery_raw_reference"](source, files, job, 1.02)
    write_json = scope["write_json"]
    publish_epoch = scope["recovery_epoch_publication"]
    tracking = scope["workflow_tracking"]
    for filename in ("resume_contract.json", "training_config.json", "epoch_metrics.json"):
        write_json(job / filename, {})

    def epoch_files(epoch):
        best = job / "best-001.pt"
        if not best.is_file():
            best.write_bytes(b"best")
        path = job / f"resume-{epoch:03d}.pt"
        path.write_bytes(f"training state {epoch}".encode())
        sha = scope["sha256_file"]
        index = {"epoch": epoch, "relative_path": path.name, "sha256": sha(path),
                 "best": {"epoch": 1, "relative_path": best.name, "sha256": sha(best)}}
        write_json(job / "latest_state.json", index)
        row = {"epoch": epoch, "optimizer_steps": epoch * 3, "fitness": 0.3, "losses": {"cls_loss": 1.0},
               "metrics": {"metrics/mAP50-95(B)": 0.3}, "learning_rates": [0.0001]}
        return index, row

    with pytest.raises(scope["WorkflowUploadPending"]), tracking(job, "n", 1.02, recovery=True, source_ref=raw_ref) as run:
        events.append(("training", 1))
        idx, row = epoch_files(1)
        publish_epoch(run, job, idx, row, raw_ref)
    original = (job / idx["relative_path"]).read_bytes()
    with tracking(job, "n", 1.02, recovery=True, source_ref=raw_ref) as run:
        publish_epoch(run, job, idx, row, raw_ref)
        events.append(("training", 2))
        publish_epoch(run, job, *epoch_files(2), raw_ref)
        scope["workflow_publish"](run, job, "recovered", {"recovered.pt": job / "best-001.pt"}, aliases=["recovered"])
    assert initializations[0]["id"] == initializations[1]["id"] != initializations[2]["id"] == initializations[3]["id"]
    assert [item["resume"] for item in initializations] == ["never", "allow", "never", "allow"]
    assert events.index(("confirmed", "raw-pruned")) < events.index(("use", raw_ref)) < events.index(("training", 1))
    assert events.index(("confirmed", "epoch-001")) < events.index(("training", 2))
    assert events.index(("confirmed", "epoch-002")) < events.index(("confirmed", "recovered"))
    assert ("upload", "epoch-001", ("epoch-001", "latest", "best")) in events
    assert ("upload", "epoch-002", ("epoch-002", "latest")) in events
    assert events.count(("metrics", 1, 3)) == 1 and (job / idx["relative_path"]).read_bytes() == original


def test_run_version_rejects_mutable_alias():
    scope = {}
    path = ROOT / "scripts/lamp_workflow_source.py"
    exec(compile(path.read_text(encoding="utf-8"), str(path), "exec"), scope)
    with pytest.raises(ValueError, match="versi tetap"):
        scope["workflow_version_ref"]("entity/project/collection:latest")
