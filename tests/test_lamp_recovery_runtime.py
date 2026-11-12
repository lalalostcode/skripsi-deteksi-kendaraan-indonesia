"""Offline orchestration, failure propagation and W&B mocks; no training/network."""
# ruff: noqa: S102 -- Execute only trusted notebook declaration cells.
import json
from pathlib import Path
from types import SimpleNamespace

import nbformat
import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK = ROOT / "notebooks/kaggle/pruning/04_lamp_yolo26ns_finetune_recovery.ipynb"


@pytest.fixture
def env(monkeypatch, tmp_path):
    import requests

    def forbidden(*a, **kw):
        pytest.fail("Orchestration mocks must not train or access the network")

    monkeypatch.setattr(requests.sessions.Session, "request", forbidden)
    monkeypatch.setattr(torch.Tensor, "backward", forbidden)
    scope = {"__name__": "recovery_mock", "display": lambda _value: None}
    for cell in nbformat.read(NOTEBOOK, as_version=4).cells:
        if cell.cell_type == "code" and set(cell.metadata.get("tags", [])) & {"config", "helpers", "imports"}:
            exec(compile(cell.source, "notebook04-declaration", "exec"), scope)
    scope.update({"TARGET_DEVICE": torch.device("cpu"), "DEVICE": "cpu", "IMGSZ": 64,
                  "WANDB_TRACKING": False, "RUN_LATENCY_BENCHMARK": False})
    torch.set_num_threads(2)
    return scope


def metrics(value=.5, parameters=100):
    return {"parameters": parameters, "gflops": 1., "checkpoint_bytes": 500,
            "validation": {"map50_95": value, "map50": value, "precision": value, "recall": value}}


def result():
    return {"model_variant": "n", "policy": "extended", "ratio": .5, "suite_id": "mock-suite",
            "structural_smoke_status": "PASS", "validation_status": "PASS", "recovery_status": "PASS",
            "recovery_validation_status": "PASS", "baseline": metrics(), "pruned": metrics(.1, 80),
            "recovered": metrics(.4, 80), "comparison": {}, "recovered_vs_baseline": {},
            "recovered_vs_pruned": {}, "checks": {}, "stages": {}, "errors": [], "warnings": [],
            "provenance": {"sha256": "1" * 64}, "dataset": {}, "environment": {"packages": {}}}


@pytest.mark.parametrize("key,value", [("RECOVERY_EPOCHS", 0), ("RECOVERY_EPOCHS", True),
    ("RECOVERY_OPTIMIZER", "auto"), ("RECOVERY_LR0", float("nan")), ("RECOVERY_AMP", True),
    ("RECOVERY_PATIENCE", 5), ("KD_ENABLED", True), ("PRUNING_RATIOS", [.51])])
def test_invalid_recovery_config_fails_and_keeps_flags(env, key, value):
    env[key] = value
    with pytest.raises(ValueError):
        env["validate_configuration"]()
    assert env["RECOVERY_ENABLED"] is True and env["FINETUNING_ENABLED"] is True


@pytest.mark.parametrize("budget", [5, 30])
@pytest.mark.parametrize("model_variant,ratio", [("n", .3), ("s", .5)])
def test_valid_config_and_new_run_identity_include_recovery_budget(env, tmp_path, monkeypatch, budget, model_variant, ratio):
    monkeypatch.setenv("WANDB_RUN_ID", "existing-baseline")
    env["RECOVERY_EPOCHS"] = budget
    env["validate_configuration"]()
    assert env["configuration"]()["recovery"]["resume"] is False
    calls = []
    r = {**result(), "model_variant": model_variant, "ratio": ratio}
    env.update(wandb_session_login=lambda: None, wandb=SimpleNamespace(init=lambda **kwargs: calls.append(kwargs)))
    env["_init_tracking"](r, tmp_path)
    env["_init_tracking"]({**r, "policy": "conservative"}, tmp_path)
    for call, policy_suffix in zip(calls, ("", "_c")):
        assert call["name"] == f"prun_lamp2_yolo26{model_variant}{policy_suffix}_ft{budget}_pruned-pr{ratio:g}"
        assert call["entity"] == "ilhamrafiqin-brawijaya-university"
        assert call["project"] == "yolom26-kd-indotraffic"
        assert call["group"] == "lamp-finetune-mock-suite"
        assert call["resume"] == "never" and call["id"] != "existing-baseline"
        assert {"recovery", f"epochs{budget}", f"pr{ratio:g}", "seed42"} <= set(call["tags"])
        assert "no-recovery" not in call["tags"]
    assert len({call["id"] for call in calls}) == 2


def test_summary_keeps_three_distinct_stages_and_deltas(env):
    r = result()
    r["recovered_vs_baseline"] = env["metric_comparison"](r["baseline"], r["recovered"])
    r["recovered_vs_pruned"] = env["metric_comparison"](r["pruned"], r["recovered"])
    suite = {"model_variant": "n", "baseline": r["baseline"], "baseline_validation_status": "PASS", "results": [r]}
    rows = env["summary_rows"](suite)
    assert [v["stage"] for v in rows] == ["baseline", "pruned", "recovered"]
    assert [v["map50_95"] for v in rows] == [.5, .1, .4]
    assert rows[-1]["delta_map50_95_pp"] == pytest.approx(-10)
    assert rows[-1]["recovery_gain_map50_95_pp"] == pytest.approx(30)


def mock_run(upload_failure=False):
    run = SimpleNamespace(id="new", url="https://wandb.invalid/new", summary={}, logs=[], artifacts=[], finishes=[])
    run.log = lambda values: run.logs.append(values)
    run.finish = lambda **kw: run.finishes.append(kw)

    def log_artifact(artifact):
        run.artifacts.append(artifact)
        def wait(**kw):
            if upload_failure:
                raise RuntimeError("upload failed")
        return SimpleNamespace(wait=wait, qualified_name=artifact.name + ":v0")

    run.log_artifact = log_artifact
    return run


class Artifact:
    def __init__(self, name, type, metadata):
        self.name, self.type, self.metadata, self.files = name, type, metadata, []

    def add_file(self, path, name=None):
        assert Path(path).is_file()
        self.files.append(name)


def test_lineage_failure_closes_created_run(env, tmp_path):
    run = mock_run()
    def failure(*a, **kw):
        raise RuntimeError("lineage failed")
    run.use_artifact = failure
    env.update({"WANDB_TRACKING": True, "_init_tracking": lambda *a: run})
    created, report = env["_start_recovery_tracking"](result(), tmp_path)
    assert created is None and report["status"] == "FAIL"
    assert run.finishes == [{"exit_code": 1}]


@pytest.mark.parametrize("fail", [False, True])
def test_uploads_keep_raw_and_recovered_distinct_and_record_failure(env, tmp_path, fail):
    (tmp_path / "recovery").mkdir()
    (tmp_path / "pruned_fp32.pt").write_bytes(b"pruned")
    (tmp_path / "recovery/recovered_fp32.pt").write_bytes(b"recovered")
    (tmp_path / "recovery/validation").mkdir()
    (tmp_path / "recovery/validation/curves.json").write_text("{}", encoding="utf-8")
    env.update({"UPLOAD_WANDB_ARTIFACTS": True, "wandb": SimpleNamespace(Artifact=Artifact)})
    run = mock_run(fail)
    report = env["_finish_recovery_tracking"](result(), tmp_path, run, {"status": "RUNNING", "uploads": [], "errors": []})
    assert report["status"] == ("FAIL" if fail else "PASS")
    assert (tmp_path / "recovery/recovered_fp32.pt").read_bytes() == b"recovered"
    if not fail:
        assert len(run.artifacts) == len(report["uploads"]) == 3
        assert len({a.name for a in run.artifacts}) == 3
        assert run.artifacts[0].files == ["pruned_fp32.pt"]
        assert run.artifacts[1].files == ["recovery/recovered_fp32.pt"]
        assert "recovery/validation/curves.json" in run.artifacts[2].files
    else:
        assert report["uploads"] == [] and run.finishes == [{"exit_code": 1}]


@pytest.mark.parametrize("missing", [True, False])
def test_missing_dataset_or_training_failure_persists_truthful_status(env, tmp_path, missing):
    folder = tmp_path / "run"
    r = result()
    r["recovered"] = {}
    def pruning(*a):
        folder.mkdir()
        return r
    def failure(*a, **kw):
        raise RuntimeError("native recovery failed sentinel")
    core = torch.nn.Linear(2, 2)
    env.update({"_run_pruning_configuration": pruning, "fine_tune_pruned_checkpoint": failure,
                "YOLO": lambda *a, **kw: SimpleNamespace(model=core)})
    args = (None, r["provenance"], folder, "extended", .5, r["baseline"],
            None if missing else {"yaml": "data.yaml", "class_names": ["car"]},
            {"names": ["car"]}, {}, "mock-suite")
    if missing:
        value = env["_run_configuration"](*args)
        assert value["recovery_status"] == "NOT_RUN" and value["stages"]["experiment"] == "PARTIAL_NOT_RUN"
    else:
        with pytest.raises(RuntimeError, match="sentinel"):
            env["_run_configuration"](*args)
    report = json.loads((folder / "recovery_status.json").read_text(encoding="utf-8"))
    assert report["recovery_status"] == ("NOT_RUN" if missing else "FAIL")
    manifest = json.loads((folder / "recovery_manifest.json").read_text(encoding="utf-8"))
    assert manifest["structural_smoke_status"] == "PASS"
    if not missing:
        assert manifest["stages"]["experiment"] == "FAIL"


def test_export_sets_recovery_metadata_without_changing_raw_flag(env, tmp_path):
    from ultralytics import YOLO
    core = YOLO("yolo26n.yaml", verbose=False).model.float().eval()
    raw = tmp_path / "pruned_fp32.pt"
    after = tmp_path / "recovered_fp32.pt"
    env["save_fp32_checkpoint"](core, raw, {"sha256": "1" * 64})
    saved = env["save_recovered_fp32_checkpoint"](core, after, {"sha256": "1" * 64},
        {"epochs_completed": 1, "best_epoch": 1, "best_fitness": .3})
    assert torch.load(raw, weights_only=False)["recovery_enabled"] is False
    payload = torch.load(after, weights_only=False)
    assert payload["recovery_enabled"] is True and payload["ema"] is payload["optimizer"] is None
    assert payload["epoch"] == 0 and payload["best_fitness"] == .3
    assert all(p.dtype == torch.float32 for p in payload["model"].parameters())
    assert saved["sha256"] == env["sha256_file"](after)
