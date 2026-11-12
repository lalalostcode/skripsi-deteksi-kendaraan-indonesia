"""Offline contract and orchestration tests for the standalone LAMP V2 notebook.

Only declaration cells are executed. W&B is mocked and no research experiment,
dataset download, optimizer, or training operation is permitted by this suite.
The structural graph integration tests live in their own policy test module.
"""
# ruff: noqa: S102 -- Only execute trusted declaration cells; never Run All.
from __future__ import annotations

import ast
import json
import re
import uuid
from pathlib import Path
from types import SimpleNamespace

import nbformat
import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK = ROOT / "notebooks/kaggle/pruning/03_lamp_yolo26n_extended_no_recovery.ipynb"


@pytest.fixture
def workdir(monkeypatch):
    folder = ROOT / "outputs/test_lamp_v2_contract" / uuid.uuid4().hex
    folder.mkdir(parents=True)
    import tempfile
    monkeypatch.setattr(tempfile, "tempdir", str(folder))
    monkeypatch.setenv("TEMP", str(folder))
    monkeypatch.setenv("TMP", str(folder))
    monkeypatch.setenv("WANDB_MODE", "disabled")
    monkeypatch.setenv("WANDB_DISABLED", "true")
    return folder


@pytest.fixture(autouse=True)
def forbid_learning_and_network(monkeypatch):
    def forbidden(*_args, **_kwargs):
        pytest.fail("Contract tests must not invoke learning or real network requests")

    import requests
    from ultralytics import YOLO
    monkeypatch.setattr(YOLO, "train", forbidden)
    monkeypatch.setattr(torch.Tensor, "backward", forbidden)
    monkeypatch.setattr(torch.autograd, "backward", forbidden)
    monkeypatch.setattr(requests.sessions.Session, "request", forbidden)
    for optimizer in (torch.optim.Optimizer, torch.optim.SGD, torch.optim.Adam, torch.optim.AdamW):
        monkeypatch.setattr(optimizer, "step", forbidden)


@pytest.fixture
def env(workdir):
    pytest.importorskip("torch_pruning", reason="Optional standalone dependency: torch-pruning==1.6.1")
    notebook = nbformat.read(NOTEBOOK, as_version=4)
    scope = {"display": lambda _value: None}
    for index, cell in enumerate(notebook.cells):
        if cell.cell_type != "code":
            continue
        tags = set(cell.metadata.get("tags", []))
        if not tags & {"config", "helpers", "imports"}:
            continue
        if "imports" in tags and "new_state" in scope:
            scope.update({"STATE": scope["new_state"](), "RUN_DIR": workdir, "WB_RUN": None})
        exec(compile(cell.source, f"{NOTEBOOK.name}#cell-{index}", "exec"), scope)
    scope.update({"RUN_DIR": workdir, "SUITE_DIR": workdir,
                  "TARGET_DEVICE": torch.device("cpu"), "DEVICE": "cpu", "IMGSZ": 64,
                  "DATASET_ROOT": workdir, "DATA_YAML": workdir / "missing.yaml",
                  "WANDB_TRACKING": False, "RUN_LATENCY_BENCHMARK": False})
    if "new_state" in scope:
        scope["STATE"] = scope["new_state"]()
    torch.set_num_threads(2)
    return scope


def test_notebook_has_fourteen_sections_and_is_standalone():
    notebook = nbformat.read(NOTEBOOK, as_version=4)
    nbformat.validate(notebook)
    headings = [cell.source for cell in notebook.cells if cell.cell_type == "markdown"]
    sections = [int(match[1]) for source in headings
                if (match := re.match(r"^#{1,2} (\d{2}) —", source))]
    assert sections == list(range(1, 15))
    calls_forbidden = {"train", "train_v2", "fit", "backward"}
    for cell in notebook.cells:
        if cell.cell_type != "code":
            continue
        assert cell.outputs == [] and cell.execution_count is None
        tree = ast.parse(cell.source)
        compile(tree, str(NOTEBOOK), "exec")
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom):
                assert (node.module or "").split(".")[0] not in {"skripsi", "scripts", "src"}
            elif isinstance(node, ast.Import):
                assert not any(alias.name.split(".")[0] in {"skripsi", "scripts", "src"}
                               for alias in node.names)
            elif isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
                assert node.func.attr not in calls_forbidden
                if node.func.attr == "step":
                    assert isinstance(node.func.value, ast.Name) and node.func.value.id == "pruner"


def test_default_suite_configuration_and_learning_disabled(env):
    assert env["MODEL_VARIANTS"] == ["n", "s"]
    assert env["PRUNING_POLICY"] == "extended"
    assert env["PRUNING_RATIOS"] == [0.10, 0.20, 0.30, 0.40, 0.50]
    assert env["WANDB_ARTIFACT"] == (
        "ilhamrafiqin-brawijaya-university/yolom26-kd-indotraffic/"
        "03_baseline_yolo26n-9uu3meve-checkpoints:v84")
    assert env["MODEL_ARTIFACTS"] == {
        "n": env["WANDB_ARTIFACT"],
        "s": "ilhamrafiqin-brawijaya-university/yolom26-kd-indotraffic/02_baseline_yolo26s-q5kp9k9d-checkpoints:v62"}
    assert env["LOCAL_CHECKPOINT_PATHS"] == {"n": None, "s": None}
    assert env["LOCAL_EXPECTED_SHA256S"] == {"n": None, "s": None}
    assert env["CHECKPOINT_SELECTOR"] == "best.pt"
    assert env["RUN_FULL_VALIDATION"] is True
    assert env["GLOBAL_PRUNING"] is True
    assert env["ITERATIVE_STEPS"] == 1
    assert env["MAX_PRUNING_RATIO"] == 0.50 and env["ROUND_TO"] is None
    assert env["SEED"] == 42
    for key in ("RECOVERY_ENABLED", "KD_ENABLED", "FINETUNING_ENABLED"):
        assert env[key] is False
    # Verify the shipped Kaggle defaults before CPU/offline fixture overrides.
    config = next(cell.source for cell in nbformat.read(NOTEBOOK, as_version=4).cells
                  if "config" in cell.metadata.get("tags", []))
    shipped = {}
    exec(compile(config, str(NOTEBOOK), "exec"), shipped)
    assert shipped["DEVICE"] == "auto" and shipped["IMGSZ"] == 640
    assert shipped["BATCH"] == 16 and shipped["WORKERS"] == 2
    assert shipped["LOCAL_CHECKPOINT_PATHS"] == {"n": None, "s": None}
    assert shipped["LOCAL_EXPECTED_SHA256S"] == {"n": None, "s": None}
    assert shipped["EDA_DPI"] == 300 and shipped["EDA_MAX_LAYERS_PER_PAGE"] == 32
    assert shipped["EDA_DISPLAY_RATIO"] == 0.5
    assert all(shipped[key] is True for key in ("RUN_FULL_VALIDATION", "RUN_LATENCY_BENCHMARK",
                                               "WANDB_TRACKING", "UPLOAD_WANDB_ARTIFACTS", "INSTALL_DEPENDENCIES"))
    env["validate_configuration"]()


@pytest.mark.parametrize("variants", [[], ["n", "n"], ["m"], ["N"], ["n", "s", "m"]])
def test_configuration_rejects_invalid_or_duplicate_model_variants(env, variants):
    env["MODEL_VARIANTS"] = variants
    with pytest.raises(ValueError):
        env["validate_configuration"]()


def test_model_specs_keep_per_model_local_paths_hashes_and_source_identity_separate(env, workdir):
    for variant in ("n", "s"):
        source = workdir / f"yolo26{variant}_best.pt"
        source.write_bytes(f"source-only-{variant}".encode())
        digest = env["sha256_file"](source)
        env["LOCAL_CHECKPOINT_PATHS"][variant] = source
        env["LOCAL_EXPECTED_SHA256S"][variant] = digest
        spec = env["model_spec"](variant)
        assert spec == {"model_variant": variant, "model_name": f"YOLO26{variant}",
                        "artifact": env["MODEL_ARTIFACTS"][variant], "local_checkpoint": source,
                        "expected_sha256": digest}
        actual, provenance = env["obtain_baseline"](workdir / f"yolo26{variant}", variant)
        assert actual == source.resolve()
        assert provenance["sha256"] == digest and provenance["model_variant"] == variant
        assert provenance["requested_artifact"] == env["MODEL_ARTIFACTS"][variant]
        config = env["configuration"]("extended", 0.3, variant)
        assert config["selected_model"]["local_checkpoint"] == str(source)
        assert config["selected_model"]["expected_sha256"] == digest
        assert config["selected_model"]["artifact"] == env["MODEL_ARTIFACTS"][variant]
    assert env["model_spec"]()["model_variant"] == "n"
    assert env["model_spec"]("s")["expected_sha256"] != env["model_spec"]("n")["expected_sha256"]
    env["LOCAL_EXPECTED_SHA256S"]["s"] = env["LOCAL_EXPECTED_SHA256S"]["n"]
    with pytest.raises(ValueError, match="SHA-256"):
        env["obtain_baseline"](workdir / "wrong-hash", "s")


@pytest.mark.parametrize("policy,expected", [
    ("both", ["conservative", "extended"]),
    ("conservative", ["conservative"]),
    ("extended", ["extended"]),
])
def test_policy_selector_yields_only_requested_jobs(env, policy, expected):
    assert env["policies_to_run"](policy) == expected
    jobs = [(selected, ratio) for selected in env["policies_to_run"](policy)
            for ratio in env["PRUNING_RATIOS"]]
    assert len(jobs) == len(expected) * 5
    assert len(jobs) == len(set(jobs))


@pytest.mark.parametrize("value", ["unknown", "", "Both"])
def test_unknown_policy_is_rejected(env, value):
    with pytest.raises(ValueError):
        env["policies_to_run"](value)


@pytest.mark.parametrize("ratios", [[], [0], [0.500001], [0.51], [0.1, 0.1], [-0.1], [float("nan")]])
def test_configuration_rejects_invalid_or_duplicate_ratios(env, ratios):
    env["PRUNING_RATIOS"] = ratios
    with pytest.raises(ValueError):
        env["validate_configuration"]()


@pytest.mark.parametrize("ratio", [0.4, 0.5])
def test_configuration_accepts_new_ratios_including_cap_boundary(env, ratio):
    env["PRUNING_RATIOS"] = [ratio]
    env["validate_configuration"]()
    assert env["MAX_PRUNING_RATIO"] == 0.5


@pytest.mark.parametrize("flag", ["RECOVERY_ENABLED", "KD_ENABLED", "FINETUNING_ENABLED"])
def test_learning_flags_cannot_be_activated(env, flag):
    env[flag] = True
    with pytest.raises(ValueError):
        env["validate_configuration"]()


@pytest.mark.parametrize("policy", ["conservative", "extended"])
@pytest.mark.parametrize("ratio,tag", [(0.1, "pr0.1"), (0.3, "pr0.3"), (0.4, "pr0.4"), (0.5, "pr0.5")])
@pytest.mark.parametrize("model_variant", ["n", "s"])
def test_run_identity_is_new_and_describes_actual_configuration(env, monkeypatch, policy, ratio, tag, model_variant):
    monkeypatch.setenv("WANDB_RUN_ID", "q5kp9k9d")
    first = env["run_identity"](policy, ratio, "suite-example", model_variant)
    second = env["run_identity"](policy, ratio, "suite-example", model_variant)
    policy_suffix = "_c" if policy == "conservative" else ""
    assert first["name"] == second["name"] == f"prun_lamp2_yolo26{model_variant}{policy_suffix}_pruned-{tag}"
    assert first["group"] == "lamp-v2-suite-example"
    assert first["resume"] == "never"
    assert uuid.UUID(first["id"]).version == 4
    assert first["id"] != second["id"] and first["id"] != "q5kp9k9d"
    assert {f"yolo26{model_variant}", "pruned", "lamp", "structured", "no-recovery", "v2", policy, tag, "seed42"} <= set(first["tags"])


def test_dataset_missing_is_not_run_but_available_invalid_data_fails(env, workdir):
    assert env["prepare_validation_dataset"](workdir / "absent.yaml", workdir, workdir) is None
    invalid = workdir / "invalid.yaml"
    invalid.write_text("names: [car]\n", encoding="utf-8")
    with pytest.raises(ValueError, match="val/names"):
        env["prepare_validation_dataset"](invalid, workdir, workdir)
    images = workdir / "images"
    images.mkdir()
    yaml_path = workdir / "data.yaml"
    yaml_path.write_text("train: images\nval: images\nnames: [car, bus]\ndownload: forbidden\n", encoding="utf-8")
    dataset = env["prepare_validation_dataset"](yaml_path, workdir, workdir)
    env["assert_class_match"](["car", "bus"], dataset["class_names"])
    with pytest.raises(ValueError, match="berbeda"):
        env["assert_class_match"](["bus", "car"], dataset["class_names"])
    assert "download" not in env["yaml"].safe_load(Path(dataset["yaml"]).read_text(encoding="utf-8"))


def test_validation_arguments_are_identical_and_explicitly_fp32(env, workdir):
    before = env["validation_arguments"]("data.yaml", workdir, "baseline")
    after = env["validation_arguments"]("data.yaml", workdir, "pruned")
    assert {key: value for key, value in before.items() if key != "name"} == {
        key: value for key, value in after.items() if key != "name"}
    assert before["split"] == "val" and before["nms"] is None
    assert before["conf"] == 0.001 and before["iou"] == 0.7 and before["max_det"] == 300
    assert before["quantize"] is None and before["augment"] is False
    assert before["plots"] is True and after["plots"] is True


def measured(*, parameters=1000, gflops=10.0, checkpoint_bytes=4000, map50_95=0.5):
    """Fractions remain fractions until explicit percentage or pp conversion."""
    return {"parameters": parameters, "gflops": gflops, "checkpoint_bytes": checkpoint_bytes,
            "validation": {"map50_95": map50_95, "map50": 0.7, "precision": 0.6, "recall": 0.5}}


def test_metric_units_distinguish_percentage_points_from_relative_change(env):
    before = measured()
    after = measured(parameters=800, gflops=7.0, checkpoint_bytes=3000, map50_95=0.4)
    compared = env["metric_comparison"](before, after)
    assert compared["parameter_reduction_pct"] == pytest.approx(20.0)
    assert compared["gflops_reduction_pct"] == pytest.approx(30.0)
    assert compared["checkpoint_reduction_pct"] == pytest.approx(25.0)
    assert compared["delta_map50_95"] == pytest.approx(-0.1)
    assert compared["delta_map50_95_pp"] == pytest.approx(-10.0)
    assert compared["relative_map50_95_pct"] == pytest.approx(-20.0)
    assert before["validation"]["map50_95"] == 0.5  # Comparisons do not mutate source evidence.


@pytest.mark.parametrize("baseline_map", [None, 0.0])
def test_relative_map_is_unknown_for_absent_or_zero_baseline(env, baseline_map):
    before, after = measured(map50_95=baseline_map), measured(map50_95=0.2)
    compared = env["metric_comparison"](before, after)
    assert compared["relative_map50_95_pct"] is None
    if baseline_map is None:
        assert compared["delta_map50_95"] is None
        assert compared["delta_map50_95_pp"] is None
    else:
        assert compared["delta_map50_95_pp"] == pytest.approx(20.0)


def test_unrun_validation_never_becomes_simulated_zero_accuracy(env):
    before, after = measured(), measured(parameters=800)
    before["validation"] = None
    after["validation"] = None
    compared = env["metric_comparison"](before, after)
    assert compared["parameter_reduction_pct"] == pytest.approx(20.0)
    assert compared["delta_map50_95"] is None
    assert compared["delta_map50_95_pp"] is None
    assert compared["relative_map50_95_pct"] is None
    json.dumps(compared, allow_nan=False)


def tracking_result(env, model_variant="n"):
    before = measured()
    after = measured(parameters=800, gflops=8.0, checkpoint_bytes=3200, map50_95=0.4)
    return {"model_variant": model_variant, "policy": "extended", "ratio": 0.3, "suite_id": "test-suite",
            "baseline": before, "pruned": after,
            "comparison": env["metric_comparison"](before, after),
            "structural_smoke_status": "PASS", "validation_status": "PASS",
            "provenance": {"sha256": "1" * 64, "requested_artifact": env["MODEL_ARTIFACTS"][model_variant]},
            "environment": {"packages": {"ultralytics": "8.4.155"}}, "dataset": {}}


class MockArtifact:
    def __init__(self, name, *, type, metadata):
        self.name, self.type, self.metadata, self.files = name, type, metadata, []

    def add_file(self, path, name=None):
        assert Path(path).is_file()
        self.files.append(name or Path(path).name)


def mock_run(*, upload_error=None):
    run = SimpleNamespace(id="new-mocked-run", url="https://wandb.invalid/test", summary={},
                          logged=[], used=[], artifacts=[], finishes=[])
    run.log = lambda values: run.logged.append(values)
    run.use_artifact = lambda name, **kwargs: run.used.append((name, kwargs))
    run.finish = lambda **kwargs: run.finishes.append(kwargs)

    def log_artifact(artifact):
        run.artifacts.append(artifact)

        def wait(**_kwargs):
            if upload_error:
                raise upload_error

        return SimpleNamespace(qualified_name="mock-project/" + artifact.name + ":v0", wait=wait)

    run.log_artifact = log_artifact
    return run


@pytest.mark.parametrize("model_variant", ["n", "s"])
@pytest.mark.parametrize("ratio", [0.3, 0.4, 0.5])
def test_tracking_init_preserves_project_but_never_reuses_baseline_run(env, workdir, monkeypatch, model_variant, ratio):
    calls = []
    monkeypatch.setenv("WANDB_RUN_ID", "9uu3meve")
    monkeypatch.setenv("WANDB_API_KEY", "sentinel-secret-must-not-be-in-config")
    env["wandb_session_login"] = lambda: None
    env["wandb"] = SimpleNamespace(init=lambda **kwargs: calls.append(kwargs) or mock_run())
    result = tracking_result(env, model_variant)
    result["ratio"] = ratio
    env["_init_tracking"](result, workdir)
    env["_init_tracking"](result, workdir)
    assert len(calls) == 2
    for call in calls:
        assert call["entity"] == "ilhamrafiqin-brawijaya-university"
        assert call["project"] == "yolom26-kd-indotraffic"
        assert call["name"] == f"prun_lamp2_yolo26{model_variant}_pruned-pr{ratio:g}"
        assert call["group"] == "lamp-v2-test-suite"
        assert call["resume"] == "never" and call["id"] != "9uu3meve"
        assert call["config"]["source_checkpoint_sha256"] == result["provenance"]["sha256"]
        assert call["config"]["selected_model"]["artifact"] == env["MODEL_ARTIFACTS"][model_variant]
        assert "sentinel-secret" not in json.dumps(call)
    assert calls[0]["id"] != calls[1]["id"]


@pytest.mark.parametrize("model_variant", ["n", "s"])
@pytest.mark.parametrize("ratio", [0.3, 0.4, 0.5])
def test_tracking_uploads_distinct_new_model_and_report_artifacts(env, workdir, model_variant, ratio):
    (workdir / "pruned_fp32.pt").write_bytes(b"mock structured checkpoint")
    (workdir / "RESTORE.md").write_text("Mock restore instruction", encoding="utf-8")
    (workdir / "pruning_manifest.json").write_text("{}", encoding="utf-8")
    (workdir / "eda").mkdir()
    (workdir / "eda/channel_comparison.svg").write_text("<svg/>", encoding="utf-8")
    (workdir / "eda/channel_masks.json").write_text("[]", encoding="utf-8")
    (workdir / "wandb").mkdir()
    (workdir / "wandb/private.json").write_text("{}", encoding="utf-8")
    run = mock_run()
    env.update({"WANDB_TRACKING": True, "UPLOAD_WANDB_ARTIFACTS": True,
                "_init_tracking": lambda *_args: run, "wandb": SimpleNamespace(Artifact=MockArtifact)})
    result = tracking_result(env, model_variant)
    result["ratio"] = ratio
    report = env["track_result"](result, workdir)
    assert report["status"] == "PASS" and len(report["uploads"]) == 2
    assert all(row["status"] == "CONFIRMED" for row in report["uploads"])
    assert run.used == [(env["MODEL_ARTIFACTS"][model_variant], {"type": "model"})]
    assert [artifact.type for artifact in run.artifacts] == ["model", "report"]
    ratio_name = f"{ratio:g}".replace(".", "p")
    assert all(f"extended-pr{ratio_name}-test-suite" in artifact.name for artifact in run.artifacts)
    assert all(artifact.name.startswith(f"yolo26{model_variant}-") for artifact in run.artifacts)
    assert len({artifact.name for artifact in run.artifacts}) == 2
    assert {"pruned_fp32.pt", "RESTORE.md"} <= set(run.artifacts[0].files)
    assert "pruning_manifest.json" in run.artifacts[1].files
    assert {"eda/channel_comparison.svg", "eda/channel_masks.json"} <= set(run.artifacts[1].files)
    assert "wandb/private.json" not in run.artifacts[1].files
    assert all(artifact.metadata["source"] == result["provenance"] for artifact in run.artifacts)
    assert run.logged[0]["delta/map50_95_pp"] == pytest.approx(-10.0)
    assert run.finishes == [{"exit_code": 0}]
    assert json.loads((workdir / "tracking_report.json").read_text(encoding="utf-8")) == report


def test_upload_failure_retains_local_pass_and_never_claims_confirmed_upload(env, workdir, monkeypatch):
    (workdir / "pruned_fp32.pt").write_bytes(b"local authoritative checkpoint")
    (workdir / "RESTORE.md").write_text("restore", encoding="utf-8")
    (workdir / "pruning_manifest.json").write_text('{"structural_smoke_status":"PASS"}', encoding="utf-8")
    monkeypatch.setenv("WANDB_API_KEY", "private-sentinel")
    run = mock_run(upload_error=RuntimeError("upload rejected private-sentinel"))
    env.update({"WANDB_TRACKING": True, "UPLOAD_WANDB_ARTIFACTS": True,
                "_init_tracking": lambda *_args: run, "wandb": SimpleNamespace(Artifact=MockArtifact)})
    result = tracking_result(env)
    report = env["track_result"](result, workdir)
    assert report["status"] == "FAIL" and report["uploads"] == []
    assert report["errors"][0]["message"] == "upload rejected [REDACTED]"
    assert result["structural_smoke_status"] == "PASS"
    assert (workdir / "pruned_fp32.pt").read_bytes() == b"local authoritative checkpoint"
    assert json.loads((workdir / "pruning_manifest.json").read_text(encoding="utf-8"))["structural_smoke_status"] == "PASS"
    persisted = (workdir / "tracking_report.json").read_text(encoding="utf-8")
    assert "private-sentinel" not in persisted
    assert json.loads(persisted)["status"] == "FAIL"
    assert run.finishes == [{"exit_code": 1}]


def test_partial_upload_keeps_only_confirmed_model_and_reports_failed_report(env, workdir):
    (workdir / "pruned_fp32.pt").write_bytes(b"local checkpoint")
    (workdir / "RESTORE.md").write_text("restore", encoding="utf-8")
    run = mock_run()

    def log_artifact(artifact):
        run.artifacts.append(artifact)

        def wait(**_kwargs):
            if artifact.type == "report":
                raise RuntimeError("report upload interrupted")

        return SimpleNamespace(qualified_name="mock/" + artifact.name + ":v0", wait=wait)

    run.log_artifact = log_artifact
    env.update({"WANDB_TRACKING": True, "UPLOAD_WANDB_ARTIFACTS": True,
                "_init_tracking": lambda *_args: run, "wandb": SimpleNamespace(Artifact=MockArtifact)})
    report = env["track_result"](tracking_result(env), workdir)
    assert report["status"] == "FAIL"
    assert [item["type"] for item in report["uploads"]] == ["model"]
    assert report["uploads"][0]["status"] == "CONFIRMED"
    assert report["errors"][0]["message"] == "report upload interrupted"
    assert run.finishes == [{"exit_code": 1}]
    assert json.loads((workdir / "tracking_report.json").read_text(encoding="utf-8")) == report


def test_disabled_tracking_records_not_run_without_contacting_wandb(env, workdir):
    def unexpected(*_args):
        pytest.fail("Disabled tracking must not initialize a W&B run")
    env.update({"WANDB_TRACKING": False, "_init_tracking": unexpected})
    report = env["track_result"](tracking_result(env), workdir)
    assert report["status"] == "NOT_RUN"
    assert json.loads((workdir / "tracking_report.json").read_text(encoding="utf-8"))["status"] == "NOT_RUN"


def test_source_core_reloads_original_weights_and_checks_source_file_hash(env, workdir):
    source = workdir / "original.pt"
    source.write_bytes(b"fixed source identity")
    digest = env["sha256_file"](source)
    calls = []

    def yolo(path, **_kwargs):
        calls.append(path)
        model = torch.nn.Linear(2, 2)
        with torch.no_grad():
            model.weight.fill_(1.0)
            model.bias.zero_()
        return SimpleNamespace(model=model)

    env["YOLO"] = yolo
    first = env["_load_source_core"](source, digest)
    with torch.no_grad():
        first.weight.fill_(-99.0)
    second = env["_load_source_core"](source, digest)
    assert first is not second and torch.equal(second.weight, torch.ones_like(second.weight))
    assert calls == [str(source), str(source)]
    source.write_bytes(b"source tampered between configurations")
    with pytest.raises(ValueError, match="sumber berubah"):
        env["_load_source_core"](source, digest)
    assert len(calls) == 2  # Fail before loading any changed checkpoint.


@pytest.mark.parametrize("model_variant", ["n", "s"])
def test_baseline_preparation_evaluates_one_fp32_copy_once_and_keeps_source_metadata(env, workdir, model_variant):
    source = workdir / "source.pt"
    torch.save({"epoch": 99, "version": "8.4.155", "best_fitness": 0.55,
                "train_args": {"seed": 42, "data": "original-v3.yaml", "epochs": 100,
                               "model": f"yolo26{model_variant}.pt", "secret_credential": "not-for-reports"}}, source)
    (workdir / "baseline").mkdir()
    core = torch.nn.Linear(2, 2).float().eval()
    core.names = {0: "car"}
    dataset = {"class_names": ["car"], "yaml": "v3.yaml", "source_yaml_sha256": "b" * 64}
    evaluation_calls, save_calls = [], []
    env["_load_source_core"] = lambda *_args: core
    architecture_calls = []
    def audit(_core, variant="n"):
        architecture_calls.append(variant)
        return {"report": {"scale": variant}}
    env["architecture_audit"] = audit
    env["forward_snapshot"] = lambda *_args: {}
    env["model_measurements"] = lambda *_args: {"parameters": 6, "gflops": 0.001}
    env["prepare_validation_dataset"] = lambda *_args: dataset

    def save(_core, path, provenance):
        assert _core is core and all(value.dtype == torch.float32 for value in _core.parameters())
        save_calls.append((path, provenance))
        path.write_bytes(b"fp32")
        return {"file_bytes": 4}

    def evaluate(path, data, folder, label):
        evaluation_calls.append((path, data, folder, label))
        return {"map50_95": 0.55}

    env.update({"save_fp32_checkpoint": save, "evaluate_checkpoint": evaluate})
    provenance = {"sha256": env["sha256_file"](source)}
    baseline, prepared_dataset, metadata = env["_prepare_baseline"](source, provenance, workdir, model_variant)
    assert prepared_dataset is dataset and len(evaluation_calls) == len(save_calls) == 1
    assert evaluation_calls[0][0] == workdir / "baseline/baseline_fp32.pt"
    assert evaluation_calls[0][0] != source
    assert baseline["validation"]["map50_95"] == 0.55 and baseline["checkpoint_bytes"] == 4
    assert metadata["validation_status"] == "PASS"
    assert metadata["model_variant"] == model_variant and architecture_calls == [model_variant]
    assert metadata["state_sha256"] == env["model_state_sha256"](core)
    assert metadata["source_checkpoint_metadata"]["epoch"] == 99
    assert metadata["source_checkpoint_metadata"]["training"]["data"] == "original-v3.yaml"
    assert "secret_credential" not in json.dumps(metadata)


@pytest.mark.parametrize("policy", ["extended", "both"])
def test_suite_prepares_baseline_once_and_dispatches_requested_independent_source_jobs(env, workdir, policy):
    env["PRUNING_POLICY"] = policy
    source = workdir / "source.pt"
    source.write_bytes(b"original baseline")
    provenance = {"sha256": env["sha256_file"](source)}
    baseline, dataset = measured(), {"class_names": ["car"]}
    base_info = {"validation_status": "PASS", "state_sha256": "initial-original-model"}
    prepared, configurations = [], []
    env["environment_report"] = lambda: {"packages": {"torch": "mock-cpu"}}

    def prepare(path, prov, folder, model_variant="n"):
        prepared.append((path, prov, folder))
        return baseline, dataset, base_info

    def configure(path, prov, folder, policy, ratio, base, data, metadata, environment, suite_id, model_variant="n"):
        configurations.append((path, prov, folder, policy, ratio, base, data, metadata, suite_id))
        assert path == source and prov == provenance
        assert base is baseline and data is dataset and metadata is base_info
        result = tracking_result(env)
        result.update({"policy": policy, "ratio": ratio, "suite_id": suite_id})
        return result

    env.update({"_prepare_baseline": prepare, "_run_configuration": configure})
    suite = env["execute_suite"](source, provenance, workdir / "suite-ratios")
    expected_policies = ["extended"] if policy == "extended" else ["conservative", "extended"]
    expected_jobs = [(selected, ratio) for selected in expected_policies for ratio in (0.1, 0.2, 0.3, 0.4, 0.5)]
    assert len(prepared) == 1 and len(configurations) == len(expected_jobs)
    assert [(entry[3], entry[4]) for entry in configurations] == expected_jobs
    assert len({entry[2] for entry in configurations}) == len(expected_jobs)
    assert all(entry[2].relative_to(workdir / "suite-ratios").as_posix() == f"{entry[3]}/pr{entry[4]:g}"
               for entry in configurations)
    assert suite["status"] == "COMPLETE" and len(suite["results"]) == len(expected_jobs)
    assert all(job["status"] == "PASS" for job in suite["jobs"])
    assert suite["v1_reference"]["ratio"] == "UNKNOWN"
    assert suite["v1_reference"]["included_in_primary_comparison"] is False
    rows = json.loads((workdir / "suite-ratios/summary/metrics.json").read_text(encoding="utf-8"))
    assert len(rows) == len(expected_jobs) + 1 and rows[0]["policy"] == "baseline"


@pytest.mark.parametrize("model_variants", [["n", "s"], ["n"], ["s"]])
def test_models_orchestration_isolates_sources_baselines_jobs_and_result_paths(env, workdir, model_variants):
    env["MODEL_VARIANTS"] = model_variants
    acquired, prepared, configured, plots, comparisons = [], [], [], [], []
    originals = {variant: measured(parameters=1000 if variant == "n" else 4000) for variant in model_variants}
    datasets = {variant: {"class_names": ["car", "bus"]} for variant in model_variants}
    metadata = {variant: {"names": ["car", "bus"], "state_sha256": f"original-{variant}",
                          "validation_status": "PASS"} for variant in model_variants}

    def acquire(folder, model_variant="n"):
        acquired.append((folder, model_variant))
        source = folder / "baseline" / f"source-{model_variant}.pt"
        source.write_bytes(f"baseline-{model_variant}".encode())
        provenance = {"requested_artifact": env["MODEL_ARTIFACTS"][model_variant],
                      "sha256": env["sha256_file"](source), "model_variant": model_variant}
        return source, provenance

    def prepare(source, provenance, folder, model_variant="n"):
        prepared.append((source, provenance, folder, model_variant))
        assert source == folder / "baseline" / f"source-{model_variant}.pt"
        assert provenance["sha256"] == env["sha256_file"](source)
        return originals[model_variant], datasets[model_variant], metadata[model_variant]

    def configure(source, provenance, folder, policy, ratio, baseline, dataset, info, _environment, suite_id,
                  model_variant="n"):
        configured.append((model_variant, folder, policy, ratio, source))
        assert baseline is originals[model_variant] and dataset is datasets[model_variant]
        assert info is metadata[model_variant] and suite_id == "models-suite"
        assert provenance["requested_artifact"] == env["MODEL_ARTIFACTS"][model_variant]
        result = tracking_result(env, model_variant)
        result.update({"policy": policy, "ratio": ratio, "baseline": baseline, "suite_id": suite_id})
        return result

    env.update({"obtain_baseline": acquire, "_prepare_baseline": prepare, "_run_configuration": configure,
                "environment_report": lambda: {"packages": {}},
                "plot_suite": lambda suite, folder: plots.append((suite, folder)),
                "plot_model_comparison": lambda suites, folder: comparisons.append((suites, folder)) or []})
    root = workdir / "models-suite"
    bundle = env["execute_models"](root)
    assert bundle["status"] == "COMPLETE" and bundle["model_variants"] == model_variants
    assert [entry[1] for entry in acquired] == model_variants
    assert [entry[3] for entry in prepared] == model_variants  # One baseline measurement/evaluation per model.
    assert len(configured) == 5 * len(model_variants)
    assert len({entry[1] for entry in configured}) == len(configured)
    for variant, folder, policy, ratio, source in configured:
        assert folder == root / f"yolo26{variant}" / policy / f"pr{ratio:g}"
        assert source == root / f"yolo26{variant}" / "baseline" / f"source-{variant}.pt"
    assert [child["model_variant"] for child in bundle["models"]] == model_variants
    assert all(child["suite_id"] == root.name and len(child["results"]) == 5 for child in bundle["models"])
    assert all(job["status"] == "PASS" for job in bundle["jobs"])
    assert len(plots) == len(model_variants) and comparisons[0][1] == root / "comparison"
    rows = json.loads((root / "comparison/metrics.json").read_text(encoding="utf-8"))
    assert len(rows) == 6 * len(model_variants) and {row["model_variant"] for row in rows} == set(model_variants)
    if len(model_variants) == 2:
        assert prepared[0][1]["sha256"] != prepared[1][1]["sha256"]


def test_models_failure_preserves_completed_model_and_pending_statuses(env, workdir):
    env["MODEL_VARIANTS"] = ["n", "s"]
    failure = RuntimeError("S artifact unavailable sentinel")
    acquired = []

    def acquire(folder, model_variant="n"):
        acquired.append(model_variant)
        if model_variant == "s":
            raise failure
        return folder / "source.pt", {"model_variant": model_variant, "sha256": "a" * 64}

    child = {"model_variant": "n", "suite_id": "models-failure", "baseline": measured(),
             "baseline_metadata": {"names": ["car"]}, "baseline_validation_status": "PASS", "results": []}
    env.update({"obtain_baseline": acquire, "execute_suite": lambda *_args: child,
                "plot_suite": lambda *_args: [],
                "plot_model_comparison": lambda *_args: pytest.fail("Comparison requires all requested models")})
    root = workdir / "models-failure"
    with pytest.raises(RuntimeError, match="S artifact unavailable sentinel") as captured:
        env["execute_models"](root)
    assert captured.value is failure and acquired == ["n", "s"]
    bundle = json.loads((root / "suite_manifest.json").read_text(encoding="utf-8"))
    assert bundle["status"] == "FAIL" and [job["status"] for job in bundle["jobs"]] == ["PASS", "FAIL"]
    assert [model["model_variant"] for model in bundle["models"]] == ["n"]
    assert bundle["errors"][0]["message"] == "S artifact unavailable sentinel"
    assert (root / "comparison/metrics.json").is_file()


def test_models_comparison_requires_identical_class_order(env, workdir):
    def execute(_source, provenance, _folder, model_variant="n", suite_id=None):
        assert provenance["model_variant"] == model_variant
        names = ["car", "bus"] if model_variant == "n" else ["bus", "car"]
        return {"model_variant": model_variant, "suite_id": suite_id, "baseline": measured(),
                "baseline_metadata": {"names": names}, "baseline_validation_status": "PASS", "results": []}

    env.update({"obtain_baseline": lambda folder, variant="n": (folder / "source.pt", {"model_variant": variant}),
                "execute_suite": execute, "plot_suite": lambda *_args: [],
                "plot_model_comparison": lambda *_args: pytest.fail("Mismatched class order cannot be compared")})
    root = workdir / "mismatched-models"
    with pytest.raises(ValueError, match="berbeda"):
        env["execute_models"](root)
    bundle = json.loads((root / "suite_manifest.json").read_text(encoding="utf-8"))
    assert bundle["status"] == "FAIL" and bundle["jobs"][1]["status"] == "FAIL"


def test_suite_failure_saves_completed_and_pending_jobs_and_reraises_original_error(env, workdir):
    env["PRUNING_POLICY"] = "both"
    baseline = measured()
    env["environment_report"] = lambda: {"packages": {}}
    env["_prepare_baseline"] = lambda *_args: (baseline, None, {"validation_status": "NOT_RUN"})
    failure = RuntimeError("suite failure sentinel")
    calls = []

    def run(*args):
        calls.append(args)
        if len(calls) == 2:
            raise failure
        result = tracking_result(env)
        result.update({"policy": args[3], "ratio": args[4]})
        return result

    env["_run_configuration"] = run
    folder = workdir / "failed-suite"
    with pytest.raises(RuntimeError, match="suite failure sentinel") as captured:
        env["execute_suite"](workdir / "source.pt", {"sha256": "a" * 64}, folder)
    assert captured.value is failure
    suite = json.loads((folder / "suite_manifest.json").read_text(encoding="utf-8"))
    assert suite["status"] == "FAIL"
    assert [job["status"] for job in suite["jobs"]] == ["PASS", "FAIL"] + ["NOT_RUN"] * 8
    assert len(suite["results"]) == 1 and len(calls) == 2
    assert suite["errors"][0]["message"] == "suite failure sentinel"
    assert (folder / "summary/metrics.json").is_file()


def test_configuration_rejects_cumulative_weights_and_persists_failure(env, workdir):
    core = torch.nn.Linear(2, 2).float().eval()
    core.names = {0: "synthetic"}
    env["_load_source_core"] = lambda *_args: core
    env["track_result"] = lambda *_args: {"status": "NOT_RUN"}
    folder = workdir / "failed-config"
    with pytest.raises(ValueError, match="Baseline awal berbeda"):
        env["_run_configuration"](workdir / "source.pt", {"sha256": "a" * 64}, folder,
                                  "extended", 0.3, measured(), None,
                                  {"state_sha256": "a-different-original-model", "names": ["synthetic"]},
                                  {"packages": {}}, "synthetic-suite")
    report = json.loads((folder / "smoke_test_report.json").read_text(encoding="utf-8"))
    assert report["structural_smoke_status"] == "FAIL"
    assert report["validation_status"] == "NOT_RUN"
    assert report["stages"]["baseline_reload"] == "FAIL"
    assert "Baseline awal berbeda" in report["errors"][0]["message"]
    assert not any(report["checks"].values())
    assert (folder / "pruning_manifest.json").is_file()


def test_available_invalid_dataset_is_validation_fail_before_any_pruning(env, workdir):
    source = workdir / "mock-source.pt"
    torch.save({"epoch": 2, "version": "mock", "train_args": {}}, source)
    core = torch.nn.Linear(2, 2).float().eval()
    core.names = {0: "synthetic"}
    invalid_yaml = workdir / "data.yaml"
    invalid_yaml.write_text("val: missing\nnames: []\n", encoding="utf-8")

    def save(_core, path, _provenance):
        path.write_bytes(b"mock FP32 baseline comparator")
        return {"file_bytes": path.stat().st_size}

    def forbidden_pruning(*_args):
        pytest.fail("Invalid baseline dataset must fail before pruning")

    env.update({"_load_source_core": lambda *_args: core,
                "architecture_audit": lambda _core, _variant="n": {"report": {}},
                "forward_snapshot": lambda *_args: {}, "model_measurements": lambda *_args: measured(),
                "save_fp32_checkpoint": save, "_run_configuration": forbidden_pruning,
                "RUN_FULL_VALIDATION": True, "DATA_YAML": invalid_yaml})
    folder = workdir / "invalid-data-suite"
    with pytest.raises(ValueError, match="val/names"):
        env["execute_suite"](source, {"sha256": env["sha256_file"](source)}, folder)
    suite = json.loads((folder / "suite_manifest.json").read_text(encoding="utf-8"))
    assert suite["status"] == "FAIL" and suite["baseline_validation_status"] == "FAIL"
    assert all(job["status"] == "NOT_RUN" for job in suite["jobs"])
    report = json.loads((folder / "baseline/validation_report.json").read_text(encoding="utf-8"))
    assert report["status"] == "FAIL" and report["type"] == "ValueError"


def synthetic_model(env, *, end2end=True, model_variant="n"):
    """Locally construct YOLO26n/S with synthetic classes, never pretrained weights."""
    from ultralytics.nn.tasks import DetectionModel, yaml_model_load
    env["seed_environment"](42)
    core = DetectionModel(yaml_model_load(f"yolo26{model_variant}.yaml"), nc=3, verbose=False).float().eval()
    core.names = {0: "synthetic_car", 1: "synthetic_bus", 2: "synthetic_truck"}
    core.end2end = end2end
    return core


@pytest.mark.parametrize("policy,group_count,independent_channels", [
    ("conservative", 13, 440), ("extended", 35, 2056),
])
@pytest.mark.parametrize("ratio", [0.1, 0.2, 0.3, 0.4, 0.5])
@pytest.mark.parametrize("model_variant", ["n", "s"])
def test_synthetic_cpu_policy_matrix_preserves_outputs_envelopes_and_capacity(
        env, policy, group_count, independent_channels, ratio, model_variant):
    core = synthetic_model(env, model_variant=model_variant)
    independent_channels *= 1 if model_variant == "n" else 2
    image = torch.zeros(1, 3, 64, 64)
    protected = env["protected_policy_snapshot"](core)
    protected_names = {name for name, module in core.named_modules()
                       if type(module).__name__ in {"Detect", "C2PSA", "PSABlock", "Attention", "AAttn"}}
    assert set(protected) == protected_names
    before_parameters = sum(value.numel() for value in core.parameters())
    pruner, audit = env["build_policy_pruner"](core, image, policy, ratio, model_variant)
    assert audit["report"]["model_variant"] == model_variant
    assert audit["report"]["group_count"] == group_count
    assert audit["independent_channels_before"] == independent_channels
    signatures = [entry["signature"] for entry in audit["eligible"].values()]
    assert len(set(signatures)) == group_count
    assert len({name for signature in signatures for name in signature}) == sum(map(len, signatures))
    assert any("detection_head" in item["reason"] for item in audit["report"]["rejected_groups"])
    assert any("attention_envelope" in item["reason"] for item in audit["report"]["rejected_groups"])
    inventory_before = env["measure_layer_inventory"](core, image, audit, policy)
    assert all(not module.training for module in core.modules())
    changes = env["apply_policy_pruning"](pruner, audit)
    assert changes and sum(value.numel() for value in core.parameters()) < before_parameters
    env["assert_layer_consistency"](core)
    env["assert_policy_protected_unchanged"](core, protected)
    snapshot = env["forward_snapshot"](core, image)
    assert set(snapshot) == {"decoded", "one2many/boxes", "one2many/scores", "one2one/boxes", "one2one/scores"}
    assert all(torch.isfinite(tensor).all() for tensor in snapshot.values())
    assert snapshot["decoded"].shape[-1] == 6
    removed = sum(entry["channels_before"] - entry["channels_after"] for entry in changes)
    assert 0 < removed < independent_channels
    assert abs(removed / independent_channels - ratio) <= 2 / independent_channels
    for entry in audit["eligible"].values():
        for module, initial in entry["output_dimensions"].items():
            assert initial / 2 <= module.out_channels <= initial
    inventory_after = env["measure_layer_inventory"](core, image, audit, policy)
    required = {"layer_name", "module_type", "region", "in_channels", "out_channels", "parameters",
                "estimated_flops", "policy", "eligible", "protection_reason", "dependency_group"}
    assert all(required <= set(row) for row in inventory_before + inventory_after)
    assert {row["layer_name"] for row in inventory_before} == {row["layer_name"] for row in inventory_after}
    before_by_name = {row["layer_name"]: row for row in inventory_before}
    physical_removed = sum(before_by_name[row["layer_name"]]["out_channels"] - row["out_channels"]
                           for row in inventory_after if row["module_type"] == "Conv2d")
    assert physical_removed == sum(entry["physical_conv_output_channels_removed"] for entry in changes)
    assert physical_removed >= removed


@pytest.mark.parametrize("model_variant", ["n", "s"])
def test_synthetic_extended_one_to_many_forward_is_valid(env, model_variant):
    core = synthetic_model(env, end2end=False, model_variant=model_variant)
    image = torch.zeros(1, 3, 64, 64)
    before = env["protected_policy_snapshot"](core)
    pruner, audit = env["build_policy_pruner"](core, image, "extended", 0.3, model_variant)
    env["apply_policy_pruning"](pruner, audit)
    env["assert_layer_consistency"](core)
    env["assert_policy_protected_unchanged"](core, before)
    assert env["forward_snapshot"](core, image)["decoded"].shape[:2] == (1, 7)


@pytest.mark.parametrize("model_variant", ["n", "s"])
def test_conservative_policy_matches_preserved_v1_on_identical_synthetic_weights(env, model_variant):
    filename = "01_lamp_yolo26n_no_recovery.ipynb" if model_variant == "n" else "02_lamp_yolo26s_no_recovery.ipynb"
    notebook = nbformat.read(NOTEBOOK.with_name(filename), as_version=4)
    selected = {"architecture_audit", "audit_group", "cap_group_importance", "build_pruner", "apply_structural_pruning"}
    declarations = [node for cell in notebook.cells if cell.cell_type == "code"
                    for node in ast.parse(cell.source).body
                    if isinstance(node, ast.FunctionDef) and node.name in selected]
    assert {node.name for node in declarations} == selected
    legacy = dict(env, PRUNING_RATIO=0.3)
    exec(compile(ast.Module(body=declarations, type_ignores=[]), "preserved-v1-declarations", "exec"), legacy)
    core_v2 = synthetic_model(env, model_variant=model_variant)
    core_v1 = env["deepcopy"](core_v2)
    image = torch.zeros(1, 3, 64, 64)
    v1_audit = legacy["architecture_audit"](core_v1)
    v1_pruner, _ = legacy["build_pruner"](core_v1, image, v1_audit)
    v1_changes = legacy["apply_structural_pruning"](v1_pruner, v1_audit)
    v2_pruner, v2_audit = env["build_policy_pruner"](core_v2, image, "conservative", 0.3, model_variant)
    v2_changes = env["apply_policy_pruning"](v2_pruner, v2_audit)
    assert [(entry["root"], entry["removed_indices"]) for entry in v2_changes] == [
        (entry["root"], entry["removed_indices"]) for entry in v1_changes]
    assert core_v1.state_dict().keys() == core_v2.state_dict().keys()
    assert all(torch.equal(value, core_v2.state_dict()[name]) for name, value in core_v1.state_dict().items())


def test_six_plots_preserve_not_run_accuracy_without_fabricated_values(env, workdir, monkeypatch):
    monkeypatch.setenv("MPLBACKEND", "Agg")
    monkeypatch.setenv("MPLCONFIGDIR", str(workdir / "mplconfig"))
    before, after = measured(), measured(parameters=800, gflops=8.0, checkpoint_bytes=3200)
    before["validation"] = after["validation"] = None
    result = tracking_result(env)
    result.update({"baseline": before, "pruned": after, "validation_status": "NOT_RUN",
                   "comparison": env["metric_comparison"](before, after)})
    suite = {"baseline": before, "results": [result], "baseline_validation_status": "NOT_RUN"}
    folder = workdir / "plots"
    reports = env["plot_suite"](suite, folder)
    assert len(reports) == 6
    states = {row["plot"]: row["status"] for row in reports}
    assert states == {"ratio_vs_map50_95": "NOT_RUN", "ratio_vs_parameter_reduction": "MEASURED",
                      "ratio_vs_gflops_reduction": "MEASURED", "map_vs_gflops": "NOT_RUN",
                      "map_vs_parameters": "NOT_RUN", "per_class_ap": "NOT_RUN"}
    assert len(list(folder.glob("*.png"))) == 6
    assert all(path.read_bytes().startswith(b"\x89PNG\r\n\x1a\n") for path in folder.glob("*.png"))
    rows = env["summary_rows"](suite)
    assert all(row["map50_95"] is None for row in rows)

