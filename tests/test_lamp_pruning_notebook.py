"""Standalone LAMP contract, failures, and real offline YOLO26n structural pruning.

No training or target W&B artifact is used. Synthetic integration checkpoints are
explicitly labelled and never substituted for the research baseline.
"""
# ruff: noqa: S102 -- Execute trusted notebook declarations; skip install, login, and run cells.
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
NOTEBOOK = ROOT / "notebooks/kaggle/pruning/01_lamp_yolo26n_no_recovery.ipynb"


@pytest.fixture
def workdir(monkeypatch):
    folder = ROOT / "outputs/test_lamp" / uuid.uuid4().hex
    folder.mkdir(parents=True)
    # Managed Windows temp ACLs are unsuitable for W&B's temporary directories.
    import tempfile
    monkeypatch.setattr(tempfile, "tempdir", str(folder))
    monkeypatch.setenv("TEMP", str(folder))
    monkeypatch.setenv("TMP", str(folder))
    monkeypatch.setenv("WANDB_MODE", "disabled")
    return folder


@pytest.fixture
def env(workdir):
    pytest.importorskip("torch_pruning", reason="Optional standalone dependency: install torch-pruning==1.6.1")
    notebook = nbformat.read(NOTEBOOK, as_version=4)
    scope = {"display": lambda _value: None}
    for cell in notebook.cells:
        if cell.cell_type == "code" and set(cell.metadata.get("tags", [])) & {"config", "helpers", "imports"}:
            if "imports" in cell.metadata.get("tags", []):
                scope.update({"STATE": scope["new_state"](), "RUN_DIR": workdir, "WB_RUN": None})
            exec(compile(cell.source, str(NOTEBOOK), "exec"), scope)
    scope.update({"STATE": scope["new_state"](), "RUN_DIR": workdir, "WB_RUN": None,
                  "TARGET_DEVICE": torch.device("cpu"), "DEVICE": "cpu", "IMGSZ": 64,
                  "DATASET_ROOT": workdir, "DATA_YAML": workdir / "missing.yaml",
                  "WANDB_TRACKING": False, "BENCHMARK_LATENCY": False})
    torch.set_num_threads(2)
    scope["seed_environment"](42)
    return scope


@pytest.fixture
def forbid_learning(monkeypatch):
    def forbidden(*_args, **_kwargs):
        pytest.fail("Training, backward, or optimizer step was invoked")

    from ultralytics import YOLO
    monkeypatch.setattr(YOLO, "train", forbidden)
    monkeypatch.setattr(torch.Tensor, "backward", forbidden)
    monkeypatch.setattr(torch.autograd, "backward", forbidden)
    for optimizer in (torch.optim.Optimizer, torch.optim.SGD, torch.optim.Adam, torch.optim.AdamW):
        monkeypatch.setattr(optimizer, "step", forbidden)


def synthetic_model(env):
    """Construct architecture locally; do not download pretrained weights."""
    from ultralytics.nn.tasks import DetectionModel, yaml_model_load
    config = yaml_model_load("yolo26n.yaml")
    core = DetectionModel(config, nc=3, verbose=False).float().eval()
    core.names = {0: "synthetic_a", 1: "synthetic_b", 2: "synthetic_c"}
    return core


def test_notebook_structure_syntax_and_no_learning_calls():
    notebook = nbformat.read(NOTEBOOK, as_version=4)
    nbformat.validate(notebook)
    headings = [cell.source for cell in notebook.cells if cell.cell_type == "markdown"]
    assert [int(re.match(r"^#{1,2} (\d{2}) —", text)[1]) for text in headings] == list(range(1, 13))
    forbidden = {"train", "train_v2", "fit", "backward", "step"}
    for cell in notebook.cells:
        if cell.cell_type != "code":
            continue
        assert cell.outputs == [] and cell.execution_count is None
        tree = ast.parse(cell.source)
        compile(tree, str(NOTEBOOK), "exec")
        assert "inatrc_kd" not in cell.source and "from skripsi" not in cell.source
        for node in ast.walk(tree):
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and node.func.attr in forbidden:
                assert node.func.attr == "step" and isinstance(node.func.value, ast.Name) and node.func.value.id == "pruner"
            if isinstance(node, ast.ExceptHandler):
                assert not (len(node.body) == 1 and isinstance(node.body[0], ast.Pass))


def test_configuration_and_pinned_actual_apis(env):
    env["validate_configuration"]()
    report = env["environment_report"]()
    assert report["packages"]["torch-pruning"] == "1.6.1"
    assert "max_pruning_ratio" in report["base_pruner_signature"]
    assert "group_reduction" in report["lamp_signature"]
    assert len(report["importance_module_sha256"]) == 64
    env["RECOVERY_ENABLED"] = True
    with pytest.raises(ValueError, match="disabled"):
        env["validate_configuration"]()


def test_checkpoint_selection_and_ambiguity(env, workdir):
    best = workdir / "nested" / "best.pt"
    best.parent.mkdir()
    best.write_bytes(b"explicit baseline copy")
    (workdir / "last.pt").write_bytes(b"last")
    assert env["select_checkpoint"](workdir, "best.pt") == best
    with pytest.raises(ValueError, match="ambigu"):
        env["select_checkpoint"](workdir, None)
    (workdir / "best.pt").write_bytes(b"second best")
    with pytest.raises(ValueError, match="2 kecocokan"):
        env["select_checkpoint"](workdir, "best.pt")
    assert env["select_checkpoint"](workdir, "nested/best.pt") == best
    assert env["select_checkpoint"](workdir / "last.pt", None).name == "last.pt"
    with pytest.raises(ValueError, match="0 kecocokan"):
        env["select_checkpoint"](workdir, "absent.pt")
    empty = workdir / "empty"
    empty.mkdir()
    with pytest.raises(FileNotFoundError):
        env["select_checkpoint"](empty)


def test_missing_best_requires_explicit_selection(env, workdir):
    (workdir / "epoch.pt").write_bytes(b"epoch")
    with pytest.raises(ValueError, match="best.pt"):
        env["select_checkpoint"](workdir, None)
    assert env["select_checkpoint"](workdir, "epoch.pt").name == "epoch.pt"


def test_provenance_hash_and_json(env, workdir):
    checkpoint = workdir / "best.pt"
    checkpoint.write_bytes(b"a known baseline copy")
    digest = env["sha256_file"](checkpoint)
    artifact = env["WANDB_ARTIFACT"]
    result = env["checkpoint_provenance"](checkpoint, artifact)
    assert result["source_verification"] == "user_asserted_local_copy"
    assert result["requested_artifact"].endswith(":v84")
    result = env["checkpoint_provenance"](checkpoint, artifact, expected_hash=digest)
    assert result["source_verification"] == "expected_sha256_matched"
    with pytest.raises(ValueError, match="SHA-256"):
        env["checkpoint_provenance"](checkpoint, artifact, expected_hash="0" * 64)
    with pytest.raises(ValueError):
        env["write_json"](workdir / "bad.json", {"map": float("nan")})


def test_online_acquisition_uses_exact_artifact_and_job_type(env, workdir):
    root = workdir / "artifact" / "nested"
    root.mkdir(parents=True)
    (root / "best.pt").write_bytes(b"mock download; no actual auth")
    calls = []
    artifact = SimpleNamespace(qualified_name=env["WANDB_ARTIFACT"], digest="mock-digest",
                               id="mock-id", version="v84", download=lambda **_kwargs: str(root.parent))
    def use_artifact(name, **kwargs):
        calls.append((name, kwargs))
        return artifact
    def init(**kwargs):
        assert kwargs["job_type"] == "lamp-pruning-smoke"
        assert kwargs["entity"] == "ilhamrafiqin-brawijaya-university"
        assert kwargs["project"] == "yolom26-kd-indotraffic"
        assert "WANDB_API_KEY" not in kwargs["config"]
        return SimpleNamespace(use_artifact=use_artifact)
    env.update({"WANDB_TRACKING": True, "LOCAL_CHECKPOINT_PATH": None,
                "wandb": SimpleNamespace(init=init, api=SimpleNamespace(api_key="existing-session"))})
    path, provenance = env["obtain_checkpoint"]()
    assert path.name == "best.pt"
    assert calls == [(env["WANDB_ARTIFACT"], {"type": "model"})]
    assert provenance["source_verification"] == "wandb_download"
    assert provenance["artifact_relative_path"] == "nested/best.pt"


def test_class_order_and_ratio_contract(env):
    env["assert_class_match"]({"1": "truck", "0": "car"}, ["car", "truck"])
    with pytest.raises(ValueError, match="berbeda"):
        env["assert_class_match"](["car", "truck"], ["truck", "car"])
    with pytest.raises(ValueError, match="Indeks"):
        env["ordered_names"]({1: "car"})
    assert env["reduction"](440, 396) == pytest.approx(0.1)
    with pytest.raises(ValueError):
        env["reduction"](0, 0)
    with pytest.raises(ValueError):
        env["reduction"](10, 11)


def test_failure_is_persisted_and_reraised(env, workdir):
    with pytest.raises(RuntimeError, match="sentinel"), env["checked_stage"]("sentinel_stage"):
        raise RuntimeError("sentinel")
    report = json.loads((workdir / "smoke_test_report.json").read_text(encoding="utf-8"))
    assert report["structural_smoke_status"] == "FAIL"
    assert report["errors"][0]["message"] == "sentinel"
    assert not any(report["checks"].values())
    env["STATE"]["structural_smoke_status"] = "PASS"
    with pytest.raises(ValueError), env["checked_stage"]("invalid_dataset", validation=True):
        raise ValueError("bad dataset")
    assert env["STATE"]["structural_smoke_status"] == "PASS"
    assert env["STATE"]["validation_status"] == "FAIL"


def test_output_rejects_nonfinite_missing_branch_and_wrong_classes(env):
    with pytest.raises(ValueError, match="NaN/Inf"):
        env["finite_tensors"]({"nested": [torch.tensor([float("inf")]) ]})
    with pytest.raises(ValueError, match="Cabang"):
        env["raw_branches"]((torch.zeros(1), {"one2many": {}}))
    with pytest.raises(ValueError, match="Output eval"):
        env["raw_branches"]({"one2many": {}})

    class Core:
        def __init__(self):
            self.model = [SimpleNamespace(nc=2, reg_max=1, end2end=False)]

        def eval(self):
            return self

        def __call__(self, _image):
            branches = {key: {"boxes": torch.zeros(1, 4, 2), "scores": torch.zeros(1, 3, 2)}
                        for key in ("one2many", "one2one")}
            return torch.zeros(1, 6, 2), branches

    with pytest.raises(ValueError, match="jumlah kelas"):
        env["forward_snapshot"](Core(), torch.zeros(1, 3, 64, 64))


def test_flops_include_attention_children_and_functional_matmuls(env):
    attention = env["Attention"](16, num_heads=2)
    core = torch.nn.Sequential(torch.nn.Conv2d(3, 16, 1), attention).eval()
    with torch.no_grad():
        child_macs, _ = env["thop_profile"](env["deepcopy"](core), inputs=(torch.zeros(1, 3, 8, 8),), verbose=False)
    measured = env["model_measurements"](core, 8)
    expected_matmul_macs = 2 * (8 * 8) ** 2 * (attention.key_dim + attention.head_dim)
    assert measured["gflops"] == pytest.approx(2 * (child_macs + expected_matmul_macs) / 1e9)
    assert child_macs > 0


def test_dataset_absent_invalid_and_class_mismatch(env, workdir):
    assert env["prepare_validation_dataset"](workdir / "missing.yaml", workdir, workdir) is None
    invalid = workdir / "invalid.yaml"
    invalid.write_text("names: [car]\n", encoding="utf-8")
    with pytest.raises(ValueError, match="val/names"):
        env["prepare_validation_dataset"](invalid, workdir, workdir)
    (workdir / "images").mkdir()
    valid = workdir / "data.yaml"
    valid.write_text("train: images\nval: images\nnames: [car]\ndownload: forbidden\n", encoding="utf-8")
    info = env["prepare_validation_dataset"](valid, workdir, workdir)
    assert "download" not in env["yaml"].safe_load(Path(info["yaml"]).read_text(encoding="utf-8"))
    assert info["class_names"] == ["car"]
    with pytest.raises(ValueError, match="berbeda"):
        env["assert_class_match"](["truck"], info["class_names"])
    left = env["validation_arguments"](info["yaml"], workdir, "before")
    right = env["validation_arguments"](info["yaml"], workdir, "after")
    assert {k: v for k, v in left.items() if k != "name"} == {k: v for k, v in right.items() if k != "name"}
    assert left["nms"] is None and left["quantize"] is None and left["split"] == "val"


@pytest.mark.parametrize("end2end", [False, True])
def test_real_lamp_pruning_fresh_process_reload_and_protection(env, workdir, forbid_learning, end2end):
    core = synthetic_model(env)
    core.end2end = end2end
    audit = env["architecture_audit"](core)
    image = torch.rand(1, 3, 64, 64)
    baseline = env["forward_snapshot"](core, image)
    before = env["model_measurements"](core, 64)
    dims_before = env["layer_dimensions"](core)
    protected_head = env["head_snapshot"](audit["head"])
    pruner, graph_report = env["build_pruner"](core, image, audit)
    assert graph_report["group_count"] == 13
    assert not any(module.training for module in core.modules())
    changes = env["apply_structural_pruning"](pruner, audit)
    assert changes and all(change["channels_after"] < change["channels_before"] for change in changes)
    env["assert_layer_consistency"](core)
    env["assert_head_unchanged"](audit["head"], protected_head)
    after = env["model_measurements"](core, 64)
    assert after["parameters"] < before["parameters"] and after["gflops"] < before["gflops"]
    assert set(env["forward_snapshot"](core, image)) == set(baseline)
    dims_after = env["layer_dimensions"](core)
    env["write_layer_report"](workdir / "layers.csv", dims_before, dims_after, audit)
    assert "dependent BN/consumer" in (workdir / "layers.csv").read_text(encoding="utf-8")
    for name, entry in audit["eligible"].items():
        assert entry["root"].out_channels >= entry["initial_channels"] / 2, name
    path = workdir / "synthetic_pruned.pt"
    saved = env["save_fp32_checkpoint"](core, path, {"source": "synthetic offline integration; NOT W&B baseline"})
    report = env["verify_fresh_reload"](path, core, image, workdir)
    assert report["direct_yolo_load"] and report["raw_parity"]
    assert all(value <= 1e-5 for value in report["max_absolute_difference"].values())
    assert report["prediction_shape"][1] == 6
    assert saved["file_bytes"] > 0
    payload = torch.load(path, map_location="cpu", weights_only=False)
    assert payload["ema"] is None and payload["optimizer"] is None
    assert all(p.dtype == torch.float32 for p in payload["model"].parameters())
    env["STATE"]["metrics"] = {"baseline": before, "pruned": after}
    env["STATE"]["manifest"] = {"source": "synthetic offline integration", "reload": report,
                                  "groups": changes, "head_unchanged": True}
    env["flush_reports"]()
    with torch.no_grad():
        next(audit["head"].parameters()).add_(1)
    with pytest.raises(ValueError, match="head berubah"):
        env["assert_head_unchanged"](audit["head"], protected_head)


def test_pruner_rejects_foreign_group_and_empty_change(env, forbid_learning):
    core = synthetic_model(env)
    audit = env["architecture_audit"](core)
    fake_group = [(SimpleNamespace(target=SimpleNamespace(module=torch.nn.Conv2d(3, 4, 1))), [0])]
    with pytest.raises(ValueError, match="allowlist"):
        env["audit_group"](fake_group, None, audit)
    entry = next(iter(audit["eligible"].values()))
    root = SimpleNamespace(target=SimpleNamespace(module=entry["root"]), handler="out")
    head_dependency = SimpleNamespace(target=SimpleNamespace(module=next(audit["head"].modules())), handler="out")
    foreign_conv = next(module for module in audit["head"].modules() if isinstance(module, torch.nn.Conv2d))
    head_dependency.target.module = foreign_conv
    graph = SimpleNamespace(is_out_channel_pruning_fn=lambda fn: fn == "out", is_in_channel_pruning_fn=lambda fn: fn == "in")
    with pytest.raises(ValueError, match="terlindungi"):
        env["audit_group"]([(root, [0]), (head_dependency, [0])], graph, audit)
    with pytest.raises(RuntimeError, match="Tidak ada group"):
        env["apply_structural_pruning"](SimpleNamespace(step=lambda **_kwargs: []), audit)
    output = env["cuda_latency"](core, torch.zeros(1, 3, 64, 64))
    assert output["status"] == "NOT_RUN"


def test_reload_failure_is_not_pass(env, workdir, monkeypatch, forbid_learning):
    core = synthetic_model(env)
    def failing_process(*_args, **_kwargs):
        return SimpleNamespace(returncode=17, stdout="", stderr="reload diagnostic sentinel")
    monkeypatch.setattr(env["subprocess"], "run", failing_process)
    with pytest.raises(RuntimeError, match="reload diagnostic sentinel"), env["checked_stage"]("reload_failure"):
        env["verify_fresh_reload"](workdir / "broken.pt", core, torch.zeros(1, 3, 64, 64), workdir)
    assert env["STATE"]["structural_smoke_status"] == "FAIL"
    assert not env["STATE"]["checks"]["checkpoint_reloaded"]
    tree = ast.parse(env["RELOAD_SCRIPT"])
    assert not any(isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
                   and node.func.attr in {"train", "backward", "step"} for node in ast.walk(tree))


@pytest.mark.slow
def test_real_validation_on_tiny_local_dataset(env, workdir, forbid_learning):
    core = synthetic_model(env)
    path = workdir / "synthetic.pt"
    env["save_fp32_checkpoint"](core, path, {"source": "synthetic evaluation only"})
    images, labels = workdir / "images", workdir / "labels"
    images.mkdir()
    labels.mkdir()
    env["Image"].new("RGB", (64, 64), "gray").save(images / "sample.jpg")
    (labels / "sample.txt").write_text("0 0.5 0.5 0.25 0.25\n", encoding="utf-8")
    data = workdir / "data.yaml"
    data.write_text("train: images\nval: images\nnames: [synthetic_a, synthetic_b, synthetic_c]\n", encoding="utf-8")
    dataset = env["prepare_validation_dataset"](data, workdir, workdir)
    env["WORKERS"] = 0
    result = env["evaluate_checkpoint"](path, dataset, workdir, "tiny_val")
    assert result["map50_95"] == 0.0
    assert len(result["per_class"]) == 3
    assert result["precision_mode"] == "FP32" and result["nms"] is None
