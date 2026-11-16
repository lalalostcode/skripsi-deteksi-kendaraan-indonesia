"""Contracts and failure cases for standalone FILKOM ablations; no online services."""
# ruff: noqa: S102 -- Execute trusted declarations and AST-selected repository functions.
import ast
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import nbformat
import pytest

from scripts.generate_kd_ablation_notebooks import NAMES, ROOT, build_notebook, tagged
from scripts.generate_selected_kd_notebooks import build_notebook as selected_book


def configuration(method, **changes):
    scope = {}
    book = build_notebook(method)
    for tag in ("config", "selection", "protocol"):
        exec("".join(tagged(book, tag)["source"]), scope)
        if tag == "config":
            scope.update(changes)
    return scope


def overlay(*names, **scope):
    tree = ast.parse((ROOT / "scripts/kd_ablation_source.py").read_text())
    code = ast.Module(body=[n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name in names], type_ignores=[])
    exec(compile(code, "ablation-overlay-test", "exec"), scope)
    return scope


@pytest.mark.parametrize("method", NAMES)
def test_schema_regeneration_standalone_and_unchanged_algorithm(method):
    book = build_notebook(method)
    assert book == json.loads((ROOT / "notebooks/filkom/recovery" / NAMES[method]).read_text(encoding="utf-8"))
    nbformat.validate(nbformat.from_dict(book))
    for c in book["cells"]:
        if c["cell_type"] == "code":
            compile("".join(c["source"]), NAMES[method], "exec")
            assert "from scripts" not in "".join(c["source"])
            assert c["outputs"] == [] and c["execution_count"] is None
    def definitions(b):
        tree = ast.parse("".join(tagged(b, "runtime")["source"]))
        names = ({"CrossKDModel", "combined_loss", "quality_focal_loss", "aligned_giou", "match_statistics", "boxes_xyxy", "align_channels"}
                 if method == "crosskd" else {"MGDModel", "MGDFeatureLoss", "neck_taps", "teacher_neck_forward"})
        return {n.name: ast.dump(n) for n in tree.body if isinstance(n, (ast.ClassDef, ast.FunctionDef)) and n.name in names}
    assert definitions(book) == definitions(selected_book("filkom", method))


@pytest.mark.parametrize("method", NAMES)
def test_server_configuration_protocol_and_input_pins(method):
    s = configuration(method)
    original = {}
    for tag in ("config", "selection", "protocol"):
        exec("".join(tagged(selected_book("filkom", "mgd"), tag)["source"]), original)
    original["SEED"] = 42
    assert s["fine_tuned_train_args"]("DATA", "PROJECT", "NAME") == original["fine_tuned_train_args"]("DATA", "PROJECT", "NAME")
    assert s["fine_tuned_val_args"]("DATA")["split"] == "val"
    assert s["SEED"] == 42 and len(s["STUDENT_PINS"]) == 1
    assert s["TEACHER_SHA256"] == original["TEACHER_SHA256"]
    assert s["OUTPUT_DIR"].is_relative_to(s["ABLATION_ROOT"])
    assert not s["AUTO_DELETE_SMOKE_ARTIFACTS"] and not s["AUTO_DELETE_LOCAL_SMOKE_WEIGHTS"]


@pytest.mark.parametrize("method,changes", [
    ("crosskd", {"KD_SCALE": 3}), ("crosskd", {"KD_CLS_WEIGHT": 2}), ("crosskd", {"KD_CLS_GAMMA": 2}),
    ("mgd", {"MGD_ALPHA": 3e-5}), ("mgd", {"MGD_MASK_RATIO": .5}),
    ("mgd", {"EVAL_SPLIT": "test"}), ("mgd", {"STUDENT_INPUTS": ["yolo26s-lamp-su1.20x-seed0-raw:v1"]}),
])
def test_invalid_selection_fails_before_bootstrap(method, changes):
    with pytest.raises(ValueError):
        configuration(method, **changes)


def test_paths_and_names_are_isolated(monkeypatch, tmp_path):
    monkeypatch.setenv("SKRIPSI_ABLATION_OUTPUT_DIR", str(tmp_path / "ablation"))
    monkeypatch.setenv("SKRIPSI_OUTPUT_DIR", str(tmp_path / "existing-main-recovery"))
    names = []
    for method, values in (("crosskd", [.5, 1., 2.]), ("mgd", [1e-5, 2e-5, 4e-5])):
        for value in values:
            changes = {"KD_SCALE": value, "KD_CLS_WEIGHT": value, "KD_BOX_WEIGHT": value} if method == "crosskd" else {"MGD_ALPHA": value}
            s = configuration(method, **changes)
            s.update(EXECUTION_ID="20261116T000000Z-abcdef12")
            s = overlay("ft_name", "ft_group", **s)
            assert s["OUTPUT_DIR"].is_relative_to(tmp_path / "ablation")
            names.append(s["ft_name"]())
            assert "ablation" in s["ft_group"]()
    assert len(set(names)) == 6


def test_comparison_requires_all_three_and_uses_final_metrics():
    import math
    s = overlay("ablation_comparison", RECOVERY_METHOD="crosskd", MODE="full_val", EPOCHS=100, math=math)
    def row(value, score, signature="same"):
        return {"method": "crosskd", "mode": "full_val", "ablation_value": value, "comparison_signature": signature,
                    "status": "PASS", "publication_status": "CONFIRMED", "logging_verified": True, "epochs_completed": 100,
                    "final_validation": {"map50_95": score}, "historical_training_map": 100-score}
    rows = [row(.5, .4), row(1., .5), row(2., .6, "different-dataset")]
    table, result = s["ablation_comparison"](rows, "same")
    assert len(table) == 3 and result["status"] == "PENDING"
    rows += [row(2., .6)]
    _, result = s["ablation_comparison"](rows, "same")
    assert result["best_values"] == [2.] and result["boundary_optimum"]
    rows += [row(1., .6)]
    assert s["ablation_comparison"](rows, "same")[1]["best_values"] == [1., 2.]
    rows += [row(1., .1)]
    assert s["ablation_comparison"](rows, "same")[1]["best_values"] == [2.]  # latest, not best-scoring attempt


def test_default_reuse_checks_provenance_not_seed_or_score_only():
    record = {"train_fingerprint": "train", "val_fingerprint": "val", "class_names": ["car"]}
    identity = {"seed": 42, "method": "mgd", "raw_artifact_ref": "raw:v1", "raw_sha256": "raw-sha",
                    "teacher_artifact_ref": "teacher:v50", "teacher_sha256": "teacher-sha"}
    proof = {**identity, "status": "PASS", "logging_verified": True, "epochs": 100,
             "hyperparameters": {"MGD_ALPHA": 2e-5}, "dataset": {**record, "label_adapter": {"policy": "mixed"}},
             "method_source_sha256": "code", "training": {"epochs": 100}, "evaluation": {"split": "val"},
             "environment": {"gpu": "A100"}}
    settings = {"training": {"epochs": 100}, "evaluation": {"split": "val"}, "environment": {"gpu": "A100"}}
    s = overlay("ablation_reuse_check", ft_identity=lambda: identity, EPOCHS=100, RECOVERY_METHOD="mgd",
                recovery_kd_settings=lambda m: {"MGD_ALPHA": 2e-5}, FILKOM_MGD_LABEL_POLICY="mixed",
                METHOD_SOURCE_SHA256="code", HISTORICAL_IMPLEMENTATIONS=["old-known"], ABLATION_SETTINGS=settings,
                ablation_protocol=lambda x: x)
    assert s["ablation_reuse_check"](proof, record)["eligible"]
    for key, value in [("seed", 0), ("raw_sha256", "other"), ("training", {}), ("environment", {}),
                       ("dataset", {**proof["dataset"], "val_fingerprint": "other"}), ("method_source_sha256", "unknown")]:
        assert not s["ablation_reuse_check"]({**proof, key: value}, record)["eligible"], key


def test_snapshot_rejects_missing_identity_before_restore(tmp_path):
    called = []
    s = overlay("ft_restore_snapshot", EXECUTION_ID="new", RESUME_SNAPSHOT_REF="entity/project/old:v1",
                wandb=SimpleNamespace(Api=lambda **k: SimpleNamespace(artifact=lambda *a, **kw: SimpleNamespace(metadata={}))),
                workflow_version_ref=lambda x: x, _ablation_restore=lambda f: called.append(f))
    with pytest.raises(ValueError, match="Snapshot"):
        s["ft_restore_snapshot"](tmp_path)
    assert not called


def test_download_rejects_hash_mismatch_and_accepts_fixed_version(tmp_path):
    def download(root):
        Path(root).mkdir(parents=True, exist_ok=True)
        Path(root, "best.pt").write_bytes(b"checkpoint")
    artifact = SimpleNamespace(qualified_name="entity/project/ckpt:v2", download=download,
                               metadata={"files_sha256": {"best.pt": hashlib.sha256(b"checkpoint").hexdigest()}})
    s = overlay("ablation_download", workflow_version_ref=lambda ref: ref,
                sha256_file=lambda p: hashlib.sha256(p.read_bytes()).hexdigest())
    assert s["ablation_download"](artifact, tmp_path) == artifact.qualified_name
    artifact.metadata["files_sha256"]["best.pt"] = "wrong"
    with pytest.raises(ValueError, match="SHA-256"):
        s["ablation_download"](artifact, tmp_path)


def test_publication_retry_keeps_immutable_journal_and_verifies_download(tmp_path):
    # Exercise the actual inherited publisher, with an interrupted first wait after commit.
    tree = ast.parse((ROOT / "scripts/pruned_finetune_selected_source.py").read_text())
    function = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "ft_publish")
    from tempfile import TemporaryDirectory
    def read(path, default=None):
        return json.loads(path.read_text()) if path.exists() else default
    def write(path, obj):
        path.write_text(json.dumps(obj))
    uploads = []
    class Artifact:
        def __init__(self, name, type, metadata):
            self.name, self.metadata = name, metadata
            self.qualified_name = "entity/project/" + name + ":v0"
            self.files = {}
            self.url = "https://example.invalid/artifact"
        def add_file(self, path, name):
            self.files[name] = Path(path).read_bytes()
        def wait(self, **kwargs):
            raise TimeoutError("Committed but caller timed out")
        def download(self, root):
            for name, content in self.files.items():
                target = Path(root, name); target.parent.mkdir(parents=True, exist_ok=True); target.write_bytes(content)
            return root
    api = SimpleNamespace(run=lambda ref: SimpleNamespace(logged_artifacts=lambda: uploads), artifact=lambda ref: uploads[0])
    scope = {"Path": Path, "TemporaryDirectory": TemporaryDirectory, "ft_read": read, "write_json": write,
                 "sha256_file": lambda p: hashlib.sha256(p.read_bytes()).hexdigest(), "workflow_version_ref": lambda x: x,
                 "WANDB_ENTITY": "entity", "WANDB_PROJECT": "project", "wandb": SimpleNamespace(Api=lambda **kw: api, Artifact=Artifact)}
    exec(compile(ast.Module(body=[function], type_ignores=[]), "inherited-publisher", "exec"), scope)
    run = SimpleNamespace(id="run", name="candidate", log_artifact=lambda a, **kw: uploads.append(a) or a)
    checkpoint = tmp_path / "best.pt"; checkpoint.write_bytes(b"model")
    with pytest.raises(TimeoutError):
        scope["ft_publish"](run, tmp_path, "checkpoints", {"best.pt": checkpoint}, {})
    receipt = scope["ft_publish"](run, tmp_path, "checkpoints", {"best.pt": checkpoint}, {})
    assert receipt["status"] == "CONFIRMED" and len(uploads) == 1
    checkpoint.write_bytes(b"changed")
    with pytest.raises(ValueError, match="berubah"):
        scope["ft_publish"](run, tmp_path, "checkpoints", {"best.pt": checkpoint}, {})


def test_mgd_alpha_only_scales_loss_and_gradient_with_identical_mask():
    import runpy

    import torch
    code = runpy.run_path(str(ROOT / "scripts/mgd_method_source.py"))["MGD_LOSS_SOURCE"]
    scope = {}
    exec(code, scope)
    torch.set_num_threads(2)
    torch.manual_seed(42)
    cls = scope["MGDFeatureLoss"]
    reference = cls(3, 4, "P3", alpha_mgd=1e-5)
    student, teacher = torch.randn(2, 3, 4, 4), torch.randn(2, 4, 4, 4)
    losses, gradients, raw = [], [], []
    for alpha in [1e-5, 2e-5, 4e-5]:
        loss_module = cls(3, 4, "P3", alpha_mgd=alpha)
        loss_module.load_state_dict(reference.state_dict())
        x = student.clone().requires_grad_(True)
        target = teacher.clone().requires_grad_(True)
        torch.manual_seed(42)
        value = loss_module(x, target)
        value.backward()
        assert target.grad is None and torch.isfinite(x.grad).all()
        losses.append(value.detach() / alpha); gradients.append(x.grad / alpha); raw.append(loss_module.last_raw_loss)
    for index in [1, 2]:
        torch.testing.assert_close(losses[0], losses[index])
        torch.testing.assert_close(gradients[0], gradients[index])
        torch.testing.assert_close(raw[0], raw[index])


def test_crosskd_scale_preserves_unweighted_losses_and_scales_gradient():
    import torch
    from torch.nn import functional as F
    tree = ast.parse("".join(tagged(build_notebook("crosskd"), "runtime")["source"]))
    names = {"combined_loss", "quality_focal_loss", "aligned_giou"}
    scope = {"torch": torch, "F": F}
    exec(compile(ast.Module(body=[n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name in names],
                            type_ignores=[]), "crosskd-original-losses", "exec"), scope)
    outputs, gradients = [], []
    for scale in [.5, 1., 2.]:
        logits = torch.tensor([-.5, .3], requires_grad=True)
        boxes = torch.tensor([[.1, .1, .8, .8]], requires_grad=True)
        parts = {"kd_cls_loss": scope["quality_focal_loss"](logits, torch.tensor([.2, .4]), 1.).mean(),
                 "kd_box_loss": (1 - scope["aligned_giou"](boxes, torch.tensor([[0., 0., 1., 1.]]))).mean()}
        wrapper = SimpleNamespace(training=True, _teacher_feats={}, _student_feats={},
                                  teacher_model=lambda x: None, kd_loss=lambda parts=parts: parts,
                                  student_model=SimpleNamespace(__call__=lambda x: None))
        class Student:
            def __call__(self, image):
                return None
            def loss(self, batch, preds):
                return torch.zeros(1), {}
        wrapper.student_model = Student()
        value, items = scope["combined_loss"](wrapper, {"img": torch.zeros(1, 3, 4, 4)}, None,
                                              {key: scale for key in parts})
        value.sum().backward()
        outputs.append(items)
        gradients.append((logits.grad / scale, boxes.grad / scale))
    for index in [1, 2]:
        for key in ["kd_cls_loss", "kd_box_loss"]:
            torch.testing.assert_close(outputs[0][key], outputs[index][key])
        torch.testing.assert_close(gradients[0][0], gradients[index][0])
        torch.testing.assert_close(gradients[0][1], gradients[index][1])


def test_verified_default_reuses_completed_state_without_training(tmp_path):
    from copy import deepcopy

    raw = SimpleNamespace(structure="pruned", state="raw-state")
    state = {"training_finished": True, "epoch": 100, "optimizer_steps": 10,
             "raw_sha256": "raw-file", "raw_state_sha256": "raw-state", "live_model": raw, "ema": raw,
             "best": {"relative_path": "checkpoints/best.pt", "sha256": "best-sha"},
             "epoch_rows": [{"epoch": i} for i in range(1, 101)]}
    result = {"kd_audit": {"teacher_unchanged": True}, "environment": {"gpu": "A100"},
              "checkpoint_artifact": {"files_sha256": {"best.pt": "best-sha"}}}
    remote = SimpleNamespace(id="historical", name="yolo26s-lamp-su1.20x-seed42-mgd", url="https://example.invalid/run",
                             summary={"status": "PASS", "logging_verified": True},
                             config={"method_hyperparameters": {}, "study_settings": {"evaluation": {"split": "val"}}})
    artifacts = [SimpleNamespace(name=remote.name + "-" + label + ":v3", qualified_name="entity/project/" + label + ":v3")
                 for label in ("results", "resume")]
    remote.logged_artifacts = lambda: artifacts
    def download(artifact, folder):
        folder.mkdir(parents=True, exist_ok=True)
        values = ({"evaluation_manifest.json": result, "training_config.json": {"effective": {}}}
                  if folder.name == "results" else
                  {"resume_contract.json": {"contract": {"implementation": "known"}},
                   "latest_state.json": {"relative_path": "resume.pt"}})
        for name, value in values.items():
            (folder / name).write_text(json.dumps(value))
        best = folder / "checkpoints/best.pt"
        best.parent.mkdir(exist_ok=True); best.write_bytes(b"best")
        return artifact.qualified_name
    checks = []
    def check(proof, record):
        checks.append(proof)
        return {"eligible": True, "reasons": []}
    def read(path):
        return json.loads(path.read_text())
    def write(path, value):
        path.write_text(json.dumps(value))
    import shutil
    s = overlay("ablation_find_default", MODE="full_val", REUSE_EXISTING_DEFAULT=True,
                ABLATION_VALUE=2e-5, RECOVERY_METHOD="mgd", EPOCHS=100, WANDB_ENTITY="entity", WANDB_PROJECT="project",
                wandb=SimpleNamespace(Api=lambda **kw: SimpleNamespace(runs=lambda *a, **k: [remote])),
                ablation_download=download, ft_read=read, write_json=write, ablation_reuse_check=check,
                ABLATION_SETTINGS={"environment": {"gpu": "A100"}}, torch=SimpleNamespace(load=lambda *a, **k: deepcopy(state)),
                ft_identity=lambda: {"raw_sha256": "raw-file"}, recovery_state_sha256=lambda r: r.state,
                recovery_structure_signature=lambda r: r.structure, sha256_file=lambda p: "best-sha", shutil=shutil,
                safe_error=str)
    reused = s["ablation_find_default"](tmp_path, {}, raw)
    assert reused["epoch"] == 100 and checks
    assert (tmp_path / "historical_best.pt").read_bytes() == b"best"
    audit = read(tmp_path / "default_reuse_audit.json")
    assert audit["status"] == "REUSED" and audit["source_run_url"] == remote.url
    # Corrupt/unverifiable packages instead request an independent rerun.
    s["sha256_file"] = lambda p: "corrupt"
    assert s["ablation_find_default"](tmp_path, {}, raw) is None
    assert read(tmp_path / "default_reuse_audit.json")["status"] == "RERUN"
