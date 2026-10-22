"""Offline PKD checks; actual Poltrada/CUDA/W&B are never substituted or claimed as tested."""
# ruff: noqa: S102 -- Only trusted notebook declarations and hash-pinned reference class are executed.
from __future__ import annotations

import ast
import base64
import copy
import gc
import hashlib
import json
import os
import subprocess
import sys
import types
import uuid
from pathlib import Path
from unittest.mock import Mock

import nbformat
import pytest
import torch
from PIL import Image
from test_serial_multiseed_notebooks import FakeWandb
from ultralytics import YOLO, settings
from ultralytics.cfg import get_cfg
from ultralytics.engine.trainer import BaseTrainer
from ultralytics.nn.distill_model import DistillationModel, FeatureHook
from ultralytics.nn.tasks import DetectionModel

ROOT = Path(__file__).resolve().parents[1]
FOLDER = ROOT / "notebooks/yolo26-kd-indotraffic/notebooks"
TARGET = FOLDER / "12_pkd_yolo26n.ipynb"
REFERENCE = ROOT / "tests/fixtures/mmrazor_pkd_v1_reference.py.txt"
NAMES = ["bus", "car", "motorcycle", "truck", "pickup", "minibus"]


def cell(notebook, key):
    return "".join(next(item for item in notebook["cells"] if item["id"] == "pkd-" + key)["source"])


@pytest.fixture(scope="session", autouse=True)
def threads():
    previous = torch.get_num_threads()
    torch.set_num_threads(2)
    yield
    torch.set_num_threads(previous)


@pytest.fixture
def workdir():
    directory = ROOT / "outputs/test_pkd" / uuid.uuid4().hex
    directory.mkdir(parents=True)
    return directory


@pytest.fixture
def environment(workdir):
    notebook = json.loads(TARGET.read_bytes())
    module = types.ModuleType("pkd_notebook_test_" + uuid.uuid4().hex)
    sys.modules[module.__name__] = module
    env = module.__dict__
    exec(compile(cell(notebook, "config"), "<pkd-config>", "exec"), env)
    declarations = ast.parse(cell(notebook, "environment"))
    declarations.body = [node for node in declarations.body if isinstance(node, (ast.Import, ast.ImportFrom, ast.FunctionDef))]
    # Tests replace W&B explicitly; importing its live SDK creates unnecessary sandbox temp directories.
    declarations.body = [node for node in declarations.body
                         if not (isinstance(node, ast.Import) and any(alias.name == "wandb" for alias in node.names))]
    exec(compile(declarations, "<pkd-safe-imports>", "exec"), env)
    dataset = ast.parse(cell(notebook, "dataset"))
    dataset.body = [node for node in dataset.body if isinstance(node, ast.FunctionDef)]
    exec(compile(dataset, "<pkd-labels>", "exec"), env)
    env.update(SESSION_DIR=workdir, ENVIRONMENT_INFO={"gpu": "CPU-SYNTHETIC-NOT-KAGGLE"},
               LAST_PHASE={"value": "initialization"}, DATASET_NAMES=NAMES,
               data_info={"nc": 6, "names": NAMES}, TEACHER_CHECKPOINT=str(workdir / "teacher.pt"),
               TEACHER_PROVENANCE={"best_pt_sha256": "mock", "version": "v62"},
               NUMERICAL_RESULT={"status": "PASS", "reference_sha256": env["REFERENCE_SHA256"]})
    for key in ("teacher", "features", "loss", "numerical", "protocol", "checkpoints", "callbacks", "diagnostic"):
        exec(compile(cell(notebook, key), "<pkd-" + key + ">", "exec"), env)
    yield env
    gc.collect()


def network(size):
    model = DetectionModel(f"yolo26{size}.yaml", nc=6, verbose=False)
    model.args = get_cfg(overrides={"imgsz": 64, "dis": 1.0, "device": "cpu", "amp": False})
    model.names = dict(enumerate(NAMES))
    return model


def example(batch=2, size=64):
    return {"img": torch.rand(batch, 3, size, size), "batch_idx": torch.arange(batch).float(),
            "cls": torch.zeros(batch, 1), "bboxes": torch.tensor([[0.5, 0.5, 0.25, 0.25]] * batch)}


def model(environment):
    return environment["PKDModel"](network("s"), network("n")).train()


def test_schema_defaults_and_no_other_method_in_objective():
    notebook = json.loads(TARGET.read_bytes())
    nbformat.validate(nbformat.from_dict(notebook))
    env = {}
    exec(compile(cell(notebook, "config"), "<config>", "exec"), env)
    assert {key: env[key] for key in ("EXPERIMENT_ID", "METHOD", "MODEL_WEIGHTS", "SEED", "EPOCHS", "IMGSZ", "BATCH",
                                     "PKD_WEIGHT", "RUN_FULL_TRAINING", "DIAGNOSTIC_BATCH", "DIAGNOSTIC_STEPS")} == {
        "EXPERIMENT_ID": "12_pkd_yolo26n", "METHOD": "pkd", "MODEL_WEIGHTS": "yolo26n.pt", "SEED": 42,
        "EPOCHS": 100, "IMGSZ": 640, "BATCH": 16, "PKD_WEIGHT": 1.0, "RUN_FULL_TRAINING": False,
        "DIAGNOSTIC_BATCH": 16, "DIAGNOSTIC_STEPS": 20}
    assert env["TEACHER_ARTIFACT"].endswith("02_baseline_yolo26s-q5kp9k9d-checkpoints:v62")
    assert env["WANDB_GROUP"] == "teacher_s_student_n_pkd"
    assert "teacher-s" in env["WANDB_TAGS"] and "teacher-m" not in env["WANDB_TAGS"]
    assert env["RESUME_STATUS"] == "RESUME_NOT_VERIFIED"
    source = cell(notebook, "loss")
    assert "loss_sl2(" not in source and "CrossKD" not in source and "attention" not in source
    for item in notebook["cells"]:
        if item["cell_type"] == "code":
            compile("".join(item["source"]), item["id"], "exec")
            assert item["outputs"] == [] and item["execution_count"] is None


def test_teacher_source_matches_native_and_crosskd_s_to_n(environment):
    expected = (
        "ilhamrafiqin-brawijaya-university/yolom26-kd-indotraffic/"
        "02_baseline_yolo26s-q5kp9k9d-checkpoints:v62"
    )
    assert environment["TEACHER_ARTIFACT"] == expected
    assert environment["TEACHER_RUN_ID"] == "q5kp9k9d"
    for filename in ("10_native_yolo26n_teacher_yolo26s.ipynb", "11_crosskd_yolo26n_teacher_yolo26s.ipynb"):
        notebook = json.loads((FOLDER / filename).read_bytes())
        values = {}
        for item in notebook["cells"]:
            if item["cell_type"] != "code":
                continue
            for node in ast.parse("".join(item["source"])).body:
                if isinstance(node, ast.Assign):
                    for target in node.targets:
                        if isinstance(target, ast.Name) and target.id in {"TEACHER_ARTIFACT", "TEACHER_RUN_ID"}:
                            values[target.id] = ast.literal_eval(node.value)
        assert values == {"TEACHER_ARTIFACT": expected, "TEACHER_RUN_ID": "q5kp9k9d"}


def test_actual_mmrazor_loss_gradients_bessel_epsilon_and_level_sum(environment):
    result = environment["numerical_verification"](REFERENCE.read_text(encoding="utf-8"))
    assert result["status"] == "PASS" and result["maximum_absolute_error"] < 1e-6
    (environment["SESSION_DIR"] / "numerical_result.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    assert result["identical_loss"] == 0 and result["opposite_loss"] > 1.9
    assert result["three_level_sum_verified"]
    # Large per-image shifts expose an incorrect normalization over H,W independently per image.
    value = torch.randn(2, 3, 5, 7)
    value[1] += 10
    normalized = environment["pkd_normalize"](value)
    torch.testing.assert_close(normalized.mean((0, 2, 3)), torch.zeros(3), atol=1e-6, rtol=0)
    assert normalized[0].mean().abs() > 0.5
    expected = (value - value.mean((0, 2, 3), keepdim=True)) / (value.std((0, 2, 3), correction=1, keepdim=True) + 1e-6)
    torch.testing.assert_close(normalized, expected, atol=1e-6, rtol=1e-5)


def test_reference_hash_mismatch_is_rejected_before_execution(environment):
    with pytest.raises(ValueError, match="Hash"):
        environment["load_mmrazor_reference"](REFERENCE.read_text(encoding="utf-8") + "\n# modified\n")


def test_fingerprint_stable_after_execution_and_detects_objective_edit(environment, monkeypatch):
    first = environment["protocol_fingerprint"]()
    for _ in range(3):
        environment["pkd_feature_loss"](torch.rand(2, 4, 5, 5), torch.rand(2, 4, 5, 5))
        assert environment["protocol_fingerprint"]() == first
    monkeypatch.setattr(environment["PKDModel"], "loss", lambda self, batch: None)
    assert environment["protocol_fingerprint"]() != first


@pytest.mark.parametrize("shape", [(1, 3, 1, 1), (3, 5, 7)])
def test_invalid_normalization_shape_rejected(environment, shape):
    with pytest.raises(ValueError):
        environment["pkd_normalize"](torch.zeros(shape))


def test_graph_taps_640_and_single_teacher_traversal_student_forward(environment, monkeypatch):
    wrapper = model(environment)
    counts = {"teacher_neck": 0, "teacher_head": 0, "student_head": 0}
    handles = [wrapper.teacher_model.model[0].register_forward_hook(lambda *a: counts.update(teacher_neck=counts["teacher_neck"] + 1)),
               wrapper.teacher_model.model[-1].register_forward_hook(lambda *a: counts.update(teacher_head=counts["teacher_head"] + 1)),
               wrapper.student_model.model[-1].register_forward_hook(lambda *a: counts.update(student_head=counts["student_head"] + 1))]
    monkeypatch.setattr(DistillationModel, "loss_sl2", lambda *a, **kw: pytest.fail("Native KD must not run"))
    try:
        assert wrapper.feats_idx == [16, 19, 22]
        assert [row["student_channels"] for row in wrapper.feature_layout] == [64, 128, 256]
        assert [row["teacher_channels"] for row in wrapper.feature_layout] == [128, 256, 512]
        assert sum(parameter.numel() for parameter in wrapper.projector.parameters()) == 172032
        wrapper.eval()
        with torch.no_grad():
            environment["teacher_neck_forward"](wrapper.teacher_model, torch.rand(1, 3, 640, 640), 22)
            wrapper.student_model(torch.rand(1, 3, 640, 640))
        for index, spatial in zip(wrapper.feats_idx, (80, 40, 20)):
            assert wrapper._teacher_feats[index].shape[-2:] == wrapper._student_feats[index].shape[-2:] == (spatial, spatial)
        counts.update(teacher_neck=0, teacher_head=0, student_head=0)
        wrapper.train()
        vector, items = wrapper(example())
        assert counts == {"teacher_neck": 1, "teacher_head": 0, "student_head": 1}
        assert torch.isfinite(vector).all() and float(items["pkd_loss"]) > 0
        assert not wrapper._teacher_feats and not wrapper._student_feats
        torch.testing.assert_close(vector[-1], items["pkd_weighted_loss"] * 2)
        torch.testing.assert_close(items["total_loss"] * 2, vector.sum().detach())
    finally:
        for handle in handles:
            handle.remove()
        wrapper._remove_feature_hooks()


def test_pkd_only_backward_student_projector_teacher_bn_and_optimizer(environment):
    wrapper = model(environment)
    try:
        before = environment["state_digest"](wrapper.teacher_model)
        vector, items = wrapper(example())
        vector[-1].backward()  # KD ONLY: detection cannot hide a broken student/projector path.
        assert environment["gradient_norm"](wrapper.student_model.parameters()) > 0
        assert environment["gradient_norm"](wrapper.projector.parameters()) > 0
        assert all(parameter.grad is None and not parameter.requires_grad for parameter in wrapper.teacher_model.parameters())
        trainer = environment["PKDTrainer"].__new__(environment["PKDTrainer"])
        optimizer = trainer.build_optimizer(wrapper, name="AdamW", lr=0.001, momentum=0.937, decay=0.0005)
        environment["audit_optimizer"](wrapper, optimizer)
        optimizer.step()
        assert environment["state_digest"](wrapper.teacher_model) == before
        assert not wrapper.teacher_model.training and torch.isfinite(items["pkd_loss"])
    finally:
        wrapper._remove_feature_hooks()


def test_detection_vector_matches_native_criterion_and_validation_is_kd_free(environment):
    wrapper = model(environment)
    baseline = copy.deepcopy(wrapper.student_model)
    for layer in baseline.modules():
        for key, hook in list(layer._forward_hooks.items()):
            if isinstance(hook, FeatureHook):
                del layer._forward_hooks[key]
    try:
        batch = example()
        expected, _items = baseline.loss(batch)
        actual, items = wrapper(batch)
        torch.testing.assert_close(actual[:-1], expected.reshape(-1), atol=1e-6, rtol=1e-5)
        assert float(items["pkd_weighted_loss"]) == pytest.approx(float(items["pkd_loss"]))
        wrapper.eval()
        with torch.no_grad():
            preds = wrapper(batch["img"])
            vector, values = wrapper.loss(batch, preds)
        assert vector[-1] == 0 and values["pkd_loss"] == 0
        assert not wrapper._teacher_feats and not wrapper._student_feats
    finally:
        wrapper._remove_feature_hooks()


def test_feature_exception_always_clears_hooks_cache(environment, monkeypatch):
    wrapper = model(environment)
    monkeypatch.setitem(environment, "pkd_feature_loss", lambda *a: (_ for _ in ()).throw(RuntimeError("injected PKD error")))
    try:
        with pytest.raises(RuntimeError, match="injected"):
            wrapper(example())
        assert wrapper._teacher_feats == wrapper._student_feats == {}
    finally:
        wrapper._remove_feature_hooks()


def test_protocol_matches_native_reference_and_blocks_oom_batch_retry(environment):
    original = json.loads((FOLDER / "05_native_yolo26n.ipynb").read_bytes())
    env = {"Path": Path, "teacher_path": Path(environment["TEACHER_CHECKPOINT"]), "RUN_NAME": "old"}
    exec(compile("".join(original["cells"][2]["source"]), "<native config>", "exec"), env)
    for item in original["cells"]:
        if item["cell_type"] != "code":
            continue
        for node in ast.parse("".join(item["source"])).body:
            if isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id == "train_args" for t in node.targets):
                exec(compile(ast.Module(body=[node], type_ignores=[]), "<native train args>", "exec"), env)
    env["train_args"].update(distill_model=environment["TEACHER_CHECKPOINT"], dis=1.0)
    old = vars(get_cfg(overrides=env["train_args"]))
    new = vars(get_cfg(overrides=environment["protocol_args"](Path("/new/run"))))
    assert {key for key in old if old[key] != new[key]} == {"name", "project"}
    trainer = environment["PKDTrainer"].__new__(environment["PKDTrainer"])
    trainer._requested_batch = 16
    trainer.batch_size, trainer.args = 8, types.SimpleNamespace(batch=8)
    with pytest.raises(RuntimeError, match="Batch"):
        trainer._build_train_pipeline()
    fake = types.SimpleNamespace(batch_size=16, args=types.SimpleNamespace(batch=16, imgsz=640), amp=True)
    environment["PKDTrainer"]._strict_batch_start(fake)
    assert fake._oom_retries == 3
    assert "super().optimizer_step()" in cell(json.loads(TARGET.read_bytes()), "protocol")
    assert BaseTrainer.optimizer_step is environment["DetectionTrainer"].optimizer_step


def test_resume_is_explicitly_not_supported(environment):
    with pytest.raises(ValueError, match="RESUME_NOT_VERIFIED"):
        environment["PKDTrainer"](overrides={"resume": True})


def teacher_fixture(environment, workdir):
    payload = b"mock teacher artifact bytes"
    checkpoint = workdir / "best.pt"
    checkpoint.write_bytes(payload)
    producer = types.SimpleNamespace(id="q5kp9k9d", path=[environment["WANDB_ENTITY"], environment["WANDB_PROJECT"], "q5kp9k9d"],
                                     config={"dataset": environment["DATA"], "model": "yolo26s.pt"})
    artifact = types.SimpleNamespace(version="v62", name="02_baseline_yolo26s-q5kp9k9d-checkpoints:v62", digest="artifact-digest",
        metadata={"epoch": 13}, logged_by=lambda: producer, download=Mock(return_value=str(workdir)),
        manifest=types.SimpleNamespace(entries={"best.pt": types.SimpleNamespace(size=len(payload), digest=base64.b64encode(hashlib.md5(payload).digest()).decode())}))
    actual = network("s")
    teacher = types.SimpleNamespace(model=actual, ckpt={"train_args": {"data": environment["DATA"], "model": "yolo26s.pt"}})
    lookup = Mock(return_value=artifact)
    environment["wandb"] = types.SimpleNamespace(Api=lambda: types.SimpleNamespace(artifact=lookup))
    environment["YOLO"] = Mock(return_value=teacher)
    return artifact, producer, teacher, lookup, payload


def test_teacher_fixed_reference_checksum_schema_classes_metadata(environment, workdir):
    artifact, _producer, _teacher, lookup, payload = teacher_fixture(environment, workdir)
    path, provenance = environment["download_teacher"]()
    lookup.assert_called_once_with(environment["TEACHER_ARTIFACT"], type="model")
    artifact.download.assert_called_once_with(root="/kaggle/working/teacher_yolo26s_v62")
    assert path == str(workdir / "best.pt")
    assert provenance["best_pt_sha256"] == hashlib.sha256(payload).hexdigest()
    assert provenance["artifact_metadata"]["epoch"] == 13 and provenance["version"] == "v62"
    assert provenance["architecture"] == "yolo26s"
    assert provenance["tensor_schema_verified"] and provenance["names"] == NAMES


@pytest.mark.parametrize("failure", ["version", "source", "manifest", "missing_file", "download", "checksum", "size", "scale", "schema", "names", "dataset"])
def test_bad_teacher_stops_without_fallback(environment, workdir, failure):
    artifact, producer, teacher, _lookup, _payload = teacher_fixture(environment, workdir)
    if failure == "version": artifact.version = "v51"
    elif failure == "source": producer.id = "other"
    elif failure == "manifest": artifact.manifest.entries.clear()
    elif failure == "missing_file": (workdir / "best.pt").unlink()
    elif failure == "download": artifact.download.side_effect = RuntimeError("download failed")
    elif failure == "checksum": artifact.manifest.entries["best.pt"].digest = "wrong"
    elif failure == "size": artifact.manifest.entries["best.pt"].size += 1
    elif failure == "scale": teacher.model.yaml["scale"] = "m"
    elif failure == "schema": teacher.model = network("n"); teacher.model.yaml["scale"] = "s"; teacher.model.yaml["yaml_file"] = "yolo26s.yaml"
    elif failure == "names": teacher.model.names = dict(enumerate(reversed(NAMES)))
    elif failure == "dataset": producer.config["dataset"] = "wrong.yaml"
    with pytest.raises((ValueError, RuntimeError, FileNotFoundError)):
        environment["download_teacher"]()
    assert all(call.args[0] != "yolo26s.pt" for call in environment["YOLO"].call_args_list)


def diagnostic_context(environment):
    return {"rows": [{"optimizer_step_performed": True, "actual_batch": 16, "imgsz": 640,
                       "allocated_bytes": 100_000_000, "iteration_seconds": 0.5,
                       "amp_scale_before": 65536, "amp_scale_after": 65536} for _ in range(20)],
            "teacher_unchanged": True, "total_vram_bytes": 16 * 1024**3,
            "peak_allocated_bytes": 1_000_000_000, "peak_reserved_bytes": 1_100_000_000,
            "train_batches_per_epoch": 100, "feature_layout": []}


def test_full_gate_requires_gpu_diagnostic_matching_code_and_teacher(environment, workdir, monkeypatch):
    checkpoint = workdir / "teacher.pt"
    checkpoint.write_bytes(b"teacher")
    environment["TEACHER_PROVENANCE"]["best_pt_sha256"] = environment["file_sha256"](checkpoint)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    with pytest.raises(RuntimeError, match="GPU"):
        environment["require_full_training"](None)
    environment["require_cuda"] = lambda: None  # Private CPU/mock gate fixture only.
    environment["RUN_FULL_TRAINING"] = True
    with pytest.raises(RuntimeError, match="diagnostic"):
        environment["require_full_training"](None)
    valid = environment["diagnostic_assessment"](diagnostic_context(environment))
    environment["require_full_training"](valid)
    environment["PKD_WEIGHT"] = 2
    with pytest.raises(RuntimeError, match="berubah"):
        environment["require_full_training"](valid)
    environment["PKD_WEIGHT"] = 1
    checkpoint.write_bytes(b"changed")
    with pytest.raises(RuntimeError, match="teacher"):
        environment["require_full_training"](valid)


def test_reexecuting_config_to_enable_full_preserves_audited_teacher(environment, workdir):
    checkpoint = workdir / "teacher.pt"
    checkpoint.write_bytes(b"audited teacher")
    environment["TEACHER_PROVENANCE"]["best_pt_sha256"] = environment["file_sha256"](checkpoint)
    environment["require_cuda"] = lambda: None  # Only a gate test, no training or GPU claim.
    report = environment["diagnostic_assessment"](diagnostic_context(environment))
    config = cell(json.loads(TARGET.read_bytes()), "config").replace("RUN_FULL_TRAINING = False", "RUN_FULL_TRAINING = True")
    exec(compile(config, "<edited config>", "exec"), environment)
    assert environment["TEACHER_CHECKPOINT"] == str(checkpoint)
    environment["require_full_training"](report)


@pytest.mark.parametrize("failure", ["memory_growth", "amp_unstable", "teacher_changed", "short_diagnostic", "batch_changed"])
def test_diagnostic_rejects_unstable_or_incomplete_result(environment, failure):
    context = diagnostic_context(environment)
    if failure == "memory_growth":
        for row in context["rows"][-4:]: row["allocated_bytes"] += 1024**3
    elif failure == "amp_unstable": context["rows"][-1]["optimizer_step_performed"] = False
    elif failure == "teacher_changed": context["teacher_unchanged"] = False
    elif failure == "short_diagnostic": context["rows"].pop()
    elif failure == "batch_changed": context["rows"][-1]["actual_batch"] = 8
    with pytest.raises(RuntimeError):
        environment["diagnostic_assessment"](context)


def test_fused_adamw_amp_skipped_update_has_no_valid_gradient_record(environment):
    env = environment
    net = torch.nn.Module()
    net.student_model = torch.nn.Linear(2, 2)
    net.projector = torch.nn.Linear(2, 1)
    net.teacher_model = torch.nn.Linear(2, 1).eval().requires_grad_(False)
    net.feature_layout = []
    optimizer = torch.optim.AdamW([p for p in net.parameters() if p.requires_grad], fused=True)
    scaler = torch.amp.GradScaler("cpu", init_scale=8)
    context = {"mode": "diagnostic", "batch": env["BATCH"], "run": Mock(config={}), "gradient": None}
    trainer = types.SimpleNamespace(model=net, optimizer=optimizer, batch_size=env["BATCH"],
        args=types.SimpleNamespace(imgsz=env["IMGSZ"]), amp=env["AMP"],
        device=torch.device("cpu"), train_loader=range(env["DIAGNOSTIC_STEPS"]))
    env["make_callbacks"](context)["on_train_start"](trainer)
    before = copy.deepcopy(net.student_model.state_dict())
    try:
        for multiplier in (float("inf"), 1.0):
            optimizer.zero_grad()
            loss = net.projector(net.student_model(torch.ones(1, 2))).sum() * multiplier
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(net.parameters(), 10)
            scaler.step(optimizer)
            scaler.update()
            if multiplier == float("inf"):
                assert context["gradient"] is None
                assert all(torch.equal(before[k], v) for k, v in net.student_model.state_dict().items())
        assert context["gradient"]["student"] > 0 and context["gradient"]["projector"] > 0
        assert any(not torch.equal(before[k], v) for k, v in net.student_model.state_dict().items())
    finally:
        context["optimizer_hook"].remove()


def configure_synthetic_cpu(env, workdir, monkeypatch):
    """Only this fixture changes data/size/device and bypasses CUDA; notebook defaults remain real."""
    from ultralytics.utils import callbacks

    monkeypatch.setattr(callbacks, "add_integration_callbacks", lambda instance: None)
    monkeypatch.setitem(settings, "wandb", False)
    dataset = workdir / "cpu_synthetic_not_poltrada"
    for split in ("train", "val"):
        for directory in ("images", "labels"):
            (dataset / split / directory).mkdir(parents=True)
        for index in range(4):
            Image.new("RGB", (64, 64), (40 + index * 10, 80, 150)).save(dataset / split / "images" / f"{index}.png")
            (dataset / split / "labels" / f"{index}.txt").write_text("0 .5 .5 .25 .25\n")
    yaml = dataset / "data.yaml"
    yaml.write_text(f"path: {dataset.as_posix()}\ntrain: train/images\nval: val/images\nnames: {json.dumps(NAMES)}\n")
    teacher, student = workdir / "synthetic_s.pt", workdir / "fixed_pretrained_n.pt"
    torch.save({"model": network("s"), "train_args": {"task": "detect", "imgsz": 64}}, teacher)
    torch.save({"model": network("n"), "train_args": {"task": "detect", "imgsz": 64}}, student)
    env.update(DATA=str(yaml), MODEL_WEIGHTS=str(student), TEACHER_CHECKPOINT=str(teacher),
               TEACHER_PROVENANCE={"best_pt_sha256": env["file_sha256"](teacher), "version": "v62"},
               BATCH=2, DIAGNOSTIC_BATCH=2, IMGSZ=64, EPOCHS=1, WORKERS=0, DEVICE="cpu", AMP=False,
               DIAGNOSTIC_STEPS=2, DIAGNOSTIC_WARMUP_STEPS=0)
    env["require_cuda"] = lambda: None  # Production CUDA gate is verified separately.
    original_args = env["protocol_args"]
    env["protocol_args"] = lambda *a, **kw: {**original_args(*a, **kw), "plots": False, "nbs": 2}
    original_val = YOLO.val
    monkeypatch.setattr(YOLO, "val", lambda model, **kw: original_val(model, **{**kw, "plots": False}))
    events = []
    wandb = FakeWandb(events)
    original_init = wandb.init

    def init(**kwargs):
        run = original_init(**kwargs)
        assert kwargs["group"] == "teacher_s_student_n_pkd"
        assert kwargs["config"]["teacher"] == "yolo26s"
        assert "teacher-s" in kwargs["tags"]
        assert kwargs["job_type"] in {"diagnostic", "training"}
        def use_artifact(reference, type):
            assert reference == env["TEACHER_ARTIFACT"] and type == "model"
            return types.SimpleNamespace(version="v62")
        run.use_artifact = use_artifact
        return run

    wandb.init = init
    env["wandb"] = wandb
    return wandb, events


@pytest.mark.slow
def test_actual_cpu_diagnostic_full_training_student_checkpoint_and_state_archives(environment, workdir, monkeypatch):
    env = environment
    wandb, events = configure_synthetic_cpu(env, workdir, monkeypatch)
    diagnostic = env["run_mode"]("diagnostic")
    assert diagnostic["passed"] and diagnostic["teacher_parameters_and_bn_unchanged"]
    assert len(diagnostic["rows"]) == 2 and diagnostic["actual_batch"] == 2
    assert all(row["gradient_norm_after_native_clipping"]["projector"] > 0 for row in diagnostic["rows"])
    env["DIAGNOSTIC_RESULT"], env["RUN_FULL_TRAINING"] = diagnostic, True
    result = env["run_mode"]("training")
    assert result["completed_epochs"] == 1 and result["status"] == "PASS"
    assert wandb.run is None and [run.kwargs["job_type"] for run in wandb.runs] == ["diagnostic", "training"]
    assert wandb.runs[0].id != wandb.runs[1].id
    assert events.index(("finish", 42, 0)) < [i for i, event in enumerate(events) if event[0] == "init"][1]
    for run in wandb.runs:
        assert [entry["artifact"].type for entry in run.uploads] == ["model", "results"]
        assert all(entry["waited"] for entry in run.uploads)
        files = run.uploads[0]["artifact"].files
        assert {"best.pt", "last.pt", "training_state_last.pt", "training_state_best.pt", "results.csv", "args.yaml",
                "teacher_provenance.json", "checkpoint_manifest.json"} <= set(files)
        status = json.loads((Path(files["best.pt"]).parent / "upload_status.json").read_bytes())
        assert status["status"] == "PASS"
        state = torch.load(files["training_state_last.pt"], map_location="cpu", weights_only=False)
        assert state["resume_status"] == "RESUME_NOT_VERIFIED" and state["projector"]
        assert state["optimizer"]["state"] and state["scheduler"]
        assert {"student", "ema_student", "ema_projector", "scaler", "rng_python", "rng_numpy", "rng_torch",
                "rng_cuda", "criterion_updates", "loader_generator_state"} <= set(state)
        # No notebook class or teacher weights in either published inference checkpoint.
        deployed = YOLO(files["best.pt"]).model
        assert not isinstance(deployed, DistillationModel) and not hasattr(deployed, "projector")
    best = Path(result["student_checkpoint"])
    archive = best.parent / "training_state_last.pt"
    script = '''import sys,torch,numpy as np
from ultralytics import YOLO
model=YOLO(sys.argv[1])
assert not hasattr(model.model,'teacher_model') and not hasattr(model.model,'projector')
result=model.predict(np.zeros((64,64,3),dtype=np.uint8),imgsz=64,device='cpu',verbose=False)
assert len(result)==1 and torch.isfinite(result[0].boxes.data).all()
state=torch.load(sys.argv[2],map_location='cpu',weights_only=False)
assert state['projector'] and state['resume_status']=='RESUME_NOT_VERIFIED'
print('PORTABLE_STUDENT_AND_ARCHIVE_PASS')'''
    process = subprocess.run([sys.executable, "-I", "-c", script, str(best), str(archive)],
                             cwd=workdir, capture_output=True, text=True, timeout=120, check=False,
                             env={**os.environ, "OMP_NUM_THREADS": "2", "YOLO_CONFIG_DIR": str(workdir / "child-config")})
    assert process.returncode == 0, process.stdout + process.stderr
    assert "PORTABLE_STUDENT_AND_ARCHIVE_PASS" in process.stdout


@pytest.mark.slow
def test_actual_cpu_upload_failure_preserves_snapshot_and_blocks_full(environment, workdir, monkeypatch):
    env = environment
    wandb, _events = configure_synthetic_cpu(env, workdir, monkeypatch)
    wandb.fail_stage, wandb.fail_seed = "epoch_upload", 42
    env["DIAGNOSTIC_RESULT"] = {"passed": True, "status": "PASS"}  # Must be invalidated by the new attempt.
    with pytest.raises(RuntimeError, match="epoch_upload"):
        env["run_mode"]("diagnostic")
    assert env["DIAGNOSTIC_RESULT"] is None and wandb.run is None
    snapshot = next(workdir.rglob("epoch_snapshots/epoch-001"))
    assert all((snapshot / name).is_file() for name in ("best.pt", "last.pt", "training_state_last.pt", "training_state_best.pt"))
    assert json.loads((snapshot / "upload_status.json").read_bytes())["status"] == "FAIL"
    assert next(workdir.rglob("failure.json")).is_file()
    env["RUN_FULL_TRAINING"] = True
    with pytest.raises(RuntimeError, match="diagnostic"):
        env["require_full_training"](env["DIAGNOSTIC_RESULT"])
