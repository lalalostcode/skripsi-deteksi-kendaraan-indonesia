"""Offline MGD source/protocol checks; never claim actual Poltrada/CUDA/W&B execution."""
# ruff: noqa: S102 -- Execute trusted notebook declarations and the SHA-pinned official reference only.
from __future__ import annotations

import ast
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
from typing import Any
from unittest.mock import Mock

import nbformat
import pytest
import torch
from PIL import Image
from test_serial_multiseed_notebooks import FakeWandb
from torch import nn
from ultralytics import YOLO, settings
from ultralytics.cfg import DEFAULT_CFG_DICT, get_cfg
from ultralytics.engine.trainer import BaseTrainer
from ultralytics.nn.distill_model import DistillationModel, FeatureHook
from ultralytics.nn.tasks import DetectionModel

from scripts.notebook_paths import notebook_path

ROOT = Path(__file__).resolve().parents[1]
TARGET = ROOT / "notebooks/kaggle/kd/teacher_s/15_mgd_yolo26n_teacher_yolo26s.ipynb"
REFERENCE = ROOT / "tests/fixtures/mgd_detection_reference.py.txt"
REFERENCE_SHA = "a91595b98b906363abff7374662638ccaa7cc1557b44ab61640e5d610b9649de"
NAMES = ["Bus", "Mobil Penumpang", "Pejalan Kaki", "Sepeda Motor", "Truck", "Unmotorized"]


def cell(book, suffix):
    return "".join(next(c for c in book["cells"] if c["id"] == "mgd-" + suffix)["source"])


def declarations(source):
    tree = ast.parse(source)
    tree.body = [node for node in tree.body if isinstance(node, (ast.Import, ast.ImportFrom, ast.FunctionDef, ast.ClassDef))]
    tree.body = [node for node in tree.body if not isinstance(node, ast.Import)
                 or not any(alias.name == "wandb" for alias in node.names)]
    return tree


@pytest.fixture(scope="session", autouse=True)
def threads():
    original = torch.get_num_threads()
    torch.set_num_threads(2)
    yield
    torch.set_num_threads(original)


@pytest.fixture
def workdir():
    path = ROOT / "outputs/test_mgd" / uuid.uuid4().hex
    path.mkdir(parents=True)
    return path


@pytest.fixture
def environment(workdir):
    book = json.loads(TARGET.read_bytes())
    module = types.ModuleType("mgd_test_" + uuid.uuid4().hex)
    sys.modules[module.__name__] = module
    env = module.__dict__
    env.update({name: value for name, value in globals().items() if not name.startswith("__")})
    exec(compile(cell(book, "config"), "<mgd-config>", "exec"), env)
    exec(compile(declarations(cell(book, "environment")), "<mgd-safe-environment>", "exec"), env)
    exec(compile(declarations(cell(book, "dataset")), "<mgd-labels>", "exec"), env)
    env.update(SESSION_DIR=workdir, DATASET_NAMES=NAMES, data_info={"nc": 6, "names": NAMES},
               TEACHER_CHECKPOINT=str(workdir / "teacher.pt"), TEACHER_PROVENANCE={"best_pt_sha256": "mock", "version": "v62"},
               LAST_PHASE={"value": "initialization"}, ENVIRONMENT_INFO={"gpu": "CPU-SYNTHETIC-NOT-KAGGLE"})
    for suffix in ("features", "loss", "protocol", "checkpoints", "callbacks", "runtime"):
        exec(compile(cell(book, suffix), "<mgd-" + suffix + ">", "exec"), env)
    yield env
    gc.collect()


def official_reference():
    payload = REFERENCE.read_bytes()
    assert hashlib.sha256(payload).hexdigest() == REFERENCE_SHA
    tree = ast.parse(payload.decode("utf-8"))
    tree.body = [node for node in tree.body if not isinstance(node, ast.ImportFrom) or node.level == 0]
    for node in tree.body:
        if isinstance(node, ast.ClassDef):
            node.decorator_list.clear()
    env = {}
    exec(compile(tree, "<official-detection-MGD>", "exec"), env)
    return env["FeatureLoss"]


def network(size):
    result = DetectionModel(f"yolo26{size}.yaml", nc=6, verbose=False)
    result.args = get_cfg(overrides={"imgsz": 64, "device": "cpu", "amp": False, "epochs": 100, "dis": 1.0})
    result.names = dict(enumerate(NAMES))
    return result


def example(batch=2):
    return {"img": torch.rand(batch, 3, 64, 64), "batch_idx": torch.arange(batch).float(),
            "cls": torch.zeros(batch, 1), "bboxes": torch.tensor([[0.5, 0.5, 0.25, 0.25]] * batch)}


def model(env):
    return env["MGDModel"](network("s"), network("n")).train()


def digest(module):
    result = hashlib.sha256()
    for key, value in module.state_dict().items():
        result.update(key.encode())
        result.update(value.detach().cpu().contiguous().reshape(-1).view(torch.uint8).numpy().tobytes())
    return result.hexdigest()


def test_schema_defaults_reference_and_one_active_seed():
    book = json.loads(TARGET.read_bytes())
    nbformat.validate(nbformat.from_dict(book))
    env = {}
    config = cell(book, "config")
    exec(compile(config, "<configuration-only>", "exec"), env)
    assert env["METHOD"] == "mgd" and env["MODEL_WEIGHTS"] == "yolo26n.pt"
    assert env["EPOCHS"] == 100 and env["BATCH"] == 16 and env["IMGSZ"] == 640
    assert env["RUN_FULL_TRAINING"] is True and env["RUN_SMOKE_TEST"] is False
    assert env["SEEDS_TO_RUN"] == [42] and env["SEED"] == 42
    for seed in (123, 2024, 7, 3407):
        assert any(line.strip().startswith(f"# {seed},") for line in config.splitlines())
    assert env["MGD_ALPHA"] == 2e-5 and env["MGD_MASK_RATIO"] == 0.65
    assert env["WANDB_GROUP"] == "T26s_S26n"
    assert env["TEACHER_ARTIFACT"].endswith("02_baseline_yolo26s-q5kp9k9d-checkpoints:v62")
    assert env["TEACHER_RUN_ID"] == "q5kp9k9d"
    assert env["RESUME_STATUS"] == "RESUME_NOT_VERIFIED"
    assert env["KEEP_LAST_LOCAL_EPOCHS"] == 1
    for c in book["cells"]:
        if c["cell_type"] == "code":
            assert c["execution_count"] is None and c["outputs"] == []
            compile("".join(c["source"]), c["id"], "exec")
    assert hashlib.sha256(REFERENCE.read_bytes()).hexdigest() == REFERENCE_SHA
    loss_source = cell(book, "loss")
    assert "loss_sl2(" not in loss_source and "pkd_normalize" not in loss_source


@pytest.mark.parametrize("channels,batch", [((3, 5), 2), ((4, 4), 1)])
def test_official_detection_loss_and_gradient_parity(environment, channels, batch):
    cin, cout = channels
    official = official_reference()(cin, cout, "reference", alpha_mgd=2e-5, lambda_mgd=0.65)
    adapted = environment["MGDFeatureLoss"](cin, cout, "adapted", alpha_mgd=2e-5, lambda_mgd=0.65)
    adapted.load_state_dict(official.state_dict(), strict=True)
    student = torch.randn(batch, cin, 5, 7)
    teacher = torch.randn(batch, cout, 5, 7)
    expected_student, actual_student = student.clone().requires_grad_(), student.clone().requires_grad_()
    actual_teacher = teacher.clone().requires_grad_()
    torch.manual_seed(21342)
    expected = official(expected_student, teacher)
    expected.backward()
    torch.manual_seed(21342)
    actual = adapted(actual_student, actual_teacher)
    actual.backward()
    torch.testing.assert_close(actual, expected, atol=1e-7, rtol=1e-5)
    torch.testing.assert_close(actual_student.grad, expected_student.grad, atol=1e-7, rtol=1e-5)
    for name, parameter in adapted.named_parameters():
        torch.testing.assert_close(parameter.grad, dict(official.named_parameters())[name].grad, atol=1e-7, rtol=1e-5)
    assert actual_teacher.grad is None
    torch.testing.assert_close(student, actual_student.detach())


def test_spatial_mask_threshold_no_rescaling_all_positions_and_sse_over_batch(environment, monkeypatch):
    block = environment["MGDFeatureLoss"](3, 3, "mask", alpha_mgd=0.25, lambda_mgd=0.65)
    block.generation = nn.Identity()
    requested = []
    def fixed_rand(shape, *args, **kwargs):
        requested.append((shape, args, kwargs))
        return torch.tensor([0.0, 0.35, 0.35001, 0.99]).reshape(1, 1, 1, 4).expand(2, 1, 1, 4).clone()
    monkeypatch.setattr(torch, "rand", fixed_rand)
    student, teacher = torch.ones(2, 3, 1, 4), torch.full((2, 3, 1, 4), 2.0)
    actual = block(student, teacher)
    # Two visible pixels contribute 1 each, two masked pixels contribute 4 each;
    # total 10 * C=3 per image. This rejects mean-MSE and masked-only reconstruction.
    torch.testing.assert_close(actual, torch.tensor(0.25 * 30.0))
    assert requested[0][0] == (2, 1, 1, 4) and "device" not in requested[0][2]
    torch.testing.assert_close(student, torch.ones_like(student))


@pytest.mark.parametrize("ratio,raw", [(0.0, 0.0), (1.0, 24.0)])
def test_zero_and_full_mask_boundaries(environment, ratio, raw):
    block = environment["MGDFeatureLoss"](3, 3, "boundaries", alpha_mgd=1.0, lambda_mgd=ratio)
    block.generation = nn.Identity()
    torch.testing.assert_close(block(torch.ones(2, 3, 2, 4), torch.ones(2, 3, 2, 4)), torch.tensor(raw))


def test_graph_taps_generators_parameter_count_and_no_native_kd(environment, monkeypatch):
    wrapper = model(environment)
    counts = {"teacher_neck": 0, "teacher_head": 0, "student_head": 0}
    handles = [wrapper.teacher_model.model[0].register_forward_hook(lambda *a: counts.update(teacher_neck=counts["teacher_neck"] + 1)),
               wrapper.teacher_model.model[-1].register_forward_hook(lambda *a: counts.update(teacher_head=counts["teacher_head"] + 1)),
               wrapper.student_model.model[-1].register_forward_hook(lambda *a: counts.update(student_head=counts["student_head"] + 1))]
    monkeypatch.setattr(DistillationModel, "loss_sl2", lambda *_a, **_k: pytest.fail("Native KD was invoked"))
    try:
        assert wrapper.feats_idx == [16, 19, 22]
        assert [row["student_channels"] for row in wrapper.feature_layout] == [64, 128, 256]
        assert [row["teacher_channels"] for row in wrapper.feature_layout] == [128, 256, 512]
        assert sum(p.numel() for p in wrapper.projector.parameters()) == 6367872
        for block in wrapper.projector:
            assert isinstance(block.align, nn.Conv2d) and block.align.bias is not None
            assert [type(layer) for layer in block.generation] == [nn.Conv2d, nn.ReLU, nn.Conv2d]
            assert all(layer.kernel_size == (3, 3) and layer.bias is not None for layer in block.generation if isinstance(layer, nn.Conv2d))
        vector, values = wrapper(example())
        assert counts == {"teacher_neck": 1, "teacher_head": 0, "student_head": 1}
        assert torch.isfinite(vector).all() and float(values["mgd_weighted_loss"]) > 0
        assert not wrapper._teacher_feats and not wrapper._student_feats
    finally:
        for handle in handles:
            handle.remove()
        wrapper._remove_feature_hooks()


@pytest.mark.parametrize("batch", [2, 1])
def test_level_sum_and_native_batch_factor_are_applied_exactly_once(environment, batch):
    wrapper = model(environment)
    try:
        vector, values = wrapper(example(batch))
        raw = sum(values["mgd_" + level] for level in ("P3", "P4", "P5"))
        torch.testing.assert_close(values["mgd_loss"], raw)
        torch.testing.assert_close(values["mgd_weighted_loss"], raw * environment["MGD_ALPHA"])
        torch.testing.assert_close(vector[-1], batch * values["mgd_weighted_loss"])
        torch.testing.assert_close(vector.sum().detach(), batch * values["total_loss"])
        assert wrapper.last_actual_batch == batch
    finally:
        wrapper._remove_feature_hooks()


def test_kd_only_backward_teacher_bn_optimizer_and_native_criterion_schedule(environment):
    wrapper = model(environment)
    try:
        before = digest(wrapper.teacher_model)
        vector, _ = wrapper(example())
        vector[-1].backward()
        assert sum(p.grad.abs().sum().item() for p in wrapper.student_model.parameters() if p.grad is not None) > 0
        for block in wrapper.projector:
            assert sum(p.grad.abs().sum().item() for p in block.align.parameters() if p.grad is not None) > 0
            assert sum(p.grad.abs().sum().item() for p in block.generation.parameters() if p.grad is not None) > 0
        assert all(p.grad is None and not p.requires_grad for p in wrapper.teacher_model.parameters())
        trainer = environment["MGDTrainer"].__new__(environment["MGDTrainer"])
        optimizer = trainer.build_optimizer(wrapper, name="AdamW", lr=0.001, momentum=0.937, decay=0.0005)
        environment["audit_optimizer"](wrapper, optimizer)
        actual = [id(p) for group in optimizer.param_groups for p in group["params"]]
        assert len(actual) == len(set(actual))
        assert set(actual) == {id(p) for p in wrapper.parameters() if p.requires_grad}
        optimizer.step()
        assert digest(wrapper.teacher_model) == before and not wrapper.teacher_model.training
        criterion = wrapper.criterion
        updates = criterion.updates
        criterion.update()
        assert criterion.updates == updates + 1 and criterion is wrapper.student_model.criterion
    finally:
        wrapper._remove_feature_hooks()


def test_detection_matches_native_and_validation_consumes_no_mask_rng(environment, monkeypatch):
    wrapper = model(environment)
    baseline = copy.deepcopy(wrapper.student_model)
    for layer in baseline.modules():
        for identifier, hook in list(layer._forward_hooks.items()):
            if isinstance(hook, FeatureHook):
                del layer._forward_hooks[identifier]
    try:
        batch = example()
        expected, _ = baseline.loss(batch)
        actual, _ = wrapper(batch)
        torch.testing.assert_close(actual[:-1], expected.reshape(-1), atol=1e-6, rtol=1e-5)
        wrapper.eval()
        for block in wrapper.projector:
            monkeypatch.setattr(block, "forward", lambda *_a, **_k: pytest.fail("Validation ran MGD"))
        monkeypatch.setitem(environment, "teacher_neck_forward", lambda *_a, **_k: pytest.fail("Validation ran teacher"))
        before = torch.random.get_rng_state().clone()
        with torch.no_grad():
            preds = wrapper(batch["img"])
            vector, values = wrapper.loss(batch, preds)
        assert vector[-1] == 0 and values["mgd_loss"] == 0
        assert torch.equal(torch.random.get_rng_state(), before)
    finally:
        wrapper._remove_feature_hooks()


def test_loss_exception_clears_feature_caches(environment, monkeypatch):
    wrapper = model(environment)
    def broken(*_args, **_kwargs):
        raise RuntimeError("injected MGD failure")
    monkeypatch.setattr(wrapper.projector[0], "forward", broken)
    try:
        with pytest.raises(RuntimeError, match="injected MGD failure"):
            wrapper(example())
        assert wrapper._teacher_feats == wrapper._student_feats == {}
    finally:
        wrapper._remove_feature_hooks()
    assert not wrapper._teacher_hooks and not wrapper._student_hooks


@pytest.mark.parametrize("filename", ["10_native_yolo26n_teacher_yolo26s.ipynb", "11_crosskd_yolo26n_teacher_yolo26s.ipynb"])
def test_effective_protocol_matches_native_and_crosskd(environment, filename):
    reference = json.loads(notebook_path("kaggle", filename).read_bytes())
    env = {"Path": Path, "Any": Any, "DEFAULT_CFG_DICT": DEFAULT_CFG_DICT,
           "SESSION_DIR": Path("/reference/run"), "TEACHER_CHECKPOINT": environment["TEACHER_CHECKPOINT"]}
    config = next("".join(c["source"]) for c in reference["cells"] if c["cell_type"] == "code" and "EXPERIMENT_ID =" in "".join(c["source"]))
    exec(compile(config, "<reference-configuration>", "exec"), env)
    env["SEED"] = 42
    for c in reference["cells"]:
        if c["cell_type"] != "code":
            continue
        selected = []
        for node in ast.parse("".join(c["source"])).body:
            if isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id == "KD_TRAIN_ARGS" for t in node.targets):
                selected.append(node)
            if isinstance(node, ast.FunctionDef) and node.name == "protocol_args":
                selected.append(node)
        if selected:
            exec(compile(ast.Module(body=selected, type_ignores=[]), "<reference-effective-args>", "exec"), env)
    env["KD_TRAIN_ARGS"]["distill_model"] = environment["TEACHER_CHECKPOINT"]
    old = vars(get_cfg(overrides=env["protocol_args"]("training")))
    new_args = environment["protocol_args"](Path("/mgd/run"))
    new_args.setdefault("model", environment["MODEL_WEIGHTS"])
    new = vars(get_cfg(overrides=new_args))
    assert {key for key in old if old[key] != new[key]} <= {"name", "project"}
    assert environment["TEACHER_ARTIFACT"] == env["TEACHER_ARTIFACT"]
    assert BaseTrainer.optimizer_step is environment["DetectionTrainer"].optimizer_step


def test_resume_is_not_silently_accepted(environment):
    with pytest.raises(ValueError, match="RESUME_NOT_VERIFIED"):
        environment["MGDTrainer"](overrides={"resume": True})


def configure_synthetic_cpu(env, workdir, monkeypatch):
    from ultralytics.utils import callbacks

    monkeypatch.setattr(callbacks, "add_integration_callbacks", lambda _instance: None)
    monkeypatch.setitem(settings, "wandb", False)
    dataset = workdir / "cpu_synthetic_not_poltrada"
    for split in ("train", "val"):
        for folder in ("images", "labels"):
            (dataset / split / folder).mkdir(parents=True)
        for index in range(4):
            Image.new("RGB", (64, 64), (40 + index * 10, 80, 150)).save(dataset / split / "images" / f"{index}.png")
            (dataset / split / "labels" / f"{index}.txt").write_text("0 .5 .5 .25 .25\n", encoding="utf-8")
    yaml_path = dataset / "data.yaml"
    yaml_path.write_text(f"path: {dataset.as_posix()}\ntrain: train/images\nval: val/images\nnames: {json.dumps(NAMES)}\n", encoding="utf-8")
    teacher, student = workdir / "synthetic_teacher_s.pt", workdir / "fixed_synthetic_initial_student_n.pt"
    torch.save({"model": network("s"), "train_args": {"task": "detect", "imgsz": 64}}, teacher)
    torch.save({"model": network("n"), "train_args": {"task": "detect", "imgsz": 64}}, student)
    teacher_sha = env["file_sha256"](teacher)
    env.update(DATA=str(yaml_path), MODEL_WEIGHTS=str(student), TEACHER_CHECKPOINT=str(teacher),
               TEACHER_SHA256=teacher_sha, PROJECT=str(workdir / "results"),
               BATCH=2, IMGSZ=64, EPOCHS=1, WORKERS=0, DEVICE="cpu", AMP=False,
               SEEDS_TO_RUN=[42, 123])
    env["require_cuda"] = lambda: None  # Explicit CPU test adapter, not a production change.
    env["download_teacher"] = lambda: (str(teacher), {"best_pt_sha256": teacher_sha, "version": "v62"})
    original_args = env["protocol_args"]
    env["protocol_args"] = lambda *args, **kwargs: {**original_args(*args, **kwargs), "plots": False, "nbs": 2}
    original_val = YOLO.val
    monkeypatch.setattr(YOLO, "val", lambda instance, **kwargs: original_val(instance, **{**kwargs, "plots": False}))
    events = []
    wandb = FakeWandb(events)
    class MGDFakeArtifact(FakeWandb.Artifact):
        def add_file(self, path, name, *, policy=None, skip_cache=None):
            if self.type == "model":
                assert policy == "immutable" and skip_cache is True
            else:
                assert policy is None and skip_cache is None
            super().add_file(path, name)
    wandb.Artifact = MGDFakeArtifact
    original_init = wandb.init

    def init(**kwargs):
        run = original_init(**kwargs)
        assert kwargs["group"] == "T26s_S26n" and kwargs["config"]["teacher"] == "yolo26s"
        assert kwargs["name"] == f"kd_mgd_T26s_S26n_seed{run.seed}"
        def use_artifact(reference, type):
            assert reference == env["TEACHER_ARTIFACT"] and type == "model"
            return types.SimpleNamespace(version="v62")
        run.use_artifact = use_artifact
        original_log_artifact = run.log_artifact
        def log_artifact(artifact, aliases):
            handle = original_log_artifact(artifact, aliases)
            def wait():
                assert all(Path(path).is_file() for path in artifact.files.values())
                result = handle.wait()
                assert all(Path(path).is_file() for path in artifact.files.values())
                return result
            return types.SimpleNamespace(wait=wait)
        run.log_artifact = log_artifact
        return run
    wandb.init = init
    env["wandb"] = wandb
    return wandb, events


@pytest.mark.slow
def test_actual_cpu_two_independent_epochs_checkpoint_reload_and_state(environment, workdir, monkeypatch):
    env = environment
    wandb, events = configure_synthetic_cpu(env, workdir, monkeypatch)
    env["EPOCHS"] = 2
    initial_students = []
    original_yolo = env["YOLO"]
    def create_yolo(path):
        instance = original_yolo(path)
        if str(path) == env["MODEL_WEIGHTS"]:
            initial_students.append(digest(instance.model))
        return instance
    env["YOLO"] = create_yolo
    results = env["run_seed_queue"]()
    assert [result["seed"] for result in results] == [42, 123]
    assert len(initial_students) == 2 and initial_students[0] == initial_students[1]
    assert wandb.run is None and len({run.id for run in wandb.runs}) == 2
    assert [run.seed for run in wandb.runs] == [42, 123]
    assert events.index(("finish", 42, 0)) < next(i for i, event in enumerate(events) if event[0] == "init" and event[1] == 123)
    for result, run in zip(results, wandb.runs):
        assert result["status"] == "PASS" and result["completed_epochs"] == result["requested_epochs"] == 2
        assert result["teacher_parameters_and_bn_unchanged"] and result["optimizer_steps"] >= 1
        assert [entry["artifact"].type for entry in run.uploads] == ["model", "model", "results"]
        assert all(entry["waited"] for entry in run.uploads)
        files = run.uploads[1]["artifact"].files
        assert not Path(run.uploads[0]["artifact"].files["best.pt"]).parent.exists()
        assert [p.name for p in (Path(result["seed_directory"]) / "epoch_snapshots").iterdir()] == ["epoch-002"]
        assert {"best.pt", "last.pt", "training_state_last.pt", "training_state_best.pt", "results.csv", "args.yaml",
                "teacher_provenance.json", "checkpoint_manifest.json"} <= set(files)
        archive = torch.load(files["training_state_last.pt"], map_location="cpu", weights_only=False)
        assert archive["resume_status"] == "RESUME_NOT_VERIFIED" and archive["projector"]
        assert archive["epoch_zero_based"] == 1
        assert any("generation.0.weight" in name for name in archive["projector"])
        assert archive["optimizer"]["state"] and archive["scheduler"]
        assert {"student", "ema_student", "ema_projector", "scaler", "rng_python", "rng_numpy", "rng_torch",
                "rng_cuda", "criterion_updates", "loader_generator_state"} <= set(archive)
        assert json.loads((Path(files["best.pt"]).parent / "upload_status.json").read_bytes())["status"] == "PASS"
        deployed = original_yolo(files["best.pt"]).model
        assert not isinstance(deployed, DistillationModel) and not hasattr(deployed, "projector")
    best = Path(results[-1]["student_checkpoint"])
    script = '''import sys,torch
from ultralytics import YOLO
m=YOLO(sys.argv[1]).model.eval()
assert not hasattr(m,'teacher_model') and not hasattr(m,'projector')
with torch.no_grad(): result=m(torch.zeros(1,3,64,64))
assert torch.isfinite(result[0]).all()
s=torch.load(sys.argv[2],map_location='cpu',weights_only=False)
assert s['projector'] and s['resume_status']=='RESUME_NOT_VERIFIED'
print('PORTABLE_MGD_STUDENT_AND_ARCHIVE_PASS')'''
    process = subprocess.run([sys.executable, "-I", "-c", script, str(best), str(best.parent / "training_state_last.pt")],
                             cwd=workdir, capture_output=True, text=True, timeout=120, check=False,
                             env={**os.environ, "OMP_NUM_THREADS": "2", "YOLO_CONFIG_DIR": str(workdir / "child-config")})
    assert process.returncode == 0, process.stdout + process.stderr
    assert "PORTABLE_MGD_STUDENT_AND_ARCHIVE_PASS" in process.stdout


@pytest.mark.slow
def test_actual_cpu_upload_failure_preserves_snapshot_and_stops_queue(environment, workdir, monkeypatch):
    env = environment
    wandb, _events = configure_synthetic_cpu(env, workdir, monkeypatch)
    wandb.fail_stage, wandb.fail_seed = "epoch_upload", 42
    with pytest.raises(RuntimeError, match="epoch_upload"):
        env["run_seed_queue"]()
    assert [run.seed for run in wandb.runs] == [42] and wandb.run is None
    snapshot = next(workdir.rglob("epoch_snapshots/epoch-001"))
    assert all((snapshot / name).is_file() for name in ("best.pt", "last.pt", "training_state_last.pt", "training_state_best.pt"))
    assert json.loads((snapshot / "upload_status.json").read_bytes())["status"] == "FAIL"
    progress = json.loads(next(workdir.rglob("multiseed_progress.json")).read_bytes())
    assert progress["status"] == "failed" and progress["failed_seed"] == 42 and progress["completed"] == []


@pytest.mark.parametrize("seeds", [[], [42, 42], [True], [-1], [2**32], [1.5]])
def test_invalid_seed_queue_is_rejected_before_any_run(environment, seeds):
    with pytest.raises(ValueError):
        environment["validate_seed_list"](seeds)


def test_full_dispatch_omits_smoke_and_changes_no_seed_history(environment, workdir):
    env = environment
    env.update(PROJECT=str(workdir), SEEDS_TO_RUN=[42, 123], require_cuda=lambda: None,
               wandb=types.SimpleNamespace(run=None))
    calls = []
    def fake_seed(seed, queue_dir, diagnostic=False):
        calls.append((seed, diagnostic))
        return {"seed": seed, "wandb_run_id": str(seed), "seed_directory": str(Path(queue_dir) / f"seed-{seed}")}
    env["run_one_seed"] = fake_seed
    result = env["run_seed_queue"]()
    assert calls == [(42, False), (123, False)] and [r["seed"] for r in result] == [42, 123]


def test_fused_adamw_amp_skip_is_not_counted_as_a_real_gradient_update(environment, workdir):
    env = environment
    wrapper = model(env)
    parameters = [p for p in wrapper.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(parameters, lr=0.001, fused=True)
    scaler = torch.amp.GradScaler("cpu", init_scale=128.0)
    data = workdir / "dataset.yaml"
    data.write_text("synthetic protocol fingerprint only", encoding="utf-8")
    env.update(BATCH=2, IMGSZ=64, AMP=True, DATA=str(data))
    context = {"run": types.SimpleNamespace(config={}), "directory": workdir,
               "diagnostic": False, "batches": 0, "best_epoch": None,
               "uploaded_epochs": set(), "logged_epochs": set()}
    callbacks = env["make_callbacks"](context)
    trainer = types.SimpleNamespace(model=wrapper, optimizer=optimizer, batch_size=2,
                                    args=types.SimpleNamespace(imgsz=64), amp=True, device=torch.device("cpu"))
    try:
        callbacks["on_train_start"](trainer)
        before = next(wrapper.student_model.parameters()).detach().clone()
        overflow = sum(p.sum() for p in parameters) * float("inf")
        scaler.scale(overflow).backward()
        scaler.unscale_(optimizer)
        scaler.step(optimizer)
        scaler.update()
        assert context["amp_skipped_steps"] == 1 and context["optimizer_steps"] == 0 and context["gradient"] is None
        torch.testing.assert_close(next(wrapper.student_model.parameters()).detach(), before)
        optimizer.zero_grad(set_to_none=True)
        finite = sum(p.square().sum() for p in parameters)
        scaler.scale(finite).backward()
        scaler.unscale_(optimizer)
        nn.utils.clip_grad_norm_(parameters, max_norm=10.0)
        scaler.step(optimizer)
        scaler.update()
        assert context["amp_skipped_steps"] == 1 and context["optimizer_steps"] == 1
        assert all(norm > 0 for norm in context["gradient"].values())
        assert not torch.equal(next(wrapper.student_model.parameters()).detach(), before)
    finally:
        if context.get("optimizer_hook") is not None:
            context["optimizer_hook"].remove()
        wrapper._remove_feature_hooks()


@pytest.mark.parametrize("setting,value", [
    ("MGD_ALPHA", 0.0), ("MGD_ALPHA", -1.0), ("MGD_ALPHA", float("inf")), ("MGD_ALPHA", float("nan")),
    ("MGD_MASK_RATIO", 1.0), ("MGD_MASK_RATIO", 1.1), ("MGD_MASK_RATIO", -0.1), ("MGD_MASK_RATIO", float("nan")),
    ("KEEP_LAST_LOCAL_EPOCHS", 0), ("KEEP_LAST_LOCAL_EPOCHS", -1), ("KEEP_LAST_LOCAL_EPOCHS", 1.5),
    ("KEEP_LAST_LOCAL_EPOCHS", True),
])
def test_research_boundaries_rejected_before_run_creation(environment, workdir, setting, value):
    env = environment
    env[setting] = value
    env["PROJECT"] = str(workdir / "not-created")
    env["require_cuda"] = Mock(side_effect=AssertionError("GPU gate should not be reached"))
    env["download_teacher"] = Mock(side_effect=AssertionError("Teacher download should not be reached"))
    env["wandb"] = types.SimpleNamespace(run=None, init=Mock())
    with pytest.raises(ValueError, match="MGD_|KEEP_LAST"):
        env["run_seed_queue"]()
    with pytest.raises(ValueError, match="MGD_|KEEP_LAST"):
        env["run_one_seed"](42, workdir)
    env["require_cuda"].assert_not_called()
    env["download_teacher"].assert_not_called()
    env["wandb"].init.assert_not_called()
    assert not (workdir / "not-created").exists() and not (workdir / "seed-42").exists()


def snapshot_directory(run, epoch, *, status="PASS", confirmed_epoch=None):
    directory = run / "epoch_snapshots" / f"epoch-{epoch:03d}"
    directory.mkdir(parents=True)
    (directory / "best.pt").write_bytes(b"synthetic snapshot")
    if status is not None:
        (directory / "upload_status.json").write_text(json.dumps({"status": status, "epoch": epoch if confirmed_epoch is None else confirmed_epoch}))
    return directory


@pytest.mark.parametrize("keep", [1, 2])
def test_confirmed_uploaded_snapshot_retention_keeps_latest_n(environment, workdir, keep):
    run = workdir / "run"
    snapshots = [snapshot_directory(run, epoch) for epoch in range(1, 5)]
    environment["KEEP_LAST_LOCAL_EPOCHS"] = keep
    environment["prune_uploaded_snapshots"]({"directory": run, "uploaded_epochs": {1, 2, 3, 4}})
    assert [path.exists() for path in snapshots] == [False] * (4 - keep) + [True] * keep


def test_cleanup_preserves_failed_unknown_mismatched_and_unconfirmed_snapshots(environment, workdir):
    run = workdir / "run"
    confirmed = snapshot_directory(run, 1)
    failed = snapshot_directory(run, 2, status="FAIL")
    unknown = snapshot_directory(run, 3, status=None)
    mismatch = snapshot_directory(run, 4, confirmed_epoch=999)
    latest = snapshot_directory(run, 5)
    not_uploaded = snapshot_directory(run, 6)
    environment["prune_uploaded_snapshots"]({"directory": run, "uploaded_epochs": {1, 2, 3, 4, 5}})
    assert not confirmed.exists()
    assert all(path.is_dir() for path in (failed, unknown, mismatch, latest, not_uploaded))


def test_cleanup_rejects_resolved_paths_outside_run_before_deletion(environment, workdir, monkeypatch):
    run = workdir / "run"
    escaped = snapshot_directory(run, 1)
    snapshot_directory(run, 2)
    outside = workdir / "outside-run"
    outside.mkdir()
    sentinel = outside / "preserve.txt"
    sentinel.write_text("never delete outside the run", encoding="utf-8")
    original_resolve = Path.resolve
    def resolve(path, *args, **kwargs):
        return outside if path == escaped else original_resolve(path, *args, **kwargs)
    monkeypatch.setattr(Path, "resolve", resolve)
    remove = Mock()
    environment["shutil"] = types.SimpleNamespace(rmtree=remove)
    with pytest.raises(RuntimeError, match="path keluar"):
        environment["prune_uploaded_snapshots"]({"directory": run, "uploaded_epochs": {1, 2}})
    remove.assert_not_called()
    assert sentinel.is_file() and escaped.is_dir()


def test_best_state_is_carried_forward_after_its_original_snapshot_was_pruned(environment, workdir):
    env = environment
    native = workdir / "native-training"
    native.mkdir()
    for name in ("last.pt", "best.pt", "results.csv", "args.yaml"):
        (native / name).write_bytes(b"synthetic native checkpoint")
    def export(source, destination):
        destination.write_bytes(source.read_bytes())
        return env["file_sha256"](destination)
    env["export_student"] = export
    env["training_state"] = lambda trainer: {"best_epoch_source": trainer.epoch + 1, "weights": torch.tensor([trainer.epoch + 1])}
    trainer = types.SimpleNamespace(epoch=0, fitness=1.0, best_fitness=1.0,
                                    last=native / "last.pt", best=native / "best.pt", save_dir=native)
    run = workdir / "retained-run"
    context = {"directory": run, "best_epoch": None, "uploaded_epochs": set()}
    for epoch in range(1, 4):
        trainer.epoch = epoch - 1
        trainer.fitness = 1.0 if epoch == 1 else 0.5
        record = env["snapshot_epoch"](trainer, context)
        (record["directory"] / "upload_status.json").write_text(json.dumps({"status": "PASS", "epoch": epoch}))
        context["uploaded_epochs"].add(epoch)
        env["prune_uploaded_snapshots"](context)
        best = torch.load(record["directory"] / "training_state_best.pt", weights_only=False)
        assert best["best_epoch_source"] == 1 and best["weights"].item() == 1
    assert context["best_epoch"] == 1
    assert context["best_state_path"].parent.name == "epoch-003"
    assert [path.name for path in (run / "epoch_snapshots").iterdir()] == ["epoch-003"]
