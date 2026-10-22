"""Offline checks for new teacher-s notebooks; never authenticate W&B or run final training."""
# Execute trusted local notebook declarations with isolated models and mocked W&B.
# ruff: noqa: S102
from __future__ import annotations

import ast
import base64
import gc
import hashlib
import json
import math
import os
import subprocess
import sys
import time
import types
import uuid
from collections.abc import Iterator
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from unittest.mock import Mock

import nbformat
import pytest
import torch
from torch import nn
from torch.nn import functional as F
from ultralytics import YOLO
from ultralytics.cfg import DEFAULT_CFG_DICT, get_cfg
from ultralytics.models.yolo.detect import DetectionTrainer
from ultralytics.nn.distill_model import DistillationModel, FeatureHook
from ultralytics.nn.tasks import DetectionModel
from ultralytics.utils.torch_utils import init_seeds, strip_optimizer, unwrap_model

from scripts.notebook_paths import notebook_path

ROOT = Path(__file__).resolve().parents[1]
NOTEBOOKS = ROOT / "notebooks/kaggle"
NAMES = ["bus", "car", "motorcycle", "truck", "pickup", "minibus"]
REFERENCE = "ilhamrafiqin-brawijaya-university/yolom26-kd-indotraffic/02_baseline_yolo26s-q5kp9k9d-checkpoints:v62"
torch.set_num_threads(2)
NOTEBOOK_IMPORTS = {
    "gc": gc, "math": math, "time": time, "contextmanager": contextmanager,
    "datetime": datetime, "timezone": timezone,
    "Any": Any, "Iterator": Iterator, "nn": nn, "F": F,
    "DEFAULT_CFG_DICT": DEFAULT_CFG_DICT, "DetectionTrainer": DetectionTrainer,
    "FeatureHook": FeatureHook, "strip_optimizer": strip_optimizer, "unwrap_model": unwrap_model,
}


@pytest.fixture(params=["10_native_yolo26n_teacher_yolo26s.ipynb", "11_crosskd_yolo26n_teacher_yolo26s.ipynb"])
def notebook(request):
    return json.loads((notebook_path("kaggle", request.param)).read_bytes())


@pytest.fixture
def workdir():
    directory = ROOT / "outputs/test_teacher_s" / uuid.uuid4().hex
    directory.mkdir(parents=True)
    return directory


def code(notebook, suffix):
    return "".join(next(cell for cell in notebook["cells"] if cell["id"].endswith("-" + suffix))["source"])


def functions_only(source):
    return ast.Module(body=[node for node in ast.parse(source).body
                            if isinstance(node, (ast.FunctionDef, ast.ClassDef))], type_ignores=[])


def namespace(notebook, workdir):
    module = types.ModuleType("teacher_s_test_" + uuid.uuid4().hex)
    sys.modules[module.__name__] = module
    env = module.__dict__
    env.update({key: value for key, value in globals().items() if not key.startswith("__")})
    env.update(NOTEBOOK_IMPORTS)
    exec(compile(code(notebook, "config"), "<config>", "exec"), env)
    exec(compile(functions_only(code(notebook, "dataset")), "<labels>", "exec"), env)
    env.update(SESSION_DIR=workdir, ENVIRONMENT_INFO={"gpu": "CPU-SYNTHETIC-NOT-KAGGLE"},
               data_info={"nc": len(NAMES), "names": NAMES}, DATASET_NAMES=NAMES,
               TEACHER_PROVENANCE={"best_pt_sha256": "mock-sha", "global_best": {"verified": True}},
               TEACHER_CHECKPOINT=str(workdir / "best.pt"), teacher_path=workdir / "best.pt")
    exec(compile(functions_only(code(notebook, "provenance")), "<provenance-functions>", "exec"), env)
    if env["METHOD"] == "crosskd":
        exec(compile(code(notebook, "cross-common"), "<source-cross-common>", "exec"), env)
        exec(compile(code(notebook, "cross-objective"), "<source-cross-objective>", "exec"), env)
    exec(compile(code(notebook, "guards"), "<protocol-guards>", "exec"), env)
    exec(compile(code(notebook, "trainer"), "<trainer>", "exec"), env)
    exec(compile(code(notebook, "logging"), "<logging>", "exec"), env)
    exec(compile(functions_only(code(notebook, "training-execute")), "<serial-seed-lifecycle>", "exec"), env)
    env["KD_TRAIN_ARGS"]["distill_model"] = env["TEACHER_CHECKPOINT"]
    return env


def test_notebook_schema_syntax_defaults_and_reference_hashes(notebook):
    nbformat.validate(nbformat.from_dict(notebook))
    env = {}
    exec(compile(code(notebook, "config"), "<config>", "exec"), env)
    expected = {"MODEL_WEIGHTS": "yolo26n.pt", "EPOCHS": 100, "IMGSZ": 640, "BATCH": 16,
                "SEED": 123, "SEEDS_TO_RUN": [123, 2024, 7, 3407], "DEVICE": 0, "WORKERS": 2, "OPTIMIZER": "AdamW", "LR": 0.001,
                "PATIENCE": 100, "AMP": True, "WEIGHT_DECAY": 0.0005, "WARMUP_EPOCHS": 3.0,
                "CLOSE_MOSAIC": 10, "EVAL_SPLIT": "val", "RUN_FULL_TRAINING": True,
                 "WANDB_ENABLED": True, "TEACHER_ARTIFACT": REFERENCE}
    assert {key: env[key] for key in expected} == expected
    if env["METHOD"] == "native":
        assert env["KD_WEIGHT"] == 1.0
        assert env["EXPERIMENT_ID"] == "10_native_yolo26n_teacher_yolo26s_dis1"
        assert env["WANDB_RUN_NAME"] == "native_T26s_S26n_dis1_seed123"
    else:
        assert env["EXPERIMENT_ID"] == "11_crosskd_yolo26n_teacher_yolo26s"
        assert env["WANDB_RUN_NAME"] == "crosskd_T26s_S26n_seed123"
    for name, digest in env["REFERENCE_SHA256"].items():
        assert hashlib.sha256((notebook_path("kaggle", name)).read_bytes()).hexdigest() == digest
    for cell in notebook["cells"]:
        if cell["cell_type"] == "code":
            compile("".join(cell["source"]), cell["id"], "exec")
            assert cell["outputs"] == [] and cell["execution_count"] is None
    assert 'RUN_FULL_TRAINING = False' not in code(notebook, "training-execute")
    assert "RUN_SMOKE_TEST" not in env
    assert not any("smoke" in cell["id"] for cell in notebook["cells"])
    assert "run_full_training()" in code(notebook, "training-execute")
    assert "smoke_report.json" not in code(notebook, "evaluation-functions")
    assert "wb_run.use_artifact(TEACHER_ARTIFACT" in code(notebook, "provenance")


def test_crosskd_objective_matches_original_exactly():
    original = json.loads((notebook_path("kaggle", "07_crosskd_yolo26n.ipynb")).read_bytes())
    new = json.loads((notebook_path("kaggle", "11_crosskd_yolo26n_teacher_yolo26s.ipynb")).read_bytes())
    original_method = next("".join(cell["source"]) for cell in original["cells"]
                           if cell["cell_type"] == "code" and "class CrossKDModel" in "".join(cell["source"]))
    old_definitions = ast.parse(original_method).body
    new_definitions = ast.parse(code(new, "cross-objective")).body
    old = [node for node in old_definitions if not isinstance(node, ast.ClassDef) or node.name != "KDTrainer"]
    old = [node for node in old if not isinstance(node, ast.Assign)]
    assert [ast.dump(node) for node in old] == [ast.dump(node) for node in new_definitions]


@pytest.mark.parametrize("filename", [
    "04_native_yolo26s.ipynb", "05_native_yolo26n.ipynb",
    "06_crosskd_yolo26s.ipynb", "07_crosskd_yolo26n.ipynb",
    "08_csakd_yolo26s.ipynb", "09_csakd_yolo26n.ipynb",
    "10_native_yolo26n_teacher_yolo26s.ipynb", "11_crosskd_yolo26n_teacher_yolo26s.ipynb",
])
def test_all_kd_notebooks_forward_dis_one(filename):
    notebook = json.loads((notebook_path("kaggle", filename)).read_bytes())
    codes = ["".join(cell["source"]) for cell in notebook["cells"] if cell["cell_type"] == "code"]
    env = {"Path": Path, "teacher_path": Path("/mock/teacher/best.pt")}
    configuration = next(source for source in codes if "EXPERIMENT_ID =" in source)
    exec(compile(configuration, "<kd-config>", "exec"), env)
    if filename.startswith(("04_", "05_", "10_")):
        assert env["EXPERIMENT_ID"].endswith("_dis1")
    else:
        assert not env["EXPERIMENT_ID"].endswith("_dis1")
    assignment = next(node for source in codes for node in ast.parse(source).body
                      if isinstance(node, ast.Assign) and any(isinstance(target, ast.Name) and target.id == "KD_TRAIN_ARGS" for target in node.targets))
    exec(compile(ast.Module(body=[assignment], type_ignores=[]), "<kd-arguments>", "exec"), env)
    assert env["KD_TRAIN_ARGS"]["dis"] == 1.0


def test_effective_training_config_matches_teacher_m_reference(notebook, workdir):
    """Compare explicit arguments AND all pinned-engine defaults against experiments 05/07."""
    env = namespace(notebook, workdir)
    reference_name = "05_native_yolo26n.ipynb" if env["METHOD"] == "native" else "07_crosskd_yolo26n.ipynb"
    reference = json.loads((notebook_path("kaggle", reference_name)).read_bytes())
    old = {"Path": Path, "teacher_path": Path("/reference/teacher_yolo26m_v50/best.pt"), "RUN_NAME": "reference_run"}
    codes = ["".join(cell["source"]) for cell in reference["cells"] if cell["cell_type"] == "code"]
    configuration = next(source for source in codes if "EXPERIMENT_ID =" in source)
    exec(compile(configuration, "<reference-config>", "exec"), old)
    old["SEED"] = env["SEED"]  # Compare the same scheduled seed against the source protocol.
    for variable in ("KD_TRAIN_ARGS", "train_args"):
        assignment = next(node for source in codes for node in ast.parse(source).body
                          if isinstance(node, ast.Assign) and any(isinstance(target, ast.Name) and target.id == variable for target in node.targets))
        exec(compile(ast.Module(body=[assignment], type_ignores=[]), "<reference-args>", "exec"), old)
    old["train_args"].update(old["KD_TRAIN_ARGS"])
    original = vars(get_cfg(overrides={**old["train_args"], "model": old["MODEL_WEIGHTS"]}))
    current = vars(get_cfg(overrides=env["protocol_args"]("new_teacher_s_run")))
    differences = {key for key in original.keys() | current.keys() if original.get(key) != current.get(key)}
    assert differences == {"project", "name", "distill_model"}
    coefficient_keys = ("KD_WEIGHT",) if env["METHOD"] == "native" else ("KD_CLS_WEIGHT", "KD_BOX_WEIGHT", "KD_CLS_GAMMA")
    assert {key: old[key] for key in coefficient_keys} == {key: env[key] for key in coefficient_keys}


def artifact_fixture(env, workdir):
    # Structure is checked with real s/n tensors in a separate test, not random model mocks.
    env["verify_yolo26s_schema"] = lambda model: None
    payload = b"offline teacher checkpoint fixture; not trained YOLO weights"
    checkpoint = workdir / "best.pt"
    checkpoint.write_bytes(payload)
    digest = base64.b64encode(hashlib.md5(payload).digest()).decode()
    entry = types.SimpleNamespace(size=len(payload), digest=digest)
    teacher = types.SimpleNamespace(
        model=types.SimpleNamespace(yaml={"scale": "s", "yaml_file": "yolo26s.yaml"},
                                    nc=6, names=NAMES),
        ckpt={"epoch": 6, "train_args": {"data": env["DATA"], "model": "yolo26s.pt"}})
    teacher.model.parameters = list
    def freeze_eval():
        teacher.model.training = False
        return types.SimpleNamespace(requires_grad_=lambda value: None)
    teacher.model.eval = freeze_eval
    producer = types.SimpleNamespace(id="q5kp9k9d", state="finished",
        path=[env["WANDB_ENTITY"], env["WANDB_PROJECT"], "q5kp9k9d"],
        config={"dataset": env["DATA"], "model": "yolo26s.pt"}, summary={"best_epoch": 7})
    # v62 deliberately contains epoch 9, best epoch 7. Versions are never inferred as epochs.
    selected = types.SimpleNamespace(version="v62", name="02_baseline_yolo26s-q5kp9k9d-checkpoints:v62",
        type="model", digest="mock-artifact-digest", manifest=types.SimpleNamespace(entries={"best.pt": entry}),
        metadata={"experiment": "02_baseline_yolo26s", "dataset": env["DATA"], "epoch": 9,
                  "best_epoch_so_far": 7, "best_fitness": 0.6},
        logged_by=lambda: producer, download=Mock(return_value=str(workdir)))
    histories = []
    for epoch in range(1, 10):
        histories.append(types.SimpleNamespace(type="model", name=f"02_baseline_yolo26s-q5kp9k9d-checkpoints:v{epoch-1}",
            version=f"v{epoch-1}", metadata={"epoch": epoch, "best_epoch_so_far": min(epoch, 7),
                                          "best_fitness": 0.6 if epoch >= 7 else 0.4},
            manifest=types.SimpleNamespace(entries={"best.pt": entry})))
    producer.logged_artifacts = lambda: iter(histories)
    env["YOLO"] = lambda path: teacher
    return selected, producer, teacher, histories, payload


def test_provenance_validates_hash_source_and_actual_epochs(notebook, workdir):
    env = namespace(notebook, workdir)
    artifact, _, _, _, payload = artifact_fixture(env, workdir)
    path, result = env["verify_teacher_artifact"](artifact)
    assert Path(path).name == "best.pt"
    assert result["best_pt_sha256"] == hashlib.sha256(payload).hexdigest()
    assert result["version"] == "v62" and result["artifact_epoch"] == 9
    assert result["actual_best_epoch"] == 7 and result["global_best"]["verified"]
    artifact.download.assert_called_once_with(root=str(workdir / "teacher_yolo26s_v62"))


@pytest.mark.parametrize("failure", ["version", "producer", "manifest", "checksum", "size", "scale", "family", "names", "dataset", "missing_epoch"])
def test_provenance_mismatches_stop_without_fallback(notebook, workdir, failure):
    env = namespace(notebook, workdir)
    artifact, producer, teacher, _, _ = artifact_fixture(env, workdir)
    if failure == "version": artifact.version = "v63"
    elif failure == "producer": producer.id = "other"
    elif failure == "manifest": artifact.manifest.entries.clear()
    elif failure == "checksum": artifact.manifest.entries["best.pt"].digest = base64.b64encode(b"x" * 16).decode()
    elif failure == "size": artifact.manifest.entries["best.pt"].size += 1
    elif failure == "scale": teacher.model.yaml["scale"] = "m"
    elif failure == "family": teacher.model.yaml["yaml_file"] = "yolo11s.yaml"
    elif failure == "names": teacher.model.names = list(reversed(NAMES))
    elif failure == "dataset": producer.config["dataset"] = "coco8.yaml"
    else: artifact.metadata.pop("epoch")
    with pytest.raises(ValueError):
        env["verify_teacher_artifact"](artifact)


@pytest.mark.parametrize("failure", ["running", "better_later", "different_final_digest", "incomplete_history", "summary_epoch"])
def test_global_best_is_not_assumed_from_version(notebook, workdir, failure):
    env = namespace(notebook, workdir)
    artifact, producer, _, history, _ = artifact_fixture(env, workdir)
    if failure == "running": producer.state = "running"
    elif failure == "better_later": history[-1].metadata["best_fitness"] = 0.8
    elif failure == "different_final_digest": history[-1].manifest = types.SimpleNamespace(entries={"best.pt": types.SimpleNamespace(digest="another")})
    elif failure == "incomplete_history": history.pop(1)
    else: producer.summary["best_epoch"] = 8
    result = env["global_best_evidence"](artifact, producer)
    assert result["verified"] is False


def test_native_run_uses_fresh_id_never_resume_and_progress_is_flushed(workdir):
    notebook = json.loads((notebook_path("kaggle", "10_native_yolo26n_teacher_yolo26s.ipynb")).read_bytes())
    env = namespace(notebook, workdir)
    tree = ast.parse(code(notebook, "training-execute"))
    init = next(node for node in ast.walk(tree) if isinstance(node, ast.Call)
                and isinstance(node.func, ast.Attribute) and isinstance(node.func.value, ast.Name)
                and node.func.value.id == "wandb" and node.func.attr == "init")
    keywords = {entry.arg: entry.value for entry in init.keywords}
    assert isinstance(keywords["id"], ast.Name) and keywords["id"].id == "NEW_WANDB_RUN_ID"
    assert ast.literal_eval(keywords["resume"]) == "never"
    fresh_id = next(node for node in ast.walk(tree) if isinstance(node, ast.Assign)
                    and any(isinstance(target, ast.Name) and target.id == "NEW_WANDB_RUN_ID" for target in node.targets))
    assert isinstance(fresh_id.value, ast.Call) and fresh_id.value.func.attr == "generate_id"
    printed = Mock()
    env["print"] = printed
    env["log_status"]("AUDIT", "metadata checkpoint")
    assert printed.call_args.kwargs["flush"] is True
    assert "[AUDIT] metadata checkpoint" in printed.call_args.args[0]


def test_native_global_best_does_not_fetch_intermediate_manifests(workdir):
    notebook = json.loads((notebook_path("kaggle", "10_native_yolo26n_teacher_yolo26s.ipynb")).read_bytes())
    env = namespace(notebook, workdir)
    selected, producer, _, history, _ = artifact_fixture(env, workdir)
    requests = []
    class LazyArtifact:
        def __init__(self, source, index):
            self.type, self.name, self.version, self.metadata = source.type, source.name, source.version, source.metadata
            self.source, self.index = source, index
        @property
        def manifest(self):
            requests.append(self.index)
            assert self.index == len(history) - 1, "Unnecessary intermediate manifest request"
            return self.source.manifest
    producer.logged_artifacts = lambda: iter(LazyArtifact(item, index) for index, item in enumerate(history))
    evidence = env["global_best_evidence"](selected, producer)
    assert evidence["verified"] is True
    assert requests == [len(history) - 1]


def build_random_wrapper(env):
    init_seeds(42, deterministic=True)
    student = DetectionModel("yolo26n.yaml", nc=6, verbose=False)
    teacher = DetectionModel("yolo26s.yaml", nc=6, verbose=False)
    for model in (student, teacher):
        model.names = dict(enumerate(NAMES))
        model.args = get_cfg(overrides={"imgsz": 64, "dis": env.get("KD_WEIGHT", 1.0)})
    if env["METHOD"] == "native":
        wrapper = DistillationModel(teacher, student)
        assert type(wrapper) is DistillationModel
        assert wrapper.loss.__func__ is DistillationModel.loss
    else:
        wrapper = env["CrossKDModel"](teacher, student)
    return wrapper.train()


def test_real_cpu_tensor_alignment_kd_gradients_optimizer_and_student_export(notebook, workdir):
    env = namespace(notebook, workdir)
    wrapper = build_random_wrapper(env)
    before = next(wrapper.teacher_model.parameters()).detach().clone()
    trainer = env["ResearchTrainer"].__new__(env["ResearchTrainer"])
    optimizer = trainer.build_optimizer(wrapper, name="AdamW", lr=0.001, momentum=0.937, decay=0.0005)
    env["audit_optimizer"](wrapper, optimizer)
    batch = {"img": torch.rand(2, 3, 64, 64), "batch_idx": torch.tensor([0., 1.]),
             "cls": torch.tensor([[0.], [1.]]), "bboxes": torch.tensor([[.5, .5, .25, .25], [.5, .5, .3, .3]])}
    vector, items = wrapper(batch)
    assert torch.isfinite(vector).all() and torch.isfinite(items["dis_loss"]).all()
    # Backpropagate KD alone: detection cannot mask a detached cross-head gradient path.
    vector[-1].sum().backward()
    student_grads = [p.grad for p in wrapper.student_model.parameters() if p.grad is not None]
    projectors = [p.grad for p in wrapper.projector.parameters() if p.grad is not None]
    assert any(g.abs().sum() > 0 for g in student_grads)
    assert any(g.abs().sum() > 0 for g in projectors)
    assert all(torch.isfinite(g).all() for g in student_grads + projectors)
    assert not wrapper.teacher_model.training
    assert all(not p.requires_grad and p.grad is None for p in wrapper.teacher_model.parameters())
    optimizer.step()
    assert torch.equal(before, next(wrapper.teacher_model.parameters()))
    raw, exported = workdir / "wrapper.pt", workdir / "student.pt"
    wrapper._teacher_feats.clear()
    wrapper._student_feats.clear()
    torch.save({"model": wrapper, "ema": None, "epoch": 0, "train_args": vars(wrapper.student_model.args)}, raw)
    env["export_student_checkpoint"](raw, exported)
    reloaded = YOLO(str(exported)).model.eval()
    assert not isinstance(reloaded, DistillationModel) and not hasattr(reloaded, "teacher_model")
    assert sum(p.numel() for p in reloaded.parameters()) == sum(p.numel() for p in wrapper.student_model.parameters())
    assert reloaded.yaml["scale"] == "n"
    with torch.no_grad():
        prediction = reloaded(torch.zeros(1, 3, 64, 64))
    prediction = prediction[0] if isinstance(prediction, tuple) else prediction
    assert torch.isfinite(prediction).all()
    # A fresh interpreter cannot import the notebook's custom CrossKDModel class.
    program = "from ultralytics import YOLO; from ultralytics.nn.distill_model import DistillationModel; import sys; m=YOLO(sys.argv[1]).model; assert not isinstance(m,DistillationModel); print('STUDENT_ONLY_RELOAD_OK')"
    result = subprocess.run([sys.executable, "-c", program, str(exported)], capture_output=True, text=True, check=False,
                            timeout=90, env={**os.environ, "PYTHONUTF8": "1"})
    assert result.returncode == 0, result.stderr
    assert "STUDENT_ONLY_RELOAD_OK" in result.stdout
    wrapper._remove_feature_hooks()


def test_protocol_arguments_and_oom_guard(notebook, workdir):
    env = namespace(notebook, workdir)
    args = env["protocol_args"]("test")
    assert args["batch"] == 16 and args["imgsz"] == 640 and args["amp"] is True
    assert args["nbs"] == 64 and args["distill_model"] == str(workdir / "best.pt")
    fake = types.SimpleNamespace(batch_size=16, args=types.SimpleNamespace(batch=16, imgsz=640), amp=True,
                                 model=types.SimpleNamespace(teacher_model=types.SimpleNamespace(training=False, parameters=list)))
    env["strict_batch_start"](fake)
    assert fake._oom_retries == 3
    fake.batch_size = 8
    with pytest.raises(RuntimeError, match="Batch"):
        env["strict_batch_start"](fake)
    fake.batch_size, fake.amp = 16, False
    with pytest.raises(RuntimeError, match="AMP"):
        env["strict_batch_start"](fake)


def test_direct_training_requires_flag_cuda_and_verified_teacher(notebook, workdir, monkeypatch):
    env = namespace(notebook, workdir)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    with pytest.raises(RuntimeError, match="CUDA"):
        env["require_full_training"]()
    env["RUN_FULL_TRAINING"] = False
    with pytest.raises(RuntimeError, match="False"):
        env["require_full_training"]()
    env["RUN_FULL_TRAINING"] = True
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    env["TEACHER_PROVENANCE"]["global_best"]["verified"] = False
    with pytest.raises(RuntimeError, match="global"):
        env["require_full_training"]()
    env["TEACHER_PROVENANCE"]["global_best"]["verified"] = True
    checkpoint = Path(env["TEACHER_CHECKPOINT"])
    checkpoint.write_bytes(b"fixture teacher")
    env["TEACHER_PROVENANCE"]["best_pt_sha256"] = env["file_hash"](checkpoint)
    env["require_full_training"]()  # Does not require SMOKE_RESULT/fingerprint or GPU smoke history.
    checkpoint.write_bytes(b"changed checkpoint")
    with pytest.raises(RuntimeError, match="SHA256"):
        env["require_full_training"]()


def test_standard_gradscaler_handles_scaled_overflow_without_changing_protocol(notebook, workdir):
    """CPU GradScaler regression: finite loss may overflow after scaling; standard step skips safely."""
    env = namespace(notebook, workdir)
    assert env["ResearchTrainer"].optimizer_step is DetectionTrainer.optimizer_step
    model = nn.Linear(1, 1, bias=False)
    with torch.no_grad():
        model.weight.fill_(1.0)
    trainer = env["ResearchTrainer"].__new__(env["ResearchTrainer"])
    trainer.model, trainer.device, trainer.ema = model, torch.device("cpu"), None
    trainer.optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    trainer.scaler = torch.amp.GradScaler("cpu", init_scale=65536.0)
    original = model.weight.detach().clone()
    loss = model.weight.sum() * 1e34
    assert torch.isfinite(loss)
    trainer.scaler.scale(loss).backward()
    assert not torch.isfinite(model.weight.grad).all()
    trainer.optimizer_step()
    assert torch.equal(original, model.weight)
    assert trainer.scaler.get_scale() == 32768.0
    trainer.scaler.scale(model.weight.sum() * 0.001).backward()
    trainer.optimizer_step()
    assert not torch.equal(original, model.weight)
    assert env["AMP"] is True and env["BATCH"] == 16 and env["IMGSZ"] == 640


def test_artifact_upload_is_synchronous_with_unique_student_snapshots(notebook, workdir):
    env = namespace(notebook, workdir)
    events = []
    artifact = types.SimpleNamespace(add_file=lambda *args, **kwargs: events.append(("file", args[0])))
    def export(source, destination):
        destination.write_bytes(b"student-only fixture")
        events.append(("export", str(destination)))
        return "mock-sha"
    def log(artifact, aliases):
        events.append(("log", aliases))
        return types.SimpleNamespace(wait=lambda: events.append(("wait",)))
    env["export_student_checkpoint"] = export
    env["log_epoch_metrics"] = lambda trainer: events.append(("metrics", trainer.epoch + 1))
    env["wandb"] = types.SimpleNamespace(Artifact=lambda **kwargs: artifact)
    env["wb_run"] = types.SimpleNamespace(id="mock-run", log_artifact=log)
    (workdir / "teacher_provenance.json").write_text("{}")
    trainer = types.SimpleNamespace(epoch=0, epochs=100, fitness=0.6, best_fitness=0.6, save_dir=workdir,
                                    last=workdir / "last.pt", best=workdir / "best.pt")
    env["save_epoch_artifact"](trainer)
    assert events[-1] == ("wait",)
    assert env["wb_state"]["uploaded_epochs"] == {1}
    assert (workdir / "epoch_snapshots/epoch-001/best.pt").is_file()
    trainer.epoch = 1
    env["save_epoch_artifact"](trainer)
    assert (workdir / "epoch_snapshots/epoch-002/best.pt").is_file()
    assert env["wb_state"]["best_epoch"] == 2


def test_native_metrics_are_sent_before_checkpoint_upload_without_duplicate(workdir, monkeypatch):
    notebook = json.loads((notebook_path("kaggle", "10_native_yolo26n_teacher_yolo26s.ipynb")).read_bytes())
    env = namespace(notebook, workdir)
    events = []
    def export(source, destination):
        destination.write_bytes(b"student-only fixture")
        events.append("export")
        return "mock-sha"
    def wait():
        assert env["wb_state"]["logged_epochs"] == {1}
        assert env["wb_state"]["uploaded_epochs"] == set()
        events.append("wait")
    env["export_student_checkpoint"] = export
    env["wandb"] = types.SimpleNamespace(Artifact=lambda **kwargs: types.SimpleNamespace(add_file=lambda *args, **kwargs: None))
    env["wb_run"] = types.SimpleNamespace(id="mock-run", log=lambda *args, **kwargs: events.append("metrics"),
        log_artifact=lambda *args, **kwargs: types.SimpleNamespace(wait=wait))
    monkeypatch.setattr(torch.cuda, "max_memory_allocated", lambda *args: 0)
    (workdir / "teacher_provenance.json").write_text("{}")
    trainer = types.SimpleNamespace(epoch=0, epochs=100, fitness=0.6, best_fitness=0.6,
        save_dir=workdir, last=workdir / "last.pt", best=workdir / "best.pt",
        tloss={"dis_loss": 0.25}, label_loss_items=lambda *args, **kwargs: {"train/dis_loss": 0.25},
        metrics={"metrics/mAP50-95(B)": 0.6}, lr={"lr/pg0": 0.001}, batch_size=16, accumulate=4,
        scaler=types.SimpleNamespace(get_scale=lambda: 32768.0))
    env["save_epoch_artifact"](trainer)
    env["log_epoch_metrics"](trainer)  # Later on_fit_epoch_end must not log the same epoch again.
    assert events == ["metrics", "export", "export", "wait"]
    assert env["wb_state"]["uploaded_epochs"] == {1}


def cpu_environment(notebook, workdir, monkeypatch):
    """Standalone synthetic CPU harness, with no study teacher or V3 substitution in notebooks."""
    from PIL import Image
    from ultralytics.utils import callbacks

    monkeypatch.setattr(callbacks, "add_integration_callbacks", lambda instance: None)
    env = namespace(notebook, workdir)
    dataset = workdir / "cpu_synthetic_not_poltrada"
    for split, count in (("train", 10), ("val", 4)):
        for child in ("images", "labels"):
            (dataset / split / child).mkdir(parents=True)
        for index in range(count):
            Image.new("RGB", (64, 64), (30 + index, 80, 150)).save(dataset / split / "images" / f"{index}.png")
            (dataset / split / "labels" / f"{index}.txt").write_text("0 .5 .5 .25 .25\n")
    yaml = dataset / "data.yaml"
    yaml.write_text(f"path: {dataset.as_posix()}\ntrain: train/images\nval: val/images\nnames: {json.dumps(NAMES)}\n")
    teacher = DetectionModel("yolo26s.yaml", nc=6, verbose=False)
    teacher.names = dict(enumerate(NAMES))
    teacher_path = workdir / "cpu_synthetic_teacher_s.pt"
    torch.save({"model": teacher, "train_args": {"task": "detect", "imgsz": 64}}, teacher_path)
    env.update(DATA=str(yaml), MODEL_WEIGHTS="yolo26n.yaml", BATCH=2, IMGSZ=64,
               AMP=False, DEVICE="cpu", WORKERS=0, PROJECT=str(workdir / "results"),
               TEACHER_CHECKPOINT=str(teacher_path), teacher_path=teacher_path)
    env["KD_TRAIN_ARGS"]["distill_model"] = str(teacher_path)
    env["TEACHER_PROVENANCE"]["best_pt_sha256"] = env["file_hash"](teacher_path)
    # Only this test harness bypasses the runtime CUDA assertion. Study notebook never does.
    class CPUHarness(env["ResearchTrainer"]):
        def _setup_train(self):
            DetectionTrainer._setup_train(self)
            env["audit_optimizer"](unwrap_model(self.model), self.optimizer)

    env["ResearchTrainer"] = CPUHarness
    env["wandb"] = types.SimpleNamespace(Artifact=lambda *args, **kwargs: types.SimpleNamespace(add_file=lambda *args, **kwargs: None))
    env["wb_run"] = types.SimpleNamespace(id="cpu-fixture", summary={}, log=Mock(), config=types.SimpleNamespace(update=lambda *args: None),
        log_artifact=lambda *args, **kwargs: types.SimpleNamespace(wait=lambda: None))
    (workdir / "teacher_provenance.json").write_text("{}")
    return env


def test_actual_yolo26s_schema_accepts_s_and_rejects_forged_n(notebook, workdir):
    env = namespace(notebook, workdir)
    teacher = DetectionModel("yolo26s.yaml", nc=6, verbose=False)
    env["verify_yolo26s_schema"](teacher)
    forged = DetectionModel("yolo26n.yaml", nc=6, verbose=False)
    forged.yaml["scale"] = "s"
    with pytest.raises(ValueError, match="Struktur tensor"):
        env["verify_yolo26s_schema"](forged)


def test_one_epoch_cpu_training_validation_and_epoch_artifacts(notebook, workdir, monkeypatch):
    """Only a 64px random-weight CPU fixture, never a 100-epoch research run."""
    env = cpu_environment(notebook, workdir, monkeypatch)
    env["EPOCHS"] = 1
    original_args = env["protocol_args"]
    env["protocol_args"] = lambda *args, **kwargs: {**original_args(*args, **kwargs), "nbs": 2, "plots": False}
    env["require_full_training"] = lambda: None  # Production CUDA/provenance gate is tested separately.
    monkeypatch.setattr(torch.cuda, "reset_peak_memory_stats", lambda *args: None)
    monkeypatch.setattr(torch.cuda, "max_memory_allocated", lambda *args: 0)
    monkeypatch.setattr(torch.cuda, "max_memory_reserved", lambda *args: 0)
    result = env["run_full_training"]()
    assert result["completed_epochs"] == 1 and result["best_epoch"] == 1
    assert Path(result["best_pt"]).is_file() and Path(result["last_pt"]).is_file()
    assert env["wb_state"]["uploaded_epochs"] == {1}
    assert env["wb_state"]["logged_epochs"] == {1}
    values = env["wb_run"].log.call_args.args[0]
    assert "train/dis_loss" in values and "metrics/mAP50-95(B)" in values
    student = YOLO(result["best_pt"]).model
    assert student.yaml["scale"] == "n" and not isinstance(student, DistillationModel)
