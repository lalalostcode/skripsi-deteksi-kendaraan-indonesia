"""Serial lifecycle checks with mocked W&B and short synthetic CPU training only."""
# ruff: noqa: S102 -- Execute trusted local notebook definitions, never setup/training cells.
from __future__ import annotations

import gc
import json
import os
import sys
import time
import types
import uuid
import weakref
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path

import pytest
import torch
from PIL import Image
from test_teacher_s_notebooks import code, cpu_environment, functions_only, namespace
from ultralytics import YOLO, settings
from ultralytics.nn.distill_model import DistillationModel
from ultralytics.nn.tasks import DetectionModel
from ultralytics.utils.torch_utils import get_flops, init_seeds

ROOT = Path(__file__).resolve().parents[1]
FOLDER = ROOT / "notebooks/yolo26-kd-indotraffic/notebooks"
FILES = ["03_baseline_yolo26n.ipynb", "10_native_yolo26n_teacher_yolo26s.ipynb",
         "11_crosskd_yolo26n_teacher_yolo26s.ipynb"]


@pytest.fixture(params=FILES, ids=lambda name: name[:2])
def notebook(request):
    return json.loads((FOLDER / request.param).read_bytes())


@pytest.fixture
def workdir():
    directory = ROOT / "outputs/test_serial_multiseed" / uuid.uuid4().hex
    directory.mkdir(parents=True)
    return directory


def load_environment(notebook, workdir):
    if notebook["cells"][0]["id"].startswith("s2n-"):
        env = namespace(notebook, workdir)
        exec(compile(code(notebook, "evaluation-functions"), "<evaluation>", "exec"), env)
    else:
        module = types.ModuleType("baseline_multiseed_" + uuid.uuid4().hex)
        sys.modules[module.__name__] = module
        env = module.__dict__
        env.update(Path=Path, json=json, os=os, time=time, gc=gc, datetime=datetime, timezone=timezone,
                   contextmanager=contextmanager, torch=torch, init_seeds=init_seeds, YOLO=YOLO)
        for index in (2, 11):
            exec(compile("".join(notebook["cells"][index]["source"]), "<baseline declarations>", "exec"), env)
        for index in (8, 13, 15):
            exec(compile(functions_only("".join(notebook["cells"][index]["source"])), "<baseline functions>", "exec"), env)
    env.update(PROJECT=str(workdir / "results"), ENVIRONMENT_INFO={"diagnostic": "CPU/mock only"}, get_flops=get_flops)
    return env


class FakeArtifact:
    def __init__(self, name=None, type=None, metadata=None, **kwargs):
        self.name, self.type, self.metadata = name, type, metadata or {}
        self.files = {}

    def add_file(self, path, name):
        assert Path(path).is_file(), path
        self.files[name] = str(path)


class FakeWandb:
    Artifact = FakeArtifact
    Settings = types.SimpleNamespace
    Image = staticmethod(str)

    def __init__(self, events, *, fail_stage=None, fail_seed=2024):
        self.run, self.events, self.runs = None, events, []
        self.fail_stage, self.fail_seed = fail_stage, fail_seed
        self.util = types.SimpleNamespace(generate_id=lambda: uuid.uuid4().hex[:8])

    def init(self, **kwargs):
        assert self.run is None, "Previous W&B run was not closed"
        assert kwargs["resume"] == "never"
        seed = kwargs["config"]["seed"]
        run = FakeRun(self, kwargs)
        self.run = run
        self.runs.append(run)
        self.events.append(("init", seed, run.id))
        return run


class FakeRun:
    def __init__(self, wandb, kwargs):
        self.wandb, self.kwargs = wandb, kwargs
        self.id, self.seed = kwargs["id"], kwargs["config"]["seed"]
        self.url, self.summary = "https://mock.invalid/" + self.id, {}
        self.config = dict(kwargs["config"])
        self.logged, self.uploads, self.finished = [], [], False

    def log(self, payload, step=None):
        assert not self.finished
        self.logged.append((dict(payload), step))
        self.wandb.events.append(("log", self.seed, step))

    def log_artifact(self, artifact, aliases):
        assert not self.finished
        entry = {"artifact": artifact, "aliases": aliases, "waited": False}
        self.uploads.append(entry)
        self.wandb.events.append(("upload", self.seed, artifact.type))

        def wait():
            stage = "epoch_upload" if artifact.type == "model" else "final_upload"
            if self.seed == self.wandb.fail_seed and self.wandb.fail_stage == stage:
                raise RuntimeError("simulated " + stage)
            entry["waited"] = True
            self.wandb.events.append(("wait", self.seed, artifact.type))

        return types.SimpleNamespace(wait=wait)

    def finish(self, exit_code=0):
        self.wandb.events.append(("finish", self.seed, exit_code))
        if self.seed == self.wandb.fail_seed and self.wandb.fail_stage == "finish":
            raise RuntimeError("simulated finish")
        if exit_code == 0:
            assert all(item["waited"] for item in self.uploads)
        self.finished = True
        self.wandb.run = None


def fake_training(env, events):
    def train():
        seed = env["SEED"]
        assert env["wb_state"]["uploaded_epochs"] == set()
        assert env["wb_state"]["logged_epochs"] == set()
        assert env["wb_state"]["best_epoch"] is None
        events.append(("new_student", seed, object()))
        directory = env["SESSION_DIR"] / "training"
        directory.mkdir()
        best = directory / "best.pt"
        best.write_bytes(f"student seed {seed}".encode())
        artifact = FakeArtifact(name=f"study-{env['wb_run'].id}-checkpoints", type="model", metadata={"seed": seed})
        artifact.add_file(str(best), "best.pt")
        env["wb_run"].log({"epoch": 1, "train/dis_loss": 0.1}, step=1)
        env["wb_run"].log_artifact(artifact, aliases=["epoch-001"]).wait()
        env["wb_state"]["uploaded_epochs"].add(1)
        env["wb_state"]["logged_epochs"].add(1)
        env["wb_state"]["best_epoch"] = 1
        return {"run_dir": str(directory), "best_pt": str(best), "best_epoch": 1}
    return train


def fake_evaluation(env):
    def evaluate(training):
        artifact = FakeArtifact(name=f"study-{env['wb_run'].id}-results", type="results", metadata={"seed": env["SEED"]})
        env["wb_run"].log_artifact(artifact, aliases=["final"]).wait()
        return {**training, "map50_95": 0.1 + env["SEED"] / 100000}
    return evaluate


def install_fake_teacher(env):
    def prepare():
        if env["METHOD"] != "none":
            env["TEACHER_PROVENANCE"] = {"global_best": {"verified": True}}
    env["prepare_seed_teacher"] = prepare


def test_default_seed_queue_metadata_artifacts_and_finish_are_isolated(notebook, workdir):
    env = load_environment(notebook, workdir)
    assert env["SEEDS_TO_RUN"] == [123, 2024] and 42 not in env["SEEDS_TO_RUN"]
    events, states, callbacks = [], [], []
    fake = env["wandb"] = FakeWandb(events)
    install_fake_teacher(env)
    training = fake_training(env, events)

    def inspect_training():
        states.append(env["wb_state"])
        callbacks.append(env["bind_seed_callback"](lambda trainer: None))
        return training()

    env["run_full_training"], env["evaluate_final_training"] = inspect_training, fake_evaluation(env)
    result = env["run_seed_queue"]()
    assert [item["seed"] for item in result] == [123, 2024]
    assert states[0] is not states[1] and callbacks[0] is not callbacks[1]
    with pytest.raises(RuntimeError, match="Callback"):
        callbacks[0](None)
    assert fake.run is None and all(run.finished for run in fake.runs)
    assert len({run.id for run in fake.runs}) == 2
    assert result[0]["seed_directory"] != result[1]["seed_directory"]
    assert events.index(("finish", 123, 0)) < next(i for i, item in enumerate(events) if item[:2] == ("init", 2024))
    for seed, run in zip([123, 2024], fake.runs):
        assert f"seed{seed}" in run.kwargs["tags"]
        assert f"seed{2024 if seed == 123 else 123}" not in run.kwargs["tags"]
        assert run.kwargs["name"].endswith(f"seed{seed}")
        assert run.kwargs["group"] == ("supervised" if env["METHOD"] == "none" else "T26s_S26n")
        assert [step for _data, step in run.logged] == [1]
        assert all(upload["artifact"].metadata["seed"] == seed for upload in run.uploads)
    progress_file = next(Path(env["PROJECT"]).rglob("multiseed_progress.json"))
    progress = json.loads(progress_file.read_bytes())
    assert progress["status"] == "completed" and progress["failed_seed"] is None
    assert [row["seed"] for row in progress["completed"]] == [123, 2024]


@pytest.mark.parametrize("stage", ["training", "epoch_upload", "evaluation", "final_upload", "finish"])
def test_failures_stop_queue_close_run_and_preserve_previous_seed(notebook, workdir, stage):
    env = load_environment(notebook, workdir)
    env["SEEDS_TO_RUN"] = [123, 2024, 7]
    events = []
    fake = env["wandb"] = FakeWandb(events, fail_stage=stage)
    install_fake_teacher(env)
    train, evaluate = fake_training(env, events), fake_evaluation(env)

    def run_training():
        if env["SEED"] == 2024 and stage == "training":
            raise RuntimeError("simulated training")
        return train()

    def run_evaluation(training):
        if env["SEED"] == 2024 and stage == "evaluation":
            raise RuntimeError("simulated evaluation")
        return evaluate(training)

    env["run_full_training"], env["evaluate_final_training"] = run_training, run_evaluation
    with pytest.raises(RuntimeError, match="simulated"):
        env["run_seed_queue"]()
    assert [run.seed for run in fake.runs] == [123, 2024]
    assert ("finish", 2024, 1) in events
    progress_file = next(Path(env["PROJECT"]).rglob("multiseed_progress.json"))
    progress = json.loads(progress_file.read_bytes())
    assert progress["status"] == "failed" and progress["failed_seed"] == 2024
    assert [row["seed"] for row in progress["completed"]] == [123]
    assert (progress_file.parent / "seed-123/seed_result.json").is_file()
    assert [row["seed"] for row in json.loads((progress_file.parent / "multiseed_results.json").read_bytes())] == [123]


@pytest.mark.parametrize("seeds", [[], [123, 123], [True], [-1], [2**32], ["123"]])
def test_invalid_queue_fails_before_wandb_init(notebook, workdir, seeds):
    env = load_environment(notebook, workdir)
    env["SEEDS_TO_RUN"] = seeds
    fake = env["wandb"] = FakeWandb([])
    with pytest.raises(ValueError):
        env["run_seed_queue"]()
    assert not fake.runs


@pytest.mark.parametrize("seeds", [[123], [7, 3407]])
def test_single_seed_and_final_stage_are_supported(notebook, workdir, seeds):
    env = load_environment(notebook, workdir)
    env["SEEDS_TO_RUN"] = seeds
    events = []
    env["wandb"] = FakeWandb(events)
    install_fake_teacher(env)
    env["run_full_training"], env["evaluate_final_training"] = fake_training(env, events), fake_evaluation(env)
    assert [row["seed"] for row in env["run_seed_queue"]()] == seeds


def test_teacher_failure_closes_run_and_never_trains(notebook, workdir):
    env = load_environment(notebook, workdir)
    if env["METHOD"] == "none":
        pytest.skip("Baseline does not use a teacher")
    events = []
    fake = env["wandb"] = FakeWandb(events)

    def prepare():
        assert env["TEACHER_PROVENANCE"] == {} and env["KD_TRAIN_ARGS"]["distill_model"] is None
        raise ValueError("simulated teacher mismatch")

    env["prepare_seed_teacher"] = prepare
    env["run_full_training"] = lambda: pytest.fail("Training must not start with a failed teacher")
    with pytest.raises(ValueError, match="teacher"):
        env["run_seed_queue"]()
    assert len(fake.runs) == 1 and fake.run is None
    assert ("finish", 123, 1) in events


@pytest.mark.slow
def test_two_serial_cpu_epochs_use_fresh_student_and_independent_wandb(notebook, workdir, monkeypatch):
    """Two independent 64px random-weight synthetic epochs, never Poltrada/Kaggle training."""
    from ultralytics.utils import callbacks

    torch.set_num_threads(2)
    monkeypatch.setattr(callbacks, "add_integration_callbacks", lambda instance: None)
    monkeypatch.setitem(settings, "wandb", False)
    if notebook["cells"][0]["id"].startswith("s2n-"):
        env = cpu_environment(notebook, workdir, monkeypatch)
        exec(compile(code(notebook, "evaluation-functions"), "<cpu evaluation>", "exec"), env)
        teacher_file = Path(env["TEACHER_CHECKPOINT"])

        def prepare_teacher():
            env["TEACHER_CHECKPOINT"], env["teacher_path"] = str(teacher_file), teacher_file
            env["TEACHER_PROVENANCE"] = {"best_pt_sha256": env["file_hash"](teacher_file), "global_best": {"verified": True}}
            env["KD_TRAIN_ARGS"]["distill_model"] = str(teacher_file)
            (env["SESSION_DIR"] / "teacher_provenance.json").write_text("{}")

        env["prepare_seed_teacher"] = prepare_teacher
        env["require_full_training"] = lambda: None  # Only this CPU fixture bypasses the production CUDA gate.
        original_args = env["protocol_args"]
        env["protocol_args"] = lambda *a, **kw: {**original_args(*a, **kw), "nbs": 2, "plots": False}
    else:
        env = load_environment(notebook, workdir)
        dataset = workdir / "synthetic"
        for split in ("train", "val"):
            for child in ("images", "labels"):
                (dataset / split / child).mkdir(parents=True)
            for index in range(2):
                Image.new("RGB", (64, 64), (30 + index, 80, 150)).save(dataset / split / "images" / f"{index}.png")
                (dataset / split / "labels" / f"{index}.txt").write_text("0 .5 .5 .25 .25\n")
        yaml = dataset / "data.yaml"
        yaml.write_text(f"path: {dataset.as_posix()}\ntrain: train/images\nval: val/images\nnames: [a,b,c,d,e,f]\n")
        env.update(DATA=str(yaml), BATCH=2, IMGSZ=64, DEVICE="cpu", WORKERS=0, AMP=False)
    env.update(EPOCHS=1, PROJECT=str(workdir / "serial_runs"), ENVIRONMENT_INFO={"synthetic": True}, get_flops=get_flops)
    # Both seeds load the exact same fixed synthetic pretrained checkpoint.
    pretrained = workdir / "fixed_student.pt"
    fixed = DetectionModel("yolo26n.yaml", nc=6, verbose=False)
    torch.save({"model": fixed, "train_args": {"task": "detect", "imgsz": 64}}, pretrained)
    del fixed
    env["MODEL_WEIGHTS"] = str(pretrained)
    events, constructed, initialized = [], [], []
    fake = env["wandb"] = FakeWandb(events)

    def create_model(weights):
        model = YOLO(weights)
        constructed.append((str(weights), weakref.ref(model)))
        if str(weights) == str(pretrained):
            def inspect_start(trainer):
                network = getattr(trainer.model, "student_model", trainer.model)
                initialized.append((env["SEED"], next(network.parameters()).detach().clone()))
                assert env["wb_state"]["uploaded_epochs"] == set()
            model.add_callback("on_train_start", inspect_start)
        return model

    env["YOLO"] = create_model
    for name in ("reset_peak_memory_stats", "max_memory_allocated", "max_memory_reserved"):
        monkeypatch.setattr(torch.cuda, name, lambda *a, **kw: 0)
    original_val = YOLO.val
    monkeypatch.setattr(YOLO, "val", lambda model, **kw: original_val(model, **{**kw, "plots": False}))
    result = env["run_seed_queue"]()
    assert [row["seed"] for row in result] == [123, 2024]
    assert [seed for seed, _weights in initialized] == [123, 2024]
    torch.testing.assert_close(initialized[0][1], initialized[1][1], rtol=0, atol=0)
    assert sum(path == str(pretrained) for path, _ref in constructed) == 2
    gc.collect()
    assert all(ref() is None for _path, ref in constructed)
    for run, row in zip(fake.runs, result):
        epochs = [step for _payload, step in run.logged if step is not None]
        assert epochs == [1]
        assert [item["artifact"].type for item in run.uploads] == ["model", "results"]
        assert all(item["artifact"].metadata["seed"] == run.seed for item in run.uploads)
        assert all(item["waited"] for item in run.uploads)
        assert Path(row["best_pt"]).is_file()
        deployed = YOLO(row["best_pt"]).model
        assert not isinstance(deployed, DistillationModel) and deployed.yaml["scale"] == "n"
    assert fake.run is None
