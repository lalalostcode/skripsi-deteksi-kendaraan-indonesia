"""Offline clone, ROCm/source mocks, serial lifecycle and synthetic CPU integration checks."""
# ruff: noqa: S102 -- Execute only trusted local notebook configuration/function declarations.
from __future__ import annotations

import ast
import gc
import hashlib
import io
import json
import os
import platform
import subprocess
import sys
import time
import types
import uuid
import weakref
from contextlib import contextmanager, redirect_stderr, redirect_stdout
from datetime import datetime, timezone
from importlib import metadata
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import MagicMock, Mock

import cv2
import nbformat
import pytest
import torch
from PIL import Image
from test_serial_multiseed_notebooks import FakeWandb, fake_evaluation, fake_training
from ultralytics import YOLO, settings
from ultralytics.cfg import get_cfg
from ultralytics.data.utils import IMG_FORMATS, check_det_dataset
from ultralytics.nn.tasks import DetectionModel
from ultralytics.utils.torch_utils import init_seeds

ROOT = Path(__file__).resolve().parents[1]
FOLDER = ROOT / "notebooks/yolo26-kd-indotraffic/notebooks"
TARGET = FOLDER / "03_baseline_yolo26n_amd_roboflow.ipynb"
ORIGINAL = FOLDER / "03_baseline_yolo26n.ipynb"
NAMES = ["bus", "car", "motorcycle", "truck", "pickup", "minibus"]


def source(key):
    notebook = json.loads(TARGET.read_bytes())
    return "".join(next(cell for cell in notebook["cells"] if cell["id"] == "amd-" + key)["source"])


def definitions(code):
    tree = ast.parse(code)
    tree.body = [node for node in tree.body if isinstance(node, (ast.FunctionDef, ast.ClassDef))]
    return compile(tree, "<amd notebook declarations>", "exec")


@pytest.fixture
def workdir():
    path = ROOT / "outputs/test_amd_baseline" / uuid.uuid4().hex
    path.mkdir(parents=True)
    return path


@pytest.fixture
def env(workdir):
    namespace = {"Path": Path, "gc": gc, "hashlib": hashlib, "io": io, "json": json, "os": os, "platform": platform,
                 "subprocess": subprocess, "sys": sys, "time": time, "metadata": metadata, "TemporaryDirectory": TemporaryDirectory,
                 "contextmanager": contextmanager, "redirect_stdout": redirect_stdout, "redirect_stderr": redirect_stderr,
                 "datetime": datetime, "timezone": timezone, "torch": torch, "YOLO": YOLO, "init_seeds": init_seeds,
                 "check_det_dataset": check_det_dataset, "IMG_FORMATS": IMG_FORMATS}
    exec(compile(source("config"), "<amd config>", "exec"), namespace)
    exec(compile(source("baseline"), "<amd baseline>", "exec"), namespace)
    for key in ("credentials", "environment", "dataset", "status", "guards", "training", "evaluation"):
        exec(definitions(source(key)), namespace)
    namespace.update(WANDB_API_KEY="test-private-wandb-key", ROBOFLOW_API_KEY="test-private-roboflow-key",
                     PROJECT=str(workdir / "results"), DATASET_DIR=workdir / "dataset",
                     ENVIRONMENT_INFO={"backend": "rocm-mock-not-real", "hip": "mock"})
    assert workdir.resolve().is_relative_to(ROOT.resolve())
    # Keep disposable pip-constraint files inside the permitted test workspace on Windows.
    namespace["TemporaryDirectory"] = lambda *a, **kw: TemporaryDirectory(*a, dir=str(workdir), **kw)
    return namespace


def synthetic_dataset(directory, *, relative=True):
    for split in ("train", "valid"):
        for child in ("images", "labels"):
            (directory / split / child).mkdir(parents=True)
        for index in range(2):
            Image.new("RGB", (64, 64), (30 + index, 80, 150)).save(directory / split / "images" / f"{index}.png")
            (directory / split / "labels" / f"{index}.txt").write_text("0 .5 .5 .25 .25\n", encoding="utf-8")
    prefix = "../" if relative else ""
    path = directory / "data.yaml"
    path.write_text(f"train: {prefix}train/images\nval: {prefix}valid/images\nnames: {json.dumps(NAMES)}\n"
                    "roboflow:\n  workspace: ilham-rafiqin\n  project: poltrada-original-raw-2\n  version: 3\n", encoding="utf-8")
    return path


def prepare_mock_queue(env, workdir):
    dataset = synthetic_dataset(workdir / "mock_data")
    env.update(DATA=str(dataset), DATASET_PROVENANCE={"source": "roboflow-mock-not-real", "version": 3,
               "yaml_sha256": hashlib.sha256(dataset.read_bytes()).hexdigest()}, require_rocm_runtime=Mock())
    events = []
    env["wandb"] = FakeWandb(events)
    env["run_full_training"], env["evaluate_final_training"] = fake_training(env, events), fake_evaluation(env)
    return events


def test_json_syntax_empty_credentials_seed_comments_and_baseline_parameters(env):
    notebook = json.loads(TARGET.read_bytes())
    nbformat.validate(nbformat.from_dict(notebook))
    for cell in notebook["cells"]:
        if cell["cell_type"] == "code":
            compile("".join(cell["source"]), cell["id"], "exec")
            assert cell["outputs"] == [] and cell["execution_count"] is None
    original = json.loads(ORIGINAL.read_bytes())
    reference = {}
    exec(compile("".join(original["cells"][2]["source"]), "<baseline config>", "exec"), reference)
    for key in ("MODEL_WEIGHTS", "EPOCHS", "IMGSZ", "BATCH", "DEVICE", "WORKERS", "OPTIMIZER", "LR", "PATIENCE", "AMP",
                "WEIGHT_DECAY", "WARMUP_EPOCHS", "CLOSE_MOSAIC", "EVAL_SPLIT", "WANDB_GROUP", "WANDB_PROJECT", "METHOD"):
        assert env[key] == reference[key], key
    assert env["SEEDS_TO_RUN"] == [7, 3407]
    assert all(f"# {seed}," in source("config") for seed in (42, 123, 2024))
    assert 'WANDB_API_KEY = ""' in source("credentials") and 'ROBOFLOW_API_KEY = ""' in source("credentials")
    complete = "\n".join("".join(cell["source"]) for cell in notebook["cells"] if cell["cell_type"] == "code")
    assert "getpass" not in complete and "kaggle_secrets" not in complete and "/kaggle/" not in complete
    assert 'wandb.login(key=WANDB_API_KEY)' in complete and 'Roboflow(api_key=ROBOFLOW_API_KEY)' in complete
    assert 'os.environ["WANDB_API_KEY"]' not in complete and 'os.environ["ROBOFLOW_API_KEY"]' not in complete
    # Evaluation changes add provenance only; no metric computation or inference protocol changes.
    new_val = next(node for node in ast.walk(ast.parse(source("evaluation"))) if isinstance(node, ast.Call) and ast.unparse(node.func) == "best_model.val")
    old_val = next(node for node in ast.walk(ast.parse("".join(original["cells"][15]["source"]))) if isinstance(node, ast.Call) and ast.unparse(node.func) == "best_model.val")
    assert ast.dump(new_val) == ast.dump(old_val)
    # Compare all effective engine defaults and explicit arguments, not just the configuration variables.
    argument_sets = []
    for text, values in (("".join(original["cells"][13]["source"]), reference), (source("training"), dict(env))):
        values.update(SESSION_DIR=Path("same-output"), DATA=reference["DATA"], SEED=42)
        call = next(node.value for node in ast.walk(ast.parse(text))
                    if isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id == "train_args" for t in node.targets))
        arguments = eval(compile(ast.Expression(call), "<train arguments>", "eval"), values)
        argument_sets.append(vars(get_cfg(overrides=arguments)))
    assert argument_sets[0] == argument_sets[1]


@pytest.mark.parametrize("key", ["WANDB_API_KEY", "ROBOFLOW_API_KEY"])
def test_empty_credentials_stop_before_download_and_wandb(env, workdir, key):
    prepare_mock_queue(env, workdir)
    env[key] = "  "
    env["Roboflow"] = Mock()
    with pytest.raises(ValueError, match=key):
        env["download_roboflow_dataset"]()
    with pytest.raises(ValueError, match=key):
        env["run_seed_queue"]()
    env["Roboflow"].assert_not_called()
    assert env["wandb"].runs == []


def fake_rocm(env):
    cuda = types.SimpleNamespace(is_available=Mock(return_value=True), set_device=Mock(), synchronize=Mock(),
        get_device_properties=Mock(return_value=types.SimpleNamespace(total_memory=16 * 1024**3, gcnArchName="gfx-mock")),
        get_device_name=Mock(return_value="AMD synthetic mock"))
    torch_mock = types.SimpleNamespace(version=types.SimpleNamespace(hip="mock-rocm"), cuda=cuda,
        ones=Mock(return_value=MagicMock()), isfinite=Mock())
    torch_mock.isfinite.return_value.all.return_value.item.return_value = True
    env.update(torch=torch_mock, platform=types.SimpleNamespace(system=lambda: "Linux"))
    return torch_mock


def test_rocm_probe_and_device_metadata(env):
    fake = fake_rocm(env)
    result = env["require_rocm_runtime"]()
    fake.ones.assert_called_once_with((8, 8), device="cuda:0")
    fake.cuda.set_device.assert_called_once_with(0)
    fake.cuda.synchronize.assert_called_once_with(0)
    assert result == {"backend": "rocm", "hip": "mock-rocm", "gpu": "AMD synthetic mock",
                      "gpu_architecture": "gfx-mock", "total_vram_bytes": 16 * 1024**3}
    env["require_training_device"](types.SimpleNamespace(type="cuda", index=0))
    for kind, index in (("cpu", None), ("cuda", 1)):
        with pytest.raises(RuntimeError, match="GPU AMD"):
            env["require_training_device"](types.SimpleNamespace(type=kind, index=index))


@pytest.mark.parametrize("failure", ["linux", "cuda-build", "unavailable", "device", "nonfinite"])
def test_invalid_runtime_rejected_before_software_install_or_run(env, failure):
    fake = fake_rocm(env)
    if failure == "linux": env["platform"].system = lambda: "Windows"
    elif failure == "cuda-build": fake.version.hip = None
    elif failure == "unavailable": fake.cuda.is_available.return_value = False
    elif failure == "device": env["DEVICE"] = 1
    elif failure == "nonfinite": fake.isfinite.return_value.all.return_value.item.return_value = False
    with pytest.raises(RuntimeError):
        env["require_rocm_runtime"]()


@pytest.mark.parametrize("mode", ["install", "already-installed", "stack-change", "restart",
                                  "desktop", "overlap", "cv2-restart", "numpy-restart", "failed-probe"])
def test_package_resolution_freezes_rocm_stack(env, mode):
    versions = {"torch": "2.14.0+rocm.mock", "torchvision": "0.29.0+rocm.mock"}
    wanted = {"ultralytics-opencv-headless": "8.4.155", "opencv-python-headless": "4.10.0.84",
              "wandb": "0.28.1", "roboflow": "1.4.0"}
    if mode in {"already-installed", "desktop", "overlap", "cv2-restart", "failed-probe"}:
        versions.update(wanted)
    if mode in {"desktop", "overlap", "cv2-restart"}:
        versions.update({"ultralytics": "8.4.155", "opencv-python": "5.0.0.93"})
    if mode == "overlap": versions["opencv-contrib-python-headless"] = "old"
    if mode == "numpy-restart": versions["numpy"] = "2.4.0"

    def version(name):
        if name not in versions: raise metadata.PackageNotFoundError(name)
        return versions[name]

    env["metadata"] = types.SimpleNamespace(version=version, PackageNotFoundError=metadata.PackageNotFoundError)
    modules = {"wandb": object()} if mode == "restart" else {"cv2.bootstrap": object()} if mode == "cv2-restart" else {}
    if mode == "numpy-restart": modules["numpy"] = object()
    env["sys"] = types.SimpleNamespace(executable="python", modules=modules)
    calls = []

    def change_packages(command):
        calls.append(command)
        if "uninstall" in command:
            assert command[:5] == ["python", "-m", "pip", "uninstall", "-y"]
            assert "torch" not in command[5:] and "torchvision" not in command[5:]
            for name in command[5:]: versions.pop(name)
            return
        constraints = Path(command[command.index("--constraint") + 1]).read_text(encoding="utf-8")
        assert constraints == "torch==2.14.0+rocm.mock\ntorchvision==0.29.0+rocm.mock\n"
        assert command[:4] == ["python", "-m", "pip", "install"]
        for package in command[6:]:
            name, pin = package.split("==")
            assert wanted[name] == pin and name not in {"torch", "torchvision", "ultralytics", "opencv-python"}
            versions[name] = pin
        if mode == "stack-change": versions["torch"] = "CUDA-build"
        if mode == "numpy-restart": versions["numpy"] = "2.3.5"

    probe = Mock(return_value=types.SimpleNamespace(returncode=1 if mode == "failed-probe" else 0,
        stderr="ImportError: libxcb.so.1", stdout='{"opencv":"4.10.0","opencv_gui":"NONE"}'))
    env["subprocess"] = types.SimpleNamespace(check_call=change_packages, run=probe)
    if mode in {"stack-change", "restart", "cv2-restart", "numpy-restart", "failed-probe"}:
        with pytest.raises(RuntimeError, match="Stack|Restart|headless gagal"):
            env["install_research_packages"]()
    else:
        assert env["install_research_packages"]() == {"opencv": "4.10.0", "opencv_gui": "NONE"}
        assert len(calls) == (0 if mode == "already-installed" else 2 if mode in {"desktop", "overlap"} else 1)
    if mode in {"desktop", "overlap", "cv2-restart"}:
        # Shared namespace providers are removed together, including the previously installed
        # headless wheel, then restored. Re-running does not reinstall desktop dependencies.
        assert calls[0][3] == "uninstall" and calls[1][3] == "install"
        assert {"ultralytics", "ultralytics-opencv-headless", "opencv-python", "opencv-python-headless"} <= set(calls[0])
        assert "ultralytics" not in versions and "opencv-python" not in versions
        assert versions["opencv-python-headless"] == "4.10.0.84"
        if mode in {"desktop", "overlap"}:
            command_count = len(calls)
            env["install_research_packages"]()
            assert len(calls) == command_count  # Rerun must not restore GUI dependencies.


@pytest.mark.parametrize("mode", ["ok", "import-error", "wrong-version", "gui", "bad-image", "bad-output"])
def test_headless_subprocess_probe_rejects_import_gui_and_codec_failures(env, monkeypatch, mode):
    fake_cv2 = types.ModuleType("cv2")
    fake_cv2.__version__ = "5.0.0" if mode == "wrong-version" else "4.10.0"
    fake_cv2.getBuildInformation = lambda: "\n  GUI: QT5\n" if mode == "gui" else "\n  GUI: NONE\n"
    fake_cv2.IMREAD_COLOR = cv2.IMREAD_COLOR
    fake_cv2.imencode = (lambda *a: (False, None)) if mode == "bad-image" else cv2.imencode
    fake_cv2.imdecode, fake_cv2.resize = cv2.imdecode, cv2.resize

    def run(command, **kwargs):
        assert command[1] == "-c" and kwargs == {"capture_output": True, "text": True, "check": False}
        if mode == "import-error":
            return types.SimpleNamespace(returncode=1, stdout="", stderr="ImportError: libxcb.so.1")
        if mode == "bad-output":
            return types.SimpleNamespace(returncode=0, stdout="not-json", stderr="")
        output = io.StringIO()
        monkeypatch.setitem(sys.modules, "cv2", fake_cv2)
        try:
            with redirect_stdout(output):
                exec(compile(command[2], "<headless probe with mocked cv2 build>", "exec"), {})
        except AssertionError as error:
            return types.SimpleNamespace(returncode=1, stdout="", stderr=str(error))
        return types.SimpleNamespace(returncode=0, stdout=output.getvalue(), stderr="")

    env["subprocess"] = types.SimpleNamespace(run=run)
    if mode == "ok":
        assert env["verify_headless_opencv"]() == {"opencv": "4.10.0", "opencv_gui": "NONE"}
    else:
        with pytest.raises(RuntimeError, match="OpenCV headless"):
            env["verify_headless_opencv"]()


def roboflow_mock(env):
    constructor = env["Roboflow"] = Mock()
    download = constructor.return_value.workspace.return_value.project.return_value.version.return_value.download
    return constructor, download


def test_download_exact_source_relative_paths_provenance_and_once_for_queue(env, workdir, monkeypatch):
    import ultralytics.data.utils as data_utils

    monkeypatch.setattr(data_utils, "check_font", lambda *a, **k: None)
    constructor, download = roboflow_mock(env)

    def fetch(*args, **kwargs):
        synthetic_dataset(Path(kwargs["location"]))
        return types.SimpleNamespace(location=kwargs["location"])

    download.side_effect = fetch
    data, info, provenance = env["download_roboflow_dataset"]()
    constructor.assert_called_once_with(api_key=env["ROBOFLOW_API_KEY"])
    constructor.return_value.workspace.assert_called_once_with("ilham-rafiqin")
    constructor.return_value.workspace.return_value.project.assert_called_once_with("poltrada-original-raw-2")
    constructor.return_value.workspace.return_value.project.return_value.version.assert_called_once_with(3)
    download.assert_called_once_with("yolo26", location=str(env["DATASET_DIR"]))
    assert Path(data) == env["DATASET_DIR"] / "data.yaml"
    assert info["train"] == str(env["DATASET_DIR"] / "train/images")
    assert provenance["class_names"] == NAMES and provenance["kaggle_dataset_equivalence"] == "NOT_VERIFIED"
    assert provenance["yaml_sha256"] == hashlib.sha256(Path(data).read_bytes()).hexdigest()
    assert env["ROBOFLOW_API_KEY"] not in json.dumps(provenance)
    env.update(DATA=data, DATASET_PROVENANCE=provenance, require_rocm_runtime=Mock())
    events = []
    env["wandb"] = FakeWandb(events)
    env["run_full_training"], env["evaluate_final_training"] = fake_training(env, events), fake_evaluation(env)
    assert [row["seed"] for row in env["run_seed_queue"]()] == [7, 3407]
    assert download.call_count == 1


@pytest.mark.parametrize("failure", ["partial", "missing-train", "empty-train", "wrong-version", "download", "missing-yaml"])
def test_bad_dataset_fails_without_overwriting_cache_or_leaking_key(env, failure, monkeypatch, capsys):
    import ultralytics.data.utils as data_utils

    monkeypatch.setattr(data_utils, "check_font", lambda *a, **k: None)
    constructor, download = roboflow_mock(env)
    destination = env["DATASET_DIR"]
    if failure == "partial": destination.mkdir()
    elif failure in {"missing-train", "empty-train", "wrong-version"}:
        yaml = synthetic_dataset(destination)
        if failure == "missing-train": yaml.write_text(yaml.read_text().replace("../train/images", "missing/images"))
        elif failure == "empty-train":
            for image in (destination / "train/images").glob("*.png"):
                image.unlink()
        else: yaml.write_text(yaml.read_text().replace("version: 3", "version: 2"))
    else:
        def fetch(*a, **kw):
            print("SDK URL ?key=" + env["ROBOFLOW_API_KEY"])
            if failure == "download": raise ConnectionError(env["ROBOFLOW_API_KEY"])
            destination.mkdir()
            return types.SimpleNamespace(location=str(destination))
        download.side_effect = fetch
    with pytest.raises((RuntimeError, FileNotFoundError, ValueError)) as caught:
        env["download_roboflow_dataset"]()
    assert env["ROBOFLOW_API_KEY"] not in str(caught.value)
    assert env["ROBOFLOW_API_KEY"] not in capsys.readouterr().out
    if failure in {"partial", "missing-train", "empty-train", "wrong-version"}:
        constructor.assert_not_called()


@pytest.mark.parametrize("change", ["batch_size", "batch", "imgsz", "amp"])
def test_strict_training_guards_and_no_oom_reduction(env, change):
    env["require_training_device"] = Mock()
    trainer = types.SimpleNamespace(device="mock", batch_size=16,
        args=types.SimpleNamespace(batch=16, imgsz=640), amp=True)
    env["strict_batch_start"](trainer)
    assert trainer._oom_retries == 3
    env["strict_batch_end"](types.SimpleNamespace(loss=torch.tensor(1.0)))
    if change in {"batch", "imgsz"}: setattr(trainer.args, change, 8)
    else: setattr(trainer, change, False if change == "amp" else 8)
    with pytest.raises(RuntimeError, match="Batch|AMP"):
        env["strict_batch_start"](trainer)
    with pytest.raises(RuntimeError, match="non-finite"):
        env["strict_batch_end"](types.SimpleNamespace(loss=torch.tensor(float("nan"))))


def test_serial_ids_state_callbacks_tags_and_code_upload_are_isolated(env, workdir):
    events = prepare_mock_queue(env, workdir)
    states, callbacks = [], []
    train = env["run_full_training"]

    def inspect():
        states.append(env["wb_state"])
        callbacks.append(env["bind_seed_callback"](lambda t: None))
        return train()

    env["run_full_training"] = inspect
    results = env["run_seed_queue"]()
    runs = env["wandb"].runs
    assert [row["seed"] for row in results] == [7, 3407]
    assert len({row["seed_directory"] for row in results}) == len({run.id for run in runs}) == 2
    assert states[0] is not states[1] and callbacks[0] is not callbacks[1]
    with pytest.raises(RuntimeError, match="Callback"):
        callbacks[0](None)
    assert events.index(("finish", 7, 0)) < next(i for i, event in enumerate(events) if event[:2] == ("init", 3407))
    for run in runs:
        assert run.kwargs["name"] == f"baseline_yolo26n_amd_seed{run.seed}"
        assert run.kwargs["group"] == "supervised"
        assert {"baseline", "amd-rocm", "roboflow-v3", f"seed{run.seed}"} <= set(run.kwargs["tags"])
        assert run.kwargs["save_code"] is False and run.kwargs["settings"].disable_code is True
        assert run.kwargs["config"]["dataset_provenance"] == env["DATASET_PROVENANCE"]
        assert env["WANDB_API_KEY"] not in repr(run.kwargs) and env["ROBOFLOW_API_KEY"] not in repr(run.kwargs)
        assert all(upload["artifact"].metadata["seed"] == run.seed for upload in run.uploads)
    assert env["wandb"].run is None


@pytest.mark.parametrize("stage", ["training", "epoch_upload", "evaluation", "final_upload", "finish"])
def test_failure_stops_queue_preserves_first_seed_and_finishes_run(env, workdir, stage):
    events = prepare_mock_queue(env, workdir)
    env["SEEDS_TO_RUN"] = [7, 3407, 42]
    env["wandb"] = FakeWandb(events, fail_stage=stage, fail_seed=3407)
    train, evaluate = env["run_full_training"], env["evaluate_final_training"]

    def train_or_fail():
        if stage == "training" and env["SEED"] == 3407: raise RuntimeError("simulated training")
        return train()

    def evaluate_or_fail(training):
        if stage == "evaluation" and env["SEED"] == 3407: raise RuntimeError("simulated evaluation")
        return evaluate(training)

    env["run_full_training"], env["evaluate_final_training"] = train_or_fail, evaluate_or_fail
    with pytest.raises(RuntimeError, match="simulated"):
        env["run_seed_queue"]()
    assert [run.seed for run in env["wandb"].runs] == [7, 3407]
    assert ("finish", 3407, 1) in events
    progress_file = next(Path(env["PROJECT"]).rglob("multiseed_progress.json"))
    progress = json.loads(progress_file.read_bytes())
    assert progress["status"] == "failed" and progress["failed_seed"] == 3407
    assert [row["seed"] for row in progress["completed"]] == [7]
    assert (progress_file.parent / "seed-7/seed_result.json").is_file()


@pytest.mark.slow
def test_two_real_synthetic_cpu_trainings_keep_fresh_students_and_checkpoint_uploads(env, workdir, monkeypatch):
    """CPU fixture only: two 64px/one-epoch synthetic runs, not AMD or Poltrada evidence."""
    from ultralytics.utils import callbacks

    previous_threads = torch.get_num_threads()
    torch.set_num_threads(2)
    monkeypatch.setattr(callbacks, "add_integration_callbacks", lambda instance: None)
    monkeypatch.setitem(settings, "wandb", False)
    dataset = synthetic_dataset(workdir / "cpu_synthetic", relative=False)
    env.update(DATA=str(dataset), EPOCHS=1, BATCH=2, IMGSZ=64, DEVICE="cpu", WORKERS=0, AMP=False,
        DATASET_PROVENANCE={"source": "CPU-SYNTHETIC-NOT-POLTRADA", "version": 3,
                            "yaml_sha256": hashlib.sha256(dataset.read_bytes()).hexdigest()},
        ENVIRONMENT_INFO={"backend": "CPU-SYNTHETIC-NOT-ROCM"},
        require_rocm_runtime=lambda: None, require_training_device=lambda device: None)
    # Only this harness bypasses the production HIP/device guards. Batch/AMP guards still run.
    pretrained = workdir / "fixed_pretrained_n.pt"
    torch.save({"model": DetectionModel("yolo26n.yaml", nc=6, verbose=False),
                "train_args": {"task": "detect", "imgsz": 64}}, pretrained)
    env["MODEL_WEIGHTS"] = str(pretrained)
    events, created, initialized = [], [], []
    env["wandb"] = FakeWandb(events)

    def model(weights):
        network = YOLO(weights)
        created.append((str(weights), weakref.ref(network)))
        if str(weights) == str(pretrained):
            def on_start(trainer):
                assert env["wb_state"]["uploaded_epochs"] == set()
                initialized.append((env["SEED"], next(trainer.model.parameters()).detach().clone()))
            network.add_callback("on_train_start", on_start)
        return network

    env["YOLO"] = model
    original_val = YOLO.val
    monkeypatch.setattr(YOLO, "val", lambda model, **kw: original_val(model, **{**kw, "plots": False}))
    try:
        result = env["run_seed_queue"]()
        assert [row["seed"] for row in result] == [7, 3407]
        assert [seed for seed, _weights in initialized] == [7, 3407]
        torch.testing.assert_close(initialized[0][1], initialized[1][1], rtol=0, atol=0)
        assert sum(path == str(pretrained) for path, _ref in created) == 2
        gc.collect()
        assert all(ref() is None for _path, ref in created)
        for run, row in zip(env["wandb"].runs, result):
            assert [step for _data, step in run.logged if step is not None] == [1]
            assert [item["artifact"].type for item in run.uploads] == ["model", "results"]
            assert all(item["waited"] for item in run.uploads)
            for item in run.uploads:
                artifact = item["artifact"]
                assert artifact.metadata["seed"] == run.seed
                assert artifact.metadata["environment"]["backend"] == "CPU-SYNTHETIC-NOT-ROCM"
                assert artifact.metadata["dataset_provenance"]["source"] == "CPU-SYNTHETIC-NOT-POLTRADA"
                assert env["WANDB_API_KEY"] not in repr(artifact.metadata)
            assert Path(row["best_pt"]).is_file() and row["epochs"] == 1
            training = json.loads((Path(row["seed_directory"]) / "training_result.json").read_bytes())
            assert training["completed_epochs"] == 1
            assert YOLO(row["best_pt"]).model.yaml["scale"] == "n"
            config = json.loads((Path(training["run_dir"]) / "config.json").read_bytes())
            assert env["WANDB_API_KEY"] not in repr(config) and env["ROBOFLOW_API_KEY"] not in repr(config)
        assert env["wandb"].run is None
    finally:
        torch.set_num_threads(previous_threads)
