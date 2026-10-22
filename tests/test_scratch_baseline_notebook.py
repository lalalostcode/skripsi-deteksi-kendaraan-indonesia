"""Scratch initialization, baseline protocol parity, serial mocks and one offline CPU epoch."""
# ruff: noqa: S102 -- Execute trusted local notebook declarations, never installation/login cells.
from __future__ import annotations

import json
import types
import uuid
from pathlib import Path
from unittest.mock import Mock

import nbformat
import pytest
import torch
from PIL import Image
from test_serial_multiseed_notebooks import FakeWandb, load_environment
from ultralytics import YOLO, settings
from ultralytics.cfg import get_cfg
from ultralytics.models.yolo.detect import DetectionTrainer
from ultralytics.utils import callbacks

from scripts.notebook_paths import notebook_path

ROOT = Path(__file__).resolve().parents[1]
FOLDER = ROOT / "notebooks/kaggle"
TARGET = notebook_path("kaggle", "03_baseline_yolo26n_scratch.ipynb")
BASELINE = notebook_path("kaggle", "03_baseline_yolo26n.ipynb")
NAMES = ["bus", "car", "motorcycle", "truck", "pickup", "minibus"]


@pytest.fixture
def notebook():
    return json.loads(TARGET.read_bytes())


@pytest.fixture
def workdir():
    directory = ROOT / "outputs/test_scratch_baseline" / uuid.uuid4().hex
    directory.mkdir(parents=True)
    return directory


@pytest.fixture
def env(notebook, workdir):
    return load_environment(notebook, workdir)


def mock_runtime(env, monkeypatch, *, fail_stage=None):
    """Exercise production training/callback/evaluation code with a tiny fake YOLO API."""
    events, constructed, trained, states = [], [], [], []
    wandb = env["wandb"] = FakeWandb(events, fail_stage=fail_stage, fail_seed=123)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr("ultralytics.utils.torch_utils.get_flops", lambda *a, **kw: 0.0)

    class FakeYOLO:
        def __init__(self, source):
            self.source = str(source)
            self.model = torch.nn.Linear(2, 2)
            self.callbacks = {}
            constructed.append(self)

        def add_callback(self, event, callback):
            self.callbacks[event] = callback

        def train(self, **kwargs):
            seed = env["SEED"]
            if seed == 123 and fail_stage == "training":
                raise RuntimeError("simulated training")
            assert env["wb_state"]["uploaded_epochs"] == set()
            assert env["wb_state"]["logged_epochs"] == set()
            assert env["wb_state"]["best_epoch"] is None
            states.append(env["wb_state"])
            trained.append((seed, self, kwargs))
            directory = Path(kwargs["project"]) / kwargs["name"]
            weights = directory / "weights"
            weights.mkdir(parents=True)
            for filename in ("best.pt", "last.pt"):
                (weights / filename).write_bytes(f"scratch seed {seed}".encode())
            (directory / "results.csv").write_text("epoch,fitness\n1,0.5\n")
            (directory / "args.yaml").write_text("pretrained: false\n")
            self.trainer = types.SimpleNamespace(
                epoch=0, last=weights / "last.pt", best=weights / "best.pt", save_dir=directory,
                fitness=0.5, best_fitness=0.5, tloss=torch.tensor(1.0), lr={"lr/pg0": 0.001},
                metrics={"metrics/mAP50-95(B)": 0.1 + seed / 100000},
                label_loss_items=lambda value, prefix: {prefix + "/box_loss": float(value)},
            )
            self.callbacks["on_model_save"](self.trainer)
            self.callbacks["on_fit_epoch_end"](self.trainer)

        def val(self, **kwargs):
            if env["SEED"] == 123 and fail_stage == "evaluation":
                raise RuntimeError("simulated evaluation")
            assert self.source == str(Path(env["SESSION_DIR"]) / "training/weights/best.pt")
            assert kwargs["split"] == "val"
            return types.SimpleNamespace(
                box=types.SimpleNamespace(mp=0.2, mr=0.3, map50=0.4, map=0.1 + env["SEED"] / 100000),
                speed={"inference": 1.0}, save_dir=Path(self.source).parents[1],
            )

    env["YOLO"] = FakeYOLO
    return types.SimpleNamespace(wandb=wandb, events=events, constructed=constructed, trained=trained, states=states)


def test_schema_defaults_and_unmodified_setup_dataset_loss(notebook):
    nbformat.validate(nbformat.from_dict(notebook))
    config = {}
    exec("".join(notebook["cells"][2]["source"]), config)
    assert config["MODEL_WEIGHTS"] == "yolo26n.yaml"
    assert config["PRETRAINED"] is False and config["SEEDS_TO_RUN"] == [42]
    assert config["SEED"] == 42 and config["EXPERIMENT_ID"] == "03_baseline_yolo26n_scratch"
    assert config["WANDB_PROJECT"] == "yolom26-kd-indotraffic"
    assert config["WANDB_GROUP"] == "supervised"
    assert {"baseline", "model-n", "from-scratch", "seed42"} <= set(config["WANDB_TAGS"])
    text = "\n".join("".join(cell["source"]) for cell in notebook["cells"])
    assert "Sudah selesai" not in text and "Hasil historis seed" not in text
    for seed in (123, 2024, 7, 3407):
        assert f"# {seed}," in "".join(notebook["cells"][2]["source"])
    for cell in notebook["cells"]:
        if cell["cell_type"] == "code":
            compile("".join(cell["source"]), cell["id"], "exec")
            assert cell["outputs"] == [] and cell["execution_count"] is None
    baseline = json.loads(BASELINE.read_bytes())
    # Installation/login, dataset and empty KD definitions are inherited verbatim.
    for index in (4, 6, 11, 17):
        assert notebook["cells"][index]["source"] == baseline["cells"][index]["source"]
    original_config = {}
    exec("".join(baseline["cells"][2]["source"]), original_config)
    exempt = {"EXPERIMENT_ID", "MODEL_WEIGHTS", "SEEDS_TO_RUN", "SEED", "WANDB_TAGS", "BASE_WANDB_TAGS"}
    assert {key: config[key] for key in original_config if key not in exempt and key != "__builtins__"} == {
        key: value for key, value in original_config.items() if key not in exempt and key != "__builtins__"}


def test_all_effective_training_arguments_match_pretrained_baseline(env, monkeypatch, workdir):
    original = load_environment(json.loads(BASELINE.read_bytes()), workdir / "pretrained")
    configs = []
    for namespace in (original, env):
        namespace["SEEDS_TO_RUN"] = [42]
        runtime = mock_runtime(namespace, monkeypatch)
        namespace["run_seed_queue"]()
        args = runtime.trained[0][2]
        configs.append(vars(get_cfg(overrides={**args, "model": namespace["MODEL_WEIGHTS"]})))
    differences = {key for key in configs[0] if configs[0][key] != configs[1][key]}
    assert differences == {"model", "pretrained", "project"}
    assert configs[0]["pretrained"] is True and configs[1]["pretrained"] is False
    assert configs[1]["epochs"] == 100 and configs[1]["amp"] is True


def test_two_seeds_have_independent_students_state_metrics_and_artifacts(env, monkeypatch):
    env["SEEDS_TO_RUN"] = [42, 123]
    runtime = mock_runtime(env, monkeypatch)
    result = env["run_seed_queue"]()
    assert [row["seed"] for row in result] == [42, 123]
    assert [run.seed for run in runtime.wandb.runs] == [42, 123]
    assert runtime.wandb.run is None and all(run.finished for run in runtime.wandb.runs)
    assert len({run.id for run in runtime.wandb.runs}) == 2
    assert len({row["seed_directory"] for row in result}) == 2
    assert len({row["map50_95"] for row in result}) == 2
    assert runtime.states[0] is not runtime.states[1]
    assert runtime.trained[0][1] is not runtime.trained[1][1]
    assert not torch.equal(runtime.trained[0][1].model.weight, runtime.trained[1][1].model.weight)
    assert [model.source for model in runtime.constructed][::2] == ["yolo26n.yaml"] * 2
    assert runtime.events.index(("finish", 42, 0)) < next(
        i for i, event in enumerate(runtime.events) if event[:2] == ("init", 123))
    names = []
    for run, row in zip(runtime.wandb.runs, result):
        assert run.kwargs["name"] == f"baseline_yolo26n_scratch_seed{run.seed}"
        assert run.kwargs["group"] == "supervised" and run.kwargs["resume"] == "never"
        assert {tag for tag in run.kwargs["tags"] if tag.startswith("seed")} == {f"seed{run.seed}"}
        assert "from-scratch" in run.kwargs["tags"]
        assert run.config["pretrained"] is False and run.config["initialization"] == "random"
        assert run.config["training_args"]["pretrained"] is False
        assert [step for _data, step in run.logged if step is not None] == [1]
        assert [upload["artifact"].type for upload in run.uploads] == ["model", "results"]
        for upload in run.uploads:
            metadata = upload["artifact"].metadata
            assert metadata["seed"] == run.seed and metadata["pretrained"] is False
            assert metadata["initialization"] == "random" and upload["waited"]
            names.append(upload["artifact"].name)
        directory = Path(row["seed_directory"])
        for path in (directory / "training_result.json", directory / "seed_result.json",
                     directory / "training/config.json", directory / "training/metrics.json"):
            content = json.loads(path.read_bytes())
            assert content["pretrained"] is False and content["initialization"] == "random"
    assert len(set(names)) == 4


@pytest.mark.parametrize("stage", ["training", "epoch_upload", "evaluation", "final_upload", "finish"])
def test_failure_stops_queue_and_preserves_completed_scratch_seed(env, monkeypatch, stage):
    env["SEEDS_TO_RUN"] = [42, 123, 2024]
    runtime = mock_runtime(env, monkeypatch, fail_stage=stage)
    with pytest.raises(RuntimeError, match="simulated"):
        env["run_seed_queue"]()
    assert [run.seed for run in runtime.wandb.runs] == [42, 123]
    assert runtime.wandb.runs[0].finished
    assert ("finish", 123, 1) in runtime.events
    if stage != "finish":
        assert runtime.wandb.run is None and runtime.wandb.runs[1].finished
    progress_path = next(Path(env["PROJECT"]).rglob("multiseed_progress.json"))
    progress = json.loads(progress_path.read_bytes())
    assert progress["status"] == "failed" and progress["failed_seed"] == 123
    assert [row["seed"] for row in progress["completed"]] == [42]
    assert (progress_path.parent / "seed-42/seed_result.json").is_file()


def test_one_epoch_cpu_scratch_has_no_pretrained_transfer(env, workdir, monkeypatch):
    """Exercise the actual trainer with 64px synthetic data; never run Poltrada or a network request."""
    torch.set_num_threads(2)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(callbacks, "add_integration_callbacks", lambda instance: None)
    monkeypatch.setattr(settings, "file_path", workdir / "ultralytics-settings.json")
    monkeypatch.setitem(settings, "wandb", False)
    monkeypatch.setattr("ultralytics.data.utils.check_font", lambda *a, **kw: None)
    monkeypatch.setattr("ultralytics.utils.checks.check_pip_update_available", lambda: None)
    request = Mock(side_effect=AssertionError("CPU scratch smoke must not use the network"))
    monkeypatch.setattr("requests.sessions.Session.request", request)

    def local_assets_only(file, *args, **kwargs):
        assert Path(file).is_file(), f"Unexpected external checkpoint: {file}"
        return str(file)

    monkeypatch.setattr("ultralytics.utils.downloads.attempt_download_asset", local_assets_only)
    dataset = workdir / "synthetic"
    for split in ("train", "val"):
        for child in ("images", "labels"):
            (dataset / split / child).mkdir(parents=True)
        for index in range(2):
            Image.new("RGB", (64, 64), (40 + index, 80, 150)).save(dataset / split / "images" / f"{index}.png")
            (dataset / split / "labels" / f"{index}.txt").write_text("0 .5 .5 .25 .25\n")
    yaml = dataset / "data.yaml"
    yaml.write_text(f"path: {dataset.as_posix()}\ntrain: train/images\nval: val/images\nnames: {NAMES}\n")
    env.update(DATA=str(yaml), EPOCHS=1, IMGSZ=64, BATCH=2, DEVICE="cpu", WORKERS=0, AMP=False)
    events, initial, models = [], [], []
    fake = env["wandb"] = FakeWandb(events)
    get_model = DetectionTrainer.get_model

    def inspect_get_model(trainer, cfg=None, weights=None, verbose=True):
        assert weights is None and trainer.args.pretrained is False
        model = get_model(trainer, cfg=cfg, weights=weights, verbose=verbose)
        assert model.yaml["scale"] == "n" and model.model[-1].nc == len(NAMES)
        assert trainer.data["names"] == dict(enumerate(NAMES))
        initial.append(next(model.parameters()).detach().clone())
        return model

    monkeypatch.setattr(DetectionTrainer, "get_model", inspect_get_model)
    train = YOLO.train
    monkeypatch.setattr(YOLO, "train", lambda model, **kw: train(model, **{**kw, "plots": False, "nbs": 2}))
    validate = YOLO.val
    monkeypatch.setattr(YOLO, "val", lambda model, **kw: validate(model, **{**kw, "plots": False}))

    def construct(source):
        model = YOLO(source)
        models.append(str(source))
        if str(source).endswith(".yaml"):
            assert not model.ckpt and model.model.yaml["scale"] == "n"
        return model

    env["YOLO"] = construct
    result = env["run_seed_queue"]()
    assert len(result) == 1 and result[0]["seed"] == 42 and len(initial) == 1
    assert models[0] == "yolo26n.yaml" and models[1] == result[0]["best_pt"]
    best = YOLO(result[0]["best_pt"]).model
    assert best.yaml["scale"] == "n" and best.model[-1].nc == len(NAMES)
    assert not torch.equal(next(best.parameters()).float(), initial[0])
    assert fake.run is None and fake.runs[0].finished
    assert fake.runs[0].config["training_args"]["pretrained"] is False
    assert [upload["artifact"].type for upload in fake.runs[0].uploads] == ["model", "results"]
    assert all(upload["artifact"].metadata["initialization"] == "random" for upload in fake.runs[0].uploads)
    assert all(upload["waited"] for upload in fake.runs[0].uploads)
    request.assert_not_called()
