"""Verify the five-seed teacher queue, including a two-seed offline CPU smoke."""
# ruff: noqa: S102 -- Execute only generated local notebook code.
from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from test_serial_multiseed_notebooks import FakeWandb

from scripts.generate_teacher_yolo26m_notebooks import SEEDS, build_notebook
from scripts.notebook_paths import MULTISEED_TEACHER, notebook_path


def read_book(platform, filename=MULTISEED_TEACHER):
    return json.loads(notebook_path(platform, filename).read_text(encoding="utf-8"))


def cell_text(book, tag):
    return ["".join(cell["source"]) for cell in book["cells"] if tag in cell["metadata"].get("tags", [])]


@pytest.mark.parametrize("platform", ["kaggle", "filkom"])
def test_teacher_generation_matches_pretrained_seed_choices(platform):
    book = read_book(platform)
    assert book == build_notebook(platform)
    environment = {}
    exec(cell_text(book, "config")[0], environment)
    current_baseline = read_book("kaggle", "02_baseline_yolo26s.ipynb")
    baseline_seeds = re.findall(r"^(?:# )?SEED = (\d+)\s*$", "".join(current_baseline["cells"][2]["source"]), re.MULTILINE)
    assert environment["SEEDS_TO_RUN"] == list(map(int, baseline_seeds)) == [42, 0, 2024, 7, 3407]
    assert environment["MODEL_WEIGHTS"] == "yolo26m.pt"
    assert (environment["EPOCHS"], environment["IMGSZ"], environment["BATCH"], environment["EVAL_SPLIT"]) == (100, 640, 16, "val")
    assert (environment["OPTIMIZER"], environment["LR"], environment["AMP"]) == ("AdamW", 0.001, True)
    assert ("kaggle_secrets" in cell_text(book, "bootstrap")[0]) == (platform == "kaggle")
    if platform == "filkom":
        assert "/kaggle" not in environment["DATA"] and "outputs/filkom/teacher" in Path(environment["PROJECT"]).as_posix()
        assert 'getpass("WANDB_API_KEY: ")' in cell_text(book, "bootstrap")[0]


class VersionedWandb(FakeWandb):
    def init(self, **kwargs):
        run = super().init(**kwargs)
        log_artifact = run.log_artifact

        def versioned(artifact, aliases):
            record = log_artifact(artifact, aliases)
            record.qualified_name = f"{kwargs['entity']}/{kwargs['project']}/{artifact.name}:v{len(run.uploads) - 1}"
            return record

        run.log_artifact = versioned
        return run


@pytest.fixture
def runtime(tmp_path, monkeypatch):
    book = read_book("filkom")
    environment = {}
    exec(cell_text(book, "config")[0], environment)
    environment.update(DEVICE="cpu", AMP=False, INSTALL_DEPENDENCIES=False, WANDB_ENABLED=False,
                       PROJECT=str(tmp_path / "results"), EPOCHS=1)
    from ultralytics import settings
    monkeypatch.setattr(settings, "file_path", tmp_path / "ultralytics-settings.json")
    monkeypatch.setitem(settings, "wandb", False)
    exec(cell_text(book, "bootstrap")[0], environment)
    # Load the same hash helper without executing the separate dataset preflight in mock runs.
    exec(cell_text(book, "dataset")[0].split('if DATA != "coco8.yaml"')[0], environment)
    for source in cell_text(book, "helpers"):
        exec(source, environment)
    return environment


def mock_experiment(runtime, tmp_path, monkeypatch, fail_seed=None):
    initial = tmp_path / "yolo26m.pt"
    initial.write_bytes(b"fixed pretrained checkpoint")
    runtime.update(MODEL_WEIGHTS=str(initial), WANDB_ENABLED=True, init_seeds=lambda *a, **kw: None)
    fake = runtime["wandb"] = VersionedWandb([], fail_seed=fail_seed, fail_stage="epoch_upload" if fail_seed is not None else None)
    monkeypatch.setattr("ultralytics.utils.torch_utils.get_flops", lambda *a, **kw: 65.0)
    loaded_models = []

    class MockYOLO:
        def __init__(self, weights):
            self.ckpt_path = weights
            self.callbacks = {}
            self.model = SimpleNamespace(parameters=lambda: [torch.empty(1)])
            if Path(weights) == initial:
                loaded_models.append(self)

        def add_callback(self, event, callback):
            self.callbacks[event] = callback

        def train(self, **args):
            assert args["pretrained"] is True and "distill_model" not in args
            assert args["seed"] == fake.run.seed and Path(self.ckpt_path) == initial
            directory = Path(args["project"]) / args["name"]
            weights = directory / "weights"
            weights.mkdir(parents=True)
            for name in ("best.pt", "last.pt"):
                (weights / name).write_bytes(f"checkpoint seed {args['seed']}".encode())
            self.trainer = SimpleNamespace(
                epoch=0, fitness=0.25, best_fitness=0.25, batch_size=args["batch"],
                amp=args["amp"], args=SimpleNamespace(imgsz=args["imgsz"]),
                best=weights / "best.pt", last=weights / "last.pt", save_dir=directory,
                label_loss_items=lambda tloss, prefix: {"train/box_loss": 1.0},
                tloss={}, lr={"lr/pg0": 0.001}, metrics={"metrics/mAP50-95(B)": 0.25},
            )
            self.callbacks["on_model_save"](self.trainer)
            self.callbacks["on_fit_epoch_end"](self.trainer)

        def val(self, **args):
            assert Path(self.ckpt_path).name == "best.pt" and args["split"] == "val"
            assert Path(args["project"]) == Path(self.ckpt_path).parents[2]
            value = 0.25 + fake.run.seed / 10000
            return SimpleNamespace(box=SimpleNamespace(mp=value, mr=value, map50=value, map=value),
                                   speed={"inference": 1.0}, save_dir=Path(args["project"]) / "evaluation")

    runtime["YOLO"] = MockYOLO
    return initial, fake, loaded_models


def test_each_seed_loads_fresh_teacher_and_finishes_its_own_artifacts(runtime, tmp_path, monkeypatch):
    initial, fake, models = mock_experiment(runtime, tmp_path, monkeypatch)
    results = runtime["run_seed_queue"]()
    assert [row["seed"] for row in results] == list(SEEDS)
    assert len({id(model) for model in models}) == len(SEEDS)
    assert len({row["wandb_run_id"] for row in results}) == len(SEEDS)
    assert len({row["seed_directory"] for row in results}) == len(SEEDS)
    assert {row["initial_checkpoint_sha256"] for row in results} == {hashlib.sha256(initial.read_bytes()).hexdigest()}
    for row, run in zip(results, fake.runs, strict=True):
        assert run.finished and all(item["waited"] for item in run.uploads)
        assert run.kwargs["name"] == f"base_teacher_yolo26m_seed{row['seed']}_smoke"
        assert f"seed{row['seed']}" in run.kwargs["tags"]
        assert row["best_artifact"]["reference"].endswith(f"-{run.id}-checkpoints:v0")
        assert row["best_artifact"]["best_pt_sha256"] == row["best_pt_sha256"]
    assert fake.run is None
    summary_path = Path(results[0]["seed_directory"]).parent / "multiseed_summary.json"
    summary = json.loads(summary_path.read_bytes())
    assert summary["seeds_completed"] == list(SEEDS)
    assert summary["metrics"]["map50_95"]["sample_std"] > 0


def test_teacher_seed_failure_preserves_completed_seed(runtime, tmp_path, monkeypatch):
    _, fake, _ = mock_experiment(runtime, tmp_path, monkeypatch, fail_seed=0)
    with pytest.raises(RuntimeError, match="simulated epoch_upload"):
        runtime["run_seed_queue"]()
    progress_file = next(Path(runtime["PROJECT"]).rglob("multiseed_progress.json"))
    progress = json.loads(progress_file.read_bytes())
    assert progress["status"] == "failed" and progress["failed_seed"] == 0
    assert [item["seed"] for item in progress["completed"]] == [42]
    assert (progress_file.parent / "seed-42/seed_result.json").is_file()
    assert [run.seed for run in fake.runs] == [42, 0] and fake.run is None


@pytest.mark.slow
def test_real_yolo26m_two_seed_cpu_training_and_reload(runtime, tmp_path, monkeypatch):
    """One 64px epoch per seed on distinct train/val fixture images."""
    from PIL import Image
    from ultralytics import YOLO
    from ultralytics.models.yolo.detect import DetectionTrainer
    from ultralytics.nn.tasks import DetectionModel
    from ultralytics.utils import callbacks

    monkeypatch.setattr(callbacks, "add_integration_callbacks", lambda instance: None)
    monkeypatch.setattr("ultralytics.data.utils.check_font", lambda *a, **kw: None)
    monkeypatch.setattr("ultralytics.utils.checks.check_pip_update_available", lambda: None)

    def forbid_network(*args, **kwargs):
        raise AssertionError("The CPU teacher smoke must remain offline.")

    monkeypatch.setattr("requests.sessions.Session.request", forbid_network)
    dataset = tmp_path / "fixture"
    names = ["Bus", "Mobil Penumpang", "Pejalan Kaki", "Sepeda Motor", "Truck", "Unmotorized"]
    for split, brightness in (("train", 30), ("val", 140)):
        for child in ("images", "labels"):
            (dataset / split / child).mkdir(parents=True)
        for index in range(2):
            Image.new("RGB", (64, 64), (brightness + index, 70, 100)).save(dataset / split / "images" / f"{index}.png")
            (dataset / split / "labels" / f"{index}.txt").write_text(f"{index} .5 .5 .25 .25\n", encoding="utf-8")
    yaml_path = dataset / "data.yaml"
    yaml_path.write_text(f"path: {dataset.as_posix()}\ntrain: train/images\nval: val/images\nnames: {json.dumps(names)}\n", encoding="utf-8")
    initial_path = tmp_path / "fixed_yolo26m.pt"
    threads = torch.get_num_threads()
    torch.set_num_threads(2)
    try:
        model = DetectionModel("yolo26m.yaml", nc=len(names), verbose=False)
        model.names = dict(enumerate(names))
        torch.save({"model": model, "train_args": {"task": "detect", "imgsz": 64}}, initial_path)
        first_weights = next(model.parameters()).detach().clone()
        del model
        runtime.update(DATA=str(yaml_path), MODEL_WEIGHTS=str(initial_path), IMGSZ=64, BATCH=2,
                       WORKERS=0, SEEDS_TO_RUN=[42, 0])
        exec(cell_text(read_book("filkom"), "dataset")[0], runtime)
        train, validate = YOLO.train, YOLO.val
        monkeypatch.setattr(YOLO, "train", lambda core, **kw: train(core, **{**kw, "plots": False, "nbs": 2}))
        monkeypatch.setattr(YOLO, "val", lambda core, **kw: validate(core, **{**kw, "plots": False}))
        get_model = DetectionTrainer.get_model
        initial_seeds = []

        def check_pretrained(trainer, cfg=None, weights=None, verbose=True):
            assert trainer.args.pretrained and weights is not None
            loaded = get_model(trainer, cfg=cfg, weights=weights, verbose=verbose)
            assert loaded.yaml["scale"] == "m" and torch.equal(next(loaded.parameters()), first_weights)
            initial_seeds.append(trainer.args.seed)
            return loaded

        monkeypatch.setattr(DetectionTrainer, "get_model", check_pretrained)
        results = runtime["run_seed_queue"]()
        assert initial_seeds == [42, 0]
        assert [row["seed"] for row in results] == [42, 0]
        for row in results:
            recovered = YOLO(row["best_pt"]).model
            assert recovered.yaml["scale"] == "m" and list(recovered.names.values()) == names
            assert not torch.equal(next(recovered.parameters()).float(), first_weights)
            assert row["completed_epochs"] == 1 and row["best_artifact"] is None
    finally:
        torch.set_num_threads(threads)
