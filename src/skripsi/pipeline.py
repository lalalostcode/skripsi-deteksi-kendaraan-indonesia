"""Explicit entrypoints. Importing this module never runs training."""

from __future__ import annotations

import gc
import hashlib
import json
import platform
import time
from copy import deepcopy
from pathlib import Path

import pandas as pd
import torch
from ultralytics import YOLO
from ultralytics.models.yolo.detect import DetectionTrainer

from .config import EXPECTED_V2_FINGERPRINT, ExperimentConfig
from .data import PreparedDataset, get_secret, prepare_dataset
from .evaluation import evaluate_checkpoint as _evaluate_checkpoint
from .models.factory import (
    build_model,
    build_model_core,
    detect_input_shapes,
    verify_model_graph,
)


class ModularDetectionTrainer(DetectionTrainer):
    """Keep the custom graph when Ultralytics normally rebuilds from YAML."""

    MODEL_TEMPLATE = None
    ARCHITECTURE = None

    def get_model(self, cfg=None, weights=None, verbose=True):
        if self.MODEL_TEMPLATE is None or self.ARCHITECTURE is None:
            raise RuntimeError("Template custom model belum diset")
        model = deepcopy(self.MODEL_TEMPLATE)
        if int(model.model[-1].nc) != int(self.data["nc"]):
            raise RuntimeError("Jumlah kelas checkpoint dan dataset berbeda")
        verify_model_graph(model, self.ARCHITECTURE)
        model.names = self.data["names"]
        return model


def _source_hash(architecture: str) -> str | None:
    if architecture == "baseline":
        return None
    source = Path(__file__).parent / "models" / f"{architecture}.py"
    return hashlib.sha256(source.read_bytes()).hexdigest()


def _package_hash() -> str:
    digest = hashlib.sha256()
    root = Path(__file__).parent
    for path in sorted(root.rglob("*.py")):
        digest.update(str(path.relative_to(root)).replace("\\", "/").encode("utf-8"))
        digest.update(hashlib.sha256(path.read_bytes()).digest())
    return digest.hexdigest()


def _environment(config: ExperimentConfig) -> dict:
    import importlib.metadata as metadata

    return {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "torch": torch.__version__,
        "ultralytics": metadata.version("ultralytics"),
        "cuda": torch.version.cuda,
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "git_commit": config.git_commit,
        "architecture_source_sha256": _source_hash(config.architecture),
        "package_sha256": _package_hash(),
    }


def preflight(config: ExperimentConfig) -> PreparedDataset:
    """Audit exact v2 and inspect one synthetic forward, without training."""
    dataset = prepare_dataset(config)
    model = build_model(config)
    core = model.model
    verify_model_graph(core, config.architecture)
    shapes = detect_input_shapes(core, config.imgsz, "cpu")
    config.reports_dir.mkdir(parents=True, exist_ok=True)
    architecture = pd.DataFrame(
        [
            {
                "index": index,
                "from": str(getattr(layer, "f", None)),
                "module": layer.__class__.__name__,
                "parameters": sum(p.numel() for p in layer.parameters()),
            }
            for index, layer in enumerate(core.model)
        ]
    )
    architecture.to_csv(
        config.reports_dir / f"{config.architecture}_model_architecture.csv",
        index=False,
    )
    report = {
        "config": config.as_dict(),
        "environment": _environment(config),
        "dataset_fingerprint": dataset.fingerprint,
        "detect_input_shapes": shapes,
        "status": "passed",
    }
    (config.reports_dir / f"{config.architecture}_preflight.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    del model
    gc.collect()
    return dataset


def _wandb_run(config: ExperimentConfig, dataset: PreparedDataset):
    key = get_secret("WANDB_API_KEY")
    if not key:
        return _NoOpRun(), False
    try:
        import os

        import wandb

        os.environ["WANDB_API_KEY"] = key
        run = wandb.init(
            project=config.wandb_project,
            entity=config.wandb_entity,
            name=config.run_name,
            group=f"yolo26n-{config.architecture}",
            job_type="research-training",
            config={
                **config.as_dict(),
                "dataset_fingerprint": dataset.fingerprint,
                **_environment(config),
            },
            mode="online",
            reinit="finish_previous",
        )
        return run, True
    except Exception as exc:
        print(
            f"[WARN] W&B tidak tersedia; hasil lokal tetap disimpan: {type(exc).__name__}: {exc}"
        )
        return _NoOpRun(), False


class _NoOpRun:
    def finish(self) -> None:
        pass


def _epoch_metrics(trainer) -> dict[str, float | int]:
    """Collect Ultralytics' per-epoch scalars, timing, and validation fitness."""
    values: dict[str, float | int] = {
        "epoch": int(trainer.epoch + 1),
        "time": float(time.time() - trainer.train_time_start),
    }
    if trainer.tloss is not None:
        for key, value in trainer.label_loss_items(trainer.tloss).items():
            values[key] = float(value)
    for key, value in (trainer.metrics or {}).items():
        try:
            # Keep the existing W&B validation names, including val/metrics/*.
            values[key if key.startswith("val/") else f"val/{key}"] = float(value)
        except (TypeError, ValueError):
            pass
    for key, value in (trainer.lr or {}).items():
        values[key] = float(value)
    if trainer.fitness is not None:
        values["val/fitness"] = float(trainer.fitness)
    if trainer.epoch_time is not None:
        values["train/epoch_time"] = float(trainer.epoch_time)
    return values


class _WandbEpochLogger:
    """Log completed epochs, excluding final validation of the best model."""

    def __init__(self, run):
        self.run = run
        self.active_epoch: int | None = None

    def on_train_epoch_start(self, trainer) -> None:
        self.active_epoch = int(trainer.epoch)

    def on_fit_epoch_end(self, trainer) -> None:
        if self.active_epoch != int(trainer.epoch):
            return
        self.active_epoch = None
        self.run.log(_epoch_metrics(trainer))


def _upload_checkpoint(
    run,
    enabled: bool,
    config: ExperimentConfig,
    checkpoint: Path,
    dataset: PreparedDataset,
    aliases: list[str],
) -> bool:
    if not enabled or not checkpoint.exists():
        return False
    try:
        import wandb

        artifact = wandb.Artifact(
            f"yolo26n-{config.architecture}-seed-{config.seed}-checkpoint",
            type="model",
            metadata={
                "architecture": config.architecture,
                "seed": config.seed,
                "dataset_fingerprint": dataset.fingerprint,
                "dataset_version": config.dataset_version,
                "git_commit": config.git_commit,
                "source_sha256": _source_hash(config.architecture),
            },
        )
        artifact.add_file(str(checkpoint), name=checkpoint.name)
        if config.architecture != "baseline":
            source = Path(__file__).parent / "models" / f"{config.architecture}.py"
            artifact.add_file(str(source), name=source.name)
        run.log_artifact(artifact, aliases=aliases).wait()
        return True
    except Exception as exc:
        print(f"[WARN] Upload checkpoint gagal: {type(exc).__name__}: {exc}")
        return False


def run_training(
    config: ExperimentConfig, dataset: PreparedDataset | None = None
) -> dict:
    """The only training entrypoint; call explicitly from a Kaggle or local notebook."""
    dataset = dataset or preflight(config)
    if dataset.fingerprint != EXPECTED_V2_FINGERPRINT:
        raise RuntimeError(
            "Training penelitian hanya menerima snapshot dataset v2 yang diaudit"
        )
    if config.run_dir.exists() and any(config.run_dir.iterdir()):
        raise RuntimeError(f"Run directory sudah berisi data: {config.run_dir}")
    config.run_dir.mkdir(parents=True, exist_ok=True)
    from .models.factory import set_seed

    set_seed(config.seed)
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
    run, wandb_enabled = _wandb_run(config, dataset)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if config.architecture == "baseline":
        model = YOLO(config.model_weights)
        trainer_class = None
    else:
        ModularDetectionTrainer.MODEL_TEMPLATE = build_model_core(config).cpu()
        ModularDetectionTrainer.ARCHITECTURE = config.architecture
        model = YOLO(config.model_weights)
        trainer_class = ModularDetectionTrainer

    def on_model_save(trainer):
        epoch = int(trainer.epoch + 1)
        if epoch % config.artifact_period == 0:
            _upload_checkpoint(
                run,
                wandb_enabled,
                config,
                Path(trainer.last),
                dataset,
                ["latest", f"epoch-{epoch}"],
            )

    if wandb_enabled:
        epoch_logger = _WandbEpochLogger(run)
        model.add_callback("on_train_epoch_start", epoch_logger.on_train_epoch_start)
        model.add_callback("on_fit_epoch_end", epoch_logger.on_fit_epoch_end)
    model.add_callback("on_model_save", on_model_save)
    started = time.perf_counter()
    status = "failed"
    error = None
    try:
        arguments = dict(
            data=str(dataset.runtime_yaml),
            epochs=config.epochs,
            patience=config.patience,
            imgsz=config.imgsz,
            batch=config.train_batch,
            device=device,
            workers=config.workers,
            optimizer=config.optimizer,
            lr0=config.lr0,
            weight_decay=config.weight_decay,
            warmup_epochs=config.warmup_epochs,
            amp=config.amp and torch.cuda.is_available(),
            seed=config.seed,
            deterministic=True,
            freeze=None,
            flipud=config.flipud,
            mixup=config.mixup,
            cutmix=config.cutmix,
            close_mosaic=config.close_mosaic,
            save=True,
            save_period=config.save_period,
            plots=True,
            val=True,
            project=str(config.output_root / "runs"),
            name=config.run_name,
            exist_ok=True,
            verbose=True,
        )
        if trainer_class is not None:
            arguments["trainer"] = trainer_class
        model.train(**arguments)
        verify_model_graph(model.model, config.architecture)
        status = "completed"
    except Exception as exc:
        error = f"{type(exc).__name__}: {exc}"
        raise
    finally:
        elapsed = time.perf_counter() - started
        metadata = {
            "architecture": config.architecture,
            "seed": config.seed,
            "status": status,
            "dataset_project": config.project,
            "dataset_version": config.dataset_version,
            "dataset_fingerprint": dataset.fingerprint,
            "duration_seconds": elapsed,
            "training_peak_vram_mb": torch.cuda.max_memory_allocated() / 1024**2
            if torch.cuda.is_available()
            else None,
            "epochs_completed": int(model.trainer.epoch + 1)
            if getattr(model, "trainer", None) is not None
            and hasattr(model.trainer, "epoch")
            else None,
            "config": config.as_dict(),
            "environment": _environment(config),
            "error": error,
        }
        (config.run_dir / "run_metadata.json").write_text(
            json.dumps(metadata, indent=2), encoding="utf-8"
        )
        best = config.run_dir / "weights" / "best.pt"
        last = config.run_dir / "weights" / "last.pt"
        _upload_checkpoint(run, wandb_enabled, config, best, dataset, ["best", status])
        _upload_checkpoint(
            run, wandb_enabled, config, last, dataset, ["latest", status]
        )
        run.finish()
        ModularDetectionTrainer.MODEL_TEMPLATE = None
        ModularDetectionTrainer.ARCHITECTURE = None
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    if not best.exists() or not last.exists():
        raise RuntimeError(
            f"Checkpoint best.pt/last.pt tidak lengkap di {config.run_dir}"
        )
    return {"run_dir": config.run_dir, "best": best, "last": last, "metadata": metadata}


def evaluate_checkpoint(
    config: ExperimentConfig,
    checkpoint: str | Path,
    dataset: PreparedDataset | None = None,
) -> dict:
    """Evaluate best.pt without calling train(); validation selects the test threshold."""
    dataset = dataset or prepare_dataset(config)
    if dataset.fingerprint != EXPECTED_V2_FINGERPRINT:
        raise RuntimeError(
            "Evaluasi penelitian hanya menerima snapshot dataset v2 yang diaudit"
        )
    checkpoint = Path(checkpoint)
    metadata_file = checkpoint.parent.parent / "run_metadata.json"
    if metadata_file.exists():
        metadata = json.loads(metadata_file.read_text(encoding="utf-8"))
        if (
            metadata.get("dataset_fingerprint") != dataset.fingerprint
            or metadata.get("architecture") != config.architecture
        ):
            raise RuntimeError(
                "Checkpoint, arsitektur, atau fingerprint dataset tidak cocok"
            )
    result = _evaluate_checkpoint(config, checkpoint, dataset)
    key = get_secret("WANDB_API_KEY")
    if key:
        try:
            import os

            import wandb

            os.environ["WANDB_API_KEY"] = key
            with wandb.init(
                project=config.wandb_project,
                entity=config.wandb_entity,
                name=f"{config.run_name}-evaluation",
                group=f"yolo26n-{config.architecture}",
                job_type="research-evaluation",
                config={**config.as_dict(), "dataset_fingerprint": dataset.fingerprint},
                reinit="finish_previous",
            ) as run:
                run.log(
                    {
                        **{
                            f"test/{k}": v
                            for k, v in result["overall"].items()
                            if isinstance(v, (int, float))
                        },
                        **{
                            f"efficiency/{k}": v
                            for k, v in result["efficiency"].items()
                            if isinstance(v, (int, float))
                        },
                    }
                )
                artifact = wandb.Artifact(
                    f"yolo26n-{config.architecture}-seed-{config.seed}-evaluation",
                    type="evaluation",
                )
                artifact.add_dir(str(result["evaluation_dir"]))
                run.log_artifact(artifact).wait()
        except Exception as exc:
            print(f"[WARN] Upload evaluasi W&B gagal: {type(exc).__name__}: {exc}")
    return result
