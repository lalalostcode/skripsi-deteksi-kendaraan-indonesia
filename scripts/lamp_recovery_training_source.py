"""Standalone ordinary fine-tuning helpers for native structurally pruned YOLO models.

The trainer retains the DetectionModel object instead of rebuilding its original YAML.
The detection criterion, gradients, optimizer, EMA, augmentations and validation remain
Ultralytics' native implementation. Only model loading, logging isolation and auditing
are customized. Best FP32 EMA is captured before SDK FP16 checkpoint serialization.
"""
import hashlib as _recovery_hashlib
import json as _recovery_json
from copy import deepcopy as _recovery_deepcopy
from datetime import UTC as _RECOVERY_UTC
from datetime import datetime as _RecoveryDateTime
from pathlib import Path as _RecoveryPath

import torch as _recovery_torch
from ultralytics import __version__ as _recovery_ultralytics_version
from ultralytics.models.yolo.detect import DetectionTrainer as _RecoveryDetectionTrainer
from ultralytics.nn.tasks import DetectionModel as _RecoveryDetectionModel


def recovery_ordered_names(names):
    """Preserve class order and reject sparse or non-integer class identifiers."""
    if isinstance(names, dict):
        if set(names) != set(range(len(names))):
            raise ValueError("Recovery class IDs must be contiguous integers starting at zero.")
        names = [names[index] for index in range(len(names))]
    if not isinstance(names, (list, tuple)) or not names or not all(isinstance(n, str) for n in names):
        raise ValueError("Recovery requires an explicit nonempty ordered class-name list.")
    return list(names)


def recovery_structure_signature(model):
    """Parameter shapes and physical Conv/BN dimensions; independent of learned values."""
    return {
        "parameters": {name: list(value.shape) for name, value in model.named_parameters()},
        "layers": {
            name: {"type": type(module).__name__, "in": module.in_channels,
                   "out": module.out_channels, "groups": module.groups}
            if isinstance(module, _recovery_torch.nn.Conv2d)
            else {"type": type(module).__name__, "features": module.num_features}
            for name, module in model.named_modules()
            if isinstance(module, (_recovery_torch.nn.Conv2d, _recovery_torch.nn.BatchNorm2d))
        },
    }


def recovery_state_sha256(model, *, parameters_only=False):
    digest = _recovery_hashlib.sha256()
    values = dict(model.named_parameters()) if parameters_only else model.state_dict()
    for name, value in sorted(values.items()):
        tensor = value.detach().cpu().contiguous()
        digest.update(name.encode())
        digest.update(str(tuple(tensor.shape)).encode())
        digest.update(str(tensor.dtype).encode())
        digest.update(tensor.numpy().tobytes())
    return digest.hexdigest()


def _recovery_write_json(path, value):
    path = _RecoveryPath(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(_recovery_json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False), encoding="utf-8")


def _recovery_native_snapshot(model):
    core = _recovery_deepcopy(model).to("cpu").float().eval()
    if type(core) is not _RecoveryDetectionModel:
        raise TypeError("Only an ordinary native DetectionModel can be serialized for recovery.")
    core.criterion = None
    return core


def _recovery_finite_state(model):
    return all(_recovery_torch.isfinite(v).all().item() for v in model.state_dict().values()
               if isinstance(v, _recovery_torch.Tensor) and v.is_floating_point())


class StructuralRecoveryTrainer(_RecoveryDetectionTrainer):
    """Native detection training with exact pruned-object loading and FP32 EMA snapshots."""

    def __init__(self, core, *, overrides, expected_names, source_sha256, epoch_callback=None):
        self.recovery_source_core = _recovery_native_snapshot(core)
        self.recovery_expected_names = recovery_ordered_names(expected_names)
        self.recovery_initial_signature = recovery_structure_signature(core)
        self.recovery_initial_state_sha256 = recovery_state_sha256(core)
        self.recovery_initial_parameters_sha256 = recovery_state_sha256(core, parameters_only=True)
        self.recovery_source_sha256 = source_sha256
        self.recovery_optimizer_steps = 0
        self.recovery_epochs = []
        self.recovery_best_core = None
        self.recovery_best_epoch = None
        self.recovery_initial_state_verified = False
        self.recovery_epoch_callback = epoch_callback
        self.recovery_filtered_callbacks = set()
        super().__init__(overrides=overrides)

    def run_callbacks(self, event):
        # BaseTrainer.__init__ dispatches integration events before returning. Filter
        # at dispatch time so W&B/Comet/Platform cannot initialize another run even
        # during construction. No global SETTINGS or callback registry is modified.
        for callback in self.callbacks.get(event, []):
            module = getattr(callback, "__module__", "")
            if module.startswith("ultralytics.utils.callbacks.") and module != "ultralytics.utils.callbacks.base":
                self.recovery_filtered_callbacks.add(module)
                continue
            callback(self)

    def setup_model(self):
        if self.resume or self.args.resume:
            raise ValueError("Recovery starts a new optimizer with resume=False.")
        if recovery_ordered_names(self.data["names"]) != self.recovery_expected_names:
            raise ValueError("Dataset class order differs from the pruned checkpoint.")
        self.model = self.recovery_source_core
        if recovery_ordered_names(self.model.names) != self.recovery_expected_names:
            raise ValueError("Checkpoint class order differs from recovery configuration.")
        self.model.criterion = None  # Native SDK builds its native criterion for this training run.
        self.model.requires_grad_(True)
        if recovery_structure_signature(self.model) != self.recovery_initial_signature:
            raise ValueError("Recovery loading changed the structurally pruned model.")
        if recovery_state_sha256(self.model) != self.recovery_initial_state_sha256:
            raise ValueError("Recovery loading changed the initial pruned weights.")
        self.recovery_initial_state_verified = True

    def get_model(self, cfg=None, weights=None, verbose=True):
        raise RuntimeError("Recovery must never rebuild the pruned model from its original YAML.")

    def optimizer_step(self):
        if not _recovery_torch.isfinite(self.loss).all().item():
            raise FloatingPointError("Recovery native detection loss became nonfinite.")
        if not all(_recovery_torch.isfinite(value).all().item() for value in self.loss_items.values()):
            raise FloatingPointError("Recovery native detection loss component became nonfinite.")
        super().optimizer_step()
        self.recovery_optimizer_steps += 1

    def _save_recovery_fp32(self, path, model):
        core = _recovery_native_snapshot(model)
        _recovery_torch.save({
            "model": core, "ema": None, "optimizer": None, "recovery_enabled": True,
            "epoch": int(self.epoch), "best_fitness": float(self.best_fitness),
            "train_args": vars(self.args).copy(), "version": _recovery_ultralytics_version,
            "date": _RecoveryDateTime.now(_RECOVERY_UTC).isoformat(),
            "recovery": {"format": "lamp-ordinary-recovery-fp32-v1",
                         "source_sha256": self.recovery_source_sha256,
                         "selection": "native validation fitness; EMA captured before SDK FP16 conversion"},
        }, path)
        return core

    def save_model(self):
        if recovery_structure_signature(self.model) != self.recovery_initial_signature:
            raise ValueError("Fine-tuning changed the pruned parameter shapes or Conv/BN dimensions.")
        if not _recovery_finite_state(self.model) or not _recovery_finite_state(self.ema.ema):
            raise FloatingPointError("Recovery model or EMA became nonfinite; no sanitized checkpoint is accepted.")
        if not _recovery_torch.isfinite(_recovery_torch.tensor(self.fitness)).item():
            raise FloatingPointError("Recovery validation fitness became nonfinite.")
        self.wdir.mkdir(parents=True, exist_ok=True)
        self._save_recovery_fp32(self.wdir / "last_fp32.pt", self.ema.ema)
        if self.best_fitness == self.fitness:
            self.recovery_best_core = self._save_recovery_fp32(self.wdir / "best_fp32.pt", self.ema.ema)
            self.recovery_best_epoch = int(self.epoch)
        row = {"epoch": int(self.epoch) + 1, "fitness": float(self.fitness),
               "losses": {k: float(v.detach().cpu()) for k, v in self.tloss.items()},
               "metrics": {k: float(v) for k, v in self.metrics.items()},
               "optimizer_steps": self.recovery_optimizer_steps,
               "learning_rates": [float(g["lr"]) for g in self.optimizer.param_groups]}
        self.recovery_epochs.append(row)
        _recovery_write_json(self.save_dir / "recovery_epoch_metrics.json", self.recovery_epochs)
        if self.recovery_epoch_callback is not None:
            self.recovery_epoch_callback(self, row)
        # Keep SDK best/last for native training diagnostics. The caller receives
        # the independently captured full-precision EMA, never these FP16 files.
        return super().save_model()


def recovery_training_arguments(checkpoint, data_yaml, output_dir, settings):
    """Explicit optimizer/protocol; callers can change the budget without changing method."""
    required = {"epochs", "optimizer", "lr0", "batch", "imgsz", "device", "seed"}
    if not required <= set(settings):
        raise ValueError(f"Missing recovery settings: {sorted(required - set(settings))}")
    if settings["optimizer"] == "auto":
        raise ValueError("Recovery requires an explicit optimizer so lr0 is not overridden.")
    if isinstance(settings["epochs"], bool) or not isinstance(settings["epochs"], int) or settings["epochs"] < 1:
        raise ValueError("Recovery epochs must be a positive integer.")
    arguments = {"model": str(checkpoint), "data": str(data_yaml), "project": str(_RecoveryPath(output_dir).parent),
                 "name": _RecoveryPath(output_dir).name, "exist_ok": False, "task": "detect", "mode": "train",
                 "resume": False, "pretrained": False, "amp": False, "quantize": None, "compile": False,
                 "distill_model": None, "freeze": None, "val": True, "save": True, "plots": True,
                 "patience": 0, "workers": 2, "cache": False, "deterministic": True, "channels_last": False}
    arguments.update(settings)
    for key, value in {"resume": False, "amp": False, "quantize": None, "compile": False,
                       "distill_model": None, "pretrained": False, "freeze": None}.items():
        if arguments[key] != value:
            raise ValueError(f"Ordinary FP32 recovery requires {key}={value!r}.")
    if not arguments["save"] or not arguments["val"]:
        raise ValueError("Recovery requires validation and checkpoint selection.")
    if not _RecoveryPath(checkpoint).is_file() or not _RecoveryPath(data_yaml).is_file():
        raise FileNotFoundError("Recovery checkpoint and dataset YAML must already exist locally.")
    return arguments


def fine_tune_pruned_checkpoint(checkpoint, data_yaml, output_dir, class_names, settings, epoch_callback=None):
    """Return (native best-EMA FP32 DetectionModel, report) after ordinary supervised fine-tuning."""
    output_dir = _RecoveryPath(output_dir)
    report_path = output_dir.parent / (output_dir.name + "_recovery_training_report.json")
    report = {"status": "RUNNING", "method": "ordinary-supervised-fine-tuning", "resume": False,
              "source_checkpoint": str(_RecoveryPath(checkpoint)), "errors": []}
    trainer = None
    try:
        arguments = recovery_training_arguments(checkpoint, data_yaml, output_dir, settings)
        source_sha256 = _recovery_hashlib.sha256(_RecoveryPath(checkpoint).read_bytes()).hexdigest()
        payload = _recovery_torch.load(checkpoint, map_location="cpu", weights_only=False)
        core = payload.get("ema") or payload.get("model")
        if type(core) is not _RecoveryDetectionModel:
            raise TypeError("Recovery input must contain a native structurally pruned DetectionModel.")
        if any(v.dtype != _recovery_torch.float32 for v in core.parameters() if v.is_floating_point()):
            raise ValueError("Recovery source must be saved in FP32; converting FP16 does not restore precision.")
        if recovery_ordered_names(core.names) != recovery_ordered_names(class_names):
            raise ValueError("Recovery source class order differs from requested classes.")
        report.update({"source_sha256": source_sha256, "arguments": arguments,
                       "initial_structure": recovery_structure_signature(core),
                       "initial_state_sha256": recovery_state_sha256(core)})
        _recovery_write_json(report_path, report)
        trainer = StructuralRecoveryTrainer(core, overrides=arguments, expected_names=class_names,
                                            source_sha256=source_sha256, epoch_callback=epoch_callback)
        trainer.train()
        if not trainer.recovery_initial_state_verified or trainer.recovery_optimizer_steps < 1:
            raise RuntimeError("Recovery did not verify the source or complete an optimizer step.")
        if trainer.recovery_best_core is None:
            raise RuntimeError("Recovery did not select a best validation EMA checkpoint.")
        if recovery_state_sha256(trainer.model, parameters_only=True) == trainer.recovery_initial_parameters_sha256:
            raise RuntimeError("Recovery completed without changing any trainable parameter.")
        recovered = trainer.recovery_best_core
        if recovery_structure_signature(recovered) != report["initial_structure"]:
            raise ValueError("Recovered model does not preserve the original pruned structure.")
        if not _recovery_finite_state(recovered):
            raise FloatingPointError("Recovered best EMA contains nonfinite weights.")
        report.update({"status": "PASS", "initial_state_verified": True, "structure_preserved": True,
                       "parameters_updated": True, "optimizer_steps": trainer.recovery_optimizer_steps,
                       "epochs_completed": len(trainer.recovery_epochs), "epochs": trainer.recovery_epochs,
                       "best_epoch": trainer.recovery_best_epoch + 1,
                       "best_selection": "native validation fitness; full-precision EMA snapshot",
                       "best_fitness": float(trainer.best_fitness),
                       "best_fp32_checkpoint": str(trainer.wdir / "best_fp32.pt"),
                       "last_fp32_checkpoint": str(trainer.wdir / "last_fp32.pt"),
                       "actual_save_dir": str(trainer.save_dir),
                       "filtered_sdk_integrations": sorted(trainer.recovery_filtered_callbacks),
                       "best_state_sha256": recovery_state_sha256(recovered)})
        _recovery_write_json(report_path, report)
        return recovered, report
    except Exception as error:
        report["status"] = "FAIL"
        report["errors"].append({"type": type(error).__name__, "message": str(error)})
        if trainer is not None:
            report.update({"optimizer_steps": trainer.recovery_optimizer_steps,
                           "epochs": trainer.recovery_epochs})
        _recovery_write_json(report_path, report)
        raise
