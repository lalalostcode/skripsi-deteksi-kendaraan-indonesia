"""FP32 epoch resume for the existing structural recovery trainer (Ultralytics 8.4.155)."""
# ruff: noqa: F821, S102 -- Shared notebook declarations; compile only the installed pinned SDK loop.
import inspect
import random
import re
import textwrap
from copy import copy
from functools import lru_cache

import numpy as np
import torch
from ultralytics.engine import trainer as _sdk_trainer


@lru_cache(maxsize=1)
def recovery_native_loop():
    """Keep SDK loss/scheduler/EMA and patch its local accumulation cursor, not the SDK module."""
    if _recovery_ultralytics_version != "8.4.155":
        raise RuntimeError("Resume struktural memerlukan Ultralytics 8.4.155.")
    source = textwrap.dedent(inspect.getsource(_RecoveryDetectionTrainer._do_train))
    digest = hashlib.sha256(source.encode()).hexdigest()
    oom = re.compile(r"^([ ]+)except RuntimeError as e:.*?(?=^\1if ni - last_opt_step >= self.accumulate:)", re.MULTILINE | re.DOTALL)
    source, count = oom.subn(lambda match: match[1] + "except RuntimeError:\n" + match[1] + "    raise\n", source)
    if count != 1:
        raise RuntimeError("Kontrak loop SDK berubah pada penanganan OOM.")
    markers = {
        "last_opt_step = -1": "last_opt_step = self.recovery_last_opt_step",
        "last_opt_step = ni": "last_opt_step = ni\n                self.recovery_last_opt_step = ni",
        "loss, self.loss_items = self.model(batch)": "loss, self.loss_items = self._forward_recovery_batch(batch)",
        "self.optimizer.zero_grad()  # zero any resumed gradients to ensure stability on train start":
            "self._restore_pending_gradients()",
        '        if RANK in {-1, 0}:\n            if self.loss_names:':
            '        self._restart_epoch_loader(epoch)\n        if RANK in {-1, 0}:\n            if self.loss_names:',
    }
    for marker, replacement in markers.items():
        if source.count(marker) != 1:
            raise RuntimeError(f"Kontrak loop SDK tidak cocok: {marker}")
        source = source.replace(marker, replacement)
    namespace = dict(vars(_sdk_trainer))
    exec(compile(source, "<ultralytics-8.4.155-epoch-resume>", "exec"), namespace)
    return namespace["_do_train"], digest


def recovery_rng(device):
    return {"python": random.getstate(), "numpy": np.random.get_state(), "torch": torch.get_rng_state(),
            "cuda": torch.cuda.get_rng_state(device) if device.type == "cuda" else None}


def recovery_restore_rng(state, device):
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"])
    if state["cuda"] is not None:
        if device.type != "cuda":
            raise ValueError("Resume CUDA memerlukan device CUDA yang sebanding.")
        torch.cuda.set_rng_state(state["cuda"], device)


def recovery_atomic_torch_save(payload, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.stem + "_" + uuid.uuid4().hex + ".pt")
    torch.save(payload, temporary)
    os.replace(temporary, path)


def recovery_committed_state(folder, contract_sha, raw_sha):
    folder = Path(folder)
    index = folder / "latest_state.json"
    if not index.is_file():
        return None, None
    record = json.loads(index.read_text(encoding="utf-8"))
    state_path = folder / record["relative_path"]
    if sha256_file(state_path) != record["sha256"]:
        raise ValueError("SHA-256 checkpoint resume tidak cocok.")
    state = torch.load(state_path, map_location="cpu", weights_only=False)
    if (state.get("format_version") != 2 or state.get("stage") != "recovery_training_state"
            or state.get("contract_sha256") != contract_sha or state.get("raw_sha256") != raw_sha
            or state.get("epoch") != record["epoch"]):
        raise ValueError("Checkpoint resume tidak cocok dengan raw dan kontrak training.")
    best = folder / state["best"]["relative_path"]
    if sha256_file(best) != state["best"]["sha256"]:
        raise ValueError("Best EMA FP32 untuk resume tidak cocok.")
    return state, record


class ResumableStructuralTrainer(StructuralRecoveryTrainer):
    """Use the native pruned object and persist the full live training state per epoch."""

    def __init__(self, raw, *, overrides, expected_names, raw_sha, folder, contract_sha,
                 val_batch, resume_state=None, committed_callback=None):
        self.recovery_folder = Path(folder)
        self.recovery_contract_sha = contract_sha
        self.recovery_resume_state = resume_state
        self.recovery_val_batch = val_batch
        self.recovery_fixed_batch = overrides["batch"]
        self.recovery_last_opt_step = -1
        self.recovery_committed_callback = committed_callback
        self.recovery_best = None
        self.recovery_raw_state = recovery_state_sha256(raw)
        if resume_state is not None:
            if resume_state["raw_state_sha256"] != self.recovery_raw_state:
                raise ValueError("State raw awal berubah sebelum resume recovery.")
            for key in ("live_model", "ema"):
                model = resume_state[key]
                if (type(model) is not _RecoveryDetectionModel or recovery_structure_signature(model) != recovery_structure_signature(raw)
                        or any(p.dtype != torch.float32 for p in model.parameters())):
                    raise ValueError("Struktur atau presisi checkpoint training tidak sesuai raw.")
        core = resume_state["live_model"] if resume_state else raw
        super().__init__(core, overrides=overrides, expected_names=expected_names, source_sha256=raw_sha)

    def get_dataloader(self, dataset_path, batch_size=16, rank=0, mode="train"):
        return super().get_dataloader(dataset_path, self.recovery_val_batch if mode == "val" else batch_size, rank, mode)

    def _build_train_pipeline(self):
        if self.batch_size != self.recovery_fixed_batch or self.args.batch != self.recovery_fixed_batch:
            raise RuntimeError("Batch efektif berubah dari kontrak recovery.")
        super()._build_train_pipeline()
        self.args.max_det = 300
        self.model.set_head_attr(max_det=300)

    def _setup_train(self):
        super()._setup_train()
        state = self.recovery_resume_state
        if not state:
            return
        self.start_epoch = state["epoch"]
        if not 0 < self.start_epoch < self.epochs:
            raise ValueError("Epoch resume di luar budget total.")
        self.optimizer.load_state_dict(state["optimizer"])
        self.scheduler.load_state_dict(state["scheduler"])
        self.scaler.load_state_dict(state["scaler"])
        self.ema.ema.load_state_dict(state["ema"].state_dict(), strict=True)
        self.ema.updates = state["ema_updates"]
        self.ema.ema.end2end = state["ema"].end2end
        self.accumulate = state["accumulate"]
        self.recovery_last_opt_step = state["last_opt_step"]
        self.recovery_optimizer_steps = state["optimizer_steps"]
        self.best_fitness = state["best"]["fitness"]
        self.recovery_best = state["best"]
        self.recovery_best_epoch = self.recovery_best["epoch"] - 1
        self.recovery_epochs = list(state["epoch_rows"])
        self.loss_names = tuple(state["loss_names"])
        self.recovery_restore_criterion(state["criterion"])
        self.recovery_restore_extra(state.get("method_state", {}))
        if self.args.close_mosaic and self.start_epoch > self.epochs - self.args.close_mosaic:
            self._close_dataloader_mosaic()
        if state.get("loader_generator") is not None:
            self.train_loader.generator.set_state(state["loader_generator"])
        recovery_restore_rng(state["rng"], self.device)

    def recovery_criterion_state(self):
        return {name: getattr(self.model.criterion, name) for name in ("updates", "o2m", "o2o")
                if hasattr(self.model.criterion, name)}

    def recovery_restore_criterion(self, state):
        self.model.criterion = self.model.init_criterion()
        for key, value in state.items():
            setattr(self.model.criterion, key, value)

    def recovery_extra_state(self):
        return {}

    def _forward_recovery_batch(self, batch):
        return self.model(batch)

    def recovery_restore_extra(self, state):
        if state:
            raise ValueError("Checkpoint memuat state metode lain.")

    def _restart_epoch_loader(self, epoch):
        # Reset the augmentation buffer and workers at each epoch boundary.
        # This makes sampling independent of the previous iterator's prefetch.
        dataset = self.train_loader.dataset
        for index in dataset.buffer:
            dataset.ims[index] = dataset.im_hw0[index] = dataset.im_hw[index] = None
        dataset.buffer.clear()
        if self.train_loader.num_workers == 0:
            # No worker fork isolates optional Albumentations' internal RNG here.
            if self.args.close_mosaic and epoch >= self.epochs - self.args.close_mosaic:
                dataset.close_mosaic(hyp=copy(self.args))
            else:
                dataset.transforms = dataset.build_transforms(hyp=copy(self.args))
        self.train_loader.generator.manual_seed((6148914691236517205 + self.args.seed + epoch) % (1 << 64))
        self.train_loader.reset()

    def _restore_pending_gradients(self):
        self.optimizer.zero_grad()
        state = self.recovery_resume_state
        if state:
            parameters = dict(self.model.named_parameters())
            for name, grad in state["gradients"].items():
                parameter = parameters[name]
                if grad.shape != parameter.shape or grad.dtype != parameter.dtype:
                    raise ValueError("Gradien resume tidak sesuai tensor raw.")
                parameter.grad = grad.to(parameter.device).clone()

    def _do_train(self):
        method, _ = recovery_native_loop()
        return method(self)

    def _handle_nan_recovery(self, epoch):
        if not torch.isfinite(self.loss).all() or not math.isfinite(self.fitness):
            raise FloatingPointError(f"Loss atau fitness nonfinite pada epoch {epoch + 1}.")
        return False

    def validate(self):
        with gflops_require_complete_nms():
            result = super().validate()
        if (self.validator.end2end or self.validator.args.quantize is not None or self.validator.args.nms is not None
                or self.test_loader.batch_size != min(self.recovery_val_batch, len(self.test_loader.dataset))):
            raise ValueError("Validator per epoch tidak memakai protokol FP32 one-to-many/batch yang dikunci.")
        return result

    def final_eval(self):
        # Notebook measures three immutable FP32 stages using the same evaluator.
        return None

    def save_model(self):
        if (recovery_structure_signature(self.model) != self.recovery_initial_signature
                or not _recovery_finite_state(self.model) or not _recovery_finite_state(self.ema.ema)):
            raise ValueError("Training mengubah struktur raw atau menghasilkan state nonfinite.")
        epoch = int(self.epoch) + 1
        folder = self.recovery_folder
        if (self.recovery_best is None or self.fitness > self.recovery_best["fitness"]
                or (not self.recovery_best.get("optimizer_steps") and self.recovery_optimizer_steps)):
            best_path = folder / "checkpoints" / f"best-{epoch:03d}-{uuid.uuid4().hex[:8]}.pt"
            best_path.parent.mkdir(parents=True, exist_ok=True)
            inference = _recovery_native_snapshot(self.ema.ema)
            inference.end2end = True
            self._save_recovery_fp32(best_path, inference)
            self.recovery_best = {"epoch": epoch, "fitness": float(self.fitness), "sha256": sha256_file(best_path),
                                  "optimizer_steps": self.recovery_optimizer_steps,
                                  "relative_path": best_path.relative_to(folder).as_posix()}
        row = {"epoch": epoch, "fitness": float(self.fitness), "optimizer_steps": self.recovery_optimizer_steps,
               "losses": {name: float(value.detach().cpu()) for name, value in self.tloss.items()},
               "metrics": {name: float(value) for name, value in self.metrics.items()},
               "learning_rates": [float(group["lr"]) for group in self.optimizer.param_groups]}
        self.recovery_epochs.append(row)
        rng = recovery_rng(self.device)
        identity_path = folder / "wandb_identity.json"
        state = {"format_version": 2, "stage": "recovery_training_state", "epoch": epoch,
                 "epochs_total": self.epochs, "raw_sha256": self.recovery_source_sha256,
                 "raw_state_sha256": self.recovery_raw_state, "contract_sha256": self.recovery_contract_sha,
                 "live_model": _recovery_native_snapshot(self.model), "ema": _recovery_native_snapshot(self.ema.ema),
                 "ema_updates": self.ema.updates, "optimizer": self.optimizer.state_dict(),
                 "scheduler": self.scheduler.state_dict(), "scaler": self.scaler.state_dict(),
                 "optimizer_steps": self.recovery_optimizer_steps, "last_opt_step": self.recovery_last_opt_step,
                 "accumulate": self.accumulate, "gradients": {name: value.grad.detach().cpu().clone()
                     for name, value in self.model.named_parameters() if value.grad is not None},
                 "criterion": self.recovery_criterion_state(), "method_state": self.recovery_extra_state(),
                 "rng": rng, "loader_generator": self.train_loader.generator.get_state(),
                 "sampler": {"completed_epoch": epoch, "policy": "seeded shuffle and fresh workers/mosaic buffer per epoch"},
                 "loss_names": list(self.loss_names), "epoch_rows": self.recovery_epochs, "best": self.recovery_best,
                 "wandb_identity": json.loads(identity_path.read_text(encoding="utf-8")) if identity_path.is_file() else None}
        path = folder / "checkpoints" / f"resume-{epoch:03d}-{uuid.uuid4().hex[:8]}.pt"
        recovery_atomic_torch_save(state, path)
        index = {"epoch": epoch, "relative_path": path.relative_to(folder).as_posix(), "sha256": sha256_file(path),
                 "best": self.recovery_best, "optimizer_steps": self.recovery_optimizer_steps}
        write_json(folder / "epoch_metrics.json", self.recovery_epochs)
        write_json(folder / "latest_state.json", index)
        if self.recovery_committed_callback:
            self.recovery_committed_callback(index, row)
        recovery_restore_rng(rng, self.device)
        return True


def recovery_fit(raw, raw_path, raw_sha, data_yaml, names, folder, settings, contract_sha, *,
                 val_batch, committed_callback=None, trainer_class=ResumableStructuralTrainer, trainer_kwargs=None):
    """Resume only from a committed epoch; retry that publication before another step."""
    folder = Path(folder)
    state, index = recovery_committed_state(folder, contract_sha, raw_sha)
    if state and committed_callback:
        write_json(folder / "epoch_metrics.json", state["epoch_rows"])
        committed_callback(index, state["epoch_rows"][-1])
    if state and state["epoch"] == settings["epochs"]:
        if (state["optimizer_steps"] < 1 or recovery_state_sha256(state["live_model"], parameters_only=True)
                == recovery_state_sha256(raw, parameters_only=True)):
            raise RuntimeError("Training selesai tanpa perubahan bobot raw.")
        return state, index
    attempt = folder / "attempts" / uuid.uuid4().hex[:10]
    arguments = recovery_training_arguments(raw_path, data_yaml, attempt, settings)
    arguments.update(conf=0.001, iou=0.7, max_det=300, nms=None)
    write_json(folder / "training_config.json", {"settings": settings, "validation_batch": val_batch,
               "sdk_loop_sha256": recovery_native_loop()[1], "raw_sha256": raw_sha})
    trainer = trainer_class(raw, overrides=arguments, expected_names=names, raw_sha=raw_sha, folder=folder,
                            contract_sha=contract_sha, val_batch=val_batch, resume_state=state,
                            committed_callback=committed_callback, **(trainer_kwargs or {}))
    with gflops_disable_label_cache():
        trainer.train()
    state, index = recovery_committed_state(folder, contract_sha, raw_sha)
    if (not trainer.recovery_initial_state_verified or not state or state["epoch"] != settings["epochs"]
            or state["optimizer_steps"] < 1 or recovery_state_sha256(state["live_model"], parameters_only=True)
            == recovery_state_sha256(raw, parameters_only=True)):
        raise RuntimeError("Training belum menyelesaikan budget atau belum mengubah bobot raw.")
    return state, index
