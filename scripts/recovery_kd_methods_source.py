"""Distillation during recovery of a native structured-pruned DetectionModel."""
# ruff: noqa: F821 -- Notebook declarations share a namespace.
import math
import re
from copy import copy as _kd_copy
from itertools import chain

import torch
from torch import nn
from ultralytics.nn.distill_model import DistillationModel
from ultralytics.nn.modules.head import Detect
from ultralytics.nn.tasks import load_checkpoint

# The existing standalone MGD implementation uses these aliases for diagnostics.
class_names = recovery_ordered_names
LAST_PHASE = {"value": "idle"}

RECOVERY_METHOD_REFERENCES = {
    "native": "https://github.com/ultralytics/ultralytics/blob/v8.4.155/ultralytics/nn/distill_model.py",
    "crosskd": "https://github.com/jbwang1997/CrossKD",
    "mgd": "https://github.com/yzd-v/MGD/tree/2c9da0b28625eb948db57afc02c824452c3910fe",
}


def recovery_kd_settings(method):
    config = {
        "native": {"NATIVE_DIS_WEIGHT": 1.0},
        "crosskd": {"KD_CLS_WEIGHT": 1.0, "KD_BOX_WEIGHT": 1.0, "KD_CLS_GAMMA": 1.0},
        "mgd": {"MGD_ALPHA": 0.00002, "MGD_MASK_RATIO": 0.65},
    }[method]
    for key, default in config.items():
        value = globals().setdefault(key, default)
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0:
            raise ValueError(f"{key} harus berupa angka finite nonnegatif.")
        config[key] = float(value)
    if method == "mgd" and not 0 <= config["MGD_MASK_RATIO"] <= 1:
        raise ValueError("MGD_MASK_RATIO harus antara 0 dan 1.")
    loss_weights = {"native": ("NATIVE_DIS_WEIGHT",), "crosskd": ("KD_CLS_WEIGHT", "KD_BOX_WEIGHT"),
                    "mgd": ("MGD_ALPHA",)}[method]
    if not any(config[key] > 0 for key in loss_weights):
        raise ValueError("Sedikitnya satu bobot loss KD harus positif.")
    return config


def recovery_prepare_method(folder):
    """Resolve a versioned teacher independently of the raw student input."""
    method = RECOVERY_METHOD
    options = recovery_kd_settings(method)
    reference = workflow_version_ref(TEACHER_ARTIFACT_REF) if TEACHER_ARTIFACT_REF else None
    if reference and TEACHER_LOCAL_PATH:
        raise ValueError("Pilih satu sumber teacher: versi W&B atau checkpoint lokal.")
    expected = TEACHER_SHA256
    if not isinstance(expected, str) or not re.fullmatch(r"[0-9a-fA-F]{64}", expected):
        raise ValueError("Isi TEACHER_SHA256 dengan hash SHA-256 checkpoint teacher yang dikunci.")
    if TEACHER_LOCAL_PATH:
        checkpoint = Path(TEACHER_LOCAL_PATH).expanduser().resolve(strict=True)
    elif reference:
        import wandb
        filename = TEACHER_CHECKPOINT_FILENAME
        if Path(filename).name != filename or not filename.endswith(".pt"):
            raise ValueError("TEACHER_CHECKPOINT_FILENAME harus nama file .pt pada root artefak.")
        workflow_login()
        artifact = wandb.Api().artifact(reference)
        folder = Path(folder) / "teacher" / workflow_digest(reference)[:16]
        checkpoint = Path(artifact.get_entry(filename).download(root=str(folder))).resolve(strict=True)
    else:
        raise ValueError("Isi TEACHER_ARTIFACT_REF versi tetap atau TEACHER_LOCAL_PATH.")
    sha = sha256_file(checkpoint)
    if sha != expected.lower():
        raise ValueError("SHA-256 checkpoint teacher berbeda dari konfigurasi.")
    identity = {"method": method, "teacher_architecture": "YOLO26m", "teacher_artifact_ref": reference, "teacher_sha256": sha,
                "teacher_checkpoint_filename": checkpoint.name, "hyperparameters": options,
                "method_reference": RECOVERY_METHOD_REFERENCES[method]}
    return KDStructuralRecoveryTrainer, {"teacher_checkpoint": checkpoint, "method_contract": identity}, identity


def recovery_kd_head(model):
    heads = [layer for layer in model.model if isinstance(layer, Detect)]
    if len(heads) != 1:
        raise ValueError("KD recovery memerlukan satu Detect head YOLO26.")
    head = heads[0]
    if (head.reg_max != 1 or head.nl != 3 or head.stride.tolist() != [8.0, 16.0, 32.0]
            or any(getattr(head, attr, None) is None for attr in ("cv2", "cv3", "one2one_cv2", "one2one_cv3"))):
        raise ValueError("KD recovery memerlukan dual head YOLO26 dengan stride P3/P4/P5 8/16/32 dan reg_max=1.")
    return head


def recovery_kd_teacher_head(model):
    yaml_name = str(model.yaml.get("yaml_file", "")).replace("\\", "/")
    if Path(yaml_name).stem.lower() not in {"yolo26", "yolo26m"} or model.yaml.get("scale") != "m":
        raise ValueError("Teacher recovery dikunci ke arsitektur YOLO26m.")
    head = recovery_kd_head(model)
    channels = [next(layer for layer in branch.modules() if isinstance(layer, nn.Conv2d)).in_channels
                for branch in head.cv2]
    if channels != [256, 512, 512]:
        raise ValueError("Dimensi fisik P3/P4/P5 teacher tidak cocok dengan YOLO26m.")
    return head


class KDStructuralRecoveryTrainer(ResumableStructuralTrainer):
    """Use existing KD losses while training and serializing the exact native student."""

    def __init__(self, raw, *, teacher_checkpoint, method_contract, **kwargs):
        self.recovery_kd_checkpoint = Path(teacher_checkpoint)
        self.recovery_kd_contract = method_contract
        self.recovery_kd_wrapper = None
        super().__init__(raw, **kwargs)

    def _create_kd_wrapper(self, model):
        if sha256_file(self.recovery_kd_checkpoint) != self.recovery_kd_contract["teacher_sha256"]:
            raise ValueError("Checkpoint teacher berubah setelah kontrak resume dikunci.")
        teacher, _ = load_checkpoint(self.recovery_kd_checkpoint, device=self.device, fuse=False)
        if type(teacher) is not _RecoveryDetectionModel:
            raise TypeError("Teacher harus DetectionModel native.")
        teacher.float().eval().requires_grad_(False)
        if recovery_ordered_names(teacher.names) != self.recovery_expected_names:
            raise ValueError("Urutan kelas teacher berbeda dari raw student.")
        if teacher.yaml.get("channels", 3) != model.yaml.get("channels", 3):
            raise ValueError("Jumlah channel citra teacher dan raw student berbeda.")
        student_head, teacher_head = recovery_kd_head(model), recovery_kd_teacher_head(teacher)
        if list(student_head.f) != list(teacher_head.f) or student_head.i != teacher_head.i:
            raise ValueError("Koneksi neck/Detect teacher tidak kompatibel dengan raw student.")
        # Training artifacts can retain the validator's one-to-many inference flag.
        # Both physical branches are verified above; Native KD needs both raw maps.
        if not model.end2end:
            raise ValueError("Raw student tidak aktif dalam mode dual-head training.")
        teacher.end2end = True
        method = self.recovery_kd_contract["method"]
        if method == "native":
            wrapper_cls = DistillationModel
        elif method == "crosskd":
            wrapper_cls = CrossKDModel
        elif method == "mgd":
            wrapper_cls = MGDModel
        else:
            raise ValueError(f"Metode KD tidak dikenal: {method}")
        # Native/CrossKD need only channel dimensions at construction. Do not use
        # a full training batch or mutate the shared trainer args for that probe.
        args = model.args
        model.args = _kd_copy(args)
        model.args.imgsz = 64
        model.args.dis = self.recovery_kd_contract["hyperparameters"].get("NATIVE_DIS_WEIGHT", 1.0)
        training = model.training
        wrapper = None
        try:
            wrapper = wrapper_cls(teacher_model=teacher, student_model=model)
            wrapper._freeze_teacher()
            self.recovery_kd_teacher_state = recovery_state_sha256(teacher)
            return wrapper
        finally:
            model.args = args
            model.train(training)
            if wrapper is not None:
                wrapper._remove_feature_hooks()
                wrapper._teacher_feats.clear()
                wrapper._student_feats.clear()
            else:
                # A failed probe must not leave capture hooks on the raw model.
                for module in model.modules():
                    DistillationModel._clear_feature_hooks(module)

    def build_optimizer(self, model, **kwargs):
        if self.recovery_kd_wrapper is not None:
            raise RuntimeError("Dilarang membangun ulang optimizer atau mengganti batch selama KD.")
        wrapper = self.recovery_kd_wrapper = self._create_kd_wrapper(model)
        # Native optimizer grouping sees student plus training-only projectors.
        # The trainer and its EMA retain the unwrapped student for all inference.
        trainable = nn.ModuleDict({"student_model": model, "projector": wrapper.projector})
        optimizer = super().build_optimizer(trainable, **kwargs)
        optimized = {id(parameter) for group in optimizer.param_groups for parameter in group["params"]}
        teacher = {id(parameter) for parameter in wrapper.teacher_model.parameters()}
        adapters = {id(parameter) for parameter in wrapper.projector.parameters()}
        if optimized & teacher or not adapters <= optimized:
            raise RuntimeError("Grup optimizer KD tidak sesuai student/adapter dan teacher frozen.")
        return optimizer

    def _forward_recovery_batch(self, batch):
        wrapper = self.recovery_kd_wrapper
        if wrapper.student_model is not self.model or not self.model.training:
            raise RuntimeError("KD hanya menerima batch training dari objek raw student aktif.")
        wrapper.train(True)  # SDK wrapper.train keeps the teacher in eval mode.
        try:
            wrapper._register_feature_hooks()
            return wrapper.loss(batch)
        finally:
            wrapper._remove_feature_hooks()
            wrapper._teacher_feats.clear()
            wrapper._student_feats.clear()

    def validate(self):
        # Validation runs the plain student EMA. Skip training-only KD terms
        # when the SDK allocates the validation loss accumulator.
        training_items = self.loss_items
        self.loss_items = {key: value for key, value in training_items.items()
                           if not key.startswith(("kd_", "mgd_"))
                           and key not in {"dis_loss", "det_total_loss", "total_loss"}}
        try:
            return super().validate()
        finally:
            self.loss_items = training_items

    def optimizer_step(self):
        if not torch.isfinite(self.loss).all() or any(not torch.isfinite(v).all() for v in self.loss_items.values()):
            raise FloatingPointError("Detection/KD loss nonfinite.")
        wrapper = self.recovery_kd_wrapper
        if any(p.grad is not None or p.requires_grad for p in wrapper.teacher_model.parameters()):
            raise RuntimeError("Teacher KD menerima gradien atau tidak frozen.")
        self.scaler.unscale_(self.optimizer)
        torch.nn.utils.clip_grad_norm_(chain(self.model.parameters(), wrapper.projector.parameters()), max_norm=10.0)
        self.scaler.step(self.optimizer)
        self.scaler.update()
        self.optimizer.zero_grad()
        self.ema.update(self.model)
        self.recovery_optimizer_steps += 1

    def recovery_extra_state(self):
        wrapper = self.recovery_kd_wrapper
        if recovery_state_sha256(wrapper.teacher_model) != self.recovery_kd_teacher_state:
            raise ValueError("Tensor atau buffer teacher berubah selama KD recovery.")
        return {"method_contract": self.recovery_kd_contract,
                "teacher_state_sha256": self.recovery_kd_teacher_state,
                "projector": {name: value.detach().cpu().clone() for name, value in wrapper.projector.state_dict().items()},
                "projector_gradients": {name: value.grad.detach().cpu().clone()
                                        for name, value in wrapper.projector.named_parameters() if value.grad is not None}}

    def recovery_restore_extra(self, state):
        if (state.get("method_contract") != self.recovery_kd_contract
                or state.get("teacher_state_sha256") != self.recovery_kd_teacher_state):
            raise ValueError("Teacher atau metode berubah saat resume KD recovery.")
        self.recovery_kd_wrapper.projector.load_state_dict(state["projector"], strict=True)

    def _restore_pending_gradients(self):
        super()._restore_pending_gradients()
        state = self.recovery_resume_state
        if state is None:
            return
        parameters = dict(self.recovery_kd_wrapper.projector.named_parameters())
        for name, grad in state["method_state"]["projector_gradients"].items():
            parameter = parameters[name]
            if grad.shape != parameter.shape or grad.dtype != parameter.dtype:
                raise ValueError("Gradien resume tidak sesuai tensor adapter KD.")
            parameter.grad = grad.to(parameter.device).clone()
