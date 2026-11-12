"""Use the source KD wrapper with the verified pruned object and native trainer."""
# ruff: noqa: F401, F821 -- Imports and declarations are shared across notebook cells.
from torch import nn
from ultralytics.nn.distill_model import DistillationModel
from ultralytics.nn.tasks import load_checkpoint

class_names = recovery_ordered_names
LAST_PHASE = {"value": "idle"}  # MGD diagnostics.


def source_matched_check_teacher(trainer, contract, *, check_digest=False):
    wrapper = trainer.model
    if not isinstance(wrapper, DistillationModel):
        raise TypeError("Trainer tidak memakai wrapper KD asal.")
    teacher = wrapper.teacher_model
    if teacher.training or any(value.requires_grad or value.grad is not None for value in teacher.parameters()):
        raise RuntimeError("Teacher harus tetap eval tanpa gradien.")
    if check_digest:
        optimizer_ids = {id(value) for group in trainer.optimizer.param_groups for value in group["params"]}
        if (recovery_state_sha256(teacher) != contract["state_sha256"]
                or any(id(value) in optimizer_ids for value in teacher.parameters())
                or any(id(value) not in optimizer_ids for value in wrapper.projector.parameters())):
            raise RuntimeError("State teacher, pemisahan optimizer teacher, atau adapter KD tidak cocok.")


def source_matched_teacher(folder, raw):
    if bool(TEACHER_ARTIFACT_REF) == bool(TEACHER_LOCAL_PATH):
        raise ValueError("Isi tepat satu sumber teacher YOLO26m v50.")
    reference = None
    if TEACHER_ARTIFACT_REF:
        reference = workflow_version_ref(TEACHER_ARTIFACT_REF)
        if reference != LOCKED_TEACHER_REF:
            raise ValueError("Teacher penelitian harus versi YOLO26m v50 yang dikunci.")
        workflow_login()
        artifact = wandb.Api().artifact(reference)
        path = Path(artifact.get_entry("best.pt").download(root=str(folder / "teacher"))).resolve(strict=True)
    else:
        path = Path(TEACHER_LOCAL_PATH).expanduser().resolve(strict=True)
    digest = sha256_file(path)
    expected = LOCKED_TEACHER_SHA256 if MODE == "full_val" else TEACHER_SHA256
    if digest != expected or digest != TEACHER_SHA256:
        raise ValueError("SHA-256 teacher YOLO26m berbeda dari referensi yang dikunci.")
    teacher, _ = load_checkpoint(path, fuse=False)
    names = recovery_ordered_names(raw.names)
    fine_tuned_check_heads(teacher, names)
    head = recovery_kd_teacher_head(teacher)
    student_head = recovery_kd_head(raw)
    if (list(head.f) != list(student_head.f) or head.i != student_head.i
            or teacher.yaml.get("channels", 3) != raw.yaml.get("channels", 3) or not _recovery_finite_state(teacher)):
        raise ValueError("Graph Detect atau tensor teacher tidak kompatibel dengan student raw.")
    hyperparameters = {
        "native": lambda: {"dis": KD_WEIGHT},
        "crosskd": lambda: {"cls_weight": KD_CLS_WEIGHT, "box_weight": KD_BOX_WEIGHT, "gamma": KD_CLS_GAMMA},
        "mgd": lambda: {"alpha": MGD_ALPHA, "mask_ratio": MGD_MASK_RATIO, "levels": ["P3", "P4", "P5"],
                        "reconstruction_precision": "FP32"},
    }[RECOVERY_METHOD]()
    return path, {"reference": reference or (LOCKED_TEACHER_REF if digest == LOCKED_TEACHER_SHA256 else None),
                  "architecture": "YOLO26m", "sha256": digest, "state_sha256": recovery_state_sha256(teacher),
                  "provenance": "VERIFIED" if digest == LOCKED_TEACHER_SHA256 else "DIAGNOSTIC_FIXTURE",
                  "hyperparameters": hyperparameters,
                  "adaptation": "Teacher YOLO26m, student YOLO26s pruned" if RECOVERY_METHOD == "mgd" else None}


class SourceMatchedKDTrainer(BaselinePrunedTrainer):
    def set_model_attributes(self):
        super().set_model_attributes()
        # Native KD is installed by the SDK after its AMP check, as in its source notebook.
        if RECOVERY_METHOD == "crosskd":
            self.model = CrossKDModel(self.args.distill_model, self.model)
        elif RECOVERY_METHOD == "mgd":
            self.model = MGDModel(self.args.distill_model, self.model)

    def build_optimizer(self, model, **kwargs):
        # Reuse native AdamW groups while excluding the frozen teacher tensors.
        trainable = torch.nn.ModuleDict({"student_model": model.student_model, "projector": model.projector})
        return super().build_optimizer(trainable, **kwargs)

    def save_model(self):
        live, ema = self.model, self.ema.ema
        for wrapper in (live, ema):
            wrapper._remove_feature_hooks()
            wrapper._student_feats.clear()
            wrapper._teacher_feats.clear()
        try:
            # Native save/fitness selection with the plain student in the model slot.
            self.model, self.ema.ema = live.student_model, ema.student_model
            return super().save_model()
        finally:
            self.model, self.ema.ema = live, ema
            for wrapper in (live, ema):
                wrapper._register_feature_hooks()
