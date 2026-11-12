"""Standalone notebook cells for detection MGD, following yzd-v/MGD's pinned source."""

MGD_FEATURES_SOURCE = '''import torch
from torch import nn
from ultralytics.nn.modules.head import Detect


def neck_taps(model: nn.Module) -> list[int]:
    """Discover every final neck feature supplied to the three-scale YOLO26 Detect."""
    heads = [layer for layer in model.model if isinstance(layer, Detect)]
    if len(heads) != 1 or not isinstance(heads[0].f, (list, tuple)) or len(heads[0].f) != 3:
        raise ValueError("Model harus memiliki satu Detect dengan tiga input neck.")
    head = heads[0]
    if [float(value) for value in head.stride] != [8.0, 16.0, 32.0]:
        raise ValueError("Urutan stride harus P3/P4/P5 = 8/16/32.")
    indices = list(head.f)
    if any(not isinstance(index, int) or not 0 <= index < head.i for index in indices):
        raise ValueError("Koneksi neck Detect tidak kompatibel.")
    return indices


def teacher_neck_forward(model: nn.Module, image: torch.Tensor, last_index: int) -> None:
    """Traverse the teacher backbone/neck once without executing its Detect head."""
    saved, value = [], image
    for layer in model.model[:last_index + 1]:
        if layer.f != -1:
            value = (saved[layer.f] if isinstance(layer.f, int)
                     else [value if index == -1 else saved[index] for index in layer.f])
        value = layer(value)
        saved.append(value if layer.i in model.save else None)
'''

MGD_LOSS_SOURCE = '''import math
from pathlib import Path
from typing import Any

import torch
from torch import nn
from torch.nn import functional as F
from ultralytics.nn.distill_model import DistillationModel
from ultralytics.nn.tasks import load_checkpoint
from ultralytics.utils.torch_utils import copy_attr


class MGDFeatureLoss(nn.Module):
    """Detection MGD: biased alignment, spatial mask, full generator, SSE/B.

    Reference: yzd-v/MGD commit 2c9da0b28625eb948db57afc02c824452c3910fe,
    det/mmdet/distillation/losses/mgd.py (not the channel-mask classification variant).
    FP32 arithmetic and explicit teacher detach preserve its mathematical objective
    while avoiding AMP overflow and accidental teacher gradients in this adapter.
    """
    def __init__(self, student_channels, teacher_channels, name,
                 alpha_mgd=0.00002, lambda_mgd=0.65):
        super().__init__()
        if not math.isfinite(alpha_mgd) or alpha_mgd < 0:
            raise ValueError("MGD_ALPHA harus finite dan nonnegative.")
        if not math.isfinite(lambda_mgd) or not 0 <= lambda_mgd <= 1:
            raise ValueError("MGD_MASK_RATIO harus finite di antara 0 dan 1.")
        if student_channels < 1 or teacher_channels < 1:
            raise ValueError("Jumlah channel fitur MGD harus positif.")
        self.alpha_mgd = float(alpha_mgd)
        self.lambda_mgd = float(lambda_mgd)
        self.name = name
        self.align = (nn.Conv2d(student_channels, teacher_channels, kernel_size=1,
                                stride=1, padding=0, bias=True)
                      if student_channels != teacher_channels else None)
        self.generation = nn.Sequential(
            nn.Conv2d(teacher_channels, teacher_channels, kernel_size=3, padding=1, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(teacher_channels, teacher_channels, kernel_size=3, padding=1, bias=True))
        self.last_mask_rate = 0.0
        self.last_raw_loss = torch.zeros(())
        self.last_aligned_shape = []
        self.last_generated_shape = []

    def forward(self, preds_S, preds_T):
        if preds_S.ndim != 4 or preds_T.ndim != 4 or preds_S.shape[0] != preds_T.shape[0]:
            raise ValueError("MGD memerlukan pasangan fitur BCHW dengan batch yang sama.")
        if preds_S.shape[-2:] != preds_T.shape[-2:]:
            raise ValueError("Grid spasial MGD harus sama; interpolasi tidak diaktifkan.")
        with torch.autocast(device_type=preds_S.device.type, enabled=False):
            student = preds_S.float()
            if self.align is not None:
                student = self.align(student)
            self.last_aligned_shape = list(student.shape)
            raw = self.get_dis_loss(student, preds_T.detach().float())
            return raw * self.alpha_mgd

    def get_dis_loss(self, preds_S, preds_T):
        if preds_S.ndim != 4 or preds_S.shape != preds_T.shape or preds_T.shape[0] < 1:
            raise ValueError("Fitur MGD harus identik BCHW setelah alignment dan batch tidak kosong.")
        with torch.autocast(device_type=preds_S.device.type, enabled=False):
            student, teacher = preds_S.float(), preds_T.detach().float()
            batch, _, height, width = teacher.shape
            # Keep the official detection implementation's CPU RNG and strict threshold.
            mat = torch.rand((batch, 1, height, width)).to(student.device)
            mat = torch.where(mat > 1 - self.lambda_mgd, 0, 1).to(student.device)
            masked_feature = torch.mul(student, mat)  # Never mutate student neck features.
            reconstructed = self.generation(masked_feature)
            raw = F.mse_loss(reconstructed, teacher, reduction="sum") / batch
        if not torch.isfinite(raw):
            raise RuntimeError("MGD reconstruction loss non-finite; proses dihentikan.")
        self.last_mask_rate = float((mat == 0).float().mean().detach())
        self.last_raw_loss = raw.detach()
        self.last_generated_shape = list(reconstructed.shape)
        return raw


class MGDModel(DistillationModel):
    """Native YOLO26 detection plus detection-MGD on all three final neck maps."""
    def __init__(self, teacher_model: str | Path | nn.Module, student_model: nn.Module):
        # Avoid allocating/calling Native KD's score-weighted projector or Detect hooks.
        nn.Module.__init__(self)
        if isinstance(teacher_model, (str, Path)):
            teacher_model = load_checkpoint(teacher_model)[0]
        device = next(student_model.parameters()).device
        self.teacher_model = teacher_model.to(device).float()
        self.student_model = student_model
        self._freeze_teacher()
        if class_names(teacher_model.names) != class_names(student_model.names):
            raise ValueError("Urutan label teacher dan student aktual berbeda.")
        self.feats_idx = neck_taps(student_model)
        if neck_taps(teacher_model) != self.feats_idx:
            raise ValueError("Koneksi neck teacher/student berbeda; diperlukan graph YOLO26 yang sama.")
        self._teacher_feats, self._student_feats = {}, {}
        self._teacher_hooks, self._student_hooks = [], []
        self._register_feature_hooks()
        self.last_actual_batch = 0
        self.last_feature_shapes = {}
        previous_training = student_model.training
        try:
            student_model.eval()
            with torch.no_grad():
                image = torch.zeros(2, student_model.yaml.get("channels", 3), 64, 64, device=device)
                teacher_neck_forward(teacher_model, image, max(self.feats_idx))
                student_model(image)
            projectors, layout = [], []
            for name, index in zip(("P3", "P4", "P5"), self.feats_idx):
                student, teacher = self._student_feats[index], self._teacher_feats[index]
                if student.shape[-2:] != teacher.shape[-2:]:
                    raise ValueError("Ukuran spasial neck berbeda; tidak ada interpolasi diam-diam.")
                cin, cout = student.shape[1], teacher.shape[1]
                projectors.append(MGDFeatureLoss(cin, cout, name,
                                                alpha_mgd=MGD_ALPHA, lambda_mgd=MGD_MASK_RATIO))
                layout.append({"level": name, "index": index, "student_channels": cin,
                               "teacher_channels": cout,
                               "projection": "identity" if cin == cout else "conv1x1-bias",
                               "generation": "conv3x3-relu-conv3x3"})
            self.projector = nn.ModuleList(projectors).to(device)
            self.feature_layout = layout
            copy_attr(self, student_model)
            self.dis = 1.0  # Engine compatibility only; no Native KD loss is invoked.
        except BaseException:
            self._remove_feature_hooks()
            raise
        finally:
            self._teacher_feats.clear()
            self._student_feats.clear()
            student_model.train(previous_training)
            self._freeze_teacher()

    def forward(self, value: torch.Tensor | dict, *args: Any, **kwargs: Any):
        if isinstance(value, dict):
            return self.loss(value, *args, **kwargs)
        try:
            return self.student_model.predict(value, *args, **kwargs)
        finally:
            self._teacher_feats.clear()
            self._student_feats.clear()

    def loss(self, batch: dict, preds: Any = None):
        self._teacher_feats.clear()
        self._student_feats.clear()
        try:
            if self.training:
                self._freeze_teacher()
                self.last_actual_batch = int(batch["img"].shape[0])
                self.last_feature_shapes = {}
                LAST_PHASE["value"] = "teacher_forward"
                with torch.no_grad():
                    teacher_neck_forward(self.teacher_model, batch["img"], max(self.feats_idx))
                LAST_PHASE["value"] = "student_forward"
                preds = self.student_model(batch["img"])
            LAST_PHASE["value"] = "detection_loss"
            detection, items = self.student_model.loss(batch, preds)
            parts, weighted_parts, mask_rates = {}, {}, {}
            if self.training:
                LAST_PHASE["value"] = "mgd_forward"
                for position, (level, index) in enumerate(zip(("P3", "P4", "P5"), self.feats_idx)):
                    module = self.projector[position]
                    student, teacher = self._student_feats[index], self._teacher_feats[index]
                    weighted_parts[level] = module(student, teacher)
                    parts[level] = module.last_raw_loss
                    mask_rates[level] = module.last_mask_rate
                    self.last_feature_shapes[level] = {
                        "student": list(student.shape), "teacher": list(teacher.shape),
                        "aligned": module.last_aligned_shape, "generated": module.last_generated_shape}
            else:
                parts = {level: detection.new_zeros(()) for level in ("P3", "P4", "P5")}
                weighted_parts = dict(parts)
                mask_rates = {level: 0.0 for level in parts}
            raw, weighted = sum(parts.values()), sum(weighted_parts.values())
            if not torch.isfinite(detection).all() or not torch.isfinite(weighted):
                raise RuntimeError("Detection/MGD loss non-finite; proses dihentikan.")
            batch_size = int(batch["img"].shape[0])
            per_image_detection = detection.detach().sum() / batch_size
            items.update({f"mgd_{level}": value.detach() for level, value in parts.items()})
            items.update({f"mgd_mask_{level}": detection.new_tensor(rate) for level, rate in mask_rates.items()})
            items.update(mgd_loss=raw.detach(), mgd_weighted_loss=weighted.detach(),
                         det_total_loss=per_image_detection,
                         total_loss=per_image_detection + weighted.detach(),
                         mgd_to_det_ratio=weighted.detach() / per_image_detection.abs().clamp_min(1e-12))
            LAST_PHASE["value"] = "backward"
            # Reference raw MGD is SSE/B; apply alpha and the native batch factor once each.
            return torch.cat((detection.reshape(-1), (weighted * batch_size).reshape(1))), items
        finally:
            self._teacher_feats.clear()
            self._student_feats.clear()
'''
