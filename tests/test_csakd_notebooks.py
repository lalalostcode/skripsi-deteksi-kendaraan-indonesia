"""Offline CSAKD regressions extracted from the actual notebook definitions.

No installation, W&B authentication, dataset download, pretrained-weight download,
or research training cell is executed. The slow tests run one diagnostic epoch on
synthetic train/val images and random weights, never Poltrada or the teacher v50.
"""

from __future__ import annotations

import ast
import copy
import gc
import hashlib
import json
import math
import os
import shutil
import subprocess
import sys
import time
import types
import uuid
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from PIL import Image
from torch.nn import functional as F
from ultralytics import YOLO
from ultralytics.cfg import get_cfg
from ultralytics.nn.tasks import DetectionModel
from ultralytics.utils.torch_utils import ModelEMA

ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK_DIR = ROOT / "notebooks/yolo26-kd-indotraffic/notebooks"
NOTEBOOKS = [NOTEBOOK_DIR / f"{prefix}_csakd_yolo26{size}.ipynb" for prefix, size in (("08", "s"), ("09", "n"))]
NOTEBOOKS.append(NOTEBOOK_DIR / "12_csakd_yolo26n_teacher_yolo26s.ipynb")
TAPS = [4, 6, 10, 16, 19, 22]


@pytest.fixture
def tmp_path():
    # Pytest's Windows 0700 temporary directories are inaccessible here.
    directory = ROOT / "outputs/test_csakd" / uuid.uuid4().hex
    directory.mkdir(parents=True)
    return directory


@pytest.fixture(scope="session", autouse=True)
def cpu_threads():
    old = torch.get_num_threads()
    torch.set_num_threads(2)
    yield
    torch.set_num_threads(old)


def definitions(path: Path, cell_indices=(11, 12)):
    """Compile only imports/classes/functions; exclude every notebook action."""
    notebook = json.loads(path.read_text(encoding="utf-8"))
    nodes = []
    for index in cell_indices:
        tree = ast.parse("".join(notebook["cells"][index]["source"]))
        nodes.extend(node for node in tree.body if isinstance(node, (ast.Import, ast.ImportFrom, ast.FunctionDef, ast.ClassDef)))
    return ast.Module(body=nodes, type_ignores=[])


def notebook_config(path):
    notebook = json.loads(path.read_text(encoding="utf-8"))
    namespace = {}
    # Config also derives tracking tags from the configured seed and KD weights.
    exec(compile("".join(notebook["cells"][2]["source"]), str(path), "exec"), namespace)  # noqa: S102
    return {key: value for key, value in namespace.items() if not key.startswith("__")}


@pytest.fixture(scope="module")
def notebook_module():
    name = "_csakd_notebook_test_definitions"
    module = types.ModuleType(name)
    module.KD_AGKA_WEIGHT = module.KD_CROSS_WEIGHT = 1.0
    module.KD_CHECKPOINT, module.KD_CHUNK_SIZE = True, 2
    sys.modules[name] = module  # Required to serialize actual notebook-defined classes.
    exec(compile(definitions(NOTEBOOKS[0]), str(NOTEBOOKS[0]), "exec"), module.__dict__)  # noqa: S102 -- Definition AST only.
    yield module
    sys.modules.pop(name, None)


def network(size, imgsz=64):
    model = DetectionModel(f"yolo26{size}.yaml", nc=6, verbose=False)
    model.args = get_cfg(overrides={"imgsz": imgsz, "dis": 1.0, "amp": False, "device": "cpu"})
    model.names = {i: f"class_{i}" for i in range(6)}
    return model


def wrapper(module, size="n", teacher_size="m"):
    result = module.CSAKDModel(network(teacher_size), network(size))
    result.train()
    return result


def batch(batch_size=2, imgsz=64):
    return {"img": torch.rand(batch_size, 3, imgsz, imgsz), "batch_idx": torch.arange(batch_size),
            "cls": torch.zeros(batch_size, 1), "bboxes": torch.tensor([[0.5, 0.5, 0.25, 0.25]] * batch_size)}


def tensors(value):
    if isinstance(value, torch.Tensor):
        yield value
    elif isinstance(value, dict):
        for item in value.values():
            yield from tensors(item)
    elif isinstance(value, (list, tuple)):
        for item in value:
            yield from tensors(item)


def assert_gradients(parameters, *, nonzero=True):
    values = [p.grad for p in parameters if p.requires_grad and p.grad is not None]
    assert values, "No gradient reached this trainable subsystem"
    assert all(torch.isfinite(g).all() for g in values)
    if nonzero:
        assert any(torch.count_nonzero(g).item() > 0 for g in values)


def assert_every_gradient(parameters):
    for parameter in parameters:
        if parameter.requires_grad:
            assert parameter.grad is not None
            assert torch.isfinite(parameter.grad).all()
            assert torch.count_nonzero(parameter.grad).item() > 0


def populate_features(model, images):
    model._teacher_feats.clear()
    model._student_feats.clear()
    with torch.no_grad():
        model.teacher_model(images)
    model.student_model(images)


def test_all_csakd_notebooks_have_identical_method_and_research_protocol():
    reference = ast.dump(definitions(NOTEBOOKS[0]), include_attributes=False)
    assert all(ast.dump(definitions(path), include_attributes=False) == reference for path in NOTEBOOKS[1:])
    for path in NOTEBOOKS:
        config = notebook_config(path)
        assert (config["EPOCHS"], config["IMGSZ"], config["BATCH"], config["SEED"], config["OPTIMIZER"], config["LR"], config["AMP"]) == (100, 640, 16, 42, "AdamW", 0.001, True)
        assert config["KD_AGKA_WEIGHT"] == config["KD_CROSS_WEIGHT"] == 1.0
        assert config["KD_CHECKPOINT"] is True and config["KD_CHUNK_SIZE"] == 2
        assert config["EVAL_SPLIT"] == "val"
        assert config["DATA"] == "/kaggle/input/datasets/ilhamrafiqin/poltrada-v3-yolo26-clean/data.yaml"


def official_oracle(s, t, c):
    # Independent transcription of OtherKD/CSAKD.py at audited author commit.
    s, t, c = s.float(), t.detach().float(), c.float()
    scores = torch.matmul(F.relu(c), F.relu(t).transpose(-2, -1)) / math.sqrt(s.size(1))
    attention = F.softmax(scores, dim=-1)
    guided = (s + torch.matmul(attention, s)) / 2
    norm_s = F.normalize(s, p=2, dim=1, eps=1e-12)
    norm_t = F.normalize(t, p=2, dim=1, eps=1e-12)
    norm_c = F.normalize(c, p=2, dim=1, eps=1e-12)
    agka = F.mse_loss(F.normalize(guided, p=2, dim=1, eps=1e-12), norm_t, reduction="mean")
    pooled = (F.adaptive_avg_pool2d(t, 1).flatten(1) * F.adaptive_avg_pool2d(s, 1).flatten(1)).sum(1) / math.sqrt(s.size(1))
    cross = (1 - torch.sigmoid(pooled.mean())) * F.mse_loss(norm_s, norm_c, reduction="mean")
    return agka, cross, attention


@pytest.mark.parametrize("scale", [1.0, 0.0, 1e-14])
def test_official_formula_values_and_gradients(notebook_module, scale):
    torch.manual_seed(42)
    s = (torch.randn(2, 5, 3, 4) * scale).requires_grad_()
    t = (torch.randn_like(s) * scale).requires_grad_()
    c = (torch.randn_like(s) * scale).requires_grad_()
    actual = notebook_module.csakd_losses(s, t, c)
    expected = official_oracle(s, t, c)
    assert expected[2].shape == (2, 5, 3, 3)
    for actual_loss, expected_loss in zip(actual, expected[:2]):
        torch.testing.assert_close(actual_loss, expected_loss)
        actual_grads = torch.autograd.grad(actual_loss, (s, c), retain_graph=True)
        oracle_grads = torch.autograd.grad(expected_loss, (s, c), retain_graph=True)
        for gradient, oracle_gradient in zip(actual_grads, oracle_grads):
            assert torch.isfinite(gradient).all()
            torch.testing.assert_close(gradient, oracle_gradient)
        assert torch.autograd.grad(actual_loss, t, allow_unused=True, retain_graph=True)[0] is None
    if scale == 0:
        assert actual[0].item() == actual[1].item() == 0  # Intended zero gradients for identical zero features.
    else:
        assert all(loss.item() > 0 for loss in actual)


def test_shape_mismatch_rejected(notebook_module):
    with pytest.raises(ValueError, match="seukuran"):
        notebook_module.csakd_losses(torch.zeros(2, 3, 4, 4), torch.zeros(2, 4, 4, 4), torch.zeros(2, 3, 4, 4))


def test_original_teacher_targets_survive_cross_hooks_and_affect_loss(notebook_module, monkeypatch):
    model = wrapper(notebook_module)
    try:
        populate_features(model, batch()["img"])
        expected = {i: model._teacher_feats[i].detach().clone() for i in TAPS}
        original_ptrs = {i: model._teacher_feats[i].data_ptr() for i in TAPS}
        observed = []
        original_loss = notebook_module.csakd_loss_terms

        def record(s, t, c):
            observed.append(t.detach().clone())
            assert not t.requires_grad
            return original_loss(s, t, c)

        monkeypatch.setattr(notebook_module, "csakd_loss_terms", record)
        targets = model.snapshot_teacher_targets()
        assert model._teacher_feats == {}
        for index in TAPS:
            assert targets[index].data_ptr() != original_ptrs[index]
        first = model.kd_loss(targets)
        for index, target in zip(TAPS, observed):
            torch.testing.assert_close(target, expected[index], rtol=0, atol=0)
        assert model._teacher_feats == {}, "Cross-forward must suppress teacher feature capture"
        # Same student features and cross modules, different ORIGINAL teacher targets.
        shifted = {index: expected[index] + 0.25 for index in TAPS}
        second = model.kd_loss(shifted)
        assert not torch.allclose(first["kd_agka_loss"], second["kd_agka_loss"], rtol=1e-5, atol=1e-8)
    finally:
        model._remove_feature_hooks()


@pytest.mark.parametrize(("teacher_size", "size"), [("m", "s"), ("m", "n"), ("s", "n")])
def test_yolo26_shapes_640_and_isolated_gradients(notebook_module, teacher_size, size):
    torch.manual_seed(42)
    model = wrapper(notebook_module, size, teacher_size)
    try:
        assert model.feats_idx[:-1] == TAPS
        model.eval()
        image = torch.rand(1, 3, 640, 640)
        with torch.no_grad():
            teacher_predictions = model.teacher_model(image)
            student_predictions = model.student_model(image)
        assert all(torch.isfinite(t).all() for t in tensors((teacher_predictions, student_predictions)))
        spatial = [80, 40, 20, 80, 40, 20]
        teacher_channels = [512, 512, 512, 256, 512, 512] if teacher_size == "m" else [256, 256, 512, 128, 256, 512]
        student_channels = [256, 256, 512, 128, 256, 512] if size == "s" else [128, 128, 256, 64, 128, 256]
        for position, index in enumerate(TAPS):
            student_feature, teacher_feature = model._student_feats[index], model._teacher_feats[index]
            assert student_feature.shape == (1, student_channels[position], spatial[position], spatial[position])
            assert teacher_feature.shape == (1, teacher_channels[position], spatial[position], spatial[position])
            assert model.projector["output"][position](student_feature).shape == teacher_feature.shape
        model.train()
        assert not model.teacher_model.training
        teacher_before = {n: v.clone() for n, v in model.teacher_model.state_dict().items()}
        cross_inputs = []

        def capture_cross_input(layer, inputs):
            if inputs[0].requires_grad:
                inputs[0].retain_grad()
                cross_inputs.append(inputs[0])

        cross_handles = [model.teacher_model.model[index].register_forward_pre_hook(capture_cross_input) for index in TAPS]
        example = batch()
        predictions = model.student_model(example["img"])
        detection, _ = model.student_model.loss(example, predictions)
        detection.sum().backward()
        assert_gradients(model.student_model.parameters())
        head = model.student_model.model[-1]
        assert_gradients(head.cv3.parameters())
        assert_gradients(head.one2one_cv3.parameters())
        for component in ("kd_agka_loss", "kd_cross_loss"):
            model.zero_grad(set_to_none=True)
            cross_inputs.clear()
            populate_features(model, example["img"])
            parts = model.kd_loss()
            assert all(torch.isfinite(value) for value in parts.values())
            forward_inputs = tuple(cross_inputs)
            assert len(forward_inputs) == len(TAPS)
            parts[component].backward()
            assert_gradients(model.student_model.parameters())
            assert_every_gradient(model.projector["input"].parameters())
            assert_every_gradient(model.projector["output"].parameters())
            assert len(cross_inputs) > len(forward_inputs), "Checkpoint must recompute cross activations"
            for incoming in forward_inputs:
                assert incoming.grad is not None and torch.isfinite(incoming.grad).all()
                assert torch.count_nonzero(incoming.grad).item() > 0
            assert all(p.grad is None and not p.requires_grad for p in model.teacher_model.parameters())
        optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=0.001)
        before_student = next(model.student_model.parameters()).detach().clone()
        optimizer.step()
        assert not torch.equal(before_student, next(model.student_model.parameters()))
        for name, before in teacher_before.items():
            torch.testing.assert_close(before, model.teacher_model.state_dict()[name], rtol=0, atol=0)
    finally:
        for handle in locals().get("cross_handles", []):
            handle.remove()
        model._teacher_feats.clear()
        model._student_feats.clear()
        model._remove_feature_hooks()
        del model
        gc.collect()


def hook_counts(model):
    return [(len(layer._forward_hooks), len(layer._forward_pre_hooks)) for net in (model.student_model, model.teacher_model) if net is not None for layer in net.model]


def test_hooks_cache_exceptions_validation_and_ema(notebook_module, monkeypatch):
    model = wrapper(notebook_module)
    try:
        before = hook_counts(model)
        model._register_feature_hooks()
        assert hook_counts(model) == before
        example = batch()
        model._teacher_feats["stale"] = torch.ones(1)
        model._student_feats["stale"] = torch.ones(1)
        loss, items = model.loss(example)
        assert all(torch.isfinite(value).all() for value in (loss, *items.values()))
        assert model._teacher_feats == model._student_feats == {}
        second_loss, _ = model.loss(batch())
        assert torch.isfinite(second_loss).all()
        assert model._teacher_feats == model._student_feats == {}
        copied = copy.deepcopy(model)
        try:
            assert hook_counts(copied) == before
            assert copied._teacher_feats == copied._student_feats == {}
        finally:
            copied._remove_feature_hooks()
        ema = ModelEMA(model)
        try:
            ema.update(model)
            assert ema.ema.teacher_model is None  # Native Ultralytics EMA deliberately drops the teacher.
            assert hook_counts(ema.ema) == before[:len(model.student_model.model)]
            ema.ema.eval()
            with torch.no_grad():
                predictions = ema.ema(example["img"])
                validation_loss, validation_items = ema.ema.loss(example, predictions)
            assert torch.isfinite(validation_loss).all()
            assert validation_items["kd_agka_loss"].item() == validation_items["kd_cross_loss"].item() == 0
            assert ema.ema._teacher_feats == ema.ema._student_feats == {}
        finally:
            ema.ema._remove_feature_hooks()
        with pytest.raises((ValueError, RuntimeError, KeyError)):
            model.kd_loss()

        def fail_loss(targets=None):
            raise RuntimeError("deliberate KD failure")

        monkeypatch.setattr(model, "kd_loss", fail_loss)
        with pytest.raises(RuntimeError, match="deliberate"):
            model.loss(example)
        assert model._teacher_feats == model._student_feats == {}
    finally:
        model._remove_feature_hooks()
        assert all(pair == (0, 0) for pair in hook_counts(model))


def test_actual_trainer_optimizer_excludes_teacher(notebook_module):
    model = wrapper(notebook_module)
    try:
        trainer = object.__new__(notebook_module.KDTrainer)
        trainer.args = get_cfg()
        trainer.data = {"nc": 6}
        optimizer = trainer.build_optimizer(model, name="AdamW", lr=0.001, decay=0.0005)
        optimized = {id(p) for group in optimizer.param_groups for p in group["params"]}
        expected = {id(p) for p in model.parameters() if p.requires_grad}
        assert optimized == expected
        assert optimized.isdisjoint({id(p) for p in model.teacher_model.parameters()})
    finally:
        model._remove_feature_hooks()


def test_oom_batch_retry_cannot_change_research_protocol(notebook_module):
    trainer = object.__new__(notebook_module.KDTrainer)
    trainer.args = SimpleNamespace(batch=16)
    trainer._requested_batch_size = 16
    trainer.batch_size = 8
    with pytest.raises(RuntimeError, match="Batch CSAKD berubah"):
        trainer._build_train_pipeline()


@pytest.mark.parametrize("size", ["s", "n"])
def test_checkpoint_chunking_preserves_combined_loss_all_gradients_and_teacher_state(notebook_module, tmp_path, size):
    """Uneven B=3 checks weighted chunk reductions and non-reentrant recomputation."""
    torch.manual_seed(53)
    direct = wrapper(notebook_module, size)
    direct.kd_checkpoint, direct.kd_chunk_size = False, 3
    optimized = copy.deepcopy(direct)
    optimized.kd_checkpoint, optimized.kd_chunk_size = True, 2
    example = batch(3)
    teacher_before = {name: value.clone() for name, value in direct.teacher_model.state_dict().items()}
    parity = {}

    def compare_gradients(objective, reference, candidate):
        # Check ALL trainable parameters, including legitimately unused ones.
        optimized_parameters = dict(candidate.named_parameters())
        errors, meaningful_relative_errors = {}, []
        for name, parameter in reference.named_parameters():
            other = optimized_parameters[name]
            if not parameter.requires_grad:
                assert parameter.grad is other.grad is None, name
                continue
            assert (parameter.grad is None) == (other.grad is None), name
            if parameter.grad is None:
                continue
            assert torch.isfinite(parameter.grad).all() and torch.isfinite(other.grad).all(), name
            delta = other.grad - parameter.grad
            reference_norm = parameter.grad.norm().item()
            relative_l2 = delta.norm().item() / max(reference_norm, 1e-20)
            # Cancellation makes some BN bias gradients mathematically near zero.
            # For these alone, relative error is uninformative; record and bound
            # the absolute FP32 difference instead. Projectors always use L2 parity.
            roundoff_only = name.endswith(".bn.bias") and reference_norm < 1e-6
            if roundoff_only:
                assert delta.abs().max().item() <= 1e-7, (name, relative_l2)
            else:
                meaningful_relative_errors.append(relative_l2)
                assert relative_l2 < 1e-4, (name, relative_l2)
            absolute_tolerance = max(1e-7 if roundoff_only else 1e-9, 1e-5 * parameter.grad.abs().max().item())
            torch.testing.assert_close(other.grad, parameter.grad, rtol=1e-4, atol=absolute_tolerance, msg=name)
            errors[name] = {"relative_l2": relative_l2, "reference_l2": reference_norm,
                            "max_absolute_difference": delta.abs().max().item(), "near_zero_bn_roundoff": roundoff_only}
        assert_every_gradient(candidate.projector.parameters())
        parity[objective] = {"max_meaningful_relative_l2": max(meaningful_relative_errors), "parameters": errors}

    try:
        direct_loss, direct_items = direct.loss(example)
        optimized_loss, optimized_items = optimized.loss(example)
        torch.testing.assert_close(optimized_loss, direct_loss, rtol=2e-5, atol=1e-7)
        for key in ("kd_agka_loss", "kd_cross_loss", "dis_loss", "det_total_loss", "kd_to_det_ratio"):
            torch.testing.assert_close(optimized_items[key], direct_items[key], rtol=2e-5, atol=1e-7)
        assert direct._teacher_feats == direct._student_feats == {}
        assert optimized._teacher_feats == optimized._student_feats == {}
        direct_loss.sum().backward()
        optimized_loss.sum().backward()
        compare_gradients("combined_detection_kd", direct, optimized)
        for component in ("kd_agka_loss", "kd_cross_loss"):
            parts = []
            for model in (direct, optimized):
                model.zero_grad(set_to_none=True)
                populate_features(model, example["img"])
                targets = model.snapshot_teacher_targets()
                parts.append(model.kd_loss(targets))
                model._student_feats.clear()
            torch.testing.assert_close(parts[1][component], parts[0][component], rtol=2e-5, atol=1e-7)
            parts[0][component].backward()
            parts[1][component].backward()
            compare_gradients(component, direct, optimized)
            assert optimized._teacher_feats == optimized._student_feats == {}
        for name, before in teacher_before.items():
            torch.testing.assert_close(before, direct.teacher_model.state_dict()[name], rtol=0, atol=0)
            torch.testing.assert_close(before, optimized.teacher_model.state_dict()[name], rtol=0, atol=0)
        assert optimized._teacher_feats == optimized._student_feats == {}, "Backward recomputation must not repopulate caches"
        (tmp_path / f"gradient_parity_{size}.json").write_text(json.dumps(parity, indent=2), encoding="utf-8")
        # Image capture must work again after the guarded cross calls.
        with torch.no_grad():
            optimized.teacher_model(example["img"])
        assert all(index in optimized._teacher_feats for index in TAPS)
    finally:
        for model in (direct, optimized):
            model._teacher_feats.clear()
            model._student_feats.clear()
            model._remove_feature_hooks()
        del direct, optimized
        gc.collect()


@pytest.mark.parametrize("chunk_size", [1, 2, 3])
def test_uneven_chunks_keep_global_aawm_gate_value_and_derivative(notebook_module, chunk_size):
    """Heterogeneous similarities detect replacing sigmoid(mean_B) with chunk gates."""
    torch.manual_seed(137)
    s = (torch.rand(5, 4, 3, 2) + 0.2).requires_grad_()
    c = (torch.rand_like(s) + 0.1).requires_grad_()
    t = torch.ones_like(s) * torch.tensor([-2.0, -0.2, 0.4, 2.0, 5.0]).view(5, 1, 1, 1)
    model = object.__new__(notebook_module.CSAKDModel)
    torch.nn.Module.__init__(model)
    model.feats_idx = [0, 1]
    model._teacher_capture = {"enabled": True}
    model._teacher_feats = {}
    model._student_feats = {0: s, ("input", 0): c}
    model.teacher_model = torch.nn.Module()
    model.teacher_model.model = torch.nn.ModuleList([torch.nn.Identity()])
    model.projector = torch.nn.ModuleDict({"input": torch.nn.ModuleList([torch.nn.Identity()]),
                                         "output": torch.nn.ModuleList([torch.nn.Identity()])})
    model.kd_checkpoint, model.kd_chunk_size = True, chunk_size
    parts = model.kd_loss({0: t})
    oracle = official_oracle(s, t, c)
    # Clear dictionaries BEFORE recomputation to expose any stale cache lookup.
    model._student_feats.clear()
    for actual, expected in zip((parts["kd_agka_loss"], parts["kd_cross_loss"]), oracle[:2]):
        torch.testing.assert_close(actual, expected, rtol=2e-5, atol=1e-7)
        actual_gradients = torch.autograd.grad(actual, (s, c), retain_graph=True)
        expected_gradients = torch.autograd.grad(expected, (s, c), retain_graph=True)
        for actual_gradient, expected_gradient in zip(actual_gradients, expected_gradients):
            torch.testing.assert_close(actual_gradient, expected_gradient, rtol=2e-5, atol=1e-7)
    assert model._teacher_capture["enabled"]
    assert model._teacher_feats == model._student_feats == {}


def test_cross_exception_restores_capture_and_snapshot_precedes_student(notebook_module):
    model = wrapper(notebook_module)
    checks = []

    def inspect_student(network, inputs):
        checks.append(model._teacher_feats == {})

    def fail_cross(layer, inputs):
        if inputs[0].requires_grad:
            raise RuntimeError("deliberate cross-forward failure")

    student_handle = model.student_model.register_forward_pre_hook(inspect_student)
    teacher_handle = model.teacher_model.model[TAPS[0]].register_forward_pre_hook(fail_cross)
    try:
        with pytest.raises(RuntimeError, match="cross-forward failure"):
            model.loss(batch())
        assert checks == [True], "Teacher caches must be released before student activations are allocated"
        assert model._teacher_feats == model._student_feats == {}
        assert model._teacher_capture["enabled"]
        with torch.no_grad():
            model.teacher_model(batch()["img"])
        assert all(index in model._teacher_feats and ("input", index) in model._teacher_feats for index in TAPS)
    finally:
        student_handle.remove()
        teacher_handle.remove()
        model._teacher_feats.clear()
        model._student_feats.clear()
        model._remove_feature_hooks()


def test_checkpoint_reduces_retained_saved_tensor_storage_cpu_proxy(notebook_module, tmp_path):
    """Counts unique saved tensor storage for KD only, never claims CUDA peak memory."""
    torch.manual_seed(91)
    model = wrapper(notebook_module)
    images = batch(3, 128)["img"]
    populate_features(model, images)
    targets = model.snapshot_teacher_targets()
    features = {key: value.detach().clone().requires_grad_() for key, value in model._student_feats.items()
                if key in TAPS or isinstance(key, tuple)}
    parameter_storages = {p.untyped_storage().data_ptr() for p in model.parameters()}
    measurements = {}
    try:
        for label, checkpoint_enabled, chunk_size in (("direct", False, 3), ("checkpoint_chunk2", True, 2)):
            model.kd_checkpoint, model.kd_chunk_size = checkpoint_enabled, chunk_size
            model._student_feats.clear()
            model._student_feats.update(features)
            saved_storages = {}

            def pack(tensor, saved=saved_storages):
                storage = tensor.untyped_storage()
                pointer = storage.data_ptr()
                if pointer and pointer not in parameter_storages:
                    saved[pointer] = storage.nbytes()
                return tensor

            with torch.autograd.graph.saved_tensors_hooks(pack, lambda tensor: tensor):
                parts = model.kd_loss(targets)
            measurements[label] = sum(saved_storages.values())
            assert all(torch.isfinite(value) for value in parts.values())
            model._student_feats.clear()
            del parts
        assert measurements["direct"] > 0
        ratio = measurements["checkpoint_chunk2"] / measurements["direct"]
        assert ratio < 0.7, measurements
        measurements["optimized_over_direct"] = ratio
        (tmp_path / "saved_tensor_storage_proxy.json").write_text(json.dumps(measurements, indent=2), encoding="utf-8")
    finally:
        model._teacher_feats.clear()
        model._student_feats.clear()
        model._remove_feature_hooks()


def test_batch_scaling_and_diagnostics_do_not_change_objective(notebook_module):
    class FakeStudent(torch.nn.Module):
        def forward(self, images):
            return images

        def loss(self, example, preds):
            return torch.tensor([2.0, 3.0], requires_grad=True), {"box_loss": torch.tensor(1.0)}

    class FakeWrapper(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.student_model = FakeStudent()
            self.teacher_model = FakeStudent()
            self._teacher_feats, self._student_feats = {}, {}

        def snapshot_teacher_targets(self):
            return {}

        def kd_loss(self, targets=None):
            return {"kd_agka_loss": torch.tensor(0.2, requires_grad=True), "kd_cross_loss": torch.tensor(0.3, requires_grad=True)}

    for count in (1, 2, 16):
        model = FakeWrapper().train()
        total, items = notebook_module.combined_loss(model, batch(count), None, {"kd_agka_loss": 1.0, "kd_cross_loss": 1.0})
        torch.testing.assert_close(total, torch.tensor([2.0, 3.0, 0.5 * count]))
        assert items["det_total_loss"].item() == pytest.approx(5.0 / count)
        assert items["kd_to_det_ratio"].item() == pytest.approx(0.5 / (5.0 / count))
        assert len(total) == 3


def save_raw_wrapper(model, target):
    torch.save({"model": copy.deepcopy(model).cpu(), "ema": None, "optimizer": {"diagnostic": True}, "epoch": 0,
                "train_args": {"task": "detect", "imgsz": 64, "data": "synthetic", "distill_model": "synthetic-teacher"}}, target)


def fresh_process_predict(checkpoint, imgsz=64):
    code = """import json, sys, torch
from ultralytics import YOLO
torch.set_num_threads(2)
model = YOLO(sys.argv[1])
assert not hasattr(model.model, 'teacher_model')
assert all(not layer._forward_hooks and not layer._forward_pre_hooks for layer in model.model.modules())
size = int(sys.argv[2])
result = model.predict(torch.rand(1,3,size,size), imgsz=size, device='cpu', verbose=False)[0]
assert torch.isfinite(result.boxes.data).all()
print('CSAKD_PORTABLE_OK', json.dumps({'shape': list(result.boxes.data.shape)}))
"""
    return subprocess.run([sys.executable, "-c", code, str(checkpoint), str(imgsz)], cwd=ROOT, capture_output=True, text=True, timeout=120, check=False)


def test_portable_checkpoint_preserves_raw_and_loads_in_fresh_process(notebook_module, tmp_path):
    model = wrapper(notebook_module)
    try:
        raw = tmp_path / "training_state_last.pt"
        save_raw_wrapper(model, raw)
        digest_before = hashlib.sha256(raw.read_bytes()).hexdigest()
        raw_attempt = subprocess.run([sys.executable, "-c", "import sys,torch; torch.load(sys.argv[1],weights_only=False)", str(raw)], cwd=ROOT, capture_output=True, text=True, timeout=120, check=False)
        assert raw_attempt.returncode != 0 and "_csakd_notebook_test_definitions" in raw_attempt.stderr
        portable = notebook_module.snapshot_student_checkpoint(raw, tmp_path / "last.pt")
        assert Path(portable).is_file()
        assert hashlib.sha256(raw.read_bytes()).hexdigest() == digest_before
        reloaded = torch.load(portable, map_location="cpu", weights_only=False)
        assert isinstance(reloaded["model"], DetectionModel)
        assert reloaded["optimizer"] is None
        assert not hasattr(reloaded["model"], "teacher_model")
        attempt = fresh_process_predict(portable)
        assert attempt.returncode == 0, attempt.stdout + attempt.stderr
        assert "CSAKD_PORTABLE_OK" in attempt.stdout
        with pytest.raises((ValueError, RuntimeError)):
            notebook_module.snapshot_student_checkpoint(raw, raw)
        corrupt = tmp_path / "corrupt.pt"
        corrupt.write_bytes(b"invalid checkpoint")
        with pytest.raises(RuntimeError, match="portabel"):
            notebook_module.snapshot_student_checkpoint(corrupt, portable)
    finally:
        model._remove_feature_hooks()


def test_wandb_callbacks_portable_and_raw_files_metrics_and_no_duplicates(tmp_path):
    notebook = json.loads(NOTEBOOKS[0].read_text(encoding="utf-8"))
    tree = ast.parse("".join(notebook["cells"][14]["source"]))
    callback_nodes = [node for node in ast.walk(tree) if isinstance(node, ast.FunctionDef)]
    config = notebook_config(NOTEBOOKS[0])
    uploaded, logged, config_updates = [], [], []

    class FakeArtifact:
        def __init__(self, **kwargs):
            self.kwargs, self.files = kwargs, {}

        def add_file(self, path, name):
            assert Path(path).is_file()
            self.files[name] = Path(path)

    class FakeConfig:
        def update(self, values, **kwargs):
            config_updates.append(dict(values))

    class FakeRun:
        id = "offline-test"
        config = FakeConfig()

        def log_artifact(self, artifact, aliases):
            entry = {"artifact": artifact, "aliases": aliases, "waited": False}
            uploaded.append(entry)
            return SimpleNamespace(wait=lambda: entry.update(waited=True))

        def log(self, payload, step):
            logged.append((payload, step))

    def snapshot(source, destination):
        destination = Path(destination)
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)
        return destination

    namespace = {**config, "Path": Path, "torch": torch, "wandb": SimpleNamespace(Artifact=FakeArtifact),
                 "snapshot_student_checkpoint": snapshot, "wb_run": FakeRun(),
                 "wb_state": {"uploaded_epochs": set(), "logged_epochs": set(), "best_epoch": None},
                 "TEACHER_PROVENANCE": {"artifact_reference": "teacher:v50", "best_pt_sha256": "diagnostic"},
                 "RUNTIME_VERSIONS": {"ultralytics": "8.4.155"}, "KD_CONFIG": {"agka_weight": 1.0, "cross_weight": 1.0},
                 "DIAGNOSTIC_ONLY": True, "effective_protocol": {}}
    exec(compile(ast.Module(body=callback_nodes, type_ignores=[]), str(NOTEBOOKS[0]), "exec"), namespace)  # noqa: S102 -- Callback definitions only.
    for filename in ("last.pt", "best.pt", "results.csv", "args.yaml"):
        (tmp_path / filename).write_bytes(b"diagnostic fixture")
    losses = {"train/box_loss": 0.2, "train/cls_loss": 0.3, "train/kd_agka_loss": 0.01,
              "train/kd_cross_loss": 0.02, "train/dis_loss": 0.03, "train/det_total_loss": 0.5, "train/kd_to_det_ratio": 0.06}
    metrics = {"metrics/precision(B)": 0.0, "metrics/recall(B)": 0.0, "metrics/mAP50(B)": 0.0, "metrics/mAP50-95(B)": 0.0}
    trainer = SimpleNamespace(epoch=0, last=tmp_path / "last.pt", best=tmp_path / "best.pt", save_dir=tmp_path,
                              fitness=0.0, best_fitness=0.0, batch_size=16, amp=False, args=SimpleNamespace(batch=16, amp=True),
                              device=torch.device("cpu"), train_loader=SimpleNamespace(batch_size=16), world_size=1,
                              tloss=torch.tensor([0.2]), lr={"lr/pg0": 0.001}, metrics=metrics,
                              label_loss_items=lambda values, prefix: losses)
    namespace["record_effective_protocol"](trainer)
    namespace["log_epoch_metrics"](trainer)  # No upload yet: no graph record.
    assert not logged
    namespace["save_epoch_artifact"](trainer)
    namespace["save_epoch_artifact"](trainer)
    namespace["log_epoch_metrics"](trainer)
    namespace["log_epoch_metrics"](trainer)  # Simulated final_eval duplicate.
    assert len(uploaded) == len(logged) == 1
    assert uploaded[0]["waited"]
    assert uploaded[0]["aliases"] == ["epoch-001", "latest", "best"]
    artifact = uploaded[0]["artifact"]
    assert set(artifact.files) == {"last.pt", "best.pt", "training_state_last.pt", "training_state_best.pt", "results.csv", "args.yaml"}
    assert artifact.files["training_state_last.pt"] == trainer.last
    assert artifact.files["last.pt"] != trainer.last
    for key in ("teacher_provenance", "runtime_versions", "kd", "effective_protocol"):
        assert key in artifact.kwargs["metadata"]
    assert config_updates
    payload, step = logged[0]
    assert step == 1 and payload["checkpoint/is_best_epoch"] == 1
    for key, value in {**losses, **metrics}.items():
        assert payload[key] == value
    trainer.batch_size = 8
    with pytest.raises(RuntimeError, match="Batch efektif"):
        namespace["record_effective_protocol"](trainer)
    trainer.batch_size = 16
    trainer.device = torch.device("cuda")
    namespace["DIAGNOSTIC_ONLY"] = False
    with pytest.raises(RuntimeError, match="AMP"):
        namespace["record_effective_protocol"](trainer)


@pytest.mark.parametrize("size", ["s", "n"])
def test_actual_poltrada_teacher_v50_cuda_amp_smoke_opt_in(notebook_module, tmp_path, monkeypatch, size):
    """Opt-in ONLY: two real training batches, one validation batch, CUDA AMP.

    Set CSAKD_GPU_SMOKE=1, CSAKD_TEACHER_CHECKPOINT, CSAKD_TEACHER_PROVENANCE_JSON,
    CSAKD_STUDENT_S_CHECKPOINT / CSAKD_STUDENT_N_CHECKPOINT, and optionally
    CSAKD_DATA_YAML. The provenance JSON must be generated by the notebook's
    verified v50 artifact retrieval, not fabricated for this smoke test.
    """
    if os.environ.get("CSAKD_GPU_SMOKE") != "1":
        pytest.skip("NOT TESTED: opt-in actual Poltrada/teacher v50 CUDA AMP requires explicit local assets")
    if not torch.cuda.is_available():
        pytest.skip("NOT TESTED: CUDA is unavailable")
    required = ("CSAKD_TEACHER_CHECKPOINT", "CSAKD_TEACHER_PROVENANCE_JSON", f"CSAKD_STUDENT_{size.upper()}_CHECKPOINT")
    if not all(os.environ.get(key) and Path(os.environ[key]).is_file() for key in required):
        pytest.skip("NOT TESTED: missing verified teacher provenance or local pretrained student/teacher checkpoint")
    import yaml
    from ultralytics.data.utils import IMG_FORMATS, check_det_dataset
    from ultralytics.utils import callbacks

    dataset_yaml = Path(os.environ.get("CSAKD_DATA_YAML", "/kaggle/input/datasets/ilhamrafiqin/poltrada-v3-yolo26-clean/data.yaml"))
    if not dataset_yaml.is_file():
        pytest.skip("NOT TESTED: original Poltrada V3 YAML is unavailable")
    teacher_path = Path(os.environ[required[0]])
    provenance = json.loads(Path(os.environ[required[1]]).read_text(encoding="utf-8"))
    artifact_reference = "ilhamrafiqin-brawijaya-university/yolom26-kd-indotraffic/01_teacher_yolo26m-fux5wdcb-checkpoints:v50"
    assert provenance["requested_artifact"] == artifact_reference
    assert provenance["version"] == "v50" and provenance["metadata_complete"]
    assert provenance["best_pt_sha256"] == hashlib.sha256(teacher_path.read_bytes()).hexdigest()
    original = check_det_dataset(str(dataset_yaml), autodownload=False)

    def image_paths(split):
        paths = original[split] if isinstance(original[split], list) else [original[split]]
        result = []
        for entry in paths:
            source = Path(entry)
            if source.is_dir():
                result.extend(path for path in source.rglob("*") if path.suffix.lstrip(".").lower() in IMG_FORMATS)
            else:
                for line in source.read_text(encoding="utf-8").splitlines():
                    candidate = Path(line)
                    result.append(candidate if candidate.is_absolute() else source.parent / candidate)
        return sorted({path.resolve() for path in result})

    train, val = image_paths("train")[:32], image_paths("val")[:16]
    assert len(train) == 32 and len(val) == 16
    for split, paths in (("train", train), ("val", val)):
        (tmp_path / f"{split}.txt").write_text("\n".join(path.as_posix() for path in paths) + "\n", encoding="utf-8")
    subset = tmp_path / "poltrada_smoke.yaml"
    subset.write_text(yaml.safe_dump({"path": str(original["path"]), "train": str(tmp_path / "train.txt"),
                                     "val": str(tmp_path / "val.txt"), "names": original["names"], "nc": original["nc"]}), encoding="utf-8")
    teacher = YOLO(str(teacher_path))
    assert teacher.model.yaml["scale"] == "m"
    assert teacher.model.names == original["names"]
    assert int(teacher.model.nc) == int(original["nc"])
    del teacher
    monkeypatch.setattr(callbacks, "add_integration_callbacks", lambda instance: None)
    model = YOLO(os.environ[required[2]])
    observations = {"batches": 0, "steps": 0}
    profile = {"diagnostic_only": True, "memory_strategy": {"kd_checkpoint": True, "kd_chunk_size": 2},
               "requested_batch": 16, "imgsz": 640, "student": f"yolo26{size}", "batches": [],
               "status": "RUNNING", "teacher_provenance": provenance, "device_name": torch.cuda.get_device_name(0)}
    torch.cuda.synchronize(0)
    torch.cuda.reset_peak_memory_stats(0)

    def memory_snapshot():
        free, total = torch.cuda.mem_get_info(0)
        return {"allocated_bytes": int(torch.cuda.memory_allocated(0)),
                "reserved_bytes": int(torch.cuda.memory_reserved(0)),
                "device_free_bytes": int(free), "device_total_bytes": int(total),
                "peak_allocated_bytes": int(torch.cuda.max_memory_allocated(0)),
                "peak_reserved_bytes": int(torch.cuda.max_memory_reserved(0))}

    def check_before_step(trainer):
        # Called after unscale/clip inside optimizer_step by PyTorch optimizer.
        assert_gradients(trainer.model.student_model.parameters())
        assert_every_gradient(trainer.model.projector.parameters())
        assert all(p.grad is None and not p.requires_grad for p in trainer.model.teacher_model.parameters())
        observations["steps"] += 1

    def setup(trainer):
        assert trainer.amp and trainer.batch_size == 16
        assert trainer.model.kd_checkpoint and trainer.model.kd_chunk_size == 2
        profile["effective_batch"], profile["effective_amp"] = int(trainer.batch_size), bool(trainer.amp)
        profile["setup"] = memory_snapshot()
        trainer.optimizer.register_step_pre_hook(lambda optimizer, args, kwargs: check_before_step(trainer))

    def start_batch(trainer):
        torch.cuda.synchronize(0)
        torch.cuda.reset_peak_memory_stats(0)
        profile["batch_started"] = time.perf_counter()

    def check_batch(trainer):
        observations["batches"] += 1
        assert torch.isfinite(trainer.loss).all()
        torch.cuda.synchronize(0)
        profile["batches"].append({"batch": observations["batches"],
                                   "phase": "warmup/setup" if observations["batches"] == 1 else "measured",
                                   "batch_elapsed_seconds": time.perf_counter() - profile["batch_started"],
                                   **memory_snapshot()})

    model.add_callback("on_train_start", setup)
    model.add_callback("on_train_batch_start", start_batch)
    model.add_callback("on_train_batch_end", check_batch)
    try:
        result = model.train(trainer=notebook_module.KDTrainer, data=str(subset), distill_model=str(teacher_path), dis=1.0,
                             epochs=1, imgsz=640, batch=16, device=0, workers=0, optimizer="AdamW", lr0=0.001, seed=42,
                             amp=True, pretrained=True, project=str(tmp_path / "gpu_runs"), name=f"v50_poltrada_{size}",
                             plots=False, save=True, val=True, compile=False)
        assert observations["batches"] == 2 and observations["steps"] > 0
        assert all(math.isfinite(float(value)) for value in result.results_dict.values())
        attempt = fresh_process_predict(Path(model.trainer.best), imgsz=640)
        assert attempt.returncode == 0, attempt.stdout + attempt.stderr
        profile["status"] = "PASS"
    except BaseException as error:
        profile["status"], profile["error"] = "FAIL", f"{type(error).__name__}: {error}"
        raise
    finally:
        final_memory = memory_snapshot()
        profile["final"] = final_memory
        snapshots = [*profile["batches"], profile.get("setup", final_memory), final_memory]
        profile["whole_run_peak_allocated_bytes"] = max(entry["peak_allocated_bytes"] for entry in snapshots)
        profile["whole_run_peak_reserved_bytes"] = max(entry["peak_reserved_bytes"] for entry in snapshots)
        (tmp_path / "gpu_memory_profile.json").write_text(json.dumps(profile, indent=2), encoding="utf-8")


@pytest.mark.slow
@pytest.mark.parametrize(("teacher_size", "size"), [("m", "s"), ("m", "n"), ("s", "n")])
def test_synthetic_one_epoch_cpu_validation_checkpoint_reload(notebook_module, tmp_path, monkeypatch, teacher_size, size):
    """Diagnostic only: random weights, synthetic train/val, 64px, two batches."""
    from ultralytics import settings
    from ultralytics.utils import callbacks

    # Suppress online integration discovery for this offline fixture.
    monkeypatch.setattr(callbacks, "add_integration_callbacks", lambda instance: None)
    monkeypatch.setitem(settings, "wandb", False)
    dataset = tmp_path / "synthetic"
    for split, count in (("train", 4), ("val", 2)):
        for child in ("images", "labels"):
            (dataset / split / child).mkdir(parents=True)
        for index in range(count):
            image = Image.new("RGB", (64, 64), (30 + 40 * index, 80, 150))
            image.save(dataset / split / "images" / f"{index}.png")
            (dataset / split / "labels" / f"{index}.txt").write_text("0 0.5 0.5 0.25 0.25\n", encoding="utf-8")
    data = dataset / "data.yaml"
    data.write_text(f"path: {dataset.as_posix()}\ntrain: train/images\nval: val/images\nnc: 6\nnames: [class_0, class_1, class_2, class_3, class_4, class_5]\n", encoding="utf-8")
    teacher_path = tmp_path / "synthetic_teacher.pt"
    torch.save({"model": network(teacher_size), "train_args": {"task": "detect", "imgsz": 64}}, teacher_path)
    model = YOLO(f"yolo26{size}.yaml")
    observations = {"batches": 0, "saves": 0, "steps": 0}
    initial_student = []

    def before_step(trainer):
        observations["steps"] += 1
        assert_gradients(trainer.model.student_model.parameters())
        assert_every_gradient(trainer.model.projector.parameters())
        assert all(p.grad is None for p in trainer.model.teacher_model.parameters())

    def setup(trainer):
        initial_student.append(next(trainer.model.student_model.parameters()).detach().clone())
        trainer.optimizer.register_step_pre_hook(lambda optimizer, args, kwargs: before_step(trainer))

    def inspect_batch(trainer):
        observations["batches"] += 1
        assert torch.isfinite(trainer.loss).all()
        assert not trainer.model.teacher_model.training
        assert all(p.grad is None for p in trainer.model.teacher_model.parameters())
        optimized = {id(p) for group in trainer.optimizer.param_groups for p in group["params"]}
        assert optimized.isdisjoint({id(p) for p in trainer.model.teacher_model.parameters()})

    def inspect_save(trainer):
        observations["saves"] += 1
        raw_hash = hashlib.sha256(Path(trainer.last).read_bytes()).hexdigest()
        destination = tmp_path / f"epoch_{trainer.epoch + 1}_student.pt"
        notebook_module.snapshot_student_checkpoint(trainer.last, destination)
        assert hashlib.sha256(Path(trainer.last).read_bytes()).hexdigest() == raw_hash
        attempt = fresh_process_predict(destination)
        assert attempt.returncode == 0, attempt.stdout + attempt.stderr

    model.add_callback("on_train_batch_end", inspect_batch)
    model.add_callback("on_model_save", inspect_save)
    model.add_callback("on_train_start", setup)
    result = model.train(trainer=notebook_module.KDTrainer, data=str(data), distill_model=str(teacher_path),
                         dis=1.0, epochs=1, imgsz=64, batch=2, nbs=2, device="cpu", workers=0, optimizer="AdamW",
                         lr0=0.001, seed=42, amp=False, pretrained=False, project=str(tmp_path / "runs"), name=f"diagnostic_{size}",
                         plots=False, save=True, val=True, mosaic=0.0, mixup=0.0, warmup_epochs=0.0, close_mosaic=0,
                         compile=False, deterministic=True)
    assert observations == {"batches": 2, "saves": 1, "steps": 2}
    assert not torch.equal(initial_student[0], next(model.trainer.model.student_model.parameters()))
    assert all(math.isfinite(float(value)) for value in result.results_dict.values())
    directory = Path(model.trainer.save_dir)
    assert (directory / "weights/best.pt").is_file() and (directory / "weights/last.pt").is_file()
    header = (directory / "results.csv").read_text(encoding="utf-8").splitlines()[0]
    for key in ("train/kd_agka_loss", "train/kd_cross_loss", "train/dis_loss", "train/det_total_loss", "train/kd_to_det_ratio",
                "metrics/precision(B)", "metrics/recall(B)", "metrics/mAP50(B)", "metrics/mAP50-95(B)"):
        assert key in header
    attempt = fresh_process_predict(directory / "weights/best.pt")
    assert attempt.returncode == 0, attempt.stdout + attempt.stderr
