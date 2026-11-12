"""Offline structural policy checks; no dataset, artifact, training, or network."""
import importlib.util
import json
import math
from copy import deepcopy
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("torch_pruning")
from ultralytics import YOLO
from ultralytics.nn.modules import SPPF

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("lamp_v2_policy", ROOT / "scripts/lamp_v2_policy_source.py")
engine = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(engine)


@pytest.fixture(autouse=True)
def no_learning(monkeypatch):
    def forbidden(*args, **kwargs):
        pytest.fail("Pruning V2 must not train, call backward, or step an optimizer.")
    monkeypatch.setattr(YOLO, "train", forbidden)
    monkeypatch.setattr(torch.Tensor, "backward", forbidden)
    monkeypatch.setattr(torch.autograd, "backward", forbidden)
    for cls in (torch.optim.Optimizer, torch.optim.SGD, torch.optim.Adam, torch.optim.AdamW):
        monkeypatch.setattr(cls, "step", forbidden)
    torch.set_num_threads(1)


@pytest.fixture(scope="module", params=["n", "s"])
def synthetic(request):
    torch.manual_seed(42)
    core = YOLO(f"yolo26{request.param}.yaml", task="detect", verbose=False).model.float().eval()
    return core


def build(core, image, policy, ratio):
    return engine.build_policy_pruner(core, image, policy, ratio, model_variant=core.yaml["scale"])


@pytest.mark.parametrize("policy,expected_groups,expected_channels", [
    ("conservative", 13, 440), ("extended", 35, 2056),
])
@pytest.mark.parametrize("ratio", [0.1, 0.2, 0.3, 0.4, 0.5])
def test_synthetic_independent_policy_runs(synthetic, policy, expected_groups, expected_channels, ratio):
    """Implementation-only synthetic coverage, not empirical dataset accuracy."""
    core = deepcopy(synthetic)
    expected_channels *= 1 if core.yaml["scale"] == "n" else 2
    image = torch.rand(1, 3, 64, 64)
    before_parameters = sum(p.numel() for p in core.parameters())
    protected = engine.protected_policy_snapshot(core)
    pruner, audit = build(core, image, policy, ratio)
    assert audit["report"]["model_variant"] == core.yaml["scale"]
    assert audit["report"]["group_count"] == expected_groups
    assert audit["independent_channels_before"] == expected_channels
    signatures = [set(entry["signature"]) for entry in audit["eligible"].values()]
    assert sum(map(len, signatures)) == len(set().union(*signatures))
    before_inventory = engine.measure_layer_inventory(core, image, audit, policy)
    assert any(row["module_type"] == "BatchNorm2d" and row["eligible"] for row in before_inventory)
    assert all(row["estimated_flops"] is not None for row in before_inventory)
    initial_mapping = deepcopy(audit["report"]["channel_mapping"])
    initial_scores = deepcopy(audit["report"]["lamp_scores"])
    assert len(initial_scores) == expected_channels
    assert all(math.isfinite(row["score"]) and not row["removed"] for row in initial_scores)
    assert all(row["kept_original_indices"] == list(range(row["channels_before"])) for row in initial_mapping)
    for entry in audit["eligible"].values():
        scores = [row for row in initial_scores if row["group_id"] == entry["group_id"]]
        count = min(entry["initial_channels"] - 1, *(initial // 2 for initial in entry["output_dimensions"].values()))
        expected = set(sorted(range(len(scores)), key=lambda i: (scores[i]["score"], i))[:count])
        assert {row["channel_idx"] for row in scores if row["capacity_eligible"]} == expected
    history = engine.apply_policy_pruning(pruner, audit)
    removed = sum(item["channels_before"] - item["channels_after"] for item in history)
    # TP floors the retained global width; requested .3 can remove 617/2056.
    assert removed == expected_channels - int(expected_channels * (1 - ratio))
    assert sum(p.numel() for p in core.parameters()) < before_parameters
    engine.assert_policy_protected_unchanged(core, protected)
    for entry in audit["eligible"].values():
        for module, initial in entry["output_dimensions"].items():
            assert module.out_channels >= 1
            assert initial - module.out_channels <= initial // 2
    with torch.no_grad():
        output = core(image)
    assert torch.isfinite(output[0]).all()
    raw = engine._v2_trace_outputs(output)
    assert len(raw) == 4 and all(torch.isfinite(value).all() for value in raw)
    after_inventory = engine.measure_layer_inventory(core, image, audit, policy)
    assert len(after_inventory) == len(before_inventory)
    mapping = audit["report"]["channel_mapping"]
    assert len(mapping) == 2 * sum(isinstance(m, torch.nn.Conv2d) for m in core.modules()) + sum(
        isinstance(m, torch.nn.BatchNorm2d) for m in core.modules())
    expected_ledger = {(row["layer"], row["dimension"]): list(row["kept_original_indices"]) for row in initial_mapping}
    for step in history:
        dimensions = step["mapped_dimensions"]
        assert len({(row["layer"], row["dimension"]) for row in dimensions}) == len(dimensions)
        for row in dimensions:
            key = row["layer"], row["dimension"]
            original = expected_ledger[key]
            assert [original[index] for index in row["current_indices"]] == row["original_indices"]
            expected_ledger[key] = [value for i, value in enumerate(original) if i not in row["current_indices"]]
        outputs = [row for row in dimensions if row["module_type"] == "Conv2d" and row["dimension"] == "out"]
        assert step["coupled_conv_output_count"] == len(outputs)
        assert step["physical_conv_output_channels_removed"] == sum(len(row["current_indices"]) for row in outputs)
    for row, mask in zip(mapping, audit["report"]["channel_masks"]):
        assert row["kept_original_indices"] == expected_ledger[(row["layer"], row["dimension"])]
        assert row["channels_after"] == len(row["kept_original_indices"]) == sum(mask["kept_mask"])
        assert sorted(row["kept_original_indices"] + row["removed_original_indices"]) == list(range(row["channels_before"]))
        assert not set(row["kept_original_indices"]) & set(row["removed_original_indices"])
    assert sum(row["removed"] for row in audit["report"]["lamp_scores"]) == removed
    assert all(row["capacity_eligible"] for row in audit["report"]["lamp_scores"] if row["removed"])
    assert all(before["score"] == after["score"] for before, after in zip(initial_scores, audit["report"]["lamp_scores"]))
    assert sum(len(row["removed_original_indices"]) for row in mapping
               if row["module_type"] == "Conv2d" and row["dimension"] == "out") == audit["report"]["physical_conv_output_channels_removed"]
    json.dumps(audit["report"], allow_nan=False)
    # The baseline fixture is never cumulatively pruned.
    assert sum(p.numel() for p in synthetic.parameters()) == before_parameters


def test_extended_adds_safe_regions_and_rejects_protected_graphs(synthetic):
    core = deepcopy(synthetic)
    pruner, audit = build(core, torch.rand(1, 3, 64, 64), "extended", 0.1)
    names = {entry["name"] for entry in audit["eligible"].values()}
    structure = audit["structure"]
    sppf_hidden = {name + ".cv1.conv" for name, module in core.named_modules()
                   if isinstance(module, SPPF)}
    assert names.intersection(sppf_hidden)
    assert any(len(entry["signature"]) > 1 for entry in audit["eligible"].values())
    rejected = audit["report"]["rejected_groups"]
    assert any("detection_head_all_dimensions" in row["reason"] for row in rejected)
    assert any("attention_envelope" in row["reason"] for row in rejected)
    assert any("CSP_split_output" in row["reason"] for row in rejected)
    assert any("unsupported_operation__SplitOp" in row["reason"] for row in rejected)
    for entry in audit["eligible"].values():
        assert not set(entry["member_dimensions"]).intersection(structure["protected"])
    head_final = structure["head"].cv2[0][-1]
    group = pruner.DG.get_pruning_group(head_final, engine.tp.prune_conv_out_channels, [0])
    with pytest.raises(ValueError, match="detection_head_all_dimensions"):
        engine._v2_validate_group(group, pruner.DG, structure, "extended")


def test_importance_nonfinite_fails_before_pruning(synthetic, monkeypatch):
    core = deepcopy(synthetic)
    original = engine.tp.importance.LAMPImportance.__call__
    monkeypatch.setattr(engine.tp.importance.LAMPImportance, "__call__",
                        lambda self, group: original(self, group) * float("nan"))
    with pytest.raises(ValueError, match="Importance LAMP invalid"):
        build(core, torch.rand(1, 3, 64, 64), "conservative", 0.1)


def test_late_invalid_proposal_keeps_partial_history(synthetic, monkeypatch):
    core = deepcopy(synthetic)
    pruner, audit = build(core, torch.rand(1, 3, 64, 64), "conservative", 0.3)
    original_step = pruner.step
    head_final = audit["structure"]["head"].cv2[0][-1]

    def proposals_then_invalid(*, interactive):
        for proposal in original_step(interactive=interactive):
            if not proposal[0].idxs:
                continue
            yield proposal
            yield pruner.DG.get_pruning_group(head_final, engine.tp.prune_conv_out_channels, [0])
            return

    monkeypatch.setattr(pruner, "step", proposals_then_invalid)
    with pytest.raises(ValueError, match="root_not_conservative_hidden_channel"):
        engine.apply_policy_pruning(pruner, audit)
    partial = audit["report"]["pruning_history"]
    assert len(partial) == 1 and partial[0]["channels_after"] < partial[0]["channels_before"]
    assert audit["report"]["independent_channels_removed"] > 0
    assert audit["report"]["pruning_status"] == "IN_PROGRESS"
    assert audit["report"]["last_proposal"]["root"].startswith(audit["structure"]["head_name"] + ".")


@pytest.mark.parametrize("policy,ratio", [("both", 0.3), ("unknown", 0.1), ("extended", 0), ("conservative", 0.6)])
def test_invalid_policy_or_ratio_rejected(synthetic, policy, ratio):
    with pytest.raises(ValueError):
        build(deepcopy(synthetic), torch.zeros(1, 3, 64, 64), policy, ratio)


def test_requested_model_variant_and_stem_are_validated(synthetic):
    core = deepcopy(synthetic)
    variant = core.yaml["scale"]
    wrong = "s" if variant == "n" else "n"
    with pytest.raises(ValueError, match="Model variant tidak cocok"):
        engine.build_policy_pruner(core, torch.rand(1, 3, 64, 64), "extended", 0.1, model_variant=wrong)
    assert engine._v2_policy_structure(core)["model_variant"] == variant
    if variant == "n":
        # The existing N-only caller signature retains its meaning.
        _, audit = engine.build_policy_pruner(core, torch.rand(1, 3, 64, 64), "conservative", 0.1)
        assert audit["model_variant"] == "n"
    first = next(m for m in core.modules() if isinstance(m, torch.nn.Conv2d))
    first.out_channels += 1
    with pytest.raises(ValueError, match="stem 16/32"):
        engine._v2_policy_structure(core, model_variant=variant)
