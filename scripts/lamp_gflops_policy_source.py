"""GFLOPs-target planner and tensor oracle embedded after the V2 policy."""
# ruff: noqa: F821 -- The standalone notebook loads V2 declarations before this cell.

import math
from contextlib import nullcontext
from copy import deepcopy
from types import SimpleNamespace

import torch
import torch_pruning as tp

GFLOPS_CAP_TIERS = ((25, 10), (35, 20), (50, 50))
GFLOPS_INTERFACE_STAGES = (0, 1, 3, 4, 5, 6, 7, 13, 16, 17, 19, 20, 22)
GFLOPS_P5_ROOT = "model.7.conv"


def _gflops_rank_space(audit, caps, protect_interfaces, exclude_p5):
    """Keep original LAMP scores and bound capacity before ranking candidates."""
    if len(caps) != 2 or any(type(cap) is not int or not 0 <= cap <= 50 for cap in caps):
        raise ValueError("Cap harus berisi dua persentase integer antara 0 dan 50.")
    structure = audit["structure"]
    interfaces = set()
    if protect_interfaces:
        for stage in GFLOPS_INTERFACE_STAGES:
            block = structure["modules"][f"model.{stage}"]
            interface = getattr(block, "cv2", block).conv
            if not isinstance(interface, torch.nn.Conv2d):
                raise TypeError(f"Output antarmuka YOLO26 tidak didukung: model.{stage}.")
            interfaces.add(interface)
    if exclude_p5:
        interfaces.add(structure["modules"][GFLOPS_P5_ROOT])

    ranked, capacities, eligible_channels = [], {}, 0
    for group in audit["eligible"].values():
        if interfaces.intersection(group["output_dimensions"]):
            continue
        width = group["initial_channels"]
        percent = caps[0] if group["root"] in structure["conservative"] else caps[1]
        capacity = min(width - 1, *(size * percent // 100 for size in group["output_dimensions"].values()))
        root = group["name"]
        scores = group["initial_lamp_scores"]
        if len(scores) != width or not all(math.isfinite(score) for score in scores):
            raise ValueError(f"Skor LAMP tidak valid: {root}.")
        eligible_channels += width
        capacities[root] = capacity
        ranked.extend(sorted((float(score), root, index) for index, score in enumerate(scores))[:capacity])
    return sorted(ranked), capacities, eligible_channels


def gflops_ranked_candidates(audit, caps, protect_interfaces=True, exclude_p5=False):
    return _gflops_rank_space(audit, caps, protect_interfaces, exclude_p5)[0]


def _gflops_prefix_plan(ranked, count):
    if type(count) is not int or not 0 <= count <= len(ranked):
        raise ValueError(f"Target {count} channel melampaui kapasitas {len(ranked)}.")
    plan = {}
    for _, root, index in ranked[:count]:
        plan.setdefault(root, []).append(index)
    return {root: sorted(indices) for root, indices in sorted(plan.items())}


def gflops_plan_key(plan):
    """Canonical identity for caching and deduplicating equal physical plans."""
    return tuple((root, tuple(sorted(indices))) for root, indices in sorted(plan.items()) if indices)


def p5_control_plans(audit, extended_10_history):
    """Use the historical 10% mask itself as the source of the P5-only dose."""
    groups = [entry for entry in audit["eligible"].values() if GFLOPS_P5_ROOT in entry["signature"]]
    if len(groups) != 1 or groups[0]["name"] != GFLOPS_P5_ROOT:
        raise ValueError("Root canonical P5 harus unik dan cocok dengan model.7.conv.")
    history = {entry["root"]: entry["removed_original_indices"] for entry in extended_10_history}
    if len(history) != len(extended_10_history) or not history.get(GFLOPS_P5_ROOT):
        raise ValueError("Kontrol extended 10% harus mencatat satu penghapusan group P5.")
    count = sum(len(indices) for indices in history.values())
    candidates = gflops_ranked_candidates(audit, (50, 50), protect_interfaces=False, exclude_p5=True)
    return {"p5-only": {GFLOPS_P5_ROOT: sorted(history[GFLOPS_P5_ROOT])},
            "p5-protected": _gflops_prefix_plan(candidates, count)}


def apply_exact_gflops_plan(pruner, audit, plan, caps=None):
    """Run the existing dependency verifier and ledger with exact finite selections."""
    groups = {entry["name"]: entry for entry in audit["eligible"].values()}
    expected_caps = {name: _v2_capacity(entry) for name, entry in groups.items()}
    if caps is not None:
        if any(name not in groups or type(cap) is not int or not 0 <= cap <= expected_caps[name]
               for name, cap in caps.items()):
            raise ValueError("Cap plan tidak sesuai dengan dependency group sumber.")
        expected_caps = {name: caps.get(name, 0) for name in groups}
        eligible_channels = sum(groups[name]["initial_channels"] for name in caps)
        audit["report"]["target_policy_eligible_channels"] = eligible_channels

    graph = pruner.DG
    selections = gflops_plan_key(plan)
    if not selections:
        raise ValueError("Plan pruning harus memiliki setidaknya satu channel.")
    for name, indices in selections:
        entry = groups.get(name)
        if (entry is None or len(indices) > expected_caps[name] or len(indices) != len(set(indices))
                or any(type(index) is not int or not 0 <= index < entry["initial_channels"] for index in indices)):
            raise ValueError(f"Plan channel invalid atau melebihi cap: {name}.")
        if entry["root"].out_channels != entry["initial_channels"]:
            raise ValueError(f"Plan harus dimulai dari indeks baseline: {name}.")

    audit["report"]["ranking"] = "Finite LAMP scores sorted by (score, root, original index), exact integer budget"
    audit["report"]["planned_original_indices"] = {name: list(indices) for name, indices in selections}
    audit["report"]["planned_group_capacities"] = expected_caps
    if caps is not None:
        permitted = {}
        for name, entry in groups.items():
            scores = entry["initial_lamp_scores"]
            permitted[name] = set(sorted(range(len(scores)), key=lambda index: (scores[index], index))[:expected_caps[name]])
        for row in audit["report"]["lamp_scores"]:
            row["capacity_eligible"] = row["channel_idx"] in permitted[row["root"]]
        if any(not set(indices).issubset(permitted[name]) for name, indices in selections):
            raise ValueError("Plan memilih channel di luar peringkat LAMP dalam batas cap.")

    def step(interactive=True):
        if not interactive:
            raise ValueError("Pruning memerlukan pemeriksaan dependency interaktif.")
        for name, indices in selections:
            entry = groups[name]
            group = graph.get_pruning_group(entry["root"], tp.prune_conv_out_channels, list(indices))
            # All coupled Conv outputs must lose exactly the same number of channels.
            for dependency, mapped in group:
                dimension = _v2_dependency_dimension(dependency, graph)
                if dimension is not None and dimension[1] == "out" and len(mapped) != len(indices):
                    raise ValueError(f"Jumlah survivor output residual tidak konsisten: {name}.")
            yield group

    history = apply_policy_pruning(SimpleNamespace(DG=graph, step=step), audit)
    applied = {entry["root"]: entry["removed_original_indices"] for entry in history}
    if gflops_plan_key(applied) != selections:
        raise ValueError("Mask aktual berbeda dari plan indeks asli.")
    return history


def search_gflops_targets(audit, baseline_gflops, measure_plan,
                         targets=(1.02, 1.05, 1.10, 1.20, 1.70), *, profile_cache=None, progress_callback=None):
    """Original finite LAMP prefix search, with resumable profiling and isolated failures."""
    if not math.isfinite(baseline_gflops) or baseline_gflops <= 0:
        raise ValueError("GFLOPs baseline harus finite dan positif.")
    targets = tuple(targets)
    if not targets or len(targets) != len(set(targets)) or any(not math.isfinite(t) or t <= 1 for t in targets):
        raise ValueError("Target GFLOPs harus unik, finite, dan lebih dari satu.")
    cached = {} if profile_cache is None else profile_cache
    cached[repr(())] = float(baseline_gflops)
    measurements, tiers, spaces, errors = {}, [], {}, []

    def profile(ranked, count, tier):
        plan = _gflops_prefix_plan(ranked, count)
        key = repr(gflops_plan_key(plan))
        if key not in cached:
            measured = float(measure_plan(plan))
            if not math.isfinite(measured) or measured <= 0 or measured > baseline_gflops + 1e-12:
                raise ValueError(f"GFLOPs tidak valid untuk tier {tier}, {count} channel.")
            cached[key] = measured
        value = cached[key]
        if not math.isfinite(value) or not 0 < value <= baseline_gflops + 1e-12:
            raise ValueError("Cache profil GFLOPs tidak valid.")
        measurements[tier, count] = {"tier": tier, "channels": count, "gflops": value,
                                     "speed_up": baseline_gflops / value}
        if progress_callback:
            progress_callback(cached, list(measurements.values()))
        return value, plan

    for tier, cap in enumerate(GFLOPS_CAP_TIERS, 1):
        ranked, capacities, eligible = _gflops_rank_space(audit, cap, True, False)
        entry = {"tier": tier, "hidden_cap_pct": cap[0], "other_cap_pct": cap[1],
                 "eligible_channels": eligible, "capacity_channels": len(ranked), "status": "NOT_RUN"}
        try:
            measured, plan = profile(ranked, len(ranked), tier)
            entry.update(status="PASS", max_gflops_speed_up=baseline_gflops / measured, gflops_at_capacity=measured)
            spaces[tier] = ranked, capacities, measured, plan
        except Exception as error:
            entry.update(status="FAIL", error=str(error))
            errors.append({"tier": tier, "stage": "capacity", "error": str(error)})
        tiers.append(entry)

    rows = []
    for target in targets:
        try:
            feasible = next((entry for entry in tiers if entry["status"] == "PASS"
                             and entry["max_gflops_speed_up"] >= target), None)
            if feasible is None:
                valid = [entry for entry in tiers if entry["status"] == "PASS"]
                if len(valid) != len(tiers):
                    raise ValueError("Kapasitas belum dapat dibuktikan karena ada profil tier yang gagal.")
                best = max(valid, key=lambda entry: (entry["max_gflops_speed_up"], -entry["tier"]))
                ranked, capacities, measured, plan = spaces[best["tier"]]
                rows.append({"target": target, "status": "TARGET_NOT_REACHED", "tier": best["tier"],
                             "channels": len(ranked), "gflops": measured, "speed_up": baseline_gflops / measured,
                             "plan": plan, "caps": capacities})
                continue
            tier = feasible["tier"]
            ranked, capacities, _, _ = spaces[tier]
            checkpoints = sorted({min(len(ranked), (feasible["eligible_channels"] * pct + 99) // 100)
                                  for pct in range(1, 51)} | {len(ranked)})
            previous, previous_flops = 0, baseline_gflops
            for checkpoint in checkpoints:
                measured, plan = profile(ranked, checkpoint, tier)
                if measured > previous_flops + 1e-12:
                    raise ValueError("GFLOPs bertambah pada prefix dalam tier.")
                if baseline_gflops / measured >= target:
                    break
                previous, previous_flops = checkpoint, measured
            for count in range(previous + 1, checkpoint + 1):
                measured, plan = profile(ranked, count, tier)
                if measured > previous_flops + 1e-12:
                    raise ValueError("GFLOPs bertambah pada refinement integer.")
                if baseline_gflops / measured >= target:
                    break
                previous_flops = measured
            rows.append({"target": target, "status": "REACHED", "tier": tier, "channels": count,
                         "gflops": measured, "speed_up": baseline_gflops / measured,
                         "plan": plan, "caps": capacities})
        except Exception as error:
            rows.append({"target": target, "status": "FAIL", "error": str(error), "plan": None})
    return {"targets": rows, "tiers": tiers, "errors": errors,
            "measurements": [measurements[key] for key in sorted(measurements)]}


def _gflops_checked_masks(audit):
    mappings = {}
    for row in audit["report"]["channel_mapping"]:
        key = row["layer"], row["dimension"]
        kept = row["kept_original_indices"]
        removed = row["removed_original_indices"]
        if (key in mappings or len(kept) != row["channels_after"] or not kept
                or kept != sorted(kept) or removed != sorted(removed)
                or sorted([*kept, *removed]) != list(range(row["channels_before"]))):
            raise ValueError(f"Ledger indeks asli tidak valid: {key}.")
        mappings[key] = row
    return mappings


def _gflops_index(tensor, axis, indices):
    return tensor.index_select(axis, torch.tensor(indices, device=tensor.device, dtype=torch.long))


def verify_gflops_oracle(baseline, pruned, image, audit):
    """Verify exact parameter slicing and functional parity against a dense oracle."""
    source, target = dict(baseline.named_modules()), dict(pruned.named_modules())
    if source.keys() != target.keys() or baseline.training or pruned.training:
        raise ValueError("Oracle memerlukan module tree yang sama dalam mode eval.")
    mappings = _gflops_checked_masks(audit)
    candidate_device = next(pruned.parameters()).device
    masked = deepcopy(baseline).to(candidate_device).eval()
    masked_modules = dict(masked.named_modules())
    verified, changed_modules, masked_inputs = 0, 0, 0
    with torch.no_grad():
        for name, module in source.items():
            current = target[name]
            if type(module) is not type(current):
                raise ValueError(f"Jenis modul berubah: {name}.")
            if isinstance(module, torch.nn.Conv2d):
                inputs, outputs = (mappings.pop((name, dimension)) for dimension in ("in", "out"))
                attributes = ("groups", "kernel_size", "stride", "padding", "dilation", "padding_mode")
                if any(getattr(module, key) != getattr(current, key) for key in attributes):
                    raise ValueError(f"Konfigurasi Conv berubah: {name}.")
                if ((inputs["channels_before"], outputs["channels_before"]) != (module.in_channels, module.out_channels)
                        or (inputs["channels_after"], outputs["channels_after"]) != (current.in_channels, current.out_channels)):
                    raise ValueError(f"Lebar Conv berbeda dari ledger: {name}.")
                dropped = inputs["removed_original_indices"]
                sliced = bool(dropped or outputs["removed_original_indices"])
                if sliced and (module.groups != 1 or current.groups != 1):
                    raise ValueError(f"Oracle tidak mendukung perubahan grouped/depthwise Conv: {name}.")
                weight, bias = module.weight, module.bias
                if sliced:
                    weight = _gflops_index(weight, 0, outputs["kept_original_indices"])
                    weight = _gflops_index(weight, 1, inputs["kept_original_indices"])
                    if bias is not None:
                        bias = _gflops_index(bias, 0, outputs["kept_original_indices"])
                    changed_modules += 1
                if dropped:
                    consumer = masked_modules[name]
                    consumer.weight.index_fill_(1, torch.tensor(dropped, device=candidate_device, dtype=torch.long), 0)
                    masked_inputs += 1
                tensors = [("weight", weight), ("bias", bias)]
            elif isinstance(module, torch.nn.BatchNorm2d):
                channels = mappings.pop((name, "channel"))
                if (channels["channels_before"] != module.num_features
                        or channels["channels_after"] != current.num_features or module.training or current.training):
                    raise ValueError(f"Lebar atau mode BN berbeda dari ledger: {name}.")
                if any(getattr(module, key) != getattr(current, key) for key in ("eps", "affine", "track_running_stats")):
                    raise ValueError(f"Konfigurasi BN berubah: {name}.")
                sliced = bool(channels["removed_original_indices"])
                changed_modules += int(sliced)
                tensors = []
                for key in ("weight", "bias", "running_mean", "running_var", "num_batches_tracked"):
                    value = getattr(module, key)
                    if sliced and value is not None and key != "num_batches_tracked":
                        value = _gflops_index(value, 0, channels["kept_original_indices"])
                    tensors.append((key, value))
            else:
                continue
            for key, value in tensors:
                actual = getattr(current, key)
                if ((value is None) != (actual is None)
                        or (value is not None and (value.dtype != actual.dtype or not torch.equal(value.to(actual.device), actual)))):
                    raise ValueError(f"Irisan tensor berbeda dari baseline: {name}.{key}.")
                verified += int(value is not None)
        if mappings:
            raise ValueError("Ledger memuat dimensi tanpa modul Conv/BN yang sesuai.")

        example = image.to(candidate_device)
        cudnn = torch.backends.cudnn
        # Dense and compact kernels can choose different accumulation paths. Use
        # FP32 for the comparison on CUDA and restore the caller's backend flags.
        precision = (cudnn.flags(enabled=cudnn.enabled, benchmark=cudnn.benchmark,
                                 deterministic=cudnn.deterministic, allow_tf32=False)
                     if candidate_device.type == "cuda" else nullcontext())
        with precision:
            oracle_outputs = _v2_trace_outputs(masked(example))
            compact_outputs = _v2_trace_outputs(pruned(example))
        metrics = {}
        names = ("one2many.boxes", "one2many.scores", "one2one.boxes", "one2one.scores")
        for name, reference, actual in zip(names, oracle_outputs, compact_outputs):
            if (reference.shape != actual.shape or not torch.isfinite(reference).all()
                    or not torch.isfinite(actual).all()
                    or not torch.allclose(reference, actual, rtol=1e-4, atol=1e-5)):
                raise ValueError(f"Forward raw berbeda dari oracle masked: {name}.")
            metrics[name] = {"shape": list(actual.shape), "max_abs_diff": float((reference - actual).abs().max())}
    return {"status": "PASS", "conv_bn_tensors_verified": verified, "changed_conv_bn_modules": changed_modules,
            "masked_conv_consumers": masked_inputs, "raw_outputs": metrics, "rtol": 1e-4, "atol": 1e-5,
            "scope": "Exact baseline slices and dense consumer-column mask parity on the supplied FP32 input."}
