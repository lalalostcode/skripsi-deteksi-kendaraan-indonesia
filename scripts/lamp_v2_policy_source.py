"""Standalone LAMP V2 policy source, embedded verbatim in the Kaggle notebook.

This file is an authoring source, not a notebook runtime dependency.
"""
import math
import warnings

import torch
import torch_pruning as tp
from ultralytics.nn.modules import Bottleneck, C2f, C3k2, Detect
from ultralytics.nn.modules.block import C2PSA, Attention, PSABlock
from ultralytics.nn.tasks import DetectionModel


def _v2_trace_outputs(output):
    if not isinstance(output, tuple) or len(output) != 2 or not isinstance(output[1], dict):
        raise ValueError("Output YOLO26 eval harus (decoded, raw).")
    if set(output[1]) != {"one2many", "one2one"}:
        raise ValueError("Graph wajib mencakup raw output kedua cabang.")
    return [output[1][branch][key] for branch in ("one2many", "one2one")
            for key in ("boxes", "scores")]


def _v2_policy_structure(core, model_variant=None):
    if not isinstance(core, DetectionModel):
        raise TypeError("Policy memerlukan DetectionModel YOLO26 unfused.")
    modules = dict(core.named_modules())
    heads = [(name, module) for name, module in modules.items() if isinstance(module, Detect)]
    if len(heads) != 1:
        raise ValueError("Diperlukan satu Detect head.")
    head_name, head = heads[0]
    if any(getattr(head, name, None) is None for name in ("cv2", "cv3", "one2one_cv2", "one2one_cv3")):
        raise ValueError("Detection head kedua cabang harus unfused dan lengkap.")
    first = next(module for module in modules.values() if isinstance(module, torch.nn.Conv2d))
    actual_variant = core.yaml.get("scale")
    if actual_variant in (None, ""):
        actual_variant = {16: "n", 32: "s"}.get(first.out_channels)
    if model_variant is not None and model_variant not in ("n", "s"):
        raise ValueError("Model variant harus n atau s.")
    if ("yolo26" not in str(core.yaml.get("yaml_file", "")).lower()
            or core.yaml.get("reg_max") != 1 or actual_variant not in ("n", "s")
            or first.out_channels != {"n": 16, "s": 32}.get(actual_variant)):
        raise ValueError("V2 memerlukan arsitektur standar YOLO26n/S dengan stem 16/32 yang sesuai.")
    if model_variant is not None and model_variant != actual_variant:
        raise ValueError(f"Model variant tidak cocok: diminta {model_variant}, checkpoint {actual_variant}.")
    envelopes = [(name, module) for name, module in modules.items()
                 if isinstance(module, (Detect, C2PSA, PSABlock, Attention))
                 or type(module).__name__ == "AAttn"]
    protected = {}
    for name, module in modules.items():
        for prefix, envelope in envelopes:
            if name == prefix or name.startswith(prefix + "."):
                protected[module] = "detection_head_all_dimensions" if isinstance(envelope, Detect) else "attention_envelope_all_dimensions"
                break
    split_outputs = {}
    for name, module in modules.items():
        if isinstance(module, (C2f, C3k2)):
            for child in (module.cv1.conv, getattr(module.cv1, "bn", None)):
                if child is not None:
                    split_outputs[child] = "CSP_split_output_dimension"
    conservative = {}
    for name, module in modules.items():
        if not isinstance(module, Bottleneck) or module in protected:
            continue
        producer, consumer = module.cv1.conv, module.cv2.conv
        bn = getattr(module.cv1, "bn", None)
        if (not isinstance(bn, torch.nn.BatchNorm2d) or producer.groups != 1 or consumer.groups != 1
                or producer.out_channels != consumer.in_channels or producer.out_channels != bn.num_features):
            raise ValueError(f"Bottleneck hidden channel tidak didukung: {name}")
        conservative[producer] = {"name": name + ".cv1.conv", "bn": bn, "consumer": consumer}
    if not conservative:
        raise ValueError("Tidak ada kandidat konservatif.")
    return {"model_variant": actual_variant, "modules": modules, "names": {module: name for name, module in modules.items()},
            "head": head, "head_name": head_name, "protected": protected,
            "split_outputs": split_outputs, "conservative": conservative}


def protected_policy_snapshot(core, model_variant=None):
    """Snapshot seluruh bobot, buffer, dan kontrak head/attention yang dilindungi."""
    structure = _v2_policy_structure(core, model_variant)
    result = {}
    for name, module in structure["modules"].items():
        if isinstance(module, (Detect, C2PSA, PSABlock, Attention)) or type(module).__name__ == "AAttn":
            result[name] = {"state": {key: value.detach().cpu().clone() for key, value in module.state_dict().items()},
                            "dimensions": {key: getattr(module, key) for key in ("nc", "no", "nl", "reg_max", "end2end", "c")
                                           if hasattr(module, key)}}
    return result


def assert_policy_protected_unchanged(core, snapshot):
    modules = dict(core.named_modules())
    for name, entry in snapshot.items():
        if name not in modules:
            raise ValueError(f"Envelope terlindungi hilang: {name}")
        module = modules[name]
        after = module.state_dict()
        if after.keys() != entry["state"].keys() or any(
                not torch.equal(value.detach().cpu(), entry["state"][key]) for key, value in after.items()):
            raise ValueError(f"Bobot/buffer envelope terlindungi berubah: {name}")
        if any(getattr(module, key) != value for key, value in entry["dimensions"].items()):
            raise ValueError(f"Kontrak envelope terlindungi berubah: {name}")


def _v2_output_signature(group, graph, names):
    """Identitas satu group adalah seluruh dimensi Conv-output yang terikat."""
    return tuple(sorted(names[dep.target.module] for dep, _ in group
                        if isinstance(dep.target.module, torch.nn.Conv2d)
                        and graph.is_out_channel_pruning_fn(dep.handler)))


def _v2_validate_group(group, graph, structure, policy):
    root = group[0].dep.target.module
    root_name = structure["names"].get(root, "<unknown>")
    entry = structure["conservative"].get(root)
    if policy == "conservative" and entry is None:
        raise ValueError("root_not_conservative_hidden_channel")
    for dep, indices in group:
        module, handler = dep.target.module, dep.handler
        name = structure["names"].get(module, type(module).__name__)
        output = graph.is_out_channel_pruning_fn(handler)
        if module in structure["protected"]:
            raise ValueError(f"{name}: {structure['protected'][module]}")
        if output and module in structure["split_outputs"]:
            raise ValueError(f"{name}: {structure['split_outputs'][module]}")
        if isinstance(module, torch.nn.Conv2d):
            if not output and not graph.is_in_channel_pruning_fn(handler):
                raise ValueError(f"{name}: unknown_conv_pruning_handler")
            if module.groups != 1:
                raise ValueError(f"{name}: grouped_or_depthwise_conv_unsupported")
            if policy == "conservative" and not (
                    (module is root and output) or (module is entry["consumer"] and graph.is_in_channel_pruning_fn(handler))):
                raise ValueError(f"{name}: outside_conservative_hidden_channel")
        elif isinstance(module, torch.nn.BatchNorm2d):
            if policy == "conservative" and module is not entry["bn"]:
                raise ValueError(f"{name}: outside_conservative_batchnorm")
        else:
            kind = type(module).__name__
            if kind not in {"_ElementWiseOp", "_ConcatOp"}:
                raise ValueError(f"{name}: unsupported_operation_{kind}")
            if kind == "_ElementWiseOp":
                operation = str(module)
                supported = ("SiluBackward", "AddBackward", "MaxPool2DWithIndicesBackward", "UpsampleNearest2DBackward", "CloneBackward")
                if not any(token.lower() in operation.lower() for token in supported):
                    raise ValueError(f"{name}: unsupported_elementwise_{operation}")
        # Torch-Pruning maps concat offsets and coupled residual identities. Reject invalid maps.
        mapped = [int(index) for index in indices]
        if len(mapped) != len(set(mapped)) or any(index < 0 for index in mapped):
            raise ValueError(f"{name}: invalid_or_duplicate_index_mapping")
        channels = graph.get_out_channels(module) if output else graph.get_in_channels(module)
        if channels is not None and any(index >= channels for index in mapped):
            raise ValueError(f"{name}: dependency_index_out_of_bounds")
    return root_name


def _v2_channel_ledger(structure):
    """Track each physical model dimension independently in original index space."""
    ledger = {}
    for name, module in structure["modules"].items():
        if isinstance(module, torch.nn.Conv2d):
            dimensions = {"in": module.in_channels, "out": module.out_channels}
        elif isinstance(module, torch.nn.BatchNorm2d):
            dimensions = {"channel": module.num_features}
        else:
            continue
        for dimension, count in dimensions.items():
            ledger[(module, dimension)] = {
                "layer": name, "module_type": type(module).__name__, "dimension": dimension,
                "channels_before": count, "kept": list(range(count)), "group_ids": set()}
    return ledger


def _v2_dependency_dimension(dependency, graph):
    module = dependency.target.module
    if isinstance(module, torch.nn.BatchNorm2d):
        return module, "channel"
    if isinstance(module, torch.nn.Conv2d):
        return module, "out" if graph.is_out_channel_pruning_fn(dependency.handler) else "in"
    return None


def _v2_refresh_channel_report(audit):
    mappings, masks = [], []
    for entry in audit["channel_ledger"].values():
        kept = set(entry["kept"])
        removed = [index for index in range(entry["channels_before"]) if index not in kept]
        row = {key: entry[key] for key in ("layer", "module_type", "dimension", "channels_before")}
        row.update(channels_after=len(entry["kept"]), kept_original_indices=list(entry["kept"]),
                   removed_original_indices=removed, dependency_group_ids=sorted(entry["group_ids"]))
        mappings.append(row)
        masks.append({"layer": row["layer"], "module_type": row["module_type"],
                      "dimension": row["dimension"], "channels_before": row["channels_before"],
                      "channels_after": row["channels_after"],
                      "dependency_group_ids": row["dependency_group_ids"],
                      "kept_mask": [index in kept for index in range(entry["channels_before"])]})
    audit["report"]["channel_mapping"] = mappings
    audit["report"]["channel_masks"] = masks


def _v2_capacity(entry):
    current = entry["root"].out_channels
    capacity = current - 1
    for module, initial in entry["output_dimensions"].items():
        capacity = min(capacity, math.floor(initial * 0.5) - (initial - module.out_channels))
    return max(0, capacity)


def build_policy_pruner(core, image, policy, ratio, model_variant="n"):
    """Discover, audit and canonicalize dependencies before global LAMP ranking."""
    if policy not in ("conservative", "extended"):
        raise ValueError("Policy harus conservative atau extended.")
    if not 0 < float(ratio) <= 0.5:
        raise ValueError("Rasio harus di dalam (0, 0.50].")
    core.eval().requires_grad_(True)
    structure = _v2_policy_structure(core, model_variant)
    importance = tp.importance.LAMPImportance(p=2, group_reduction="mean", bias=False)
    initial_ignored = [module for module in structure["modules"].values()
                       if isinstance(module, torch.nn.Conv2d) and
                       (module in structure["protected"] or module in structure["split_outputs"]
                        or (policy == "conservative" and module not in structure["conservative"]))]
    with torch.enable_grad(), warnings.catch_warnings(record=True) as captured:
        warnings.simplefilter("always")
        pruner = tp.pruner.BasePruner(core, image, importance=importance,
            global_pruning=True, pruning_ratio=float(ratio), iterative_steps=1,
            max_pruning_ratio=0.50, round_to=None, root_module_types=[torch.nn.Conv2d],
            ignored_layers=initial_ignored,
            forward_fn=lambda model, example: model(example), output_transform=_v2_trace_outputs)
    graph = pruner.DG
    eligible, rejected, seen, seen_outputs = {}, [], set(), set()
    # Inspect every Conv root explicitly, including rejected roots. The later canonical
    # scan follows BasePruner's own graph order to preserve V1 ranking semantics.
    for name, module in structure["modules"].items():
        if not isinstance(module, torch.nn.Conv2d):
            continue
        if module in initial_ignored:
            reason = structure["protected"].get(module) or structure["split_outputs"].get(module) or "root_not_conservative_hidden_channel"
            rejected.append({"root": name, "reason": reason})
            continue
        group = graph.get_pruning_group(module, tp.prune_conv_out_channels, list(range(module.out_channels)))
        try:
            _v2_validate_group(group, graph, structure, policy)
        except ValueError as error:
            rejected.append({"root": name, "reason": str(error)})
    rejected_roots = {entry["root"] for entry in rejected}
    ignored = [module for name, module in structure["modules"].items()
               if isinstance(module, torch.nn.Conv2d) and name in rejected_roots]
    for group in graph.get_all_groups(ignored_layers=ignored, root_module_types=[torch.nn.Conv2d]):
        root_name = _v2_validate_group(group, graph, structure, policy)
        signature = _v2_output_signature(group, graph, structure["names"])
        if not signature or signature in seen:
            continue
        overlap = seen_outputs.intersection(signature)
        if overlap:
            raise ValueError(f"Canonical dependency groups overlap output dimensions: {sorted(overlap)}")
        seen.add(signature)
        seen_outputs.update(signature)
        root = group[0].dep.target.module
        score = importance(group)
        if score is None or score.numel() != root.out_channels or not torch.isfinite(score).all():
            raise ValueError(f"Importance LAMP invalid: {root_name}")
        dimensions = {dep.target.module: int(dep.target.module.out_channels)
                      for dep, _ in group if isinstance(dep.target.module, torch.nn.Conv2d)
                      and graph.is_out_channel_pruning_fn(dep.handler)}
        membership = {dep.target.module: ("out" if graph.is_out_channel_pruning_fn(dep.handler) else "in")
                      for dep, _ in group
                      if isinstance(dep.target.module, (torch.nn.Conv2d, torch.nn.BatchNorm2d))}
        capacity = min(root.out_channels - 1, *(math.floor(initial * 0.5) for initial in dimensions.values()))
        permitted = set(torch.argsort(score, stable=True)[:capacity].detach().cpu().tolist())
        eligible[root] = {"name": root_name, "root": root, "signature": signature,
                          "initial_channels": root.out_channels, "output_dimensions": dimensions,
                          "member_dimensions": membership,
                          "initial_lamp_scores": score.detach().cpu().tolist(),
                          "initial_capacity_mask": [index in permitted for index in range(root.out_channels)],
                          "group_id": f"{policy}-group-{len(eligible):03d}",
                          "dependencies": [str(dep) for dep, _ in group]}
    if not eligible:
        raise ValueError("Audit tidak menemukan group aman untuk policy ini.")
    pruner.ignored_layers = ignored
    audit = {"structure": structure, "eligible": eligible, "policy": policy, "ratio": float(ratio),
             "model_variant": structure["model_variant"], "channel_ledger": _v2_channel_ledger(structure),
             "independent_channels_before": sum(entry["initial_channels"] for entry in eligible.values()),
             "physical_conv_output_channels_before": sum(sum(entry["output_dimensions"].values()) for entry in eligible.values())}
    audit["report"] = {"model_variant": structure["model_variant"], "policy": policy,
         "requested_ratio": float(ratio), "max_pruning_ratio": 0.5,
         "importance": "LAMPImportance(p=2, group_reduction='mean', bias=False)",
         "ranking": "BasePruner global one-step; capacity exclusions are +inf sentinels, finite eligible LAMP scores unchanged",
         "group_count": len(eligible), "independent_channels_before": audit["independent_channels_before"],
         "physical_conv_output_channels_before": audit["physical_conv_output_channels_before"],
         "eligible_groups": [{key: value for key, value in entry.items() if key not in ("root", "output_dimensions", "member_dimensions")}
                             for entry in eligible.values()],
         "rejected_groups": rejected, "warnings": [str(item.message) for item in captured],
         "lamp_scores": [{"group_id": entry["group_id"], "root": entry["name"],
                          "channel_idx": index, "score": score,
                          "capacity_eligible": entry["initial_capacity_mask"][index], "removed": False}
                         for entry in eligible.values() for index, score in enumerate(entry["initial_lamp_scores"])],
         "channel_mapping_status": "RECORDED_FROM_LIVE_DEPENDENCY_GRAPH",
         "channel_mapping_index_space": "Original model dimension indices; current indices translated before every successful prune"}
    for entry in eligible.values():
        group = graph.get_pruning_group(entry["root"], tp.prune_conv_out_channels,
                                        list(range(entry["initial_channels"])))
        for dependency, _ in group:
            dimension = _v2_dependency_dimension(dependency, graph)
            if dimension is not None:
                audit["channel_ledger"][dimension]["group_ids"].add(entry["group_id"])
    _v2_refresh_channel_report(audit)

    def capacity_limited_lamp(group):
        _v2_validate_group(group, graph, structure, policy)
        entry = eligible.get(group[0].dep.target.module)
        if entry is None:
            raise ValueError("Proposal berakar di luar group yang diaudit.")
        score = importance(group)
        current = entry["root"].out_channels
        if score is None or score.numel() != current or not torch.isfinite(score).all():
            raise ValueError("Importance LAMP tidak finite sebelum cap.")
        # Bound every coupled output dimension, preserving the original LAMP score.
        capacity = _v2_capacity(entry)
        permitted = torch.argsort(score, stable=True)[:capacity]
        capped = torch.full_like(score, float("inf"))
        capped[permitted] = score[permitted]
        return capped
    pruner.importance = capacity_limited_lamp
    return pruner, audit


def apply_policy_pruning(pruner, audit):
    """Audit each concrete proposal before irreversible tensor shape mutation."""
    history = []
    graph, structure = pruner.DG, audit["structure"]
    # The runtime report retains this live list even if a later proposal raises.
    audit["report"]["pruning_history"] = history
    audit["report"]["pruning_status"] = "IN_PROGRESS"
    for group in pruner.step(interactive=True):
        audit["report"]["last_proposal"] = {
            "root": structure["names"].get(group[0].dep.target.module, "<unknown>"),
            "removed_indices": [int(index) for index in group[0].idxs]}
        name = _v2_validate_group(group, graph, structure, audit["policy"])
        entry = audit["eligible"].get(group[0].dep.target.module)
        if entry is None:
            raise ValueError("Group proposal tidak terdapat pada audit.")
        indices = [int(index) for index in group[0].idxs]
        if not indices:
            continue
        if _v2_output_signature(group, graph, structure["names"]) != entry["signature"]:
            raise ValueError("Dimensi output proposal berbeda dari group canonical yang diaudit.")
        if not graph.check_pruning_group(group):
            raise ValueError(f"Dependency group tidak valid: {name}")
        mapped_dimensions = {}
        for dependency, mapped in group:
            dimension = _v2_dependency_dimension(dependency, graph)
            if dimension is None:
                continue
            # A residual/concat may reach a physical dimension along multiple paths.
            # The ledger records its union once, in the current dimension's space.
            mapped_dimensions.setdefault(dimension, set()).update(int(index) for index in mapped)
        dimension_records = []
        for dimension, mapped in mapped_dimensions.items():
            ledger = audit["channel_ledger"][dimension]
            current_indices = sorted(mapped)
            if any(index < 0 or index >= len(ledger["kept"]) for index in current_indices):
                raise ValueError(f"Ledger dependency index out of bounds: {ledger['layer']}:{dimension[1]}")
            dimension_records.append({"layer": ledger["layer"], "module_type": ledger["module_type"],
                "dimension": dimension[1], "current_indices": current_indices,
                "original_indices": [ledger["kept"][index] for index in current_indices],
                "channels_before": len(ledger["kept"]), "channels_after": len(ledger["kept"]) - len(mapped)})
        physical_removed = 0
        touched = []
        for (module, dimension), mapped in mapped_dimensions.items():
            if isinstance(module, torch.nn.Conv2d) and dimension == "out":
                if module not in entry["output_dimensions"]:
                    raise ValueError("Proposal mengubah output di luar group canonical.")
                initial = entry["output_dimensions"][module]
                remaining = module.out_channels - len(mapped)
                if remaining < 1 or initial - remaining > math.floor(initial * 0.5):
                    raise ValueError(f"Cap 50%/minimum satu channel dilanggar: {structure['names'][module]}")
                physical_removed += len(mapped)
                touched.append({"layer": structure["names"][module], "before": module.out_channels, "after": remaining})
        before = entry["root"].out_channels
        root_ledger = audit["channel_ledger"][(entry["root"], "out")]
        original_root_indices = [root_ledger["kept"][index] for index in indices]
        record = {"root": name, "group_id": entry["group_id"], "removed_indices": indices,
                  "removed_original_indices": original_root_indices, "mapped_dimensions": dimension_records,
                  "channels_before": before, "channels_after": before - len(indices),
                  "physical_conv_output_channels_removed": physical_removed,
                  "coupled_conv_output_count": len(touched),
                  "output_changes": touched, "dependencies": [str(dep) for dep, _ in group]}
        group.prune()
        for (module, dimension), mapped in mapped_dimensions.items():
            ledger = audit["channel_ledger"][(module, dimension)]
            ledger["kept"] = [original for index, original in enumerate(ledger["kept"]) if index not in mapped]
            ledger["group_ids"].add(entry["group_id"])
            actual = (module.in_channels if dimension == "in" else module.out_channels) if isinstance(
                module, torch.nn.Conv2d) else module.num_features
            if len(ledger["kept"]) != actual:
                raise ValueError(f"Ledger/model channel mismatch: {ledger['layer']}:{dimension}")
        removed_set = set(original_root_indices)
        for row in audit["report"]["lamp_scores"]:
            if row["group_id"] == entry["group_id"] and row["channel_idx"] in removed_set:
                row["removed"] = True
        _v2_refresh_channel_report(audit)
        history.append(record)
        audit["report"]["independent_channels_after"] = sum(entry["root"].out_channels for entry in audit["eligible"].values())
        audit["report"]["independent_channels_removed"] = sum(item["channels_before"] - item["channels_after"] for item in history)
        audit["report"]["physical_conv_output_channels_removed"] = sum(item["physical_conv_output_channels_removed"] for item in history)
    if not history:
        raise RuntimeError("Tidak ada channel struktural dihapus.")
    audit["report"]["pruning_status"] = "PASS"
    return history


def measure_layer_inventory(core, image, audit, policy):
    """Per-layer Conv/BN estimates at the actual input size, on an unfused model."""
    shapes, handles = {}, []
    for name, module in core.named_modules():
        if isinstance(module, (torch.nn.Conv2d, torch.nn.BatchNorm2d)):
            def hook(layer, args, output, name=name):
                shapes[name] = list(output.shape)
            handles.append(module.register_forward_hook(hook))
    try:
        with torch.no_grad():
            core(image)
    finally:
        for handle in handles:
            handle.remove()
    structure = audit["structure"]
    membership, member_dimensions = {}, {}
    for entry in audit["eligible"].values():
        for module, dimension in entry["member_dimensions"].items():
            membership.setdefault(module, []).append(entry["group_id"])
            member_dimensions.setdefault(module, set()).add(dimension)
    rejected = {item["root"]: item["reason"] for item in audit["report"]["rejected_groups"]}
    backbone_count = len(core.yaml.get("backbone", []))
    rows = []
    for name, module in core.named_modules():
        if not isinstance(module, (torch.nn.Conv2d, torch.nn.BatchNorm2d)):
            continue
        shape = shapes.get(name, [])
        if isinstance(module, torch.nn.Conv2d):
            inputs, outputs = module.in_channels, module.out_channels
            operations = (2 * math.prod(shape) * (inputs // module.groups) * math.prod(module.kernel_size)) if shape else None
        else:
            inputs = outputs = module.num_features
            operations = 2 * math.prod(shape) if shape else None
        stage = next((i for i, block in enumerate(core.model)
                      if any(child is module for child in block.modules())), None)
        reason = structure["protected"].get(module) or structure["split_outputs"].get(module) or rejected.get(name)
        rows.append({"layer_name": name, "module_type": type(module).__name__,
                     "region": "detection_head" if module in structure["protected"] and str(reason).startswith("detection")
                               else ("backbone" if stage is not None and stage < backbone_count else "neck"),
                     "in_channels": inputs, "out_channels": outputs,
                     "parameters": sum(parameter.numel() for parameter in module.parameters(recurse=False)),
                     "estimated_flops": operations, "flops_estimate_method": "Conv:2*elements*Cin/groups*Kh*Kw; BN:2*elements; functional attention excluded",
                     "policy": policy, "eligible": module in membership,
                     "protection_reason": reason or ("not_in_eligible_dependency_group" if module not in membership else ""),
                     "dependency_group": ";".join(membership.get(module, [])),
                     "eligible_dimensions": ";".join(sorted(member_dimensions.get(module, set()))), "output_shape": shape})
    return rows
