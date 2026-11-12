"""Standalone analysis helpers embedded in notebook 03 by its generator.

All figures describe measured structural changes, not activation feature maps.
Missing historical mappings, scores, and validation values are never fabricated.
"""
import csv as _eda_csv
import json as _eda_json
import math as _eda_math
from pathlib import Path as _EdaPath

import matplotlib.pyplot as _eda_plt
import numpy as _eda_np
from matplotlib.colors import ListedColormap as _EdaColors


def _eda_write_json(path, value):
    path = _EdaPath(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(_eda_json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")


def _eda_write_csv(path, rows):
    path = _EdaPath(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    keys = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = _eda_csv.DictWriter(stream, fieldnames=keys)
        writer.writeheader()
        writer.writerows({key: _eda_json.dumps(value, ensure_ascii=False) if isinstance(value, (dict, list)) else value
                         for key, value in row.items()} for row in rows)


def _eda_finite(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool) and _eda_math.isfinite(value)


def _eda_save_figure(fig, path):
    path = _EdaPath(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path.with_suffix(".png"), dpi=int(globals().get("EDA_DPI", 300)), bbox_inches="tight")
    fig.savefig(path.with_suffix(".svg"), bbox_inches="tight")
    _eda_plt.close(fig)
    return {"png": path.with_suffix(".png").name, "svg": path.with_suffix(".svg").name}


def layer_eda_rows(inventory_before, inventory_after):
    """Join actual unfused inventory by module name, preserving baseline order."""
    previous = {row["layer_name"]: row for row in inventory_before}
    current = {row["layer_name"]: row for row in inventory_after}
    if len(previous) != len(inventory_before) or len(current) != len(inventory_after):
        raise ValueError("EDA inventory has duplicate module names.")
    if set(previous) != set(current):
        raise ValueError("EDA before/after module names differ; layer alignment is unsafe.")
    rows = []
    for name, before in previous.items():
        after = current[name]
        if before["module_type"] != after["module_type"]:
            raise ValueError(f"EDA module type changed at {name}.")
        if after["out_channels"] > before["out_channels"] or after["in_channels"] > before["in_channels"]:
            raise ValueError(f"Pruning unexpectedly increased channels at {name}.")
        parts = name.split(".")
        stage = before.get("stage", ".".join(parts[:2]))
        row = {"layer_name": name, "module_type": before["module_type"], "stage": stage,
               "region": before.get("region", "unknown"), "policy": before.get("policy"),
               "eligible": before.get("eligible", False), "protection_reason": before.get("protection_reason", ""),
               "dependency_group": before.get("dependency_group", "")}
        for key in ("in_channels", "out_channels", "parameters", "estimated_flops"):
            row[key + "_before"] = before.get(key)
            row[key + "_after"] = after.get(key)
            base, value = before.get(key), after.get(key)
            row[key + "_removed"] = base - value if _eda_finite(base) and _eda_finite(value) else None
            row[key + "_reduction_pct"] = 100 * (base - value) / base if _eda_finite(base) and _eda_finite(value) and base else None
        if (row["protection_reason"] and str(row["protection_reason"]).startswith(("detection", "attention"))
                and (row["out_channels_removed"] or row["in_channels_removed"])):
            raise ValueError(f"Protected head/attention channel dimension changed at {name}.")
        rows.append(row)
    return rows


def stage_eda_rows(rows):
    """Physical Conv-output sums; these are distinct from independent group channels."""
    stages = {}
    for row in rows:
        key = (str(row["stage"]), row["region"])
        stage = stages.setdefault(key, {"stage": row["stage"], "region": row["region"], "conv_layers": 0,
                                      "physical_conv_outputs_before": 0, "physical_conv_outputs_after": 0,
                                      "parameters_before": 0, "parameters_after": 0,
                                      "estimated_flops_before": 0, "estimated_flops_after": 0})
        if row["module_type"] == "Conv2d":
            stage["conv_layers"] += 1
            for suffix in ("before", "after"):
                stage["physical_conv_outputs_" + suffix] += row["out_channels_" + suffix]
        for metric in ("parameters", "estimated_flops"):
            for suffix in ("before", "after"):
                value = row[metric + "_" + suffix]
                if _eda_finite(value):
                    stage[metric + "_" + suffix] += value
    for row in stages.values():
        row["physical_conv_outputs_removed"] = row["physical_conv_outputs_before"] - row["physical_conv_outputs_after"]
        for key in ("physical_conv_outputs", "parameters", "estimated_flops"):
            base, after = row[key + "_before"], row[key + "_after"]
            row[key + "_reduction_pct"] = 100 * (base - after) / base if base else None
    return list(stages.values())


def make_channel_figure(rows, title="Channel comparison"):
    """Three panels share exact Conv order and Y limits, as in the reference image."""
    if not rows or any(row["module_type"] != "Conv2d" for row in rows):
        raise ValueError("Channel diagram requires a nonempty ordered Conv2d inventory.")
    count = len(rows)
    width = max(16, min(32, count * 0.16))
    fig, axes = _eda_plt.subplots(3, 1, figsize=(width, 12), sharex=True, sharey=True)
    indices = _eda_np.arange(count)
    before = [row["out_channels_before"] for row in rows]
    after = [row["out_channels_after"] for row in rows]
    upper = max(max(before), max(after), 1) * 1.12
    axes[0].bar(indices, before, color="blue", width=0.82, label="Before Pruning")
    axes[1].bar(indices, after, color="red", width=0.82, label="After Pruning")
    axes[2].bar(indices, before, color="blue", width=0.82, label="Before Pruning")
    axes[2].bar(indices, after, color="red", width=0.82, label="After Pruning")
    for ax, heading in zip(axes, ("Before Pruning", "After Pruning", "Channel Comparison")):
        ax.set_title(heading)
        ax.set_ylabel("Output channels")
        ax.set_ylim(0, upper)
        ax.set_xlim(-0.8, count - 0.2)
        ax.grid(axis="y", alpha=0.2)
        ax.set_axisbelow(True)
        ax.legend(loc="upper right")
        ax.tick_params(axis="x", labelbottom=True)
        ax.set_xticks(indices, [row["layer_name"] for row in rows], rotation=90, fontsize=6 if count > 32 else 8)
        ax.set_xlabel("Conv module (baseline order)")
    fig.suptitle(title, y=1.005, fontsize=14)
    fig.tight_layout()
    return fig


def _eda_channel_plots(conv_rows, folder, title):
    _eda_write_csv(folder / "channel_plot_data.csv", conv_rows)
    files = [_eda_save_figure(make_channel_figure(conv_rows, title), folder / "channels_overview")]
    page_size = min(32, max(1, int(globals().get("EDA_MAX_LAYERS_PER_PAGE", 32))))
    for offset in range(0, len(conv_rows), page_size):
        page = offset // page_size + 1
        files.append(_eda_save_figure(make_channel_figure(conv_rows[offset:offset + page_size],
                     f"{title} — layers {offset + 1}–{min(offset + page_size, len(conv_rows))}"),
                     folder / f"channels_detail_{page:02d}"))
    fig, ax = _eda_plt.subplots(figsize=(max(14, min(32, len(conv_rows) * 0.16)), 5))
    ax.bar(range(len(conv_rows)), [row["out_channels_reduction_pct"] for row in conv_rows], color="red")
    ax.set_xticks(range(len(conv_rows)), [row["layer_name"] for row in conv_rows], rotation=90, fontsize=6)
    ax.set_ylabel("Output channels removed (%)")
    ax.set_title(title + " — physical Conv-output reduction")
    ax.set_ylim(0, 100)
    fig.tight_layout()
    files.append(_eda_save_figure(fig, folder / "channel_reduction_percent"))
    return {"status": "MEASURED", "conv_layers": len(conv_rows), "detail_page_size": page_size, "files": files}


def _eda_mapping_plots(audit, conv_rows, folder):
    mapping = audit.get("channel_mapping")
    if not mapping:
        return {"status": "UNAVAILABLE", "reason": "Original channel indices were not recorded; no reconstruction attempted."}
    _eda_write_json(folder / "channel_mapping.json", mapping)
    _eda_write_csv(folder / "channel_mapping.csv", mapping)
    masks = audit.get("channel_masks")
    if masks:
        _eda_write_json(folder / "channel_masks.json", masks)
    by_layer = {row["layer"]: row for row in mapping if row.get("module_type") == "Conv2d"
                and row.get("dimension") in {"out", "out_channels", "output", "output_channels"}}
    selected = []
    for row in conv_rows:
        item = by_layer.get(row["layer_name"])
        if item is None:
            return {"status": "UNAVAILABLE", "reason": "Mapping does not cover every Conv output.", "layer": row["layer_name"]}
        before = int(item["channels_before"])
        kept, removed = item.get("kept_original_indices"), item.get("removed_original_indices")
        if not isinstance(kept, list) or not isinstance(removed, list):
            raise TypeError(f"Missing original-index ledger for {row['layer_name']}.")
        if (len(set(kept)) != len(kept) or len(set(removed)) != len(removed)
                or set(kept) & set(removed) or set(kept) | set(removed) != set(range(before))
                or len(kept) != item["channels_after"]
                or before != row["out_channels_before"] or len(kept) != row["out_channels_after"]):
            raise ValueError(f"Original-index ledger disagrees with measured channels at {row['layer_name']}.")
        selected.append((row["layer_name"], before, set(removed)))
    page_size = min(32, max(1, int(globals().get("EDA_MAX_LAYERS_PER_PAGE", 32))))
    files = []
    for offset in range(0, len(selected), page_size):
        page = selected[offset:offset + page_size]
        width = max(item[1] for item in page)
        data = _eda_np.full((len(page), width), _eda_np.nan)
        for index, (_name, count, removed) in enumerate(page):
            data[index, :count] = 0
            if removed:
                data[index, sorted(removed)] = 1
        fig, ax = _eda_plt.subplots(figsize=(16, max(4, len(page) * 0.25)))
        cmap = _EdaColors(["#4477cc", "#ee3333"]).with_extremes(bad="#eeeeee")
        plot = ax.imshow(data, interpolation="nearest", aspect="auto", cmap=cmap, vmin=0, vmax=1)
        ax.set_yticks(range(len(page)), [item[0] for item in page], fontsize=7)
        ax.set_xlabel("Original Conv-output channel index (before any pruning)")
        ax.set_title("Original-index mask: blue kept, red removed, grey outside layer width")
        fig.colorbar(plot, ax=ax, ticks=[0, 1]).ax.set_yticklabels(["Kept", "Removed"])
        fig.tight_layout()
        files.append(_eda_save_figure(fig, folder / f"channel_mask_{offset // page_size + 1:02d}"))
    return {"status": "MEASURED", "conv_layers": len(selected), "files": files,
            "meaning": "Physical Conv-output masks, including coupled dimensions; never summed as independent channels."}


def _eda_lamp_plots(audit, folder):
    scores = audit.get("lamp_scores")
    if not scores:
        return {"status": "UNAVAILABLE", "reason": "Finite pre-cap LAMP scores were not recorded."}
    if any(not _eda_finite(row.get("score")) for row in scores):
        raise ValueError("LAMP EDA requires finite original scores; capacity sentinels must not be plotted.")
    _eda_write_csv(folder / "lamp_scores.csv", scores)
    _eda_write_json(folder / "lamp_scores.json", scores)
    fig, ax = _eda_plt.subplots(figsize=(10, 5))
    for removed, color, label in ((False, "blue", "Kept"), (True, "red", "Removed")):
        values = [row["score"] for row in scores if bool(row.get("removed")) == removed]
        if values:
            ax.hist(values, bins=min(50, max(5, int(len(values) ** 0.5))), alpha=0.55, color=color, label=label)
    ax.set_xlabel("Finite LAMP score before capacity mask")
    ax.set_ylabel("Independent root channels")
    ax.set_title("LAMP selection — descriptive score distribution")
    ax.legend()
    fig.tight_layout()
    files = [_eda_save_figure(fig, folder / "lamp_score_distribution")]
    capacities = {}
    for row in scores:
        group = capacities.setdefault(row["group_id"], {"channels": 0, "capacity_eligible": 0, "removed": 0})
        group["channels"] += 1
        group["capacity_eligible"] += bool(row.get("capacity_eligible"))
        group["removed"] += bool(row.get("removed"))
    capacity_rows = [{"group_id": key, **value} for key, value in capacities.items()]
    _eda_write_csv(folder / "lamp_group_capacity.csv", capacity_rows)
    if any(row["removed"] > row["capacity_eligible"] for row in capacity_rows):
        raise ValueError("LAMP selection removed channels outside the audited capacity mask.")
    fig, ax = _eda_plt.subplots(figsize=(max(12, min(24, len(capacity_rows) * .45)), 6))
    positions = _eda_np.arange(len(capacity_rows))
    blocked = [row["channels"] - row["capacity_eligible"] for row in capacity_rows]
    retained = [row["capacity_eligible"] - row["removed"] for row in capacity_rows]
    removed = [row["removed"] for row in capacity_rows]
    ax.bar(positions, blocked, color="grey", label="Protected by capacity cap")
    ax.bar(positions, retained, bottom=blocked, color="blue", label="Capacity eligible, kept")
    ax.bar(positions, removed, bottom=_eda_np.array(blocked) + retained, color="red", label="Removed")
    ax.set_xticks(positions, [row["group_id"] for row in capacity_rows], rotation=90, fontsize=7)
    ax.set_ylabel("Independent root channels")
    ax.set_title("Group capacity and actual LAMP selection")
    ax.legend()
    fig.tight_layout()
    files.append(_eda_save_figure(fig, folder / "lamp_group_capacity"))
    return {"status": "MEASURED", "finite_scores": len(scores),
            "capacity_eligible": sum(bool(row.get("capacity_eligible")) for row in scores),
            "removed": sum(bool(row.get("removed")) for row in scores), "files": files}


def _eda_efficiency_plots(stages, result, folder):
    files = []
    for metric, label in (("physical_conv_outputs", "Physical Conv-output channels"),
                          ("parameters", "Parameters (Conv + BN)"),
                          ("estimated_flops", "Estimated layer FLOPs (Conv + BN)")):
        fig, ax = _eda_plt.subplots(figsize=(14, 5))
        positions = _eda_np.arange(len(stages))
        ax.bar(positions - .2, [row[metric + "_before"] for row in stages], width=.4, color="blue", label="Before")
        ax.bar(positions + .2, [row[metric + "_after"] for row in stages], width=.4, color="red", label="After")
        ax.set_xticks(positions, [str(row["stage"]) for row in stages], rotation=90)
        ax.set_ylabel(label)
        ax.set_title("Measured stage inventory — " + label)
        ax.legend()
        fig.tight_layout()
        files.append(_eda_save_figure(fig, folder / ("stage_" + metric)))
    baseline, pruned = result.get("baseline", {}), result.get("pruned", {})
    summary = {"independent_eligible_channels_before": result.get("eligible_channels_before"),
               "independent_eligible_channels_after": result.get("eligible_channels_after"),
               "actual_independent_channel_ratio": result.get("actual_eligible_channel_ratio"),
               "physical_conv_outputs_before": sum(row["physical_conv_outputs_before"] for row in stages),
               "physical_conv_outputs_after": sum(row["physical_conv_outputs_after"] for row in stages),
               "baseline": {key: baseline.get(key) for key in ("parameters", "gflops", "checkpoint_bytes", "latency")},
               "pruned": {key: pruned.get(key) for key in ("parameters", "gflops", "checkpoint_bytes", "latency")},
               "comparison": result.get("comparison", {}),
               "latency_scope": "raw model forward, not end-to-end",
               "layer_flops_scope": "Conv/BN estimate; model GFLOPs also counts functional attention"}
    _eda_write_json(folder / "efficiency_summary.json", summary)
    return {"status": "MEASURED", "files": files, **summary}


def _eda_evaluation_plots(result, folder):
    statuses, files = {}, []
    for identity in ("baseline", "pruned"):
        validation = (result.get(identity) or {}).get("validation")
        if not validation:
            statuses[identity] = {"status": "NOT_RUN", "reason": "Dataset validation was not run."}
            continue
        target = folder / identity
        target.mkdir(parents=True, exist_ok=True)
        per_class = validation.get("per_class") or []
        if per_class:
            _eda_write_csv(target / "per_class_metrics.csv", per_class)
            labels = [row.get("name", str(row.get("class_id"))) for row in per_class]
            fig, axes = _eda_plt.subplots(1, 3, figsize=(21, 5))
            for key, label in (("ap50", "AP50"), ("ap50_95", "AP50–95")):
                indices = [index for index, row in enumerate(per_class) if _eda_finite(row.get(key))]
                if indices:
                    axes[0].plot(indices, [per_class[index][key] for index in indices], "o-", label=label)
            for key, label in (("precision", "Precision"), ("recall", "Recall"), ("f1", "F1")):
                indices = [index for index, row in enumerate(per_class) if _eda_finite(row.get(key))]
                if indices:
                    axes[1].plot(indices, [per_class[index][key] for index in indices], "o-", label=label)
            support_indices = [index for index, row in enumerate(per_class) if _eda_finite(row.get("support"))]
            if support_indices:
                axes[2].bar(support_indices, [per_class[index]["support"] for index in support_indices], color="grey")
            else:
                axes[2].text(.5, .5, "Class support unavailable", ha="center", va="center", transform=axes[2].transAxes)
            for ax in axes:
                ax.set_xticks(range(len(labels)), labels, rotation=45, ha="right")
                if ax.lines:
                    ax.legend()
            for ax in axes[:2]:
                ax.set_ylim(0, 1)
                ax.set_ylabel("Measured metric")
            axes[2].set_ylabel("Ground-truth instance support")
            fig.suptitle(identity + " — measured class metrics; missing values are omitted")
            fig.tight_layout()
            files.append(_eda_save_figure(fig, target / "class_metrics"))
        curves = validation.get("curves")
        if curves:
            _eda_write_json(target / "curves.json", curves)
        curve_status = _eda_curve_plots(curves, target) if curves else {"status": "UNAVAILABLE"}
        confusion = validation.get("confusion_matrix") or {}
        confusion_status = {"status": "UNAVAILABLE", "reason": "No measured confusion matrix was collected."}
        if (confusion.get("measured") is True or confusion.get("status") == "MEASURED") and confusion.get("matrix") is not None:
            matrix = _eda_np.asarray(confusion["matrix"], dtype=float)
            labels = confusion.get("labels", [])
            if matrix.ndim != 2 or matrix.shape != (len(labels), len(labels)) or not _eda_np.isfinite(matrix).all():
                raise ValueError("Confusion matrix dimensions/labels/values are invalid.")
            _eda_write_json(target / "confusion_matrix.json", confusion)
            _eda_write_csv(target / "confusion_matrix.csv", [
                {"predicted_class": labels[index], **{label: float(value) for label, value in zip(labels, row)}}
                for index, row in enumerate(matrix)])
            fig, ax = _eda_plt.subplots(figsize=(9, 8))
            plotted = ax.imshow(matrix, cmap="Blues")
            ax.set_xticks(range(len(labels)), labels, rotation=45, ha="right")
            ax.set_yticks(range(len(labels)), labels)
            ax.set_xlabel(confusion.get("x_label", "True class"))
            ax.set_ylabel(confusion.get("y_label", "Predicted class"))
            conf = confusion.get("confidence_threshold", confusion.get("conf"))
            matching_iou = confusion.get("matching_iou_threshold", confusion.get("iou"))
            ax.set_title(f"{identity}: measured confusion matrix\nconf={conf}, matching IoU={matching_iou}")
            fig.colorbar(plotted, ax=ax)
            fig.tight_layout()
            files.append(_eda_save_figure(fig, target / "confusion_matrix"))
            confusion_status = {"status": "MEASURED", "confidence_threshold": conf, "matching_iou_threshold": matching_iou,
                                "validation_iou": confusion.get("validation_iou")}
        statuses[identity] = {"status": "MEASURED", "per_class": "MEASURED" if per_class else "UNAVAILABLE",
                              "curves": curve_status, "confusion_matrix": confusion_status}
    return {"status": "MEASURED" if any(row["status"] == "MEASURED" for row in statuses.values()) else "NOT_RUN",
            "measurements": statuses, "files": files,
            "interpretation": "Descriptive association; layer changes do not establish a cause of mAP changes."}


def _eda_curve_plots(curves, folder):
    entries = curves if isinstance(curves, list) else curves.get("curves", curves.get("items", []))
    count, files = 0, []
    for entry in entries:
        if not isinstance(entry, dict) or entry.get("status") in {"UNAVAILABLE", "NOT_RUN"}:
            continue
        x, y = entry.get("x"), entry.get("y")
        if x is None or y is None:
            continue
        xs, ys = _eda_np.asarray(x, dtype=float), _eda_np.asarray(y, dtype=float)
        if xs.ndim != 1 or not xs.size or not _eda_np.isfinite(xs).all() or not _eda_np.isfinite(ys).all():
            continue
        if ys.ndim == 1:
            ys = ys[None, :]
        if ys.ndim != 2 or ys.shape[-1] != len(xs):
            continue
        fig, ax = _eda_plt.subplots(figsize=(10, 6))
        labels = entry.get("labels") or entry.get("class_names") or []
        for index, values in enumerate(ys):
            ax.plot(xs, values, label=str(labels[index]) if index < len(labels) else f"series {index}", alpha=.75)
        ax.set_xlabel(entry.get("x_label", entry.get("xlabel", "x")))
        ax.set_ylabel(entry.get("y_label", entry.get("ylabel", "y")))
        ax.set_title(entry.get("name", "Measured validation curve"))
        if len(ys) <= 20:
            ax.legend()
        fig.tight_layout()
        files.append(_eda_save_figure(fig, folder / f"evaluation_curve_{count:02d}"))
        count += 1
    return {"status": "MEASURED" if count else "UNAVAILABLE", "curve_count": count, "files": files}


def _build_run_eda(result, run_dir, inventory_before, inventory_after):
    """Export ordered EDA from actual run records; failures propagate to run reports."""
    folder = _EdaPath(run_dir) / "eda"
    folder.mkdir(parents=True, exist_ok=True)
    rows = layer_eda_rows(inventory_before, inventory_after)
    conv_rows = [row for row in rows if row["module_type"] == "Conv2d"]
    stages = stage_eda_rows(rows)
    _eda_write_csv(folder / "layer_eda.csv", rows)
    _eda_write_csv(folder / "stage_eda.csv", stages)
    audit = result.get("audit") or {}
    title = f"{result.get('model_name', 'YOLO26' + result.get('model_variant', 'n'))} {result.get('policy', '')} pr{result.get('ratio', 0):g}"
    report = {"status": "MEASURED", "model_variant": result.get("model_variant", "n"),
              "model_name": result.get("model_name", "YOLO26n"), "policy": result.get("policy"), "ratio": result.get("ratio"),
              "section_order": ["identity_coverage", "channels", "masks_lamp", "efficiency", "validation_tradeoffs"],
              "identity_coverage": {"provenance": result.get("provenance"), "conv_layers": len(conv_rows),
                                    "bn_layers": sum(row["module_type"] == "BatchNorm2d" for row in rows),
                                    "eligible_layers": sum(bool(row["eligible"]) for row in rows),
                                    "protected_layers": sum(bool(row["protection_reason"]) for row in rows),
                                    "group_count": result.get("pruned_group_count"),
                                    "protection_reasons": sorted({str(row["protection_reason"]) for row in rows if row["protection_reason"]})},
              "channels": _eda_channel_plots(conv_rows, folder, title),
              "channel_mapping": _eda_mapping_plots(audit, conv_rows, folder),
              "lamp_scores": _eda_lamp_plots(audit, folder),
              "efficiency": _eda_efficiency_plots(stages, result, folder),
              "validation_tradeoffs": _eda_evaluation_plots(result, folder),
              "notes": ["Channel figures include all Conv outputs in original module order.",
                        "Independent root-channel counts differ from physical coupled Conv-output changes.",
                        "No activation feature maps, simulated accuracy, or reconstructed historical channel indices."]}
    _eda_write_json(folder / "eda_status.json", report)
    return report


def build_run_eda(result, run_dir, inventory_before, inventory_after):
    """Persist an explicit failed EDA report before propagating analysis errors."""
    try:
        return _build_run_eda(result, run_dir, inventory_before, inventory_after)
    except Exception as error:
        _eda_write_json(_EdaPath(run_dir) / "eda/eda_status.json",
                        {"status": "FAIL", "model_variant": result.get("model_variant", "n"),
                         "policy": result.get("policy"), "ratio": result.get("ratio"),
                         "error": {"type": type(error).__name__, "message": str(error)}})
        raise


def plot_model_comparison(suites, output_dir):
    """Compare N/S only where each point has measured accuracy and efficiency."""
    folder = _EdaPath(output_dir)
    folder.mkdir(parents=True, exist_ok=True)
    suites = list(suites.values()) if isinstance(suites, dict) else list(suites)
    points = []
    for suite in suites:
        variant = suite.get("model_variant", "n")
        name = suite.get("model_name", "YOLO26" + variant)
        baseline = suite.get("baseline") or {}
        for ratio, policy, metrics in [(0., "baseline", baseline)] + [
                (run["ratio"], run.get("policy", "extended"), run.get("pruned") or {})
                for run in suite.get("results", [])]:
            validation = metrics.get("validation") or {}
            latency = metrics.get("latency") or {}
            points.append({"model_variant": variant, "model_name": name, "ratio": ratio, "policy": policy,
                           "map50_95": validation.get("map50_95"), "parameters": metrics.get("parameters"),
                           "gflops": metrics.get("gflops"),
                           "latency_ms": latency.get("mean_ms") if latency.get("status") in {"PASS", "MEASURED"} else None,
                           "latency_p05_ms": latency.get("p05_ms") if latency.get("status") in {"PASS", "MEASURED"} else None,
                           "latency_p95_ms": latency.get("p95_ms") if latency.get("status") in {"PASS", "MEASURED"} else None})
    _eda_write_csv(folder / "model_comparison.csv", points)
    statuses = []
    for metric, label in (("parameters", "Parameters"), ("gflops", "GFLOPs"), ("latency_ms", "Raw model forward latency (ms)")):
        valid = [row for row in points if _eda_finite(row.get(metric)) and _eda_finite(row.get("map50_95"))]
        if not valid:
            statuses.append({"plot": "map_vs_" + metric, "status": "NOT_RUN", "reason": "Paired measured accuracy/efficiency is unavailable."})
            continue
        fig, ax = _eda_plt.subplots(figsize=(10, 7))
        for variant in dict.fromkeys(row["model_variant"] for row in valid):
            rows = [row for row in valid if row["model_variant"] == variant]
            ax.scatter([row[metric] for row in rows], [row["map50_95"] for row in rows], label="YOLO26" + variant)
            for row in rows:
                if (metric == "latency_ms" and _eda_finite(row.get("latency_p05_ms"))
                        and _eda_finite(row.get("latency_p95_ms"))):
                    ax.hlines(row["map50_95"], row["latency_p05_ms"], row["latency_p95_ms"], alpha=.5)
                ax.annotate(f"{row['model_name']} {row['policy']} pr{row['ratio']:g}", (row[metric], row["map50_95"]),
                            xytext=(5, 5), textcoords="offset points", fontsize=8)
        ax.set_xlabel(label)
        ax.set_ylabel("Measured mAP50–95")
        ax.set_title("N/S compression–accuracy comparison")
        ax.legend()
        fig.tight_layout()
        files = _eda_save_figure(fig, folder / ("map_vs_" + metric))
        statuses.append({"plot": "map_vs_" + metric, "status": "MEASURED", "points": len(valid), "files": files})
    report = {"plots": statuses, "point_count": len(points), "scope": "Each model uses its own measured baseline; no missing accuracy is replaced with zero."}
    _eda_write_json(folder / "comparison_status.json", report)
    return report
