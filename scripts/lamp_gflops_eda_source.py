"""Cross-configuration analysis for notebook 05, following notebook 03's EDA.

Consumes completed run records and their original inventories. It never prunes,
validates, selects a student, or supplies a value for a missing measurement.
"""
# ruff: noqa: F821 -- Shared EDA and evaluation declarations precede this cell.

import pandas as pd


def _gflops_eda_csv(path):
    if not path.is_file() or not path.stat().st_size or not path.read_text(encoding="utf-8").strip():
        return []
    frame = pd.read_csv(path)
    return frame.astype(object).where(pd.notna(frame), None).to_dict("records")


def _gflops_eda_delta(after, before, scale=1):
    return scale * (after - before) if _eda_finite(after) and _eda_finite(before) else None


def gflops_eda_tables(bundle, suite_dir):
    """Join classes by ID and keep model baselines, denominators and statuses explicit."""
    tables = {key: [] for key in (
        "metrics", "sources", "architecture", "coverage", "layers", "stages", "groups", "ledger",
        "per_class_metrics", "validation_protocol", "verification", "target_tiers", "target_search",
        "search_profiles", "p5_contrasts", "p5_per_class", "student_eligibility",
    )}
    for model in bundle.get("models", []):
        identity = {"model": model["model_name"], "model_variant": model["model_variant"]}
        folder = _EdaPath(suite_dir) / model["model_name"].lower()
        baseline_report = model.get("baseline_report", {})
        baseline = baseline_report.get("reserialized", model.get("baseline", {}))
        baseline_val = baseline.get("validation", {}) if baseline.get("validation_status") == "PASS" else {}
        baseline_classes = {row["class_id"]: row for row in baseline_val.get("per_class", [])}
        source = model.get("source", {})
        tables["sources"].append({**identity, **source,
                                   "baseline_validation_parity_status": baseline_report.get("validation_parity_status", "NOT_RUN")})
        for row in baseline_report.get("architecture", {}).get("stages", []):
            tables["architecture"].append({**identity, **row})
        records = {row["configuration_id"]: row for row in model.get("rows", [])}
        processed = set()
        stages = [(f"baseline-{kind}", "baseline", baseline_report.get(kind, {}), None)
                  for kind in ("source", "reserialized")]
        stages.extend((run["configuration_id"], "pruned", run.get("pruned", {}), run)
                      for run in model.get("results", []))
        for config_id, stage, metric, run in stages:
            if not metric and run is None:
                continue
            processed.add(config_id)
            audit = (run or {}).get("audit", {})
            info = {**identity, "configuration_id": config_id, "stage": stage,
                    "policy": run["policy"] if run else "baseline", "ratio": run.get("ratio") if run else 0}
            validation_status = run.get("validation_status", "NOT_RUN") if run else metric.get("validation_status", "NOT_RUN")
            val = metric.get("validation", {}) if validation_status == "PASS" else {}
            latency = metric.get("latency", {})
            row = {**records.get(config_id, {}), **info,
                   **{key: metric.get(key) for key in ("parameters", "gflops", "checkpoint_bytes")},
                   **{key: val.get(key) for key in ("map50_95", "map50", "precision", "recall")},
                   "baseline_map50_95": baseline_val.get("map50_95"), "validation_status": validation_status,
                   "evaluation_mode": val.get("evaluation_mode", bundle.get("mode")),
                   "dataset_fingerprint": val.get("dataset_fingerprint"),
                   "dataset_provenance_status": val.get("dataset_provenance_status"),
                   "validation_image_count": val.get("validation_image_count"),
                   "latency_status": latency.get("status", "NOT_RUN"),
                   "latency_median_ms": latency.get("median_ms") if latency.get("status") in {"PASS", "MEASURED"} else None,
                   "latency_p95_ms": latency.get("p95_ms") if latency.get("status") in {"PASS", "MEASURED"} else None,
                   "gflops_speedup": baseline.get("gflops") / metric["gflops"]
                   if _eda_finite(metric.get("gflops")) and metric["gflops"] > 0 and baseline.get("gflops") else None,
                   "latency_speedup": baseline["latency"]["median_ms"] / latency["median_ms"]
                   if latency.get("status") in {"PASS", "MEASURED"} and latency.get("median_ms", 0) > 0
                   and baseline.get("latency", {}).get("status") in {"PASS", "MEASURED"}
                   and baseline["latency"].get("median_ms") else None,
                   "delta_map50_95_pp": _gflops_eda_delta(val.get("map50_95"), baseline_val.get("map50_95"), 100),
                   "pruned_group_count": len((run or {}).get("groups", [])) if run else 0}
            if run is None:
                row.update(structural_status="BASELINE", actual_eligible_ratio=0, eda_status="NOT_RUN")
                reload = baseline_report.get("reload", {})
                row["reload_status"] = ("PASS" if config_id == "baseline-reserialized"
                                        and reload.get("direct_yolo_load") is True and reload.get("raw_parity") is True
                                        else row.get("reload_status", "NOT_RUN"))
            for key, label in (("parameters", "parameter"), ("gflops", "gflops"), ("checkpoint_bytes", "checkpoint")):
                delta = _gflops_eda_delta(metric.get(key), baseline.get(key), -100)
                row[label + "_reduction_pct"] = delta / baseline[key] if delta is not None and baseline.get(key) else None
            delta = _gflops_eda_delta(val.get("map50_95"), baseline_val.get("map50_95"), 100)
            row["relative_map50_95_pct"] = delta / baseline_val["map50_95"] if delta is not None and baseline_val.get("map50_95") else None
            tables["metrics"].append(row)
            protocol = {key: val.get(key) for key in (
                "split", "evaluation_mode", "dataset_fingerprint", "dataset_provenance_status", "validation_image_count",
            )}
            tables["validation_protocol"].append({**info, "validation_status": validation_status, **protocol,
                                                   **val.get("validation_arguments_effective", {})})
            # The reserialized baseline is the one common reference for all pruned students.
            if config_id != "baseline-source":
                for entry in val.get("per_class", []):
                    prior = baseline_classes.get(entry["class_id"], {})
                    if prior and prior.get("name") != entry.get("name"):
                        raise ValueError(f"Nama/ID kelas tidak konsisten pada {model['model_name']} / {config_id}.")
                    tables["per_class_metrics"].append({**info, **entry,
                        "validation_status": validation_status, "evaluation_mode": row["evaluation_mode"],
                        "baseline_ap50_95": prior.get("ap50_95"), "baseline_ap50": prior.get("ap50"),
                        **{f"delta_{key}_pp": _gflops_eda_delta(entry.get(key), prior.get(key), 100)
                           for key in ("ap50_95", "ap50", "precision", "recall", "f1")}})
            if run is None:
                continue
            eda_dir = folder / run.get("artifact_configuration_id", config_id) / "eda"
            layers = _gflops_eda_csv(eda_dir / "layer_eda.csv")
            convs = [layer for layer in layers if layer["module_type"] == "Conv2d"]
            denominator = audit.get("target_policy_eligible_channels", audit.get("independent_channels_before"))
            removed = audit.get("independent_channels_removed")
            coverage = {**info, "graph_eligible_groups": len(audit.get("eligible_groups", [])),
                        "graph_rejected_roots": len(audit.get("rejected_groups", [])),
                        "selected_groups": len(run.get("groups", [])),
                        "graph_independent_channels_before": audit.get("independent_channels_before"),
                        "policy_eligible_channels": denominator, "independent_channels_removed": removed,
                        "policy_eligible_channels_after": denominator - removed
                        if _eda_finite(denominator) and _eda_finite(removed) else None,
                        "actual_eligible_ratio": run.get("actual_eligible_channel_ratio"),
                        "physical_conv_outputs_removed": audit.get("physical_conv_output_channels_removed"),
                        "inventory_status": "MEASURED" if layers else "UNAVAILABLE",
                        "conv_layers": len(convs) if layers else None,
                        "changed_conv_outputs": sum(layer["out_channels_removed"] > 0 for layer in convs) if layers else None}
            tables["coverage"].append(coverage)
            tables["layers"].extend({**info, **layer} for layer in layers)
            tables["stages"].extend({**info, **item} for item in _gflops_eda_csv(eda_dir / "stage_eda.csv"))
            tables["ledger"].extend({**info, **item} for item in audit.get("channel_mapping", []))
            scores = {}
            for item in audit.get("lamp_scores", []):
                scores.setdefault(item["root"], []).append(item)
            for entry in audit.get("eligible_groups", []):
                values = scores.get(entry["name"], [])
                finite = sorted(item["score"] for item in values if _eda_finite(item.get("score")))
                tables["groups"].append({**info, "root": entry["name"], "group_id": entry["group_id"],
                    "decision": "GRAPH_ELIGIBLE", "initial_channels": entry["initial_channels"],
                    "capacity_eligible": sum(bool(item.get("capacity_eligible")) for item in values),
                    "removed": sum(bool(item.get("removed")) for item in values),
                    "score_min": finite[0] if finite else None,
                    "score_median": float(_eda_np.median(finite)) if finite else None,
                    "score_max": finite[-1] if finite else None,
                    "coupled_output_layers": entry.get("signature"), "reason": ""})
            tables["groups"].extend({**info, **entry, "decision": "GRAPH_REJECTED"}
                                    for entry in audit.get("rejected_groups", []))
            tables["verification"].append({**info,
                "structural_status": run.get("structural_smoke_status", "NOT_RUN"),
                "oracle_status": run.get("oracle", {}).get("status", "NOT_RUN"),
                "reproducibility_status": run.get("reproducibility_status", "NOT_RUN"),
                "reload_status": run.get("reload_status", "NOT_RUN"), "validation_status": validation_status,
                "baseline_validation_parity_status": baseline_report.get("validation_parity_status", "NOT_RUN"),
                "latency_status": row["latency_status"], "eda_status": run.get("eda_status", "NOT_RUN"),
                "checkpoint_sha256": run.get("saved_pruned", {}).get("sha256")})
            tables["student_eligibility"].append({**info,
                "target_speedup": row.get("target_speedup"), "target_status": row.get("target_status"),
                "gflops_speedup": row.get("gflops_speedup"), "delta_map50_95_pp": row["delta_map50_95_pp"],
                "evaluation_mode": row["evaluation_mode"], "dataset_provenance_status": row["dataset_provenance_status"],
                "latency_status": row["latency_status"], "qualifies": _gflops_complete_student_row(row)})
        tables["metrics"].extend({**row, "stage": "pruned"} for config_id, row in records.items() if config_id not in processed)
        search = model.get("target_search", {})
        for key, source_key in (("target_tiers", "tiers"), ("search_profiles", "measurements"), ("target_search", "targets")):
            tables[key].extend({**identity, **{k: v for k, v in item.items() if k not in {"plan", "caps"}}}
                               for item in search.get(source_key, []))
        for key in ("p5_contrasts", "p5_per_class"):
            tables[key].extend({**identity, **entry} for entry in _gflops_eda_csv(folder / f"{key}.csv"))
    return tables


def _gflops_eda_finish(fig, path, count, reason="Pengukuran berpasangan belum tersedia."):
    files = _eda_save_figure(fig, path)
    return {"plot": path.name, "status": "MEASURED" if count else "NOT_RUN",
            "measured_values": count, "reason": None if count else reason, **files}


def _gflops_eda_panel(ax, rows, x, y, xlabel, ylabel):
    valid = [row for row in rows if _eda_finite(row.get(x)) and _eda_finite(row.get(y))]
    if valid:
        ax.plot([row[x] for row in valid], [row[y] for row in valid], "o-")
    else:
        ax.text(.5, .5, "NOT_RUN", ha="center", transform=ax.transAxes)
    ax.set(xlabel=xlabel, ylabel=ylabel)
    ax.grid(alpha=.2)
    return len(valid)


def gflops_eda_model_plots(tables, model_name, folder):
    """Notebook 03 sweep/AP plots, with computational targets shown separately."""
    folder = _EdaPath(folder)
    rows = [row for row in tables["metrics"] if row["model"] == model_name and row["configuration_id"] != "baseline-source"]
    protocol = next((row for row in tables["validation_protocol"] if row["model"] == model_name
                     and row.get("validation_status") == "PASS"), {})
    scope = " · " + str(protocol.get("evaluation_mode", "NOT_RUN"))
    if protocol.get("validation_image_count") is not None:
        scope += f" · {protocol['validation_image_count']} citra"
    if protocol.get("imgsz") is not None:
        scope += f" · {protocol['imgsz']} px"
    statuses = []
    controls = sorted((row for row in rows if row["policy"] in {"baseline", "extended"}), key=lambda row: row["ratio"])
    fig, axes = _eda_plt.subplots(1, 3, figsize=(17, 5))
    count = sum(_gflops_eda_panel(ax, controls, "ratio", metric, "Extended eligible-channel ratio", label)
                for ax, metric, label in zip(axes, ("map50_95", "parameter_reduction_pct", "gflops_reduction_pct"),
                                            ("mAP50–95", "Parameter reduction (%)", "GFLOPs reduction (%)")))
    fig.suptitle(model_name + scope + "\nKontrol extended 10–50%; target GFLOPs dianalisis terpisah")
    fig.tight_layout()
    statuses.append(_gflops_eda_finish(fig, folder / "extended_sweep", count))

    fig, axes = _eda_plt.subplots(2, 2, figsize=(17, 10))
    count = 0
    for ax, metric in zip(axes.flat, ("map50_95", "map50", "precision", "recall")):
        measured = [(index, row[metric]) for index, row in enumerate(rows) if _eda_finite(row.get(metric))]
        if measured:
            ax.scatter(*zip(*measured))
        else:
            ax.text(.5, .5, "NOT_RUN", ha="center", transform=ax.transAxes)
        ax.set_xticks(range(len(rows)), [row["configuration_id"] for row in rows], rotation=65, ha="right", fontsize=8)
        ax.set(ylabel=metric, ylim=(0, 1))
        ax.grid(axis="y", alpha=.2)
        if metric == "map50_95" and rows and _eda_finite(rows[0].get("baseline_map50_95")):
            ax.axhline(rows[0]["baseline_map50_95"] - .02, color="red", linestyle="--", label="baseline − 2 pp")
            ax.legend()
        count += len(measured)
    fig.suptitle(model_name + scope + "\nMetrik setiap konfigurasi")
    fig.tight_layout()
    statuses.append(_gflops_eda_finish(fig, folder / "configuration_metrics", count))

    classes = [row for row in tables["per_class_metrics"] if row["model"] == model_name]
    class_names = dict(sorted({row["class_id"]: row["name"] for row in classes}.items()))
    configs = [row["configuration_id"] for row in rows]
    fig, axes = _eda_plt.subplots(1, 2, figsize=(max(14, len(class_names) * 2.5), max(5, len(configs) * .45)))
    count = 0
    for ax, metric, label in zip(axes, ("ap50_95", "delta_ap50_95_pp"), ("AP50–95", "Delta AP50–95 vs baseline (pp)")):
        values = {(row["configuration_id"], row["class_id"]): row.get(metric) for row in classes}
        data = _eda_np.array([[values.get((config, class_id)) if _eda_finite(values.get((config, class_id))) else _eda_np.nan
                              for class_id in class_names] for config in configs], dtype=float)
        measured = int(_eda_np.isfinite(data).sum())
        if measured:
            options = {"vmin": 0, "vmax": 1, "cmap": "Blues"} if metric == "ap50_95" else {
                "vmin": -max(1, float(_eda_np.nanmax(_eda_np.abs(data)))),
                "vmax": max(1, float(_eda_np.nanmax(_eda_np.abs(data)))), "cmap": "RdBu"}
            plot = ax.imshow(_eda_np.ma.masked_invalid(data), aspect="auto", **options)
            fig.colorbar(plot, ax=ax)
            for i, j in zip(*_eda_np.where(_eda_np.isfinite(data))):
                ax.text(j, i, f"{data[i, j]:.3f}" if metric == "ap50_95" else f"{data[i, j]:+.2f}",
                        ha="center", va="center", fontsize=7)
        else:
            ax.text(.5, .5, "NOT_RUN", ha="center", transform=ax.transAxes)
        ax.set_xticks(range(len(class_names)), list(class_names.values()), rotation=45, ha="right")
        ax.set_yticks(range(len(configs)), configs)
        ax.set_title(label + " · sel kosong = tidak tersedia")
        count += measured
    fig.suptitle(model_name + scope + "\nAP per kelas untuk seluruh konfigurasi")
    fig.tight_layout()
    statuses.append(_gflops_eda_finish(fig, folder / "per_class_ap_all", count))

    for metric, label in (("parameters", "Parameters"), ("gflops", "GFLOPs"), ("latency_median_ms", "Median latency raw forward (ms)")):
        fig, ax = _eda_plt.subplots(figsize=(11, 6))
        valid = [row for row in rows if _eda_finite(row.get(metric)) and _eda_finite(row.get("map50_95"))]
        for row in valid:
            ax.scatter(row[metric], row["map50_95"])
            ax.annotate(row["configuration_id"], (row[metric], row["map50_95"]), xytext=(4, 4), textcoords="offset points", fontsize=8)
        if not valid:
            ax.text(.5, .5, "NOT_RUN", ha="center", transform=ax.transAxes)
        ax.set(xlabel=label, ylabel="mAP50–95", title=model_name + scope + "\nAkurasi dan efisiensi")
        ax.grid(alpha=.2)
        fig.tight_layout()
        statuses.append(_gflops_eda_finish(fig, folder / ("tradeoff_" + metric), len(valid)))

    profiles = [row for row in tables["search_profiles"] if row["model"] == model_name]
    tiers = [row for row in tables["target_tiers"] if row["model"] == model_name]
    fig, axes = _eda_plt.subplots(1, 2, figsize=(15, 5))
    for tier in sorted({row["tier"] for row in profiles}):
        values = sorted((row for row in profiles if row["tier"] == tier), key=lambda row: row["channels"])
        axes[0].plot([row["channels"] for row in values], [row["speed_up"] for row in values], ".-", label=f"tier {tier}")
    if profiles:
        axes[0].legend()
    else:
        axes[0].text(.5, .5, "NOT_RUN", ha="center", transform=axes[0].transAxes)
    tiers = [row for row in tiers if _eda_finite(row.get("max_gflops_speed_up"))]
    if tiers:
        axes[1].bar([str(row["tier"]) for row in tiers], [row["max_gflops_speed_up"] for row in tiers])
        for target in sorted({row["target"] for row in tables["target_search"] if row["model"] == model_name}):
            axes[1].axhline(target, linestyle="--", label=f"target {target:g}×")
        axes[1].legend()
    else:
        axes[1].text(.5, .5, "NOT_RUN", ha="center", transform=axes[1].transAxes)
    axes[0].set(xlabel="Channel independen dihapus", ylabel="GFLOPs speed-up", title="Kandidat pencarian, tanpa evaluasi mAP")
    axes[1].set(xlabel="Tier cap", ylabel="Maksimum GFLOPs speed-up", title="Kapasitas dengan proteksi aktif")
    fig.suptitle(model_name + scope + "\nPencarian target komputasi")
    fig.tight_layout()
    statuses.append(_gflops_eda_finish(fig, folder / "target_search_capacity", len(profiles) + len(tiers)))
    return statuses


def _gflops_build_analysis(bundle, suite_dir):
    """Export analysis for every completed configuration independently of detail display."""
    folder = _EdaPath(suite_dir) / "comparison/analysis"
    folder.mkdir(parents=True, exist_ok=True)
    tables = gflops_eda_tables(bundle, suite_dir)
    for key, rows in tables.items():
        _eda_write_csv(folder / f"{key}.csv", rows)
        _eda_write_json(folder / f"{key}.json", rows)
    plots = []
    for model in bundle.get("models", []):
        plots.extend({"model": model["model_name"], **row}
                     for row in gflops_eda_model_plots(tables, model["model_name"], folder / model["model_name"].lower()))
    report = {"status": "PASS" if bundle.get("models") else "NOT_RUN", "mode": bundle.get("mode"),
              "table_rows": {key: len(rows) for key, rows in tables.items()}, "plots": plots,
              "interpretation": "Smoke checks the pipeline. Full validation selects a student. Layer distributions are descriptive.",
              "selection": bundle.get("selection"),
              "display_configurations": list(globals().get("EDA_CONFIGURATIONS", ())),
              "detail_configurations": list(globals().get("EDA_DETAIL_CONFIGURATIONS", ())), "tables": tables}
    _eda_write_json(folder / "analysis_status.json", {key: value for key, value in report.items() if key != "tables"})
    return report


def gflops_build_analysis(bundle, suite_dir):
    path = _EdaPath(suite_dir) / "comparison/analysis/analysis_status.json"
    _eda_write_json(path, {"status": "RUNNING"})
    try:
        return _gflops_build_analysis(bundle, suite_dir)
    except Exception as error:
        _eda_write_json(path, {"status": "FAIL", "error": str(error), "type": type(error).__name__})
        raise
