"""EDA for measured baseline, raw pruned, and recovered DetectionModels."""
# ruff: noqa: F821 -- Notebook cells supply shared persistence and EDA declarations.

RECOVERY_STAGE_STYLES = {
    "baseline": ("Baseline", "#2863AD"),
    "raw_pruned": ("Raw pruned", "#BA492E"),
    "recovered": ("Recovered", "#217854"),
}
RECOVERY_ACCURACY_FIELDS = {"map50_95": "mAP50–95", "map50": "mAP50", "precision": "Precision", "recall": "Recall"}
RECOVERY_CLASS_FIELDS = {"ap50_95": "AP50–95", "ap50": "AP50", "precision": "Precision", "recall": "Recall", "f1": "F1"}
RECOVERY_LATENCY_CONTRACT = ("device", "precision", "batch", "imgsz", "fusion", "scope", "warmup", "repeats")
RECOVERY_EDA_METHODS = {"finetune": "Fine-tuning standar", "native": "Native KD", "crosskd": "CrossKD", "mgd": "MGD"}


def _recovery_eda_num(value):
    return value if _eda_finite(value) else None


def _recovery_eda_ratio(before, after):
    return before / after if before is not None and after is not None and after > 0 else None


def _recovery_eda_delta(after, before):
    return 100 * (after - before) if after is not None and before is not None else None


def _recovery_eda_metadata(result):
    return {key: result.get(key) for key in ("model_variant", "method", "platform", "mode", "configuration_id",
                                             "target_speedup", "target_status", "alias_of", "raw_sha256")}


def recovery_comparison_rows(result):
    """Use successful measurements only and join class data by stable ID."""
    stages = result.get("stages", {})
    metadata = _recovery_eda_metadata(result)
    rows, per_class_by_stage, class_names = [], {}, {}
    latency_rows = [stage.get("latency", {}) for stage in stages.values()]
    latency_contracts = [tuple(item.get(key) for key in RECOVERY_LATENCY_CONTRACT)
                         for item in latency_rows if item.get("status") in {"PASS", "MEASURED"}]
    comparable_latency = all(contract == latency_contracts[0] for contract in latency_contracts)
    for stage in RECOVERY_STAGE_STYLES:
        measured = stages.get(stage, {})
        validation_status = measured.get("validation_status", "NOT_RUN")
        validation = measured.get("validation", {}) if validation_status == "PASS" else {}
        latency = measured.get("latency", {})
        latency_status = latency.get("status", "NOT_RUN")
        if latency_status in {"PASS", "MEASURED"} and not comparable_latency:
            latency_status = "NOT_COMPARABLE"
        latency = latency if latency_status in {"PASS", "MEASURED"} else {}
        rows.append({**metadata, "stage": stage, "validation_status": validation_status, "latency_status": latency_status,
                     "validation_image_count": validation.get("validation_image_count"),
                     "validation_fingerprint": validation.get("dataset_fingerprint"),
                     "validation_device": validation.get("validation_arguments_effective", {}).get("device"),
                     "latency_device": latency.get("device"), "latency_scope": latency.get("scope"),
                     **{key: _recovery_eda_num(validation.get(key)) for key in RECOVERY_ACCURACY_FIELDS},
                     **{key: _recovery_eda_num(measured.get(key)) for key in ("parameters", "gflops", "checkpoint_bytes", "conv_output_channels")},
                     "latency_median_ms": _recovery_eda_num(latency.get("median_ms")),
                     "latency_p95_ms": _recovery_eda_num(latency.get("p95_ms"))})
        indexed = {}
        for item in validation.get("per_class", []):
            class_id, name = item["class_id"], item["name"]
            if class_id in indexed or class_names.setdefault(class_id, name) != name:
                raise ValueError(f"Identitas kelas tidak konsisten: {stage}, {class_id}, {name}")
            indexed[class_id] = item
        per_class_by_stage[stage] = indexed
    baseline, raw, recovered = rows
    class_rows = []
    for row in rows:
        stage = row["stage"]
        row["gflops_speedup"] = _recovery_eda_ratio(baseline["gflops"], row["gflops"])
        row["latency_speedup"] = _recovery_eda_ratio(baseline["latency_median_ms"], row["latency_median_ms"])
        for key in RECOVERY_ACCURACY_FIELDS:
            row[key + "_drop_pp"] = _recovery_eda_delta(baseline[key], row[key])
            row[key + "_recovery_gain_pp"] = _recovery_eda_delta(recovered[key], raw[key]) if stage == "recovered" else None
        for key, name in (("parameters", "parameter"), ("gflops", "gflops"), ("conv_output_channels", "channel"),
                          ("checkpoint_bytes", "checkpoint")):
            ratio = _recovery_eda_ratio(row[key], baseline[key])
            row[name + "_reduction_pct"] = None if ratio is None else 100 * (1 - ratio)
        for class_id, name in sorted(class_names.items()):
            record = per_class_by_stage[stage].get(class_id, {})
            class_row = {**metadata, "stage": stage, "class_id": class_id, "name": name,
                         "support": _recovery_eda_num(record.get("support")),
                         **{key: _recovery_eda_num(record.get(key)) for key in RECOVERY_CLASS_FIELDS}}
            for key in ("ap50_95", "ap50"):
                class_row[key + "_drop_pp"] = _recovery_eda_delta(
                    _recovery_eda_num(per_class_by_stage["baseline"].get(class_id, {}).get(key)), class_row[key])
                class_row[key + "_recovery_gain_pp"] = _recovery_eda_delta(
                    class_row[key], _recovery_eda_num(per_class_by_stage["raw_pruned"].get(class_id, {}).get(key))
                ) if stage == "recovered" else None
            class_rows.append(class_row)
    return rows, class_rows


def recovery_epoch_rows(rows):
    """Flatten epoch records for plotting and logging without inventing missing loss values."""
    return [{"epoch": row["epoch"], "optimizer_steps": row.get("optimizer_steps"), "fitness": row.get("fitness"),
             **{f"train/{name.removeprefix('train/')}": _recovery_eda_num(value)
                for name, value in row.get("losses", {}).items()},
             **{name: _recovery_eda_num(value) for name, value in row.get("metrics", {}).items()},
             **{f"lr/pg{index}": _recovery_eda_num(value) for index, value in enumerate(row.get("learning_rates", []))}}
            for row in rows]


def _recovery_draw_bar_panel(ax, records, *, title, unit, bounded=False):
    groups = list(dict.fromkeys(record["group"] for record in records))
    labels = {record["group"]: record["group_label"] for record in records}
    positions = {group: index for index, group in enumerate(groups)}
    width = .24
    plotted = 0
    for offset, (stage, (_label, color)) in enumerate(RECOVERY_STAGE_STYLES.items()):
        observed = [record for record in records if record["stage"] == stage and _eda_finite(record["value"])]
        if observed:
            bars = ax.bar([positions[record["group"]] + (offset - 1) * width for record in observed],
                          [record["value"] for record in observed], width=width, color=color)
            if len(groups) <= 4:
                ax.bar_label(bars, labels=[f"{record['value']:.4g}" for record in observed], fontsize=7, padding=2)
            plotted += len(observed)
    if not plotted:
        ax.text(.5, .55, "Belum terukur", ha="center", va="center", transform=ax.transAxes, color="#57616B")
    ax.set_xticks(list(positions.values()), [labels[group] for group in groups],
                  rotation=30 if len(groups) > 4 else 0, ha="right" if len(groups) > 4 else "center")
    ax.set_title(title)
    ax.set_ylabel(unit)
    if bounded:
        ax.set_ylim(0, 1.1)
    else:
        ax.margins(y=.15)
    ax.set_axisbelow(True)
    ax.grid(axis="y", alpha=.18)
    return plotted


def recovery_plot_stage_bars(result, folder, rows=None, class_rows=None):
    """Export exact plot inputs and separate axes for each unit of measurement."""
    from matplotlib.patches import Patch

    folder = _EdaPath(folder)
    if rows is None or class_rows is None:
        rows, class_rows = recovery_comparison_rows(result)
    figures, data = [], []
    metadata = _recovery_eda_metadata(result)
    variant = result.get("model_variant", "?")
    target = result.get("target_speedup")
    target_label = f"{target:.2f}×" if _eda_finite(target) else "—"
    status = result.get("target_status") or "NOT_RUN"
    method = RECOVERY_EDA_METHODS.get(result.get("method"), result.get("method") or "Fine-tuning standar")
    achieved = rows[1]["gflops_speedup"]
    achieved_label = f"{achieved:.3f}×" if _eda_finite(achieved) else "belum terukur"

    def record(panel, group, label, row, key, unit):
        return {**metadata, "panel": panel, "group": group, "group_label": label, "stage": row["stage"],
                "metric": key, "value": row[key], "unit": unit}

    def plot(name, panels, *, group_cols=2):
        nrows = (len(panels) + group_cols - 1) // group_cols
        fig, axes = _eda_plt.subplots(nrows, group_cols, figsize=(6.3 * group_cols, 3.5 * nrows + 1.6), squeeze=False)
        for ax, (title, unit, records, bounded) in zip(axes.flat, panels):
            _recovery_draw_bar_panel(ax, records, title=title, unit=unit, bounded=bounded)
            data.extend({"figure": name, **item} for item in records)
        for ax in list(axes.flat)[len(panels):]:
            ax.set_visible(False)
        fig.suptitle(f"YOLO26{variant}  |  {method}  |  Target {target_label}  |  Aktual {achieved_label}\n{status}",
                     fontsize=13, y=.99)
        fig.legend(handles=[Patch(facecolor=color, label=label) for label, color in RECOVERY_STAGE_STYLES.values()],
                   loc="upper center", bbox_to_anchor=(.5, .885), ncol=3, frameon=False)
        fig.text(.5, .01, "Nilai yang belum diukur dibiarkan kosong. Tinggi nol hanya untuk nilai nol yang terukur.",
                 ha="center", fontsize=9, color="#57616B")
        fig.tight_layout(rect=(0, .025, 1, .82))
        figures.append(_eda_save_figure(fig, folder / f"comparison_{name}"))

    def panel(title, unit, metrics, bounded=False):
        return (title, unit, [record(title, key, label, row, key, unit)
                              for key, label in metrics for row in rows], bounded)

    plot("accuracy", [panel("Metrik deteksi", "Proporsi", tuple(RECOVERY_ACCURACY_FIELDS.items()), True)], group_cols=1)
    plot("complexity", [panel(title, unit, ((key, label),)) for key, title, unit, label in (
        ("parameters", "Parameter", "Jumlah parameter", "Parameter"),
        ("gflops", "Komputasi model", "GFLOPs", "Komputasi"),
        ("checkpoint_bytes", "Checkpoint FP32", "Byte", "Ukuran"),
        ("conv_output_channels", "Jumlah channel", "Total output Conv2d", "Channel"))])
    plot("latency", [panel("Raw forward FP32 · batch 1", "Milidetik", (("latency_median_ms", "p50"), ("latency_p95_ms", "p95"))),
                     panel("Speed-up terhadap baseline", "Rasio (×)", (("gflops_speedup", "GFLOPs"), ("latency_speedup", "Latency p50")))])
    plot("changes", [panel("Penurunan metrik terhadap baseline", "Poin persentase",
                           tuple((key + "_drop_pp", label) for key, label in RECOVERY_ACCURACY_FIELDS.items())),
                     panel("Reduksi terhadap baseline", "%", (("parameter_reduction_pct", "Parameter"),
                           ("gflops_reduction_pct", "GFLOPs"), ("checkpoint_reduction_pct", "Checkpoint"),
                           ("channel_reduction_pct", "Channel"))),
                     panel("Recovery gain terhadap raw pruned", "Poin persentase",
                           tuple((key + "_recovery_gain_pp", label) for key, label in RECOVERY_ACCURACY_FIELDS.items()))])
    plot("per_class", [(label + " per kelas", "Proporsi", [record(label, str(row["class_id"]), row["name"], row, key, "Proporsi")
                          for row in class_rows], True) for key, label in list(RECOVERY_CLASS_FIELDS.items())[:2]])
    write_json(folder / "plot_data.json", data)
    write_csv(folder / "plot_data.csv", data)
    return figures


def _recovery_epoch_plots(rows, folder):
    losses = sorted({key.split("/", 1)[1] for row in rows for key in row if key.startswith(("train/", "val/"))})
    groups = [(key, (f"train/{key}", f"val/{key}"), "Loss") for key in losses]
    groups.extend((
        ("Akurasi validation", ("metrics/mAP50-95(B)", "metrics/mAP50(B)", "metrics/precision(B)", "metrics/recall(B)"), "Proporsi"),
        ("Learning rate", sorted({key for row in rows for key in row if key.startswith("lr/")}), "Learning rate"),
        ("Optimizer", ("optimizer_steps",), "Jumlah langkah kumulatif"),
    ))
    files = []
    for index, (title, keys, ylabel) in enumerate(groups):
        observed = {key: [(row["epoch"], row[key]) for row in rows if _eda_finite(row.get(key))] for key in keys}
        if not any(observed.values()):
            continue
        fig, ax = _eda_plt.subplots(figsize=(9, 4))
        for key, points in observed.items():
            if points:
                ax.plot(*zip(*points), label=key, marker="." if len(points) < 10 else None)
        ax.set(title=title, xlabel="Epoch", ylabel=ylabel)
        ax.grid(alpha=.18)
        ax.legend(frameon=False)
        fig.tight_layout()
        files.append(_eda_save_figure(fig, folder / f"epochs_{index:02d}"))
    return files


def _recovery_stage_validation_plots(result, folder):
    reports = {}
    for stage, (label, _color) in RECOVERY_STAGE_STYLES.items():
        measured = result.get("stages", {}).get(stage, {})
        validation = measured.get("validation", {}) if measured.get("validation_status") == "PASS" else {}
        if not validation:
            reports[stage] = {"status": "NOT_RUN"}
            continue
        target = folder / stage
        curves = validation.get("curves", {})
        write_json(target / "curves.json", curves)
        curve_report = _eda_curve_plots(curves, target)
        matrix = validation.get("confusion_matrix", {})
        matrix_files = []
        if matrix.get("status") == "MEASURED":
            write_json(target / "confusion_matrix.json", matrix)
            labels = matrix["labels"]
            for field, title, cmap in (("matrix", "Jumlah prediksi", "Blues"),
                                       ("normalized_by_true", "Normalisasi per kelas aktual", "Purples")):
                values = _eda_np.asarray(matrix[field], dtype=float)
                if values.shape != (len(labels), len(labels)) or not _eda_np.isfinite(values).all():
                    raise ValueError(f"Confusion matrix tidak valid: {stage}/{field}")
                write_csv(target / (field + ".csv"), [{"predicted_class": labels[i], **dict(zip(labels, row.tolist()))}
                                                     for i, row in enumerate(values)])
                fig, ax = _eda_plt.subplots(figsize=(9, 8))
                im = ax.imshow(values, cmap=cmap, vmin=0, vmax=1 if field == "normalized_by_true" else None)
                ax.set_xticks(range(len(labels)), labels, rotation=35, ha="right")
                ax.set_yticks(range(len(labels)), labels)
                ax.set(xlabel="Kelas aktual", ylabel="Kelas prediksi",
                       title=f"{label} · {title}\nconfidence {matrix.get('confidence_threshold')} | matching IoU {matrix.get('matching_iou_threshold')}")
                fig.colorbar(im, ax=ax)
                fig.tight_layout()
                matrix_files.append(_eda_save_figure(fig, target / field))
        reports[stage] = {"status": "PASS" if matrix_files and curve_report["status"] == "MEASURED" else "PARTIAL",
                          "curves": curve_report, "confusion_matrix": matrix_files}
    return reports


def _recovery_pruning_diagnostics(result, files, folder):
    if "inventories.json" not in files:
        return {"status": "NOT_RUN", "reason": "Manifest input belum menyertakan inventory pruning."}
    inventories = recovery_read_json(files["inventories.json"])
    rows = layer_eda_rows(inventories["before"], inventories["after"])
    conv = [row for row in rows if row["module_type"] == "Conv2d"]
    stages = stage_eda_rows(rows)
    audit = recovery_read_json(files["dependency_audit.json"]) if "dependency_audit.json" in files else {}
    write_csv(folder / "layer_eda.csv", rows)
    write_csv(folder / "stage_eda.csv", stages)
    channels = _eda_channel_plots(conv, folder, f"YOLO26{result['model_variant']} · struktur baseline dan raw")
    mapping = _eda_mapping_plots(audit, conv, folder)
    importance = _eda_lamp_plots(audit, folder)
    return {"status": "PASS" if mapping["status"] == importance["status"] == "MEASURED" else "PARTIAL",
            "channels": channels, "channel_mapping": mapping, "lamp_scores": importance}


def recovery_build_alias_eda(result, job_folder):
    """Write target-specific summary plots even when the raw checkpoint is shared."""
    folder = _EdaPath(job_folder) / "eda"
    rows, class_rows = recovery_comparison_rows(result)
    write_json(folder / "stage_metrics.json", rows)
    write_csv(folder / "stage_metrics.csv", rows)
    write_json(folder / "per_class_metrics.json", class_rows)
    write_csv(folder / "per_class_metrics.csv", class_rows)
    figures = recovery_plot_stage_bars(result, folder, rows, class_rows)
    if result.get("alias_of"):
        write_json(folder / "eda_status.json", {"status": result.get("eda_status", "NOT_RUN"), "bars": figures,
                   "canonical_diagnostics": f"../../{result['alias_of']}/eda"})
    return figures


def recovery_build_eda(result, folder, epoch_rows, files):
    folder = _EdaPath(folder)
    target = folder / "eda"
    bars = recovery_build_alias_eda(result, folder)
    epochs = recovery_epoch_rows(epoch_rows)
    write_json(target / "epoch_metrics.json", epochs)
    write_csv(target / "epoch_metrics.csv", epochs)
    report = {"status": "PASS", "bars": bars, "epochs": _recovery_epoch_plots(epochs, target / "training"),
              "validation": _recovery_stage_validation_plots(result, target / "validation"),
              "pruning_diagnostics": _recovery_pruning_diagnostics(result, files, target / "pruning")}
    if (not report["epochs"] or report["pruning_diagnostics"]["status"] != "PASS"
            or any(value["status"] != "PASS" for value in report["validation"].values())):
        report["status"] = "PARTIAL"
    write_json(target / "eda_status.json", report)
    return report


def recovery_log_comparison(run, result, folder):
    """Log the exact local tables and comparison figures to the recovery run."""
    import wandb

    rows, classes = recovery_comparison_rows(result)
    payload = {}
    for name, records in (("three_stage_metrics", rows), ("three_stage_per_class", classes)):
        if records:
            keys = list(dict.fromkeys(key for row in records for key in row))
            payload[f"recovery/{name}"] = wandb.Table(columns=keys, data=[[row.get(key) for key in keys] for row in records])
    for path in sorted((_EdaPath(folder) / "eda").glob("comparison_*.png")):
        payload[f"recovery/{path.stem}"] = wandb.Image(str(path))
    run.log(payload)


def recovery_suite_tables(report, folder):
    """Export summaries for all completed/failed targets and the measured trade-offs."""
    folder = _EdaPath(folder)
    rows, classes, summary = [], [], []
    for job in report.get("jobs", []):
        metrics, per_class = recovery_comparison_rows(job)
        rows.extend(metrics)
        classes.extend(per_class)
        recovered = metrics[-1]
        summary.append({**_recovery_eda_metadata(job), **{key: job.get(key) for key in (
            "status", "input_status", "training_status", "structural_status", "reload_status", "validation_status",
            "latency_status", "eda_status", "upload_status", "completed_epochs", "best_epoch", "raw_artifact_ref")},
            "actual_gflops_speedup": job.get("actual_gflops_speedup"), "baseline_map50_95": metrics[0]["map50_95"],
            "raw_map50_95": metrics[1]["map50_95"], "recovered_map50_95": recovered["map50_95"],
            "map50_95_recovery_gain_pp": recovered["map50_95_recovery_gain_pp"],
            "parameters": recovered["parameters"], "gflops": recovered["gflops"],
            "latency_median_ms": recovered["latency_median_ms"], "latency_p95_ms": recovered["latency_p95_ms"]})
    for name, table in (("recovery_summary", summary), ("stage_metrics", rows), ("per_class_metrics", classes)):
        write_json(folder / (name + ".json"), table)
        write_csv(folder / (name + ".csv"), table)
    if rows:
        fig, axes = _eda_plt.subplots(1, 3, figsize=(14, 4.3))
        for ax, (metric, xlabel) in zip(axes, (("gflops", "GFLOPs"), ("parameters", "Parameter"),
                                             ("latency_median_ms", "Raw forward p50 (ms)"))):
            for stage, (label, color) in RECOVERY_STAGE_STYLES.items():
                points = {(row[metric], row["map50_95"]) for row in rows if row["stage"] == stage
                          and _eda_finite(row[metric]) and _eda_finite(row["map50_95"])}
                if points:
                    ax.scatter(*zip(*sorted(points)), label=label, color=color, alpha=.85)
            ax.set(xlabel=xlabel, ylabel="mAP50–95")
            ax.grid(alpha=.18)
            if ax.collections:
                ax.legend(frameon=False, fontsize=8)
            else:
                ax.text(.5, .5, "Belum terukur", ha="center", transform=ax.transAxes)
        fig.suptitle("Trade-off semua target pada suite dan perangkat yang sama")
        fig.tight_layout()
        _eda_save_figure(fig, folder / "recovery_tradeoffs")
    return {"summary": summary, "stage_metrics": rows, "per_class_metrics": classes}
