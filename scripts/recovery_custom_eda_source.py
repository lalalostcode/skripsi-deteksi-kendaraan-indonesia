"""Explicit three-condition charts, Tables, verified EDA packages and W&B Reports.

This presentation layer never loads, trains, or changes a model. It consumes the
measurements made by the recovery evaluator, with their missing-value statuses.
"""
# ruff: noqa: F821 -- Embedded after the notebook's evaluator declarations.
import csv as _ceda_csv
import hashlib as _ceda_hashlib
import json as _ceda_json
import math as _ceda_math
import os as _ceda_os
from pathlib import Path as _CedaPath

CEDA_STAGES = ("baseline", "raw_pruned", "recovered")
CEDA_COLORS = ("#2863AD", "#BA492E", "#217854")
CEDA_METRICS = {
    "map50_95": ("mAP50–95", "Proporsi", 1, "01_accuracy"),
    "map50": ("mAP50", "Proporsi", 1, "01_accuracy"),
    "precision": ("Precision", "Proporsi", 1, "01_accuracy"),
    "recall": ("Recall", "Proporsi", 1, "01_accuracy"),
    "parameters": ("Parameter", "Juta parameter", 1e-6, "02_efficiency"),
    "gflops": ("GFLOPs", "GFLOPs", 1, "02_efficiency"),
    "conv_output_channels": ("Channel Conv2d", "Jumlah channel", 1, "02_efficiency"),
    "checkpoint_bytes": ("Ukuran checkpoint", "MiB", 1 / 2**20, "02_efficiency"),
    "latency_median_ms": ("Latency p50", "ms · raw forward batch 1", 1, "03_latency"),
    "latency_p95_ms": ("Latency p95", "ms · raw forward batch 1", 1, "03_latency"),
    "gflops_speedup": ("Speed-up GFLOPs", "× terhadap baseline asal", 1, "03_latency"),
    "latency_speedup": ("Speed-up latency", "× terhadap baseline asal", 1, "03_latency"),
}
CEDA_CLASS_METRICS = {"ap50_95": "AP50–95", "ap50": "AP50", "precision": "Precision", "recall": "Recall", "f1": "F1", "support": "Support"}
CEDA_METHODS = {"finetune": "Fine-tuned", "native": "Native KD", "crosskd": "CrossKD", "mgd": "MGD"}


def ceda_number(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool) and _ceda_math.isfinite(value)


def ceda_hash(path):
    value = _ceda_hashlib.sha256()
    with _CedaPath(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(2**20), b""):
            value.update(chunk)
    return value.hexdigest()


def ceda_read(path, default=None):
    path = _CedaPath(path)
    return _ceda_json.loads(path.read_text(encoding="utf-8")) if path.is_file() else default


def ceda_write(path, value):
    path = _CedaPath(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + ".tmp")
    temp.write_text(_ceda_json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    _ceda_os.replace(temp, path)


def ceda_csv(path, rows):
    columns = list(dict.fromkeys(key for row in rows for key in row))
    with _CedaPath(path).open("w", encoding="utf-8", newline="") as stream:
        writer = _ceda_csv.DictWriter(stream, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def ceda_identity(result, rows):
    first = rows[0] if rows else {}
    seed = result.get("seed", result.get("recovery_seed", first.get("seed", globals().get("SEED"))))
    method = result.get("method", first.get("method", globals().get("RECOVERY_METHOD", "finetune")))
    target = result.get("target_speedup", first.get("target_speedup"))
    actual = result.get("raw_gflops_speedup", result.get("actual_gflops_speedup"))
    if not ceda_number(actual) and len(rows) > 1:
        actual = rows[1].get("gflops_speedup")
    return {"seed": seed, "method": method, "model_variant": result.get("model_variant", first.get("model_variant", "s")),
            "target_speedup": target, "actual_gflops_speedup": actual,
            "platform": result.get("platform", first.get("platform", globals().get("PLATFORM"))),
            "mode": result.get("mode", first.get("mode", globals().get("MODE"))),
            "target_status": result.get("target_status", first.get("target_status", "NOT_RUN")),
            "raw_artifact_ref": result.get("raw_artifact_ref", first.get("raw_artifact_ref")),
            "raw_sha256": result.get("raw_sha256", first.get("raw_sha256")),
            "baseline_artifact_ref": result.get("baseline_artifact_ref", first.get("baseline_artifact_ref")),
            "baseline_sha256": result.get("baseline_sha256", result.get("baseline_source_sha256", first.get("baseline_sha256"))),
            "baseline_fp32_sha256": result.get("baseline_fp32_sha256", first.get("baseline_fp32_sha256")),
            "teacher_artifact_ref": result.get("teacher_artifact_ref", first.get("teacher_artifact_ref")),
            "teacher_sha256": result.get("teacher_sha256", first.get("teacher_sha256"))}


def ceda_title(identity):
    def speed(value):
        return f"{value:.3f}×" if ceda_number(value) else "NOT_RUN"
    target = f"{identity['target_speedup']:.2f}x" if ceda_number(identity['target_speedup']) else "NOT_RUN"
    return (f"YOLO26{identity.get('model_variant', 's')} · Seed {identity['seed']} · su{target} · aktual {speed(identity['actual_gflops_speedup'])}"
            f"\n{CEDA_METHODS.get(identity['method'], identity['method'])} · {identity['platform']} · {identity['mode']}")


def ceda_rows(result):
    compare = globals().get("fine_tuned_comparison") or recovery_comparison_rows
    rows, classes = compare({**result, "stages": result.get("stages", {})})
    identity = ceda_identity(result, rows)
    return [{**identity, **row} for row in rows], [{**identity, **row} for row in classes]


def ceda_status(row, metric):
    if ceda_number(row.get(metric)):
        return "MEASURED"
    if metric.startswith("latency"):
        return row.get("latency_status") or "NOT_RUN"
    if metric in {"map50_95", "map50", "precision", "recall"}:
        value = row.get("validation_status") or "NOT_RUN"
        return "NOT_MEASURED" if value == "PASS" else value
    return "NOT_RUN"


def ceda_build(result, folder, *, rows=None, classes=None):
    """One axis per metric; null measurements are labelled instead of drawn as zero."""
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch

    folder = _CedaPath(folder)
    target = folder / "custom_eda"
    target.mkdir(parents=True, exist_ok=True)
    if rows is None or classes is None:
        rows, classes = ceda_rows(result)
    identity = ceda_identity(result, rows)
    rows = [{**identity, **row} for row in rows]
    classes = [{**identity, **row} for row in classes]
    if [row["stage"] for row in rows] != list(CEDA_STAGES):
        raise ValueError("EDA custom memerlukan tepat tiga kondisi berurutan: baseline/raw_pruned/recovered.")
    digest = _ceda_hashlib.sha256(_ceda_json.dumps({"rows": rows, "classes": classes,
        "renderer": globals().get("CUSTOM_EDA_SHA256", "standalone")}, sort_keys=True, allow_nan=False).encode()).hexdigest()
    receipt_path = folder / "custom_eda_receipt.json"
    previous = ceda_read(receipt_path, {})
    if previous.get("input_sha256") == digest and previous.get("status") == "PASS":
        if any(not (target / name).is_file() or ceda_hash(target / name) != value for name, value in previous["files_sha256"].items()):
            raise ValueError("Berkas EDA custom berubah setelah verifikasi.")
        return previous
    # A separate package keeps pre-existing immutable results artifacts untouched.
    labels = ["Baseline asal", "Raw pruned", CEDA_METHODS.get(identity["method"], "Hasil recovery")]
    long, deltas, images = [], [], {}

    def save(fig, name, key):
        fig.suptitle(ceda_title(identity), fontsize=10, y=.99)
        fig.tight_layout(rect=(0, .08, 1, .85 if fig.legends else .96))
        fig.text(.5, .015, "Biru: baseline asal  |  Merah: raw pruned  |  Hijau: hasil recovery\n"
                 "NOT_RUN/NOT_COMPARABLE bukan nilai nol.", ha="center", fontsize=8, color="#52606D")
        fig.savefig(target / f"{name}.png", dpi=globals().get("EDA_DPI", 120), bbox_inches="tight")
        fig.savefig(target / f"{name}.svg", bbox_inches="tight", metadata={"Date": None})
        plt.close(fig)
        images[key] = f"{name}.png"

    for metric, (title, unit, scale, section) in CEDA_METRICS.items():
        fig, ax = plt.subplots(figsize=(7.8, 4.8))
        for i, row in enumerate(rows):
            value = row.get(metric)
            status = ceda_status(row, metric)
            long.append({**identity, "stage": row["stage"], "condition": labels[i], "metric": metric,
                "value": value, "display_value": value * scale if ceda_number(value) else None, "unit": unit, "status": status,
                "latency_device": row.get("latency_device"), "validation_fingerprint": row.get("validation_fingerprint")})
            if ceda_number(value):
                bar = ax.bar(i, value * scale, color=CEDA_COLORS[i], width=.6)
                ax.bar_label(bar, labels=[f"{value * scale:.5g}"], padding=4, fontsize=10)
            else:
                ax.text(i, .07, status, transform=ax.get_xaxis_transform(), ha="center", fontsize=9, rotation=15, color="#52606D")
        ax.set(title=title, ylabel=unit, xticks=range(3), xticklabels=labels)
        ax.set_xlim(-.6, 2.6)
        if metric in {"map50_95", "map50", "precision", "recall"}:
            ax.set_ylim(0, 1.1)
        else:
            ax.margins(y=.22)
        ax.grid(axis="y", alpha=.2)
        ax.set_axisbelow(True)
        save(fig, metric, f"custom_eda/{section}/{metric}")

    for metric in ("map50_95", "map50", "precision", "recall"):
        base, raw, recovered = [row.get(metric) for row in rows]
        for stage, value in zip(CEDA_STAGES, (base, raw, recovered), strict=True):
            deltas.append({**identity, "stage": stage, "metric": metric,
                "delta_vs_baseline_pp": 100 * (value - base) if ceda_number(value) and ceda_number(base) else None,
                "gain_vs_raw_pp": 100 * (value - raw) if ceda_number(value) and ceda_number(raw) else None,
                "status": "MEASURED" if ceda_number(value) and ceda_number(base) and ceda_number(raw) else "NOT_RUN"})
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.9))
    for ax, field, title in zip(axes, ("delta_vs_baseline_pp", "gain_vs_raw_pp"),
            ("Delta terhadap baseline asal", "Gain terhadap raw pruned"), strict=True):
        for i, stage in enumerate(CEDA_STAGES):
            items = [r for r in deltas if r["stage"] == stage]
            valid = [(j, r[field]) for j, r in enumerate(items) if ceda_number(r[field])]
            if valid:
                ax.bar([j + (i - 1) * .24 for j, _ in valid], [v for _, v in valid], width=.24, color=CEDA_COLORS[i])
        ax.axhline(0, color="#52606D", linewidth=.7)
        ax.set(title=title, ylabel="Poin persentase", xticks=range(4), xticklabels=["mAP50–95", "mAP50", "P", "R"])
        ax.grid(axis="y", alpha=.2)
    fig.legend([Patch(facecolor=c) for c in CEDA_COLORS], labels, loc="upper center", bbox_to_anchor=(.5, .89), ncol=3, frameon=False)
    save(fig, "delta_gain", "custom_eda/04_delta_gain/comparison")

    class_order = sorted({(r["class_id"], r["name"]) for r in classes})
    for metric, label in CEDA_CLASS_METRICS.items():
        fig, ax = plt.subplots(figsize=(max(8.5, len(class_order) * 1.1), 5))
        indexed = {(r["stage"], r["class_id"]): r for r in classes}
        for i, stage in enumerate(CEDA_STAGES):
            for j, (class_id, _name) in enumerate(class_order):
                value = indexed.get((stage, class_id), {}).get(metric)
                if ceda_number(value):
                    ax.bar(j + (i - 1) * .25, value, width=.25, color=CEDA_COLORS[i])
                else:
                    ax.text(j + (i - 1) * .25, .025, "N/A", transform=ax.get_xaxis_transform(), ha="center", fontsize=6, rotation=90)
        if not class_order:
            ax.text(.5, .5, "NOT_RUN · evaluasi per kelas belum tersedia", ha="center", transform=ax.transAxes)
        ax.set(title=f"{label} per kelas", ylabel="Objek" if metric == "support" else "Proporsi",
               xticks=range(len(class_order)), xticklabels=[name for _, name in class_order])
        if metric != "support":
            ax.set_ylim(0, 1.1)
        ax.grid(axis="y", alpha=.2)
        ax.legend([Patch(facecolor=c) for c in CEDA_COLORS], labels, ncol=3, loc="upper center", fontsize=8)
        save(fig, f"class_{metric}", f"custom_eda/05_per_class/{metric}")
    for name, records in (("stage_metrics", rows), ("per_class_metrics", classes), ("measurements", long), ("delta_gain", deltas)):
        ceda_write(target / f"{name}.json", records)
        ceda_csv(target / f"{name}.csv", records)
    ceda_write(target / "comparison_manifest.json", {"format_version": 1, "identity": identity, "input_sha256": digest,
        "custom_eda_sha256": globals().get("CUSTOM_EDA_SHA256"), "stage_order": CEDA_STAGES, "images": images,
        "meaning": "Baseline asal → raw pruned → hasil recovery; nilai kosong mempertahankan status pengukuran."})
    receipt = {"status": "PASS", "input_sha256": digest, "identity": identity, "images": images,
               "files_sha256": {p.name: ceda_hash(p) for p in sorted(target.iterdir()) if p.is_file()}}
    ceda_write(receipt_path, receipt)
    return receipt


def ceda_workspace_sdk():
    import importlib.metadata
    import subprocess
    import sys
    try:
        installed = importlib.metadata.version("wandb-workspaces")
    except importlib.metadata.PackageNotFoundError:
        installed = None
    if installed != "0.4.5":
        if not globals().get("INSTALL_DEPENDENCIES", True):
            raise RuntimeError("Halaman EDA W&B memerlukan wandb-workspaces==0.4.5.")
        subprocess.run([sys.executable, "-m", "pip", "install", "--no-deps", "wandb-workspaces==0.4.5"], check=True)
    import wandb_workspaces.reports.v2 as reports
    return reports


def ceda_report_spec(entity, project, run_id, name, identity, media_keys, *, summary_prefix="custom_eda/values"):
    """Reports use an explicit run-ID filter, keeping concurrent sessions separate."""
    from wandb_workspaces import expr
    wr = ceda_workspace_sdk()
    runset = wr.Runset(entity=entity, project=project, name=name, filters=[expr.Metric("ID") == run_id])
    labels = ["Baseline asal", "Raw pruned", CEDA_METHODS.get(identity["method"], "Hasil recovery")]
    blocks = [wr.H1("Baseline asal → raw pruned → hasil recovery"), wr.P(ceda_title(identity).replace("\n", " · ")),
        wr.P(f"Run {run_id}. Warna: baseline biru, raw merah, recovery hijau. Baris kosong berarti belum terukur, "
             "bukan nilai nol. Full validation dan smoke dibedakan oleh mode. Latency dibandingkan hanya pada protokol/perangkat yang sama.")]
    for section, heading in (("01_accuracy", "Akurasi deteksi"), ("02_efficiency", "Parameter, GFLOPs, channel, dan checkpoint"),
                             ("03_latency", "Latency dan speed-up")):
        panels = []
        for metric, (title, unit, _scale, group) in CEDA_METRICS.items():
            if group != section:
                continue
            fields = [f"{summary_prefix}/{stage}/{metric}" if summary_prefix else f"{stage}/{metric}" for stage in CEDA_STAGES]
            # Native bars retain the exact original unit in summary (bytes and parameter counts).
            unit = {"parameters": "Jumlah parameter", "checkpoint_bytes": "Byte"}.get(metric, unit)
            panels.append(wr.BarPlot(title=f"{title} · tiga kondisi", metrics=fields, orientation="v", title_y=unit,
                line_titles=dict(zip(fields, labels, strict=True)), line_colors=dict(zip(fields, CEDA_COLORS, strict=True)),
                aggregate=False, layout=wr.Layout(x=(len(panels) % 2) * 12, y=(len(panels) // 2) * 7, w=12, h=7)))
        section_images = sorted(key for key in media_keys if key.startswith(f"custom_eda/{section}/"))
        if section_images:
            panels.append(wr.MediaBrowser(title="Diagram tiga kondisi dan status pengukuran", media_keys=section_images,
                num_columns=2, mode="gallery", layout=wr.Layout(y=14, w=24, h=12)))
        blocks.extend([wr.H2(heading), wr.PanelGrid(runsets=[runset], panels=panels)])
    categories = [
        ("Delta terhadap baseline dan gain recovery", [key for key in media_keys if "delta_gain" in key or "comparison_changes" in key]),
        ("AP, precision, recall, F1, dan support per kelas", [key for key in media_keys if "05_per_class/" in key or "comparison_per_class" in key]),
        ("Channel per layer, ledger, heatmap, dan importance LAMP", [key for key in media_keys if any(s in key.lower() for s in ("channel", "heatmap", "importance", "lamp", "dependency", "physical"))]),
        ("Kurva PR/F1 dan confusion matrix · masing-masing kondisi", [key for key in media_keys if any(s in key.lower() for s in ("confusion", "pr_curve", "f1_curve", "precision_curve", "recall_curve", "/matrix", "normalized_by_true", "evaluation_curve", "/curves/"))]),
        ("Training, validation, loss KD, dan learning rate", [key for key in media_keys if any(s in key.lower() for s in ("epoch_", "training/", "training_curves", "kd_curves", "results"))]),
        ("Trade-off akurasi dan efisiensi", [key for key in media_keys if "tradeoff" in key.lower() or "trade_off" in key.lower()]),
    ]
    for heading, keys in categories:
        blocks.append(wr.H2(heading))
        panels = []
        if heading.startswith("Training,"):
            blocks.append(wr.P("Kurva berikut memakai history per epoch dan sudah dapat dibaca selama training. "
                "Batang hijau pada evaluasi tiga kondisi tetap menunggu evaluasi checkpoint best sesudah recovery selesai."))
            series = [
                ("Loss deteksi per epoch", ["train/box_loss", "train/cls_loss", "train/l1_loss"]),
                ("Validation per epoch · bukan evaluasi akhir tiga kondisi", ["metrics/mAP50-95(B)", "metrics/mAP50(B)", "metrics/precision(B)", "metrics/recall(B)"]),
                ("Learning rate", ["lr/pg0", "lr/pg1", "lr/pg2"]),
                ("Fitness per epoch", ["fitness"]),
            ]
            if identity["method"] != "finetune":
                series.append(("Komponen KD dan total loss", ["train/dis_loss", "train/mgd_weighted_loss", "train/det_total_loss", "train/total_loss"]))
            for i, (title, metrics) in enumerate(series):
                panels.append(wr.LinePlot(title=title, x="epoch", y=metrics, smoothing_type="none",
                    layout=wr.Layout(x=(i % 2) * 12, y=(i // 2) * 7, w=12, h=7)))
        if keys:
            panels.append(wr.MediaBrowser(title=heading, media_keys=sorted(set(keys)), num_columns=2, mode="gallery",
                layout=wr.Layout(y=21 if panels else 0, w=24, h=12)))
        if panels:
            blocks.append(wr.PanelGrid(runsets=[runset], panels=panels))
        else:
            blocks.append(wr.P("NOT_RUN · panel belum tersedia pada run ini. Panel final diunggah sesudah evaluasi tiga kondisi selesai."))
    return wr.Report(entity=entity, project=project, title=f"EDA · {name} · {run_id[:8]}",
                     description="Perbandingan baseline asal, raw pruned, dan hasil recovery.", blocks=blocks, width="fluid")


def ceda_save_report(entity, project, run_id, name, identity, media_keys, folder, *, summary_prefix="custom_eda/values"):
    path = _CedaPath(folder) / "custom_eda_report.json"
    old = ceda_read(path, {})
    report = ceda_report_spec(entity, project, run_id, name, identity, media_keys, summary_prefix=summary_prefix)
    wr = ceda_workspace_sdk()
    if old.get("run_id") == run_id and old.get("url"):
        existing = wr.Report.from_url(old["url"].replace("\\", "/"))
        owners = {block.runsets[0].filters for block in existing.blocks if isinstance(block, wr.PanelGrid) and block.runsets}
        if owners and all(run_id in value for value in owners):
            existing.blocks, existing.title, existing.description = report.blocks, report.title, report.description
            report = existing
        else:
            ceda_write(_CedaPath(folder) / "custom_eda_report_conflict.json", {
                "old_url": old["url"], "reason": "Existing report filters belong to a different run; create an independent report.",
                "run_id": run_id, "observed_filters": sorted(owners)})
    # SDK 0.4.5's save() otherwise generates its server key from Python's seeded
    # random.random(). Identically seeded kernels can overwrite the same report.
    # Bind only this report instance's model name, preserving every RNG state.
    to_model = report._to_model

    def unique_model():
        model = to_model()
        model.name = "eda" + _ceda_hashlib.sha256(f"{entity}/{project}/{run_id}".encode()).hexdigest()[:24]
        return model

    object.__setattr__(report, "_to_model", unique_model)
    try:
        report.save()
    finally:
        # Pydantic validate_assignment may clear non-field instance overrides
        # when save() assigns the returned report ID.
        if "_to_model" in vars(report):
            object.__delattr__(report, "_to_model")
    # SDK 0.4.5 joins URL components with os.path.join on Windows.
    url = report.url.replace("\\", "/")
    ceda_write(path, {"status": "PENDING_VERIFICATION", "run_id": run_id, "url": url, "report_id": report.id})
    loaded = wr.Report.from_url(url)
    owners = [block.runsets[0].filters for block in loaded.blocks if isinstance(block, wr.PanelGrid) and block.runsets]
    if (loaded.id != report.id or len(loaded.blocks) != len(report.blocks) or loaded.title != report.title
            or not owners or any(run_id not in value for value in owners)):
        raise ValueError("Halaman EDA W&B belum terverifikasi setelah disimpan.")
    receipt = {"status": "CONFIRMED", "run_id": run_id, "url": url, "report_id": report.id,
               "blocks": len(loaded.blocks), "identity": identity}
    ceda_write(path, receipt)
    return receipt


def ceda_publish(run, folder, receipt):
    """The separate EDA artifact is retryable and checked by downloading every file."""
    import tempfile

    import wandb
    folder = _CedaPath(folder)
    path = folder / "custom_eda_publication.json"
    hashes = receipt["files_sha256"]
    previous = ceda_read(path, {})
    if previous.get("run_id") == run.id and previous.get("files_sha256") == hashes and previous.get("status") == "CONFIRMED":
        return previous
    api = wandb.Api(timeout=60)
    record = {"run_id": run.id, "name": f"{run.name}-eda", "status": "PENDING", "files_sha256": hashes,
              "input_sha256": receipt["input_sha256"]}
    ceda_write(path, record)
    remote = None
    for candidate in api.run(f"{run.entity}/{run.project}/{run.id}").logged_artifacts():
        if candidate.name.rsplit(":", 1)[0] == record["name"] and candidate.metadata.get("files_sha256") == hashes:
            remote = candidate
            break
    if remote is None:
        artifact = wandb.Artifact(record["name"], type="results", metadata={**receipt["identity"],
            "stage": "custom-three-condition-eda", "files_sha256": hashes, "input_sha256": receipt["input_sha256"],
            "custom_eda_sha256": globals().get("CUSTOM_EDA_SHA256")})
        for name in hashes:
            artifact.add_file(str(folder / "custom_eda" / name), name=name)
        uploaded = run.log_artifact(artifact, aliases=["latest", "three-conditions"])
        uploaded.wait(timeout=600)
        remote = api.artifact(uploaded.qualified_name)
    reference = remote.qualified_name
    if not reference.rsplit(":", 1)[-1].startswith("v"):
        raise ValueError("Publikasi EDA memerlukan versi W&B tetap :vN.")
    with tempfile.TemporaryDirectory(prefix="verify_custom_eda_", dir=folder) as cache:
        downloaded = _CedaPath(remote.download(root=cache))
        if any(ceda_hash(downloaded / name) != digest for name, digest in hashes.items()):
            raise ValueError("SHA-256 paket EDA W&B tidak cocok.")
    record.update(status="CONFIRMED", reference=reference, url=remote.url)
    ceda_write(path, record)
    return record


def ceda_log(run, result, folder):
    import wandb
    receipt = ceda_build(result, folder)
    folder = _CedaPath(folder)
    root = folder / "custom_eda"
    for name in ("stage_metrics", "per_class_metrics", "measurements", "delta_gain"):
        values = ceda_read(root / f"{name}.json", [])
        columns = list(dict.fromkeys(key for row in values for key in row))
        if columns:
            run.log({f"custom_eda/tables/{name}": wandb.Table(columns=columns, data=[[r.get(k) for k in columns] for r in values])})
    for key, name in receipt["images"].items():
        run.log({key: wandb.Image(str(root / name), caption=ceda_title(receipt["identity"]))})
    scalars = {}
    for row in ceda_read(root / "measurements.json"):
        prefix = f"custom_eda/values/{row['stage']}/{row['metric']}"
        scalars[prefix + "_status"] = row["status"]
        scalars[prefix] = row["value"] if ceda_number(row["value"]) else None
    run.summary.update(scalars)
    media = set(receipt["images"]) | set(ceda_read(folder / "wandb_media.json", {}).get("keys", []))
    # N01's legacy comparison logger has no media-index file.
    media |= {f"recovery/{p.stem}" for p in (folder / "eda").glob("comparison_*.png")}
    publication = ceda_publish(run, folder, receipt)
    report = ceda_save_report(run.entity, run.project, run.id, run.name, receipt["identity"], sorted(media), folder)
    status = {"custom_eda_status": "PASS", "custom_eda_report_status": "CONFIRMED", "custom_eda_report_url": report["url"], "custom_eda_artifact_ref": publication["reference"],
              "custom_eda_sha256": globals().get("CUSTOM_EDA_SHA256"), "custom_eda_image_count": len(receipt["images"])}
    run.summary.update(status)
    ceda_write(folder / "custom_eda_logging.json", {**status, "run_id": run.id, "image_keys": sorted(receipt["images"]),
        "table_keys": [f"custom_eda/tables/{n}" for n in ("stage_metrics", "per_class_metrics", "measurements", "delta_gain")]})
    result["custom_eda"] = {**status, "publication": publication}
    return result["custom_eda"]


def ceda_display(result, folder):
    import pandas as pd
    from IPython.display import Image, Markdown, display
    receipt = ceda_build(result, folder)
    folder = _CedaPath(folder)
    display(Markdown("### Baseline asal → raw pruned → hasil recovery\n\n" + ceda_title(receipt["identity"]).replace("\n", " · ")))
    stages = pd.DataFrame(ceda_read(folder / "custom_eda/stage_metrics.json"))
    display(stages[["stage", "validation_status", "map50_95", "map50", "precision", "recall"]])
    display(stages[["stage", "parameters", "gflops", "conv_output_channels", "checkpoint_bytes",
                    "latency_median_ms", "latency_p95_ms", "latency_status", "gflops_speedup"]])
    classes = pd.DataFrame(ceda_read(folder / "custom_eda/per_class_metrics.json"))
    if not classes.empty:
        display(classes[["stage", "name", "ap50_95", "ap50", "precision", "recall", "f1", "support"]])
    changes = pd.DataFrame(ceda_read(folder / "custom_eda/delta_gain.json"))
    display(changes[["stage", "metric", "delta_vs_baseline_pp", "gain_vs_raw_pp", "status"]])
    for name in receipt["images"].values():
        display(Image(filename=str(folder / "custom_eda" / name)))
    record = ceda_read(folder / "custom_eda_report.json", {})
    if record.get("url"):
        display(Markdown(f"**[Buka EDA custom tiga kondisi di W&B]({record['url']})**"))
    else:
        print("Halaman W&B: NOT_RUN. Aktifkan logging dan selesaikan evaluasi/publikasi untuk memperoleh tautannya.")
