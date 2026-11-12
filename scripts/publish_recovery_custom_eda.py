"""Backfill visible EDA for finished recovery runs, without training or touching checkpoints.

Active runs only accept --reports-only: a public summary update and a live report,
never a second SDK session using their run ID.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path

from scripts import recovery_custom_eda_source as eda

ENTITY = "ilhamrafiqin-brawijaya-university"
PROJECT = "yolom26-kd-indotraffic"


def backfill(api, run_id, output, *, reports_only=False):
    import wandb
    source = api.run(f"{ENTITY}/{PROJECT}/{run_id}")
    summary = dict(source.summary)
    folder = output / run_id
    folder.mkdir(parents=True, exist_ok=True)
    identity = {"seed": summary.get("seed", source.config.get("seed")),
                "method": summary.get("method", source.config.get("method", "finetune")),
                "target_speedup": summary.get("target_speedup", source.config.get("target_speedup")),
                "actual_gflops_speedup": summary.get("actual_gflops_speedup", source.config.get("actual_gflops_speedup")),
                "platform": summary.get("platform", source.config.get("platform")),
                "mode": summary.get("mode", source.config.get("mode")), "target_status": summary.get("target_status")}
    if not (folder / "custom_eda_report.json").exists() and summary.get("custom_eda_report_url"):
        eda.ceda_write(folder / "custom_eda_report.json", {"run_id": run_id, "url": summary["custom_eda_report_url"]})
    if reports_only:
        keys = [key for key, value in summary.items() if isinstance(value, dict) and value.get("_type", "").startswith("image")]
        report = eda.ceda_save_report(ENTITY, PROJECT, run_id, source.name, identity, keys, folder, summary_prefix="")
        report_status = "CONFIRMED" if source.state == "finished" and summary.get("custom_eda_status") == "PASS" else "LIVE_REPORT"
        source.summary.update({"custom_eda_report_url": report["url"], "custom_eda_report_status": report_status,
                               "custom_eda_report_note": "Laporan membaca run asal; evaluasi tiga kondisi mengisi panel setelah selesai."})
        return {"status": report_status, "run_id": run_id, "url": report["url"], "training_started": False}
    if source.state != "finished" or summary.get("status") != "PASS":
        raise ValueError(f"Run {run_id} masih aktif/belum PASS; gunakan --reports-only agar kernel training tetap independen.")
    reference = summary["results_artifact_ref"]
    artifact = api.artifact(reference)
    manifest_path = artifact.get_path("evaluation_manifest.json").download(root=str(folder / "source"))
    expected = artifact.metadata.get("files_sha256", {})
    if not expected.get("evaluation_manifest.json") or eda.ceda_hash(manifest_path) != expected["evaluation_manifest.json"]:
        raise ValueError("SHA-256 manifest evaluasi sumber tidak cocok dengan metadata W&B.")
    result = json.loads(Path(manifest_path).read_text(encoding="utf-8"))
    # Reuse the exact measured tables; this command performs no inference or training.
    for name in ("stage_metrics", "per_class_metrics"):
        local = artifact.get_path(f"eda/{name}.json").download(root=str(folder / "source"))
        if not expected.get(f"eda/{name}.json") or eda.ceda_hash(local) != expected[f"eda/{name}.json"]:
            raise ValueError(f"SHA-256 tabel EDA sumber tidak cocok: {name}")
        if name == "stage_metrics":
            rows = json.loads(Path(local).read_text(encoding="utf-8"))
        else:
            classes = json.loads(Path(local).read_text(encoding="utf-8"))
    result.update({key: value for key, value in identity.items() if value is not None})
    eda.ceda_rows = lambda _value: (rows, classes)
    keys = [key for key, value in summary.items() if isinstance(value, dict) and value.get("_type", "").startswith("image")]
    eda.ceda_write(folder / "wandb_media.json", {"keys": keys})
    with wandb.init(entity=ENTITY, project=PROJECT, id=run_id, resume="must", mode="online", dir=str(folder),
                    settings=wandb.Settings(disable_code=True, disable_git=True)) as run:
        record = eda.ceda_log(run, result, folder)
        run.summary["custom_eda_source_results_ref"] = reference
        run.summary["custom_eda_backfill_only"] = True
    # Api.run caches objects; a new reader must observe the just-finished SDK session.
    current = wandb.Api(timeout=60).run(f"{ENTITY}/{PROJECT}/{run_id}")
    for key in eda.ceda_read(folder / "custom_eda_logging.json")["image_keys"]:
        if current.summary.get(key, {}).get("_type") != "image-file":
            raise ValueError(f"Gambar W&B belum terverifikasi: {key}")
    receipt = eda.ceda_read(folder / "custom_eda_receipt.json")
    for key, name in receipt["images"].items():
        if current.summary[key].get("sha256") != eda.ceda_hash(folder / "custom_eda" / name):
            raise ValueError(f"SHA-256 gambar W&B tidak cocok: {key}")
    if current.summary.get("custom_eda_report_url") != record["custom_eda_report_url"]:
        raise ValueError("Tautan report belum tersimpan pada run asal.")
    return {**record, "status": "PASS", "run_id": run_id, "source_results_artifact_ref": reference,
            "training_started": False, "original_epoch_count": summary.get("epochs_completed"),
            "new_epoch_count": current.summary.get("epochs_completed")}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-ids", nargs="+", required=True)
    parser.add_argument("--reports-only", action="store_true")
    parser.add_argument("--output", type=Path, default=Path("outputs/pruning/recovery_custom_eda_20261108"))
    args = parser.parse_args()
    if not os.environ.get("WANDB_API_KEY"):
        from dotenv import dotenv_values
        os.environ["WANDB_API_KEY"] = dotenv_values(Path(__file__).resolve().parents[1] / ".env", interpolate=False)["WANDB_API_KEY"]
    os.environ.update(MPLBACKEND="Agg", WANDB_MODE="online")
    import wandb
    eda.CUSTOM_EDA_SHA256 = hashlib.sha256(Path(eda.__file__).read_bytes()).hexdigest()
    eda.EDA_DPI = 100
    args.output.mkdir(parents=True, exist_ok=True)
    receipt = args.output / ("live_reports.json" if args.reports_only else "verification.json")
    records = eda.ceda_read(receipt, {})
    api = wandb.Api(timeout=60)
    for run_id in args.run_ids:
        record = backfill(api, run_id, args.output, reports_only=args.reports_only)
        records[run_id] = record
        eda.ceda_write(receipt, records)
        print("CUSTOM_EDA", run_id, record["status"], record.get("custom_eda_report_url", record.get("url")), flush=True)


if __name__ == "__main__":
    main()
