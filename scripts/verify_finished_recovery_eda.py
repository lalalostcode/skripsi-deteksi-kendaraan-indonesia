"""Audit presentation-only changes to completed user runs, including epoch history."""
from __future__ import annotations

import json
import os
from pathlib import Path

from scripts import recovery_custom_eda_source as eda

ENTITY = "ilhamrafiqin-brawijaya-university"
PROJECT = "yolom26-kd-indotraffic"
ROOT = Path(__file__).resolve().parents[1]


def main():
    if not os.environ.get("WANDB_API_KEY"):
        from dotenv import dotenv_values
        os.environ["WANDB_API_KEY"] = dotenv_values(ROOT / ".env", interpolate=False)["WANDB_API_KEY"]
    import wandb

    api = wandb.Api(timeout=60)
    root = ROOT / "outputs/pruning/recovery_custom_eda_20261108"
    records = eda.ceda_read(root / "verification.json")
    audits = {}
    wr = eda.ceda_workspace_sdk()
    for run_id, record in records.items():
        run = api.run(f"{ENTITY}/{PROJECT}/{run_id}")
        assert run.state == "finished" and run.summary["status"] == "PASS"
        history = run.history(keys=["epoch", "fitness"], samples=max(500, (run.lastHistoryStep or 0) + 1), pandas=False)
        epochs = [row["epoch"] for row in history]
        assert len(epochs) == len(set(epochs)) == run.summary["epochs_completed"] == record["original_epoch_count"] == 100
        source = api.artifact(record["source_results_artifact_ref"])
        for filename in ("evaluation_manifest.json", "eda/stage_metrics.json", "eda/per_class_metrics.json"):
            assert eda.ceda_hash(root / run_id / "source" / filename) == source.metadata["files_sha256"][filename]
        rows = eda.ceda_read(root / run_id / "source/eda/stage_metrics.json")
        assert [row["stage"] for row in rows] == ["baseline", "raw_pruned", "recovered"]
        assert len({json.dumps(row["validation_fingerprint"], sort_keys=True) for row in rows}) == 1
        assert rows[1]["parameters"] == rows[2]["parameters"] and rows[1]["gflops"] == rows[2]["gflops"]
        report_ref = eda.ceda_read(root / run_id / "custom_eda_report.json")
        report = wr.Report.from_url(report_ref["url"])
        grids = [block for block in report.blocks if isinstance(block, wr.PanelGrid)]
        assert all(run_id in grid.runsets[0].filters for grid in grids)
        assert len([panel for grid in grids for panel in grid.panels if isinstance(panel, wr.BarPlot)]) == 12
        assert len([panel for grid in grids for panel in grid.panels if isinstance(panel, wr.LinePlot)]) == 4
        publication = eda.ceda_read(root / run_id / "custom_eda_publication.json")
        assert publication["status"] == "CONFIRMED"
        assert api.artifact(publication["reference"]).metadata["files_sha256"] == publication["files_sha256"]
        models = api.artifact(run.summary["checkpoint_artifact_ref"])
        hashes = models.metadata["files_sha256"]
        audits[run_id] = {"status": "PASS", "run_id": run_id, "run_name": run.name, "epochs": epochs,
            "no_duplicate_epochs": True, "no_training_during_backfill": True,
            "checkpoint_artifact_ref": models.qualified_name, "checkpoint_hashes_from_original_publication_metadata": hashes,
            "source_results_artifact_ref": source.qualified_name, "source_table_sha256": source.metadata["files_sha256"]["eda/stage_metrics.json"],
            "custom_eda_artifact_ref": publication["reference"], "custom_eda_file_hashes": publication["files_sha256"],
            "report": report_ref, "measured_stage_metrics": rows,
            "scope": "Original user measurements read from fixed artifacts; EDA package download verified separately. No full evaluation repeated."}
        print("FINISHED_RUN_EDA", run_id, "PASS", len(epochs), flush=True)
    target = ROOT / "reports/pruning/recovery/finished_recovery_eda_audit_20261108.json"
    eda.ceda_write(target, {"status": "PASS", "rows": audits})


if __name__ == "__main__":
    main()
