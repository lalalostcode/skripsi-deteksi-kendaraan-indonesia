"""Regression tests for scientific figures and source tables in recovery notebooks."""
# ruff: noqa: S102 -- Execute repository notebook declarations, without notebook side effects.
import csv
import json
from pathlib import Path

import matplotlib
import pytest

matplotlib.use("Agg")
ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def eda():
    scope = {"EDA_DPI": 35}
    for name in ("lamp_v2_eda", "lamp_workflow", "standard_finetuning_eda"):
        path = ROOT / "scripts" / f"{name}_source.py"
        exec(compile(path.read_text(encoding="utf-8"), str(path), "exec"), scope)
    return scope


def measured_job():
    def stage(ap, params, flops, latency):
        return {"parameters": params, "gflops": flops, "checkpoint_bytes": 4 * params,
                "conv_output_channels": params // 10, "validation_status": "PASS",
                "validation": {"map50_95": ap, "map50": ap + .08, "precision": .74, "recall": .63,
                    "per_class": [{"class_id": 1, "name": "Bus", "ap50_95": None, "support": 0},
                                  {"class_id": 0, "name": "Car", "ap50_95": ap, "ap50": ap + .08, "support": 5}]},
                "latency": {"status": "PASS", "median_ms": latency, "p95_ms": latency + 1,
                            "device": "cuda:0", "precision": "FP32", "batch": 1, "imgsz": [640, 640],
                            "fusion": False, "scope": "raw model forward", "warmup": 50, "repeats": 200}}

    return {"model_variant": "s", "platform": "filkom", "mode": "full_val", "configuration_id": "target-1.20x",
            "target_speedup": 1.20, "target_status": "REACHED", "stages": {
                "baseline": stage(.65, 1000, 12, 10), "raw_pruned": stage(.60, 900, 10, 9),
                "recovered": stage(.64, 900, 10, 9.2)}}


def test_recovery_bars_and_csv_match_measured_stages(eda, tmp_path):
    job = measured_job()
    rows, classes = eda["recovery_comparison_rows"](job)
    baseline, raw, recovered = rows
    assert raw["map50_95_drop_pp"] == pytest.approx(5)
    assert recovered["map50_95_recovery_gain_pp"] == pytest.approx(4)
    assert baseline["map50_95_recovery_gain_pp"] is None
    assert recovered["gflops_speedup"] == pytest.approx(1.2)
    assert recovered["latency_speedup"] == pytest.approx(10 / 9.2)
    assert raw["gflops_reduction_pct"] == pytest.approx(100 / 6)
    assert [record["class_id"] for record in classes] == [0, 1] * 3
    assert [record["ap50_95"] for record in classes if record["class_id"] == 1] == [None] * 3
    assert len(eda["recovery_build_alias_eda"](job, tmp_path)) == 5
    root = tmp_path / "eda"
    graph = json.loads((root / "plot_data.json").read_text(encoding="utf-8"))
    with (root / "stage_metrics.csv").open(encoding="utf-8", newline="") as stream:
        table = {record["stage"]: record for record in csv.DictReader(stream)}
    panel = [record for record in graph if record["figure"] == "accuracy"]
    for record in panel:
        assert record["value"] == float(table[record["stage"]][record["metric"]])
    figure, axis = eda["_eda_plt"].subplots()
    assert eda["_recovery_draw_bar_panel"](axis, panel, title="Accuracy", unit="Proporsi") == 12
    assert [bar.get_height() for bar in axis.patches] == [table_row[key] for table_row in rows
                                                         for key in eda["RECOVERY_ACCURACY_FIELDS"]]
    eda["_eda_plt"].close(figure)
    assert len(list(root.glob("comparison_*.png"))) == len(list(root.glob("comparison_*.svg"))) == 5


def test_missing_and_mismatched_measurements_are_not_plotted_as_zero(eda):
    job = measured_job()
    job["stages"]["raw_pruned"]["validation_status"] = "FAIL"
    job["stages"]["baseline"]["latency"]["batch"] = 4
    rows, classes = eda["recovery_comparison_rows"](job)
    assert rows[1]["map50_95"] is None
    assert rows[2]["map50_95_recovery_gain_pp"] is None
    assert all(row["latency_status"] == "NOT_COMPARABLE" and row["latency_speedup"] is None for row in rows)
    assert all(row["ap50_95"] is None for row in classes if row["stage"] == "raw_pruned")


def test_epoch_exports_use_actual_losses_validation_and_optimizer_steps(eda, tmp_path):
    epoch_rows = [{"epoch": 1, "optimizer_steps": 2, "fitness": .52, "losses": {"box_loss": 2.0, "kd_loss": 0.5},
                   "metrics": {"val/box_loss": 2.5, "metrics/mAP50-95(B)": .51}, "learning_rates": [1e-4]}]
    flattened = eda["recovery_epoch_rows"](epoch_rows)
    assert flattened == [{"epoch": 1, "optimizer_steps": 2, "fitness": .52, "train/box_loss": 2.0,
                          "train/kd_loss": 0.5, "val/box_loss": 2.5, "metrics/mAP50-95(B)": .51, "lr/pg0": 1e-4}]
    assert len(eda["_recovery_epoch_plots"](flattened, tmp_path)) == 5
