"""Numeric regression checks for cross-configuration notebook 05 analysis."""
# ruff: noqa: S102 -- Execute only trusted standalone notebook declarations.
from __future__ import annotations

import json
from copy import deepcopy
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import pytest

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def eda():
    scope = {"EDA_DPI": 40}
    for name in ("lamp_v2_eda_source", "lamp_gflops_evaluation_source", "lamp_gflops_eda_source"):
        path = ROOT / "scripts" / f"{name}.py"
        exec(compile(path.read_text(encoding="utf-8"), str(path), "exec"), scope)
    return scope


def measured_bundle():
    models = []
    for variant, base_map in (("n", .5), ("s", .6)):
        validation = {"map50_95": base_map, "map50": .7, "precision": .8, "recall": .6,
                      "evaluation_mode": "smoke", "validation_image_count": 128,
                      "per_class": [{"class_id": 1, "name": "B", "ap50_95": .7, "support": 3},
                                    {"class_id": 0, "name": "A", "ap50_95": base_map, "support": 4}]}
        baseline = {"parameters": 1000, "gflops": 2., "checkpoint_bytes": 4000,
                    "validation_status": "PASS", "validation": validation, "latency": {"status": "NOT_RUN"}}
        pruned = {**baseline, "parameters": 900, "gflops": 1.8, "checkpoint_bytes": 3600,
                  "validation": {**validation, "map50_95": base_map - .03, "per_class": [
                      {"class_id": 0, "name": "A", "ap50_95": base_map - .05, "support": 4},
                      {"class_id": 1, "name": "B", "ap50_95": None, "support": 0}]}}
        run = {"configuration_id": "target-1.1x", "policy": "gflops-target", "ratio": .1,
               "pruned": pruned, "validation_status": "PASS", "groups": [{"root": "a"}],
               "actual_eligible_channel_ratio": .1,
               "audit": {"independent_channels_before": 1000, "target_policy_eligible_channels": 200,
                         "independent_channels_removed": 20, "physical_conv_output_channels_removed": 30}}
        models.append({"model_name": f"YOLO26{variant}", "model_variant": variant,
                       "baseline": baseline, "baseline_report": {"reserialized": baseline,
                           "source": {**baseline, "checkpoint_bytes": 2000}, "validation_parity_status": "PASS"},
                       "results": [run], "rows": [{"configuration_id": "target-1.1x", "target_speedup": 1.1,
                           "target_status": "REACHED", "structural_status": "PASS", "reload_status": "PASS"}]})
    return {"mode": "smoke", "models": models, "selection": {"status": "NOT_RUN", "selected": None}}


def test_all_variants_class_id_join_and_distinct_channel_denominators(eda, tmp_path):
    tables = eda["gflops_eda_tables"](measured_bundle(), tmp_path)
    assert len(tables["metrics"]) == 6  # Source, reserialized baseline and student for N/S.
    for variant in ("n", "s"):
        rows = [row for row in tables["per_class_metrics"] if row["model_variant"] == variant]
        assert len(rows) == 4  # One canonical baseline per model, never duplicated for every student.
        student = {row["class_id"]: row for row in rows if row["stage"] == "pruned"}
        assert student[0]["delta_ap50_95_pp"] == pytest.approx(-5.)
        assert student[1]["delta_ap50_95_pp"] is None and student[1]["ap50_95"] is None
        coverage = next(row for row in tables["coverage"] if row["model_variant"] == variant)
        assert (coverage["graph_independent_channels_before"], coverage["policy_eligible_channels"],
                coverage["policy_eligible_channels_after"], coverage["physical_conv_outputs_removed"]) == (1000, 200, 180, 30)
        assert coverage["inventory_status"] == "UNAVAILABLE" and coverage["conv_layers"] is None
    for row in tables["metrics"]:
        if row["stage"] == "pruned":
            assert row["delta_map50_95_pp"] == pytest.approx(-3.)
            assert row["gflops_speedup"] == pytest.approx(2 / 1.8)
            assert row["checkpoint_reduction_pct"] == 10.
            assert row["latency_speedup"] is None
    assert not any(row["qualifies"] for row in tables["student_eligibility"])


def test_failed_validation_payload_is_not_used_and_class_identity_mismatch_fails(eda, tmp_path):
    bundle = measured_bundle()
    bundle["models"][0]["results"][0]["validation_status"] = "FAIL"
    tables = eda["gflops_eda_tables"](bundle, tmp_path)
    row = next(row for row in tables["metrics"] if row["model_variant"] == "n" and row["stage"] == "pruned")
    assert row["map50_95"] is None and row["delta_map50_95_pp"] is None
    assert not any(row["model_variant"] == "n" and row["stage"] == "pruned" for row in tables["per_class_metrics"])
    bundle = measured_bundle()
    bundle["models"][0]["results"][0]["pruned"]["validation"]["per_class"][0]["name"] = "Different class"
    with pytest.raises(ValueError, match="Nama/ID kelas"):
        eda["gflops_build_analysis"](bundle, tmp_path)
    report = json.loads((tmp_path / "comparison/analysis/analysis_status.json").read_text(encoding="utf-8"))
    assert report["status"] == "FAIL"


def test_unmeasured_latency_and_search_have_not_run_panels_without_fake_points(eda, tmp_path):
    bundle = measured_bundle()
    # A stale numeric value in a NOT_RUN record must not become a measured point.
    bundle["models"][0]["results"][0]["pruned"]["latency"]["median_ms"] = 11.
    original = deepcopy(bundle)
    tables = eda["gflops_eda_tables"](bundle, tmp_path)
    plots = eda["gflops_eda_model_plots"](tables, "YOLO26n", tmp_path)
    statuses = {row["plot"]: row for row in plots}
    assert statuses["tradeoff_latency_median_ms"]["status"] == "NOT_RUN"
    assert statuses["tradeoff_latency_median_ms"]["measured_values"] == 0
    assert statuses["target_search_capacity"]["status"] == "NOT_RUN"
    assert statuses["per_class_ap_all"]["measured_values"] == 6  # AP and delta, with a missing class left empty.
    assert (tmp_path / "per_class_ap_all.png").is_file()
    assert bundle == original
