"""Offline numeric and figure contracts for standalone pruning EDA helpers."""
from __future__ import annotations

import csv
import importlib.util
import json
import uuid
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.colors
import matplotlib.pyplot as plt
import pytest

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "scripts/lamp_v2_eda_source.py"


@pytest.fixture
def eda():
    spec = importlib.util.spec_from_file_location("lamp_v2_eda_test", SOURCE)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module.EDA_DPI = 40  # Production defaults to 300 DPI; keep offline fixture exports small.
    return module


@pytest.fixture
def folder():
    path = ROOT / "outputs/test_lamp_v2_eda" / uuid.uuid4().hex
    path.mkdir(parents=True)
    return path


def inventory(count=3):
    before, after = [], []
    for index in range(count):
        protected = index == count - 1
        base = {"layer_name": f"model.{index}.conv", "module_type": "Conv2d",
                "region": "detection_head" if protected else "backbone", "policy": "extended",
                "in_channels": 8, "out_channels": 16, "parameters": 128, "estimated_flops": 1024,
                "eligible": not protected, "dependency_group": "" if protected else f"g{index}",
                "protection_reason": "detection_head" if protected else ""}
        before.append(base)
        after.append({**base, "out_channels": 16 if protected else 12,
                      "parameters": 128 if protected else 96, "estimated_flops": 1024 if protected else 768})
    return before, after


def result(before, after, *, measured=False):
    mappings = []
    scores = []
    for index, (previous, current) in enumerate(zip(before, after)):
        count = current["out_channels"]
        mappings.append({"layer": previous["layer_name"], "module_type": "Conv2d", "dimension": "out",
                         "channels_before": previous["out_channels"], "channels_after": count,
                         "kept_original_indices": list(range(count)),
                         "removed_original_indices": list(range(count, previous["out_channels"])),
                         "dependency_group_ids": [f"g{index}"] if count < 16 else []})
        if count < 16:
            scores.extend({"group_id": f"g{index}", "root": previous["layer_name"], "channel_idx": channel,
                           "score": .001 * (channel + 1), "capacity_eligible": channel >= 8,
                           "removed": channel >= count} for channel in range(16))
    validation = {"map50_95": .5, "per_class": [
        {"class_id": 0, "name": "Bus", "ap50": .7, "ap50_95": .5,
         "precision": .8, "recall": .6, "f1": 2 * .8 * .6 / 1.4, "support": 3}],
        "curves": {"status": "MEASURED", "curves": [{"name": "Precision–Recall", "class_names": ["Bus"],
                   "class_ids": [0], "x": [0., .5, 1.], "y": [[1., .7, .3]],
                   "x_label": "Recall", "y_label": "Precision", "status": "MEASURED"}]},
        "confusion_matrix": {"status": "MEASURED", "matrix": [[2, 1], [1, 0]],
                             "labels": ["Bus", "background"], "confidence_threshold": .001,
                             "matching_iou_threshold": .45, "validation_iou": .7}}
    return {"model_variant": "s", "model_name": "YOLO26s", "policy": "extended", "ratio": .3,
            "pruned_group_count": 2, "eligible_channels_before": 32, "eligible_channels_after": 24,
            "actual_eligible_channel_ratio": .25,
            "baseline": {"parameters": 1000, "gflops": 2., "checkpoint_bytes": 4000,
                         **({"validation": validation} if measured else {})},
            "pruned": {"parameters": 800, "gflops": 1.7, "checkpoint_bytes": 3200,
                       **({"validation": validation} if measured else {})},
            "comparison": {"parameter_reduction_pct": 20.},
            "audit": {"channel_mapping": mappings, "lamp_scores": scores}}


def test_three_panel_reference_plot_has_identical_order_limits_and_true_heights(eda):
    before, after = inventory()
    rows = eda.layer_eda_rows(before, after)
    fig = eda.make_channel_figure(rows, "Reference-style comparison")
    axes = fig.axes
    assert len(axes) == 3
    assert [ax.get_title() for ax in axes] == ["Before Pruning", "After Pruning", "Channel Comparison"]
    assert len({ax.get_ylim() for ax in axes}) == 1
    assert [[label.get_text() for label in ax.get_xticklabels()] for ax in axes] == [[row["layer_name"] for row in rows]] * 3
    assert [bar.get_height() for bar in axes[0].patches] == [16, 16, 16]
    assert [bar.get_height() for bar in axes[1].patches] == [12, 12, 16]
    assert [bar.get_height() for bar in axes[2].patches] == [16, 16, 16, 12, 12, 16]
    assert axes[0].patches[0].get_facecolor() == matplotlib.colors.to_rgba("blue")
    assert axes[1].patches[0].get_facecolor() == matplotlib.colors.to_rgba("red")
    plt.close(fig)


def test_inventory_matches_by_name_not_after_order_and_checks_protection(eda):
    before, after = inventory()
    rows = eda.layer_eda_rows(before, list(reversed(after)))
    assert [row["layer_name"] for row in rows] == [row["layer_name"] for row in before]
    assert rows[-1]["out_channels_removed"] == 0
    after[-1]["out_channels"] = 15
    with pytest.raises(ValueError, match="Protected"):
        eda.layer_eda_rows(before, after)


@pytest.mark.parametrize("mutation", ["missing", "duplicate", "growth"])
def test_inventory_rejects_misaligned_or_nonpruning_changes(eda, mutation):
    before, after = inventory()
    if mutation == "missing":
        after.pop()
    elif mutation == "duplicate":
        after.append(after[0])
    else:
        after[0]["out_channels"] = 17
    with pytest.raises(ValueError):
        eda.layer_eda_rows(before, after)


def test_run_exports_csv_figures_and_preserves_missing_validation(eda, folder):
    before, after = inventory()
    report = eda.build_run_eda(result(before, after), folder, before, after)
    output = folder / "eda"
    assert report["section_order"] == ["identity_coverage", "channels", "masks_lamp", "efficiency", "validation_tradeoffs"]
    assert report["model_name"] == "YOLO26s"
    assert report["validation_tradeoffs"]["status"] == "NOT_RUN"
    assert report["channel_mapping"]["status"] == "MEASURED"
    assert report["lamp_scores"]["finite_scores"] == 32
    assert report["efficiency"]["physical_conv_outputs_before"] == 48
    assert report["efficiency"]["physical_conv_outputs_after"] == 40
    assert report["efficiency"]["independent_eligible_channels_before"] == 32
    with (output / "channel_plot_data.csv").open(newline="", encoding="utf-8") as stream:
        exported = list(csv.DictReader(stream))
    assert [int(row["out_channels_after"]) for row in exported] == [12, 12, 16]
    assert [float(row["out_channels_reduction_pct"]) for row in exported] == [25., 25., 0.]
    assert (output / "channels_overview.png").is_file()
    assert (output / "channels_overview.svg").is_file()
    assert (output / "channels_detail_01.png").is_file()
    assert not list(output.glob("**/confusion_matrix.png"))
    assert json.loads((output / "eda_status.json").read_text(encoding="utf-8"))["status"] == "MEASURED"


def test_detail_pages_never_exceed_32_layers(eda, folder, monkeypatch):
    before, after = inventory(65)
    rows = eda.layer_eda_rows(before, after)
    figures = []
    def capture(fig, path):
        figures.append((Path(path).name, len(fig.axes[0].patches)))
        plt.close(fig)
        return {"png": Path(path).name + ".png", "svg": Path(path).name + ".svg"}
    monkeypatch.setattr(eda, "_eda_save_figure", capture)
    eda.EDA_MAX_LAYERS_PER_PAGE = 100  # Runtime config may reduce, but never exceed requested cap.
    report = eda._eda_channel_plots(rows, folder, "detail test")
    assert report["detail_page_size"] == 32
    assert [count for name, count in figures if name.startswith("channels_detail")] == [32, 32, 1]


def test_mapping_requires_true_partition_and_agreement_with_inventory(eda, folder):
    before, after = inventory()
    run = result(before, after)
    run["audit"]["channel_mapping"][0]["removed_original_indices"] = [12, 12, 14, 15]
    with pytest.raises(ValueError, match="ledger disagrees"):
        eda._eda_mapping_plots(run["audit"], eda.layer_eda_rows(before, after), folder)
    assert eda._eda_mapping_plots({}, eda.layer_eda_rows(before, after), folder)["status"] == "UNAVAILABLE"


def test_sentinel_scores_are_rejected_not_drawn(eda, folder):
    before, after = inventory()
    run = result(before, after)
    run["audit"]["lamp_scores"][0]["score"] = float("inf")
    with pytest.raises(ValueError, match="finite original scores"):
        eda._eda_lamp_plots(run["audit"], folder)
    assert eda._eda_lamp_plots({}, folder)["status"] == "UNAVAILABLE"


def test_analysis_failure_is_persisted_and_propagated(eda, folder):
    before, after = inventory()
    after.pop()
    with pytest.raises(ValueError, match="alignment is unsafe"):
        eda.build_run_eda({}, folder, before, after)
    report = json.loads((folder / "eda/eda_status.json").read_text(encoding="utf-8"))
    assert report["status"] == "FAIL"
    assert report["error"]["type"] == "ValueError"


def test_measured_validation_exports_curves_class_support_and_matching_thresholds(eda, folder):
    before, after = inventory()
    report = eda._eda_evaluation_plots(result(before, after, measured=True), folder)
    assert report["status"] == "MEASURED"
    assert report["measurements"]["pruned"]["curves"]["curve_count"] == 1
    matrix = report["measurements"]["pruned"]["confusion_matrix"]
    assert matrix["matching_iou_threshold"] == .45
    assert matrix["validation_iou"] == .7
    assert matrix["confidence_threshold"] == .001
    assert (folder / "pruned/evaluation_curve_00.svg").is_file()
    with (folder / "pruned/per_class_metrics.csv").open(encoding="utf-8") as stream:
        classes = list(csv.DictReader(stream))
    assert classes[0]["support"] == "3"


def test_cross_model_comparison_never_draws_missing_accuracy_as_zero(eda, folder):
    before, after = inventory()
    run = result(before, after)
    suite = {"model_variant": "s", "model_name": "YOLO26s", "baseline": run["baseline"], "results": [run]}
    report = eda.plot_model_comparison([suite], folder / "missing")
    assert all(row["status"] == "NOT_RUN" for row in report["plots"])
    assert not list((folder / "missing").glob("*.png"))
    run = result(before, after, measured=True)
    suite.update({"baseline": run["baseline"], "results": [run]})
    report = eda.plot_model_comparison([suite], folder / "measured")
    assert [row["status"] for row in report["plots"]] == ["MEASURED", "MEASURED", "NOT_RUN"]
    assert report["plots"][0]["points"] == 2
