"""Keep three-stage measurements, missing statuses and visible W&B panels auditable."""
# ruff: noqa: S102 -- Execute trusted notebook declarations without the training cells.
import ast
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import matplotlib
import pytest

from scripts.generate_standard_finetuning_notebooks import DESTINATIONS, build_notebook

matplotlib.use("Agg")
ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def eda():
    scope = {"EDA_DPI": 35, "CUSTOM_EDA_SHA256": "test-renderer", "SEED": 2024}
    for name in ("lamp_v2_eda", "lamp_workflow", "standard_finetuning_eda", "recovery_custom_eda"):
        path = ROOT / "scripts" / f"{name}_source.py"
        exec(compile(path.read_text(encoding="utf-8"), str(path), "exec"), scope)
    return scope


def measurements():
    def stage(ap, params, flops):
        return {"parameters": params, "gflops": flops, "checkpoint_bytes": params * 4,
            "conv_output_channels": 500 if params == 900 else 600, "validation_status": "PASS",
            "validation": {"map50_95": ap, "map50": ap + .1, "precision": .75, "recall": .65,
                "per_class": [{"class_id": 0, "name": "Car", "ap50_95": ap, "ap50": ap + .1,
                               "precision": .75, "recall": .65, "f1": .69642857, "support": 20}]},
            "latency": {"status": "NOT_RUN", "device": "cpu"}}
    return {"seed": 2024, "method": "crosskd", "target_speedup": 1.2, "target_status": "REACHED",
            "platform": "filkom", "mode": "smoke", "stages": {"baseline": stage(.6, 1000, 12),
                 "raw_pruned": stage(.4, 900, 10), "recovered": stage(.55, 900, 10)}}


def test_each_metric_has_three_exact_values_statuses_and_immutable_cache(eda, tmp_path):
    result = measurements()
    report = eda["ceda_build"](result, tmp_path)
    assert len(report["images"]) == 19
    assert len(list((tmp_path / "custom_eda").glob("*.svg"))) == 19
    rows = eda["ceda_read"](tmp_path / "custom_eda/measurements.json")
    assert len(rows) == 36
    assert [r["value"] for r in rows if r["metric"] == "map50_95"] == [.6, .4, .55]
    assert [r["value"] for r in rows if r["metric"] == "gflops"] == [12, 10, 10]
    assert all(r["seed"] == 2024 and r["method"] == "crosskd" for r in rows)
    assert all(r["value"] is None and r["status"] == "NOT_RUN" for r in rows if r["metric"].startswith("latency"))
    changes = eda["ceda_read"](tmp_path / "custom_eda/delta_gain.json")
    recovered = next(r for r in changes if r["stage"] == "recovered" and r["metric"] == "map50_95")
    assert recovered["delta_vs_baseline_pp"] == pytest.approx(-5)
    assert recovered["gain_vs_raw_pp"] == pytest.approx(15)
    # A retry reuses identical PNG/SVG bytes rather than changing an immutable package.
    assert eda["ceda_build"](result, tmp_path) == report
    (tmp_path / "custom_eda/map50.png").write_bytes(b"modified")
    with pytest.raises(ValueError, match="berubah"):
        eda["ceda_build"](result, tmp_path)


def test_partial_comparison_and_true_measured_zero_are_distinct(eda, tmp_path):
    result = measurements()
    result["stages"].pop("recovered")
    result["stages"]["raw_pruned"]["validation"]["map50_95"] = 0.
    rows, classes = eda["ceda_rows"](result)
    assert rows[1]["map50_95"] == 0 and rows[2]["map50_95"] is None
    assert eda["ceda_status"](rows[1], "map50_95") == "MEASURED"
    assert eda["ceda_status"](rows[2], "map50_95") == "NOT_RUN"
    assert eda["ceda_build"](result, tmp_path, rows=rows, classes=classes)["status"] == "PASS"
    text = (tmp_path / "custom_eda/map50_95.svg").read_text(encoding="utf-8")
    assert "NOT_RUN" in text and "Seed 2024" in text and "su1.20x" in text


def test_notebook_display_includes_all_six_class_panels_and_readable_metric_tables(eda, tmp_path, monkeypatch):
    import IPython.display as ipy
    import pandas as pd
    outputs = []
    monkeypatch.setattr(ipy, "display", outputs.append)
    monkeypatch.setattr(ipy, "Image", lambda filename: SimpleNamespace(image=filename))
    eda["ceda_display"](measurements(), tmp_path)
    images = [item.image for item in outputs if isinstance(item, SimpleNamespace)]
    assert len(images) == 19 and len([name for name in images if "class_" in name]) == 6
    frames = [item for item in outputs if isinstance(item, pd.DataFrame)]
    assert len(frames) == 4
    assert frames[0].columns.tolist() == ["stage", "validation_status", "map50_95", "map50", "precision", "recall"]
    assert {"stage", "name", "f1", "support"} <= set(frames[2].columns)


def test_report_has_run_id_filter_twelve_native_charts_and_all_diagnostic_sections(eda):
    wr = pytest.importorskip("wandb_workspaces.reports.v2")
    result = measurements()
    rows, _ = eda["ceda_rows"](result)
    identity = eda["ceda_identity"](result, rows)
    media = ["visuals/eda/pruning/channel_flow", "visuals/eda/validation/raw_pruned/evaluation_curve_00",
             "visuals/eda/validation/recovered/normalized_by_true", "visuals/eda/training/kd_curves",
             "visuals/eda/tradeoff_gflops", "custom_eda/05_per_class/f1", "custom_eda/04_delta_gain/comparison"]
    report = eda["ceda_report_spec"]("entity", "project", "exact-run-id", "example", identity, media)
    grids = [block for block in report.blocks if isinstance(block, wr.PanelGrid)]
    bars = [panel for block in grids for panel in block.panels if isinstance(panel, wr.BarPlot)]
    assert len(bars) == 12
    assert all(len(bar.metrics) == 3 and set(bar.line_colors.values()) == set(eda["CEDA_COLORS"]) for bar in bars)
    assert all("exact-run-id" in block.runsets[0].filters for block in grids)
    assert len([block for block in report.blocks if isinstance(block, wr.H2)]) == 9
    lines = [panel for block in grids for panel in block.panels if isinstance(panel, wr.LinePlot)]
    assert len(lines) == 5 and all(line.x == "epoch" for line in lines)
    assert "train/total_loss" in lines[-1].y
    # Confusion matrices and PR/F1 graphs must not silently disappear due to file naming.
    keys = {key for block in grids for panel in block.panels if isinstance(panel, wr.MediaBrowser) for key in panel.media_keys}
    assert set(media) <= keys


def test_windows_report_url_is_normalized_before_readback(eda, tmp_path):
    loaded = []
    class Grid:
        def __init__(self):
            self.runsets = [SimpleNamespace(filters="ID == 'run'")]
    report = SimpleNamespace(id="id", title="name", url="https://wandb.ai/entity\\project\\reports\\EDA--id",
                             blocks=[Grid()], save=lambda: None, _to_model=lambda: SimpleNamespace(name=None))
    wr = SimpleNamespace(PanelGrid=Grid, Report=SimpleNamespace(from_url=lambda url: loaded.append(url) or report))
    eda.update(ceda_workspace_sdk=lambda: wr, ceda_report_spec=lambda *a, **k: report)
    record = eda["ceda_save_report"]("entity", "project", "run", "name", {}, [], tmp_path)
    assert record["status"] == "CONFIRMED" and record["url"] == "https://wandb.ai/entity/project/reports/EDA--id"
    assert loaded == [record["url"]]


def test_report_server_names_are_unique_for_identically_seeded_sessions(eda, tmp_path, monkeypatch):
    import random
    wr = pytest.importorskip("wandb_workspaces.reports.v2")
    names = []
    originals = {}
    monkeypatch.setattr(wr.Report, "url", property(lambda self: f"https://wandb.ai/entity/project/reports/EDA--{self.id}"))

    def spec(_entity, _project, run_id, _name, _identity, _keys, **_kw):
        from wandb_workspaces import expr
        report = wr.Report(entity="entity", project="project", title=run_id,
            blocks=[wr.PanelGrid(runsets=[wr.Runset(filters=[expr.Metric("ID") == run_id])])])
        # Save is mocked; the actual instance _to_model used by SDK save() is exercised.
        def save():
            names.append(report._to_model().name)
        report.id = run_id
        object.__setattr__(report, "save", save)
        originals[run_id] = report
        return report

    eda["ceda_report_spec"] = spec
    eda["ceda_workspace_sdk"] = lambda: SimpleNamespace(PanelGrid=wr.PanelGrid,
        Report=SimpleNamespace(from_url=lambda url: originals[url.rsplit("--", 1)[-1]]))
    for run_id in ("runA", "runB"):
        random.seed(42)
        before = random.getstate()
        eda["ceda_save_report"]("entity", "project", run_id, "name", {}, [], tmp_path / run_id)
        # Constructing panel IDs consumes randomness, but the server key does not depend on it.
        assert before != random.getstate()
    assert len(set(names)) == 2 and all(name.startswith("eda") for name in names)


@pytest.mark.parametrize("platform", ["kaggle", "filkom"])
@pytest.mark.parametrize("method,variant", DESTINATIONS)
def test_all_ten_notebooks_keep_core_and_add_visible_eda(platform, method, variant):
    import nbformat
    book = build_notebook(platform, method, variant)
    path = ROOT / "notebooks" / platform / "recovery" / DESTINATIONS[method, variant]
    assert book == json.loads(path.read_text(encoding="utf-8"))
    nbformat.validate(nbformat.from_dict(book))
    assert len(book["metadata"]["research_workflow"]["custom_eda_sha256"]) == 64
    runtime = "\n".join("".join(c["source"]) for c in book["cells"] if c["cell_type"] == "code")
    for cell in book["cells"]:
        if cell["cell_type"] == "code":
            compile("".join(cell["source"]), str(path), "exec")
    assert runtime.index("IMPLEMENTATION_SHA256 =") < runtime.index("CUSTOM_EDA_SHA256 =")
    assert "ceda_display(" in runtime and "ceda_log(" in runtime
    headings = [line for c in book["cells"] if c["cell_type"] == "markdown" for line in "".join(c["source"]).splitlines() if line.startswith("#")]
    assert len(headings) == (7 if variant == "n" else 6)
    assert "recovery_build_eda" in runtime and "_recovery_pruning_diagnostics" in runtime


def test_original_pruning_notebooks_and_unrelated_notebooks_not_rewritten():
    before = json.loads((ROOT / "outputs/recovery_custom_eda_before_20261108.json").read_text(encoding="utf-8"))
    allowed = {f"notebooks/{platform}/recovery/{filename}" for platform in ("kaggle", "filkom") for filename in DESTINATIONS.values()}
    for name, digest in before.items():
        if name not in allowed:
            assert hashlib.sha256((ROOT / name).read_bytes()).hexdigest() == digest, name


def test_custom_layer_does_not_redefine_any_training_or_optimizer_function():
    path = ROOT / "scripts/recovery_custom_eda_selected_source.py"
    definitions = {n.name for n in ast.parse(path.read_text(encoding="utf-8")).body if isinstance(n, ast.FunctionDef)}
    assert definitions == {"ft_eda", "ft_log_results", "ft_registry", "ft_verify_logging"}
