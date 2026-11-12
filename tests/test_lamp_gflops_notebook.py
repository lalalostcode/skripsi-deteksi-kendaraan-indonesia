"""Offline notebook 05 contracts and focused structural regressions."""
# ruff: noqa: S102 -- These tests execute only trusted authoring declarations.
from __future__ import annotations

import ast
import hashlib
import json
import runpy
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace

import nbformat
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("torch_pruning")

ROOT = Path(__file__).resolve().parents[1]
PRUNING = ROOT / "notebooks/kaggle/pruning"
NOTEBOOKS = {"n": PRUNING / "05_lamp_yolo26n_gflops_no_recovery.ipynb",
             "s": PRUNING / "06_lamp_yolo26s_gflops_no_recovery.ipynb"}
GENERATOR = ROOT / "scripts/generate_lamp_gflops_notebook.py"


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


@pytest.fixture
def engine():
    def write_json(path, value):
        Path(path).write_text(json.dumps(value), encoding="utf-8")

    scope = {"sha256_file": digest, "write_json": write_json, "ordered_names": list}
    for name in ("lamp_v2_policy_source", "lamp_gflops_policy_source", "lamp_gflops_evaluation_source"):
        path = ROOT / "scripts" / f"{name}.py"
        exec(compile(path.read_text(encoding="utf-8"), str(path), "exec"), scope)
    return SimpleNamespace(**scope)


def test_gflops_crossing_uses_smallest_integer_and_preserves_first_feasible_tier(engine):
    nn = torch.nn
    hidden, other = nn.Conv2d(3, 100, 1), nn.Conv2d(3, 100, 1)
    interfaces = {f"model.{stage}": SimpleNamespace(conv=nn.Conv2d(3, 1, 1))
                  for stage in engine.GFLOPS_INTERFACE_STAGES}
    audit = {"structure": {"modules": interfaces, "conservative": {hidden}}, "eligible": {}}
    for name, layer in (("z.hidden", hidden), ("a.other", other)):
        audit["eligible"][layer] = {"root": layer, "name": name, "initial_channels": 100,
                                    "output_dimensions": {layer: 100}, "initial_lamp_scores": [1.0] * 100}
    ranked = engine.gflops_ranked_candidates(audit, (25, 10))
    assert ranked[:11] == [(1.0, "a.other", i) for i in range(10)] + [(1.0, "z.hidden", 0)]
    calls = set()

    def measure(plan):
        key = engine.gflops_plan_key(plan)
        assert key not in calls
        calls.add(key)
        return 100 - 0.2 * sum(map(len, plan.values()))

    high_targets = (1.30, 1.40, 1.50, 1.60, 1.70)
    result = engine.search_gflops_targets(audit, 100, measure, (1.10, 1.20, *high_targets))
    assert [tier["capacity_channels"] for tier in result["tiers"]] == [35, 55, 100]
    assert [(row["status"], row["tier"], row["channels"]) for row in result["targets"]] == [
        ("REACHED", 2, 46), ("REACHED", 3, 84), *[("TARGET_NOT_REACHED", 3, 100)] * len(high_targets),
    ]
    assert result["targets"][0]["plan"] == {"a.other": list(range(20)), "z.hidden": list(range(26))}
    assert len({engine.gflops_plan_key(row["plan"]) for row in result["targets"][2:]}) == 1
    with pytest.raises(ValueError, match="melampaui kapasitas"):
        engine._gflops_prefix_plan(ranked, 36)


def test_profile_resume_keeps_successful_budgets_and_retries_failed_tier(engine):
    hidden = torch.nn.Conv2d(3, 100, 1)
    interfaces = {f"model.{stage}": SimpleNamespace(conv=torch.nn.Conv2d(3, 1, 1))
                  for stage in engine.GFLOPS_INTERFACE_STAGES}
    audit = {"structure": {"modules": interfaces, "conservative": {hidden}}, "eligible": {
        hidden: {"root": hidden, "name": "hidden", "initial_channels": 100,
                 "output_dimensions": {hidden: 100}, "initial_lamp_scores": [1.0] * 100}}}
    journal = {}

    def checkpoint(cache, points):
        journal.update(cache)

    def initial(plan):
        amount = sum(map(len, plan.values()))
        if amount == 50:
            raise RuntimeError("synthetic profile failure at maximum capacity")
        return 100 - amount / 2

    partial = engine.search_gflops_targets(audit, 100, initial, profile_cache={}, progress_callback=checkpoint)
    assert len(partial["targets"]) == 5
    assert partial["targets"][0]["status"] == "REACHED"
    assert partial["targets"][-1]["status"] == "FAIL"
    measured = set(journal)
    retried = []

    def recovered(plan):
        key = engine.gflops_plan_key(plan)
        assert key not in measured
        retried.append(key)
        return 100 - sum(map(len, plan.values())) / 2

    complete = engine.search_gflops_targets(audit, 100, recovered, profile_cache=journal)
    assert retried
    assert complete["targets"][0] == partial["targets"][0]
    assert complete["targets"][-1]["status"] == "TARGET_NOT_REACHED"


@pytest.mark.parametrize("variant", ["n", "s"])
def test_target_controller_retains_alias_rows_and_continues_after_job_failure(engine, tmp_path, variant):
    measured = {"parameters": 100, "gflops": 10, "checkpoint_bytes": 1000,
                "latency": {"status": "NOT_RUN"}, "validation_status": "PASS"}
    baseline = {"source": measured, "reserialized": measured, "checkpoint": {"path": "baseline.pt", "sha256": "base"},
                "validation_parity_status": "PASS"}
    high_targets = (1.30, 1.40, 1.50, 1.60, 1.70)
    targets = (1.02, 1.05, 1.10, 1.20, *high_targets)
    plans = ([0], [0], [0, 1], [0, 1, 2], *[[0, 1, 2, 3]] * len(high_targets))
    search = {"targets": [{"target": target, "plan": {"conv": plan},
                           "status": "TARGET_NOT_REACHED" if target in high_targets else "REACHED"}
                          for target, plan in zip(targets, plans, strict=True)]}
    jobs = []

    def experiment(path, provenance, variant, image, baseline_report, dataset, folder, job):
        target = job["target"]
        jobs.append(target)
        if target == 1.10:
            raise RuntimeError("candidate evaluation error")
        result = {"model_name": f"YOLO26{variant}", "model_variant": variant, "configuration_id": job["id"],
                  "policy": job["policy"], "ratio": .10, "target_speedup": target, "target_status": job["status"],
                  "pruned": {"gflops": 8, "validation": {}}, "status": "PASS", "structural_smoke_status": "PASS",
                  "reload_status": "PASS", "validation_status": "PASS", "eda_status": "PASS", "comparison": {},
                  "saved_pruned": {"path": str(folder / "pruned_fp32.pt"), "sha256": str(target)}}
        return result, scope["gflops_result_row"](result, baseline_report, dataset)

    scope = {"json": json, "torch": torch, "traceback": __import__("traceback"), "deepcopy": deepcopy,
             "MODE": "smoke", "SEED": 42, "IMGSZ": 32, "TARGET_DEVICE": torch.device("cpu"),
             "RUN_DIAGNOSTIC_CONTROLS": False, "GFLOPS_TARGETS": targets, "safe_error": str,
             "obtain_baseline": lambda *_: ("baseline.pt", {}), "gflops_plan_key": engine.gflops_plan_key}
    path = NOTEBOOKS[variant]
    book = nbformat.read(path, as_version=4)
    for marker in ("def workflow_status(", "def gflops_run_model("):
        source = next(cell.source for cell in book.cells if cell.cell_type == "code" and marker in cell.source)
        exec(compile(source, str(path), "exec"), scope)
    scope.update(gflops_baseline=lambda *_: baseline, gflops_load_source=lambda *_: object(),
                 gflops_make_graph=lambda *_: (None, {}), search_gflops_targets=lambda *_, **__: search,
                 gflops_experiment=experiment)
    suite = scope["gflops_run_model"](variant, tmp_path, None)
    rows = {row["target_speedup"]: row for row in suite["rows"] if row.get("target_speedup")}
    assert list(rows) == list(targets)
    assert jobs == [1.02, 1.10, 1.20, 1.70]
    assert rows[1.02]["checkpoint_sha256"] == rows[1.05]["checkpoint_sha256"]
    assert all(rows[target]["target_status"] == "TARGET_NOT_REACHED" for target in high_targets)
    assert {rows[target]["artifact_configuration_id"] for target in high_targets} == {"target-1.70x"}
    assert {rows[target]["checkpoint_sha256"] for target in high_targets} == {"1.7"}
    assert {rows[target]["gflops_speedup"] for target in high_targets} == {1.25}
    assert rows[1.10]["status"] == "FAIL" and rows[1.20]["status"] == "PASS"
    assert suite["status"] == "PARTIAL"


def test_p5_control_uses_exact_original_indices_and_same_global_budget(engine):
    nn = torch.nn
    p5, other = nn.Conv2d(3, 8, 1), nn.Conv2d(3, 10, 1)
    p5_root = engine.GFLOPS_P5_ROOT
    audit = {"structure": {"modules": {p5_root: p5}, "conservative": set()}, "eligible": {}}
    for name, layer in ((p5_root, p5), ("other", other)):
        width = layer.out_channels
        audit["eligible"][layer] = {"root": layer, "name": name, "signature": (name,),
                                    "initial_channels": width, "output_dimensions": {layer: width},
                                    "initial_lamp_scores": list(range(width))}
    controls = engine.p5_control_plans(audit, [
        {"root": p5_root, "removed_original_indices": [7, 3]},
        {"root": "other", "removed_original_indices": [9]},
    ])
    assert controls == {"p5-only": {p5_root: [3, 7]}, "p5-protected": {"other": [0, 1, 2]}}


@pytest.mark.parametrize("variant", ["n", "s"])
def test_exact_p5_plan_rebuilds_the_same_graph_and_ledger(engine, variant):
    from ultralytics import YOLO

    threads = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        torch.manual_seed(42)
        baseline = YOLO(f"yolo26{variant}.yaml", task="detect", verbose=False).model.float().eval()
        baseline.end2end = True
        image = torch.rand(1, 3, 64, 64)
        snapshot = engine.protected_policy_snapshot(baseline)
        plan = {engine.GFLOPS_P5_ROOT: [2, 8, 14]}
        runs = []
        for _ in range(2):
            core = deepcopy(baseline)
            pruner, audit = engine.build_policy_pruner(core, image, "extended", 0.1, variant)
            width = audit["structure"]["modules"][engine.GFLOPS_P5_ROOT].out_channels
            with pytest.raises(ValueError, match="melebihi cap"):
                engine.apply_exact_gflops_plan(pruner, audit, plan, caps={engine.GFLOPS_P5_ROOT: 2})
            assert audit["structure"]["modules"][engine.GFLOPS_P5_ROOT].out_channels == width
            history = engine.apply_exact_gflops_plan(pruner, audit, plan)
            assert history[0]["removed_original_indices"] == plan[engine.GFLOPS_P5_ROOT]
            assert audit["report"]["independent_channels_removed"] == 3
            assert engine.verify_gflops_oracle(baseline, core, image, audit)["status"] == "PASS"
            engine.assert_policy_protected_unchanged(core, snapshot)
            runs.append((core.state_dict(), audit["report"]["channel_mapping"]))
        assert runs[0][1] == runs[1][1]
        assert all(torch.equal(tensor, runs[1][0][name]) for name, tensor in runs[0][0].items())
    finally:
        torch.set_num_threads(threads)


class BranchedConv(torch.nn.Module):
    def __init__(self, width):
        super().__init__()
        nn = torch.nn
        self.left = nn.Conv2d(1, width, 1, bias=True)
        self.bn = nn.BatchNorm2d(width)
        self.right = nn.Conv2d(1, width, 1, bias=True)
        self.act = nn.SiLU()
        self.consumer = nn.Conv2d(width, 1, 1, bias=True)
        self.concat_consumer = nn.Conv2d(width + 1, 1, 1, bias=True)

    def forward(self, value):
        residual = self.act(self.bn(self.left(value))) + self.right(value)
        direct = self.consumer(residual)
        concat = self.concat_consumer(torch.cat((value, residual), dim=1))
        return direct, {"one2many": {"boxes": direct, "scores": concat},
                        "one2one": {"boxes": concat, "scores": direct}}


def test_oracle_covers_bn_silu_bias_residual_concat_and_fanout(engine):
    dense, compact = BranchedConv(2).eval(), BranchedConv(1).eval()
    with torch.no_grad():
        for module in (dense.left, dense.right, dense.consumer, dense.concat_consumer):
            module.weight.fill_(2)
            module.bias.fill_(0.25)
        dense.bn.weight.fill_(1)
        dense.bn.bias.copy_(torch.tensor([0.0, 1.0]))
        dense.bn.running_mean.copy_(torch.tensor([0.0, 0.5]))
        dense.bn.running_var.fill_(1)
        for name in ("left", "right"):
            source, target = getattr(dense, name), getattr(compact, name)
            target.weight.copy_(source.weight[:1])
            target.bias.copy_(source.bias[:1])
        for name in ("weight", "bias", "running_mean", "running_var"):
            getattr(compact.bn, name).copy_(getattr(dense.bn, name)[:1])
        compact.consumer.weight.copy_(dense.consumer.weight[:, :1])
        compact.consumer.bias.copy_(dense.consumer.bias)
        compact.concat_consumer.weight.copy_(dense.concat_consumer.weight[:, :2])
        compact.concat_consumer.bias.copy_(dense.concat_consumer.bias)

    def mapping(layer, dimension, before, kept):
        return {"layer": layer, "dimension": dimension, "channels_before": before,
                "channels_after": len(kept), "kept_original_indices": kept,
                "removed_original_indices": [index for index in range(before) if index not in kept]}

    rows = [mapping(name, dimension, before, kept) for name, dimension, before, kept in (
        ("left", "in", 1, [0]), ("left", "out", 2, [0]), ("bn", "channel", 2, [0]),
        ("right", "in", 1, [0]), ("right", "out", 2, [0]),
        ("consumer", "in", 2, [0]), ("consumer", "out", 1, [0]),
        ("concat_consumer", "in", 3, [0, 1]), ("concat_consumer", "out", 1, [0]),
    )]
    audit = {"report": {"channel_mapping": rows}}
    image = torch.ones(1, 1, 2, 2)
    report = engine.verify_gflops_oracle(dense, compact, image, audit)
    assert report["status"] == "PASS" and report["masked_conv_consumers"] == 2
    assert len(report["raw_outputs"]) == 4

    before_bn_mask = deepcopy(dense)
    with torch.no_grad():
        before_bn_mask.left.weight[1].zero_()
        before_bn_mask.left.bias[1].zero_()
        before_bn_mask.right.weight[1].zero_()
        before_bn_mask.right.bias[1].zero_()
        assert not torch.allclose(before_bn_mask(image)[0], compact(image)[0])

    # Offset zero would mask the input concatenated before the residual rather than its removed channel.
    bad = deepcopy(audit)
    bad["report"]["channel_mapping"][7] = mapping("concat_consumer", "in", 3, [1, 2])
    with pytest.raises(ValueError, match="Forward raw berbeda"):
        engine.verify_gflops_oracle(dense, compact, image, bad)


def test_dataset_smoke_is_fixed_fingerprinted_and_cannot_select_a_student(engine, tmp_path):
    root = tmp_path / "dataset"
    images = root / "valid/images"
    labels = root / "valid/labels"
    images.mkdir(parents=True)
    labels.mkdir(parents=True)
    for index in range(140):
        (images / f"{index:03}.jpg").write_bytes(str(index).encode())
    source = root / "data.yaml"
    source.write_text("names: [person]\nval: ../valid/images\n", encoding="utf-8")
    first = engine.gflops_prepare_dataset("smoke", source, root, tmp_path / "one", seed=42)
    second = engine.gflops_prepare_dataset("smoke", source, root, tmp_path / "two", seed=42)
    assert first["image_count"] == 128 and first["full_val_image_count"] == 140
    assert first["fingerprint"] == second["fingerprint"] and first["image_files"] == second["image_files"]
    (labels / f"{Path(first['image_files'][0]).stem}.txt").write_text("", encoding="utf-8")
    changed = engine.gflops_prepare_dataset("smoke", source, root, tmp_path / "three", seed=42)
    assert changed["fingerprint"] != first["fingerprint"]
    full = engine.gflops_prepare_dataset("full_val", source, root, tmp_path / "full", seed=42)
    assert full["image_count"] == 140
    assert full["provenance_status"] == "UNVERIFIED"
    assert engine.gflops_select_student([], "smoke")["status"] == "NOT_RUN"

    train, test = root / "train/images", root / "test/images"
    train.mkdir(parents=True)
    test.mkdir(parents=True)
    (train / "train.jpg").write_bytes(b"training")
    held_out = test / "test.jpg"
    held_out.write_bytes(b"held out")
    source.write_text("names: [person]\nval: ../valid/images\ntrain: train/images\ntest: test/images\n",
                      encoding="utf-8")
    test_args = {"seed": 42, "expected_fingerprint": full["fingerprint"], "dataset_id": engine.POLTRADA_V3_ID}
    independent = engine.gflops_prepare_dataset("test", source, root, tmp_path / "test", **test_args)
    assert independent["provenance_status"] == "VERIFIED"
    assert independent["split"] == "test" and independent["test_content_disjoint_from_train_val"] is True
    cache = held_out.with_suffix(".npy")
    cache.write_bytes(b"stale cache")
    with pytest.raises(ValueError, match="Cache gambar"):
        engine.gflops_prepare_dataset("test", source, root, tmp_path / "test", **test_args)
    cache.unlink()
    held_out.write_bytes((images / "000.jpg").read_bytes())
    with pytest.raises(ValueError, match="berbagi konten gambar dengan full val"):
        engine.gflops_prepare_dataset("test", source, root, tmp_path / "test", **test_args)


def test_student_selection_uses_full_val_speed_target_before_other_metrics(engine):
    baseline = {"evaluation_mode": "full_val", "dataset_provenance_status": "VERIFIED",
                "dataset_fingerprint": "fingerprint", "structural_status": "PASS", "reload_status": "PASS",
                "validation_status": "PASS", "latency_status": "MEASURED", "target_speedup": 1.2,
                "gflops_speedup": 1.2, "map50_95": 0.58, "baseline_map50_95": 0.60,
                "gflops": 5.0, "parameters": 2_000_000, "latency_median_ms": 11.0, "model_variant": "n"}
    rows = [baseline, {**baseline, "model_variant": "s", "map50_95": 0.59},
            {**baseline, "target_speedup": 1.1, "map50_95": 0.60},
            {**baseline, "target_speedup": 1.3, "gflops_speedup": 1.3, "map50_95": 0.57}]
    assert engine.gflops_select_student([baseline], "full_val")["status"] == "SELECTED"
    assert engine.gflops_select_student(rows, "full_val")["selected"] == rows[1]
    unverified = {**baseline, "dataset_provenance_status": "UNVERIFIED"}
    assert engine.gflops_select_student([unverified], "full_val")["status"] == "NO_QUALIFYING_STUDENT"


def candidate_export(folder, variant, *, map50_95=.58):
    folder.mkdir()
    student, baseline = folder / "student.pt", folder / "baseline.pt"
    student.write_bytes(f"student {variant}".encode())
    baseline.write_bytes(f"baseline {variant}".encode())
    row = {"model_variant": variant, "evaluation_mode": "full_val", "dataset_provenance_status": "VERIFIED",
           "dataset_fingerprint": "dataset", "structural_status": "PASS", "reload_status": "PASS",
           "validation_status": "PASS", "latency_status": "MEASURED", "target_speedup": 1.2,
           "gflops_speedup": 1.2, "map50_95": map50_95, "baseline_map50_95": .60,
           "gflops": 5., "parameters": 2_000_000, "latency_median_ms": 11.,
           "checkpoint_path": "/old-session/student.pt", "checkpoint_relative_path": "student.pt",
           "baseline_checkpoint_path": "/old-session/baseline.pt", "baseline_checkpoint_relative_path": "baseline.pt",
           "checkpoint_sha256": digest(student), "baseline_checkpoint_sha256": digest(baseline)}
    payload = {"format_version": 1, "mode": "full_val", "status": "PASS", "evaluation_status": "PASS",
               "model_variants": [variant], "dataset_provenance_status": "VERIFIED", "dataset_fingerprint": "dataset",
               "comparison_contract": {"implementation_sha256": "method", "gpu": "T4", "seed": 42}, "rows": [row]}
    path = folder / "student_candidates.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path, payload


def test_split_exports_require_both_models_and_relocate_hash_checked_student(engine, tmp_path):
    nano, _ = candidate_export(tmp_path / "n", "n")
    small, _ = candidate_export(tmp_path / "s", "s", map50_95=.59)
    assert engine.gflops_select_candidate_files([nano])["status"] == "NOT_RUN"
    result = engine.gflops_select_candidate_files([nano, small])
    assert result["status"] == "SELECTED" and result["selected"]["model_variant"] == "s"
    assert result["selected"]["checkpoint_path"] == str((small.parent / "student.pt").resolve())
    assert set(result["selected"]["selection_source_sha256"]) == {str(nano.resolve()), str(small.resolve())}
    assert engine.gflops_lock_student(result, tmp_path / "student_lock.json")["status"] == "LOCKED"


@pytest.mark.parametrize("mutation", ["fingerprint", "gpu", "smoke", "failed", "duplicate", "checkpoint"])
def test_split_selection_rejects_incomparable_or_incomplete_exports(engine, tmp_path, mutation):
    nano, _ = candidate_export(tmp_path / "n", "n")
    small, payload = candidate_export(tmp_path / "s", "s", map50_95=.59)
    if mutation == "fingerprint":
        payload["dataset_fingerprint"] = "other dataset"
    elif mutation == "gpu":
        payload["comparison_contract"]["gpu"] = "P100"
    elif mutation == "smoke":
        payload["mode"] = "smoke"
    elif mutation == "failed":
        payload["status"] = "FAIL"
    elif mutation == "duplicate":
        payload["model_variants"] = ["n"]
    elif mutation == "checkpoint":
        (small.parent / "student.pt").write_bytes(b"changed")
    small.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError):
        engine.gflops_select_candidate_files([nano, small])


def test_nms_timeout_rejects_partial_validation_and_removes_handler(engine):
    from ultralytics.utils import LOGGER

    original_handlers = list(LOGGER.handlers)
    with pytest.raises(RuntimeError, match="Validasi parsial karena NMS timeout"), engine.gflops_require_complete_nms():
        LOGGER.warning("NMS time limit 2.300s exceeded")
    assert LOGGER.handlers == original_handlers


def test_label_cache_is_bypassed_without_altering_source(engine, tmp_path):
    from ultralytics.data import dataset as sdk_dataset

    path = tmp_path / "labels.cache"
    path.write_bytes(b"stale cache")
    original = sdk_dataset.load_dataset_cache_file
    payload = {}
    with engine.gflops_disable_label_cache():
        with pytest.raises(FileNotFoundError):
            sdk_dataset.load_dataset_cache_file(path)
        sdk_dataset.save_dataset_cache_file("", path, payload, "1")
        assert payload == {"version": "1"}
    assert sdk_dataset.load_dataset_cache_file is original
    assert path.read_bytes() == b"stale cache"


@pytest.mark.parametrize("variant", ["n", "s"])
@pytest.mark.parametrize("platform", ["kaggle", "filkom"])
def test_notebook_is_standalone_and_matches_generator_without_touching_old_notebooks(variant, platform):
    path = ROOT / "notebooks" / platform / "pruning" / NOTEBOOKS[variant].name
    old = {path: digest(path) for path in PRUNING.glob("0[1234]_*.ipynb")}
    assert len(old) == 4
    authoring = runpy.run_path(str(GENERATOR))
    assert authoring["build_notebook"](variant, platform) == json.loads(path.read_text(encoding="utf-8"))
    assert old == {path: digest(path) for path in old}
    notebook = nbformat.read(path, as_version=4)
    nbformat.validate(notebook)
    config_cell = next(cell for cell in notebook.cells if "config" in cell.metadata.get("tags", []))
    config = {}
    exec(config_cell.source, config)
    assert config["MODE"] == "full_val" and config["MODEL_VARIANTS"] == [variant]
    assert config["NOTEBOOK_VARIANT"] == variant
    expected_output = f"lamp_gflops_yolo26{variant}" if platform == "kaggle" else f"outputs/filkom/pruning/yolo26{variant}"
    assert expected_output in Path(config["OUTPUT_DIR"]).as_posix()
    expected_targets = (1.02, 1.05, 1.10, 1.20, 1.30, 1.40, 1.50, 1.60, 1.70)
    assert (config["IMGSZ"], config["BATCH"], config["GFLOPS_TARGETS"]) == (640, 16, expected_targets)
    assert config["EDA_CONFIGURATIONS"] == ()
    exec((ROOT / "scripts/lamp_gflops_runtime_source.py").read_text(encoding="utf-8"), config)
    config["configuration"] = dict
    assert config["gflops_config"]()["notebook_variant"] == variant
    historical_cell = next(cell for cell in notebook.cells if "HISTORICAL_EXTENDED_CSV =" in cell.source)
    assignment = next(node for node in ast.parse(historical_cell.source).body
                      if isinstance(node, ast.Assign) and node.targets[0].id == "HISTORICAL_EXTENDED_CSV")
    assert ast.literal_eval(assignment.value) == (ROOT / "reports/pruning/lamp_gflops_historical_unverified.csv").read_text(encoding="utf-8")
    for cell in notebook.cells:
        if cell.cell_type != "code":
            continue
        assert cell.outputs == [] and cell.execution_count is None
        for node in ast.walk(ast.parse(cell.source)):
            if isinstance(node, ast.Import):
                assert all(alias.name.split(".")[0] not in {"scripts", "src", "skripsi"} for alias in node.names)
            elif isinstance(node, ast.ImportFrom):
                assert (node.module or "").split(".")[0] not in {"scripts", "src", "skripsi"}
            elif isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
                assert node.func.attr not in {"train", "train_v2", "fit", "backward"}
