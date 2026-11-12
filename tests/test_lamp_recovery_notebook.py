"""Offline authoring contracts for notebook04; never launch recovery or network."""
from __future__ import annotations

import ast
import hashlib
import json
import re
import runpy
from pathlib import Path

import nbformat

ROOT = Path(__file__).resolve().parents[1]
GENERATOR = ROOT / "scripts/generate_lamp_recovery_notebook.py"
NOTEBOOK = ROOT / "notebooks/kaggle/pruning/04_lamp_yolo26ns_finetune_recovery.ipynb"


def authoring():
    return runpy.run_path(str(GENERATOR))


def source(cell):
    return "".join(cell["source"])


def configuration(book):
    config = next(source(c) for c in book["cells"]
                  if "config" in c.get("metadata", {}).get("tags", []))
    namespace = {}
    exec(compile(config, "trusted-notebook04-config", "exec"), namespace)  # noqa: S102
    return namespace


def test_generator_matches_saved_notebook_and_preserves_old_notebooks():
    previous = list((ROOT / "notebooks/kaggle/pruning").glob("0[123]_*.ipynb"))
    before = {p: hashlib.sha256(p.read_bytes()).hexdigest() for p in previous}
    built = authoring()["build_notebook"]()
    assert built == json.loads(NOTEBOOK.read_text(encoding="utf-8"))
    assert before == {p: hashlib.sha256(p.read_bytes()).hexdigest() for p in previous}


def test_fourteen_sections_code_syntax_and_standalone_runtime():
    book = nbformat.read(NOTEBOOK, as_version=4)
    nbformat.validate(book)
    headings = [int(m[1]) for c in book.cells if c.cell_type == "markdown"
                and (m := re.match(r"^#{1,2} (\d{2}) —", c.source))]
    assert headings == list(range(1, 15))
    assert len({cell.id for cell in book.cells}) == len(book.cells)
    for cell in book.cells:
        if cell.cell_type != "code":
            continue
        assert cell.outputs == [] and cell.execution_count is None
        tree = ast.parse(cell.source)
        compile(tree, str(NOTEBOOK), "exec")
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                assert all(alias.name.split(".")[0] not in {"scripts", "src", "skripsi"}
                           for alias in node.names)
            elif isinstance(node, ast.ImportFrom):
                assert (node.module or "").split(".")[0] not in {"scripts", "src", "skripsi"}


def test_configuration_retains_both_models_five_ratios_and_explicit_finetune_budget():
    env = configuration(authoring()["build_notebook"]())
    assert env["MODEL_VARIANTS"] == ["n", "s"]
    assert env["PRUNING_POLICY"] == "extended"
    assert env["PRUNING_RATIOS"] == [.1, .2, .3, .4, .5]
    assert len([(m, ratio) for m in env["MODEL_VARIANTS"] for ratio in env["PRUNING_RATIOS"]]) == 10
    assert env["MODEL_ARTIFACTS"]["n"].endswith("03_baseline_yolo26n-9uu3meve-checkpoints:v84")
    assert env["MODEL_ARTIFACTS"]["s"].endswith("02_baseline_yolo26s-q5kp9k9d-checkpoints:v62")
    assert env["RECOVERY_ENABLED"] is True and env["FINETUNING_ENABLED"] is True
    assert env["KD_ENABLED"] is False
    assert env["RECOVERY_EPOCHS"] == 5
    assert env["RECOVERY_OPTIMIZER"] == "AdamW" and env["RECOVERY_LR0"] == .0001
    assert env["RECOVERY_PATIENCE"] == 0 and env["RECOVERY_AMP"] is False
    assert env["RECOVERY_BATCH"] == 16 and env["RECOVERY_WORKERS"] == 2
    assert env["RECOVERY_WARMUP_EPOCHS"] == 3 and env["RECOVERY_CLOSE_MOSAIC"] == 1
    assert env["RECOVERY_WEIGHT_DECAY"] == .0005
    assert env["IMGSZ"] == 640 and env["SEED"] == 42
    assert env["MAX_PRUNING_RATIO"] == .5 and env["ITERATIVE_STEPS"] == 1
    assert env["EDA_DISPLAY_RATIO"] == .5
    assert all(env[key] is True for key in ("RUN_FULL_VALIDATION", "RUN_LATENCY_BENCHMARK",
                                           "WANDB_TRACKING", "UPLOAD_WANDB_ARTIFACTS"))
    assert str(env["OUTPUT_DIR"]).replace("\\", "/").endswith("pruning/lamp_v2_finetune")


def test_recovery_declarations_load_before_their_first_run():
    texts = [source(cell) for cell in authoring()["build_notebook"]()["cells"]
             if cell["cell_type"] == "code"]
    original_validation = next(i for i, text in enumerate(texts) if "def validate_configuration(" in text)
    recovery_config = next(i for i, text in enumerate(texts)
                           if "RECOVERY_EPOCHS" in text and "def " in text
                           and i > original_validation)
    validation_run = next(i for i, text in enumerate(texts) if text.startswith("validate_configuration()"))
    training = next(i for i, text in enumerate(texts) if "def fine_tune_pruned_checkpoint(" in text)
    dispatch = next(i for i, text in enumerate(texts) if text.startswith("BUNDLE = execute_models("))
    assert original_validation < recovery_config < validation_run
    assert training < dispatch
    assert any("def _run_configuration(" in text for text in texts[training + 1:dispatch])


def test_pruning_engine_is_embedded_unchanged_and_no_recovery_claims_are_scoped():
    book = authoring()["build_notebook"]()
    policy = (ROOT / "scripts/lamp_v2_policy_source.py").read_text(encoding="utf-8").strip()
    assert any(source(c) == policy for c in book["cells"])
    markdown = "\n".join(source(c) for c in book["cells"] if c["cell_type"] == "markdown")
    assert "bukan replikasi penuh" in markdown
    assert "optimizer baru" in markdown and "resume=False" in markdown
    assert "recovery belum dijalankan" not in markdown
    assert "Tidak ada training," not in markdown
    assert "https://docs.ultralytics.com/usage/engine/" in markdown
    assert "https://docs.ultralytics.com/modes/train/" in markdown
    assert "https://doi.org/10.1016/j.jvcir.2025.104560" in markdown


def test_results_exports_include_stage_and_recovered_class_metrics():
    texts = [source(cell) for cell in authoring()["build_notebook"]()["cells"]
             if cell["cell_type"] == "code"]
    summary = next(text for text in texts if "SUMMARY = pd.DataFrame(" in text)
    assert '"stage"' in summary
    per_class = next(text for text in texts if "PER_CLASS = []" in text)
    assert 'for stage in ("pruned", "recovered"):' in per_class
    assert 'result.get(stage, {}).get("validation")' in per_class
    repro = next(text for text in texts if '"reproducibility.json"' in text)
    assert '"recovery_statuses"' in repro and '"No recovery"' not in repro
