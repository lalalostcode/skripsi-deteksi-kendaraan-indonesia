"""Check baseline protocol parity and reject invalid raw inputs before deserialization."""
# ruff: noqa: S102 -- Execute trusted notebook cells and declarations from this repository.
import ast
import hashlib
import json
from pathlib import Path

import pytest

from scripts.generate_pruned_fine_tuned_notebooks import (
    BASELINE,
    METHOD_NOTEBOOKS,
    build_notebook,
)
from scripts.notebook_paths import PRUNED_FINE_TUNED, ROOT, notebook_path


def execute_cells(book, tags=None, **overrides):
    scope = {}
    for cell in book["cells"]:
        if cell["cell_type"] == "code" and (tags is None or set(tags) & set(cell["metadata"]["tags"])):
            exec("".join(cell["source"]), scope)
            if "config" in cell["metadata"]["tags"]:
                scope.update(overrides)
    return scope


@pytest.mark.parametrize("platform", ["kaggle", "filkom"])
@pytest.mark.parametrize("method", METHOD_NOTEBOOKS)
def test_generator_and_empty_input(platform, method):
    book = json.loads(notebook_path(platform, METHOD_NOTEBOOKS[method][0]).read_text(encoding="utf-8"))
    assert book == build_notebook(platform, method)
    scope = execute_cells(book, {"config", "selection", "protocol"})
    assert len(scope["STUDENT_INPUTS"]) == 1 and scope["SEED"] == 42
    assert scope["RESUME_RUN_DIR"] is None and scope["RESUME_SNAPSHOT_REF"] is None
    assert len(scope.get("ACTIVE_STUDENT_INPUTS", scope["STUDENT_PINS"])) == (7 if method == "finetune" else 5)


def test_historical_fine_tuned_snapshots():
    for platform, digest in (("kaggle", "a0e1f290122bb1664f03317f8b9784821029bfafcdec26bfd531638cb5df0911"),
                             ("filkom", "c0c8ff73dd445db123c632a5f75808e26c5dd4a67b9fe7e3277e75fe49ab7014")):
        assert hashlib.sha256(notebook_path(platform, PRUNED_FINE_TUNED).read_bytes()).hexdigest() == digest


@pytest.mark.parametrize("method", METHOD_NOTEBOOKS)
def test_explicit_and_effective_protocol_match_sources(method):
    from ultralytics.cfg import get_cfg

    scope = execute_cells(build_notebook("filkom", method), {"config", "selection", "protocol"})
    baseline = json.loads(BASELINE.read_text(encoding="utf-8"))
    config = {}
    exec("".join(baseline["cells"][2]["source"]), config)
    args, validation = {}, {}
    for index, variable, result in ((13, "train_args", args), (15, "metrics", validation)):
        nodes = ast.parse("".join(baseline["cells"][index]["source"])).body
        expression = next(node.value for node in nodes if isinstance(node, ast.Assign)
                          and isinstance(node.targets[0], ast.Name) and node.targets[0].id == variable)
        result.update({kw.arg: eval(ast.unparse(kw.value), {**config, "RUN_NAME": "training"}) for kw in expression.keywords})
    args.update(data="test.yaml", project="results")
    validation.update(data="test.yaml")
    extras = {"finetune": {}, "native": {"dis": 1.0}, "crosskd": {"dis": 1.0, "compile": False},
              "mgd": {"dis": 1.0, "compile": False, "resume": False}}[method]
    teacher = None if method == "finetune" else "teacher.pt"
    if teacher:
        args.update(distill_model=teacher, **extras)
        assert scope["TEACHER_ARTIFACT_REF"].endswith(":v50")
    current = scope["fine_tuned_train_args"]("test.yaml", "results", "training", teacher)
    assert vars(get_cfg(overrides=current)) == vars(get_cfg(overrides=args))
    assert current == args and scope["fine_tuned_val_args"]("test.yaml") == validation
    scope["MODE"] = "smoke"
    assert scope["fine_tuned_train_args"]("test.yaml", "results", "training", teacher) == {
        **args, "epochs": 1, "amp": False, "nbs": args["batch"],
    }
    if method == "crosskd":
        assert [scope[key] for key in ("KD_CLS_WEIGHT", "KD_BOX_WEIGHT", "KD_CLS_GAMMA")] == [1.0, 1.0, 1.0]
    if method == "mgd":
        assert (scope["MGD_ALPHA"], scope["MGD_MASK_RATIO"]) == (2e-5, 0.65)


@pytest.mark.parametrize(("variant", "bad_hash", "error"), [("n", False, "YOLO26s"), ("s", True, "SHA-256")])
def test_wrong_raw_rejected_before_training(tmp_path, variant, bad_hash, error):
    path = tmp_path / "raw_pruned_manifest.json"
    file = tmp_path / "model.pt"
    file.write_bytes(b"Not a PyTorch checkpoint.")
    digest = hashlib.sha256(file.read_bytes()).hexdigest()
    manifest = {"model_variant": variant, "format_version": 2, "stage": "raw_pruned", "target_speedup": 1.02,
                "target_status": "REACHED", "configuration_id": "target-1.02x", "structural_status": "PASS",
                "oracle_status": "PASS", "reload_status": "PASS", "source_baseline": {"sha256": "v62", "requested_artifact": "ref"},
                "raw_checkpoint": {"sha256": "0" * 64 if bad_hash else digest, "relative_path": file.name},
                "baseline_checkpoint": {"artifact_path": file.name, "sha256": digest}}
    path.write_text(json.dumps(manifest), encoding="utf-8")
    source = ROOT / "scripts/pruned_fine_tuned_source.py"
    loader = next(node for node in ast.parse(source.read_text(encoding="utf-8")).body
                  if isinstance(node, ast.FunctionDef) and node.name == "fine_tuned_source")
    scope = {"Path": Path, "RAW_ARTIFACT_REF": None, "RAW_MANIFEST_PATH": path, "EXPECTED_RAW_SHA256": None,
             "BASELINE_SHA256": "v62", "BASELINE_REF": "ref", "recovery_read_json": lambda p: json.loads(p.read_text()),
             "sha256_file": lambda p: hashlib.sha256(p.read_bytes()).hexdigest()}
    exec(compile(ast.Module(body=[loader], type_ignores=[]), str(source), "exec"), scope)
    with pytest.raises(ValueError, match=error):
        scope["fine_tuned_source"](tmp_path)


@pytest.mark.parametrize("wrong_ref", [True, False])
def test_wrong_teacher_rejected_before_loading(tmp_path, wrong_ref):
    path = tmp_path / "teacher.pt"
    path.write_bytes(b"Not a PyTorch checkpoint.")
    source = ROOT / "scripts/source_matched_kd_source.py"
    resolver = next(node for node in ast.parse(source.read_text(encoding="utf-8")).body
                    if isinstance(node, ast.FunctionDef) and node.name == "source_matched_teacher")
    scope = {"Path": Path, "TEACHER_ARTIFACT_REF": "entity/project/teacher:v51" if wrong_ref else None,
             "TEACHER_LOCAL_PATH": None if wrong_ref else path, "LOCKED_TEACHER_REF": "entity/project/teacher:v50",
             "MODE": "full_val", "LOCKED_TEACHER_SHA256": "a" * 64, "TEACHER_SHA256": "a" * 64,
             "sha256_file": lambda p: hashlib.sha256(p.read_bytes()).hexdigest(), "workflow_version_ref": str}
    exec(compile(ast.Module(body=[resolver], type_ignores=[]), str(source), "exec"), scope)
    with pytest.raises(ValueError, match="v50" if wrong_ref else "SHA-256"):
        scope["source_matched_teacher"](tmp_path, None)
