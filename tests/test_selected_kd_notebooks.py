"""Input identity, teacher locking, KD protocol, and partial cross-method summaries."""
# ruff: noqa: S102 -- Execute trusted declarations and generated configuration cells.
import ast
import hashlib
import json
from types import SimpleNamespace

import pytest

from scripts.generate_pruned_fine_tuned_notebooks import (
    METHOD_NOTEBOOKS,
    TEACHER_REF,
    TEACHER_SHA,
    notebook_path,
)
from scripts.generate_selected_kd_notebooks import METHODS, build_notebook


def configuration(method="native", **changes):
    book = build_notebook("filkom", method)
    scope = {}
    for cell in book["cells"]:
        tags = cell.get("metadata", {}).get("tags", [])
        if set(tags) & {"config", "selection", "protocol"}:
            exec("".join(cell["source"]), scope)
            if "config" in tags:
                scope.update(changes)
    return scope


@pytest.mark.parametrize("platform", ["kaggle", "filkom"])
@pytest.mark.parametrize("method", METHODS)
def test_generated_notebooks_and_five_locked_inputs(platform, method):
    import nbformat
    book = build_notebook(platform, method)
    assert book == json.loads(notebook_path(platform, METHOD_NOTEBOOKS[method][0]).read_text(encoding="utf-8"))
    nbformat.validate(nbformat.from_dict(book))
    for cell in book["cells"]:
        if cell["cell_type"] == "code":
            compile("".join(cell["source"]), method, "exec")
    scope = configuration(method)
    assert {v["seed"] for v in scope["STUDENT_PINS"].values()} == {42, 0, 2024, 7, 3407}
    assert all(v["target"] == 1.2 for v in scope["STUDENT_PINS"].values())
    assert scope["TEACHER_ARTIFACT_REF"] == TEACHER_REF and scope["TEACHER_SHA256"] == TEACHER_SHA
    assert scope["EPOCHS"] == 100 and scope["AMP"] is True
    assert len([c for c in book["cells"] if c["cell_type"] == "markdown"]) == 6


@pytest.mark.parametrize("active", [[], ["a", "b"], ["yolo26s-lamp-su1.10x-seed42-raw:v0"]])
def test_invalid_selection_rejected_before_model_loading(active):
    with pytest.raises(ValueError, match="tepat satu"):
        configuration(STUDENT_INPUTS=active)


@pytest.mark.parametrize("method", METHODS)
@pytest.mark.parametrize("seed", [42, 0, 2024, 7, 3407])
def test_seed_and_teacher_identity_names(method, seed):
    name = next(key for key, pin in configuration(method)["STUDENT_PINS"].items() if pin["seed"] == seed)
    scope = configuration(method, STUDENT_INPUTS=[name])
    runtime = next("".join(c["source"]) for c in build_notebook("filkom", method)["cells"]
                   if "runtime" in c.get("metadata", {}).get("tags", []))
    names = {"ft_select", "ft_identity", "ft_name", "ft_group", "recovery_kd_settings"}
    nodes = [n for n in ast.parse(runtime).body if isinstance(n, ast.FunctionDef) and n.name in names]
    original, override = [n for n in nodes if n.name == "ft_identity"]
    original.name = "_selected_identity"
    scope.update(math=__import__("math"), workflow_digest=lambda value: hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest())
    exec(compile(ast.Module(body=[n for n in nodes if n is not override], type_ignores=[]), "identity", "exec"), scope)
    exec(compile(ast.Module(body=[override], type_ignores=[]), "kd-identity", "exec"), scope)
    scope["KD_DEFAULT_SETTINGS"] = {m: {} for m in METHODS}
    identity = scope["ft_identity"]()
    assert identity["seed"] == scope["SEED"] == seed and identity["teacher_artifact_ref"] == TEACHER_REF
    assert scope["ft_name"]() == f"yolo26s-lamp-su1.20x-seed{seed}-{method}"
    assert scope["ft_group"]() == "yolo26s-lamp-kd-20261107T020459Z_45575a7f"
    scope["TEACHER_SHA256"] = "0" * 64
    with pytest.raises(ValueError, match="Teacher"):
        scope["ft_identity"]()


def test_empty_group_remains_readable():
    source = notebook_path("filkom", METHOD_NOTEBOOKS["native"][0])
    code = next("".join(c["source"]) for c in json.loads(source.read_text(encoding="utf-8"))["cells"]
                if "group-display" in c.get("metadata", {}).get("tags", []))
    import pandas as pd
    tables = []
    exec(code, {"LOAD_GROUP_RESULTS": True, "ft_group_results": list, "display": tables.append, "pd": pd})
    assert len(tables) == 1 and tables[0].empty


def test_group_reader_keeps_fifteen_cases_and_separates_methods():
    import pandas as pd

    from scripts.generate_pruned_fine_tuned_notebooks import ROOT
    scope = configuration()
    source = ast.parse((ROOT / "scripts/pruned_kd_selected_source.py").read_text(encoding="utf-8"))
    nodes = [n for n in source.body if isinstance(n, ast.FunctionDef) and n.name == "ft_group_results"
             or isinstance(n, ast.Assign) and any(isinstance(t, ast.Name) and t.id == "KD_DEFAULT_SETTINGS" for t in n.targets)]
    runs = []
    exec(compile(ast.Module(body=nodes, type_ignores=[]), "group-reader", "exec"), scope)
    for method in METHODS:
        for name, pin in scope["STUDENT_PINS"].items():
            summary = {"status": "PASS", "logging_verified": True, "method": method,
                       "raw_artifact_ref": "entity/project/" + name, "raw_sha256": pin["raw_sha256"],
                       "teacher_sha256": TEACHER_SHA, "seed": pin["seed"], "platform": "filkom",
                       "recovered_map50_95": 0.2, "media": {"path": "ignored"}}
            config = {"protocol": "selected-raw-kd-v1", "comparison_signature": "comparable",
                      "method_hyperparameters": scope["KD_DEFAULT_SETTINGS"][method], "environment": {"gpu": None}}
            runs.append(SimpleNamespace(summary=summary, config=config, created_at="2026-11-07", url="https://example.invalid/run"))
    runs.append(SimpleNamespace(summary={**runs[0].summary, "teacher_sha256": "changed"}, config=runs[0].config,
                                created_at="2026-11-08", url="https://example.invalid/rejected"))
    tables = []
    scope.update(pd=pd, workflow_login=lambda: None, ft_group=lambda: "group", workflow_digest=lambda _: "comparable",
                 kd_comparison_settings=dict, recovery_kd_settings=lambda m: scope["KD_DEFAULT_SETTINGS"][m],
                 wandb=SimpleNamespace(Api=lambda **_: SimpleNamespace(runs=lambda *args, **kwargs: runs)), display=tables.append)
    rows = scope["ft_group_results"]()
    assert len(rows) == 15 and all("media" not in row and row["gpu"] == "CPU" for row in rows)
    assert set(tables[0][("recovered_map50_95", "count")]) == {5}


def test_missing_wandb_staging_directories_are_recreated(tmp_path, monkeypatch):
    from wandb.sdk.data_types import _private

    import wandb
    from scripts.generate_pruned_fine_tuned_notebooks import ROOT
    media, artifacts = tmp_path / "media", tmp_path / "artifacts"
    monkeypatch.setattr(_private, "MEDIA_TMP", SimpleNamespace(name=str(media)))
    monkeypatch.setattr(wandb.Artifact, "_TMP_DIR", SimpleNamespace(name=str(artifacts)))
    source = ast.parse((ROOT / "scripts/pruned_kd_selected_source.py").read_text(encoding="utf-8"))
    node = next(n for n in source.body if isinstance(n, ast.FunctionDef) and n.name == "kd_media_staging")
    scope = {"Path": type(tmp_path), "wandb": wandb}
    exec(compile(ast.Module(body=[node], type_ignores=[]), "media-staging", "exec"), scope)
    scope["kd_media_staging"]()
    assert media.is_dir() and artifacts.is_dir()
    media.rmdir(); artifacts.rmdir()
    scope["kd_media_staging"]()
    assert media.is_dir() and artifacts.is_dir()


@pytest.mark.parametrize("change", ["hash", "namespace"])
def test_restoration_receipt_cannot_replace_input_content_or_namespace(tmp_path, monkeypatch, change):
    from scripts import generate_selected_kd_notebooks as generator
    receipt = generator.base.ROOT / "reports/pruning/lamp_yolo26s_raw_restored_20261107.json"
    record = json.loads(receipt.read_text(encoding="utf-8"))
    if change == "hash":
        record["rows"][0]["raw_sha256"] = "0" * 64
    else:
        record["rows"][0]["artifact_ref"] = "other/project/" + record["rows"][0]["artifact_ref"].rsplit("/", 1)[-1]
    target = tmp_path / "reports/pruning/lamp_yolo26s_raw_restored_20261107.json"
    target.parent.mkdir(parents=True)
    target.write_text(json.dumps(record), encoding="utf-8")
    monkeypatch.setattr(generator.base, "ROOT", tmp_path)
    with pytest.raises(ValueError, match="berbeda"):
        generator.selected_pins()
