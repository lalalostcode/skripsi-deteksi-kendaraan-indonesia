"""Repair raw URIs and resume I/O while preserving the committed training identity."""
# ruff: noqa: S102 -- Execute only trusted configuration and persistence declarations.
import ast
import hashlib
import json
from contextlib import contextmanager
from pathlib import Path

import pytest

from scripts.generate_pruned_fine_tuned_notebooks import build_selected_notebook

ROOT = Path(__file__).resolve().parents[1]


def config(**changes):
    scope = {}
    for cell in build_selected_notebook("filkom")["cells"]:
        tags = cell.get("metadata", {}).get("tags", [])
        if {"config", "selection", "protocol"} & set(tags):
            exec("".join(cell["source"]), scope)
            if "config" in tags:
                scope.update(changes)
    return scope


def binding_functions(scope):
    sources = [("pruned_finetune_selected_source.py", {"ft_select", "ft_identity"}),
               ("finetune_artifact_binding_source.py", {"ft_resolved_raw_ref", "ft_binding_select_resume"})]
    for filename, names in sources:
        tree = ast.parse((ROOT / "scripts" / filename).read_text(encoding="utf-8"))
        tree.body = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name in names]
        exec(compile(tree, filename, "exec"), scope)


@pytest.mark.parametrize("seed", [42, 0, 2024, 7, 3407])
def test_legacy_local_resume_retains_v0_identity_and_resolves_identical_v1(seed, tmp_path):
    old = f"yolo26s-lamp-su1.20x-seed{seed}-raw:v0"
    live = old.replace(":v0", ":v1")
    pin = config()["STUDENT_PINS"][old]
    ref = "ilhamrafiqin-brawijaya-university/yolom26-kd-indotraffic/" + old
    (tmp_path / "resume_contract.json").write_text(json.dumps({"contract": {"raw_artifact_ref": ref, "raw_sha256": pin["raw_sha256"]}}), encoding="utf-8")
    scope = config(RESUME_RUN_DIR=tmp_path, STUDENT_INPUTS=[live])
    binding_functions(scope)
    assert scope["STUDENT_INPUTS"] == [old] and scope["SEED"] == seed
    assert scope["ft_identity"]()["raw_artifact_ref"] == ref
    assert scope["ft_identity"]()["raw_sha256"] == pin["raw_sha256"]
    assert scope["ft_resolved_raw_ref"]() == ref.replace(":v0", ":v1")


def test_remote_snapshot_alias_rejects_another_seed_or_hash():
    scope = config(STUDENT_INPUTS=["yolo26s-lamp-su1.20x-seed2024-raw:v1"])
    binding_functions(scope)
    pin = scope["STUDENT_PINS"]["yolo26s-lamp-su1.20x-seed2024-raw:v0"]
    scope["ft_binding_select_resume"]("entity/project/yolo26s-lamp-su1.20x-seed2024-raw:v0", pin["raw_sha256"])
    assert scope["SEED"] == 2024 and scope["STUDENT_INPUTS"][0].endswith("seed2024-raw:v0")
    with pytest.raises(ValueError, match="Input snapshot"):
        scope["ft_binding_select_resume"]("entity/project/yolo26s-lamp-su1.20x-seed7-raw:v0", pin["raw_sha256"])
    with pytest.raises(ValueError, match="Input snapshot"):
        scope["ft_binding_select_resume"]("entity/project/yolo26s-lamp-su1.20x-seed2024-raw:v0", "0" * 64)


def test_original_ft_training_implementation_hash_preserved():
    book = build_selected_notebook("filkom")
    runtime = next("".join(c["source"]) for c in book["cells"] if "runtime" in c.get("metadata", {}).get("tags", []))
    hashes = [n.value.value for n in ast.parse(runtime).body if isinstance(n, ast.Assign)
              and any(isinstance(t, ast.Name) and t.id == "IMPLEMENTATION_SHA256" for t in n.targets)]
    assert hashes == ["027164184f1c39433043b73ba3e41d91cb5e62f63e508050c339cad8513b30c3"]


def postprocess_scope():
    def read(path, default=None):
        return json.loads(path.read_text(encoding="utf-8")) if path.is_file() else default

    def write(path, value):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(value), encoding="utf-8")

    scope = {"Path": Path, "contextmanager": contextmanager, "ft_read": read, "write_json": write,
             "sha256_file": lambda p: hashlib.sha256(p.read_bytes()).hexdigest(), "POSTPROCESSING_SHA256": "io-hash",
             "ft_eda": lambda *args: pytest.fail("Committed EDA must not be rendered again"),
             "ft_publish": lambda *_: None, "ft_tracking": lambda *_: None,
             "_recovery_eda_metadata": lambda _: {}, "ft_identity": lambda: {"seed": 42}}
    exec((ROOT / "scripts/recovery_postprocess_source.py").read_text(encoding="utf-8"), scope)
    return scope


def test_pending_publication_reuses_verified_eda_bytes(tmp_path):
    scope = postprocess_scope()
    rows = [{"stage": "baseline", "map50_95": .6}]
    scope["fine_tuned_comparison"] = lambda _: (rows, [])
    for name, value in (("eda/eda_status.json", {"status": "PASS"}), ("eda/stage_metrics.json", rows)):
        scope["write_json"](tmp_path / name, value)
    (tmp_path / "eda/chart.png").write_bytes(b"original-render")
    files = {p.relative_to(tmp_path).as_posix(): scope["sha256_file"](p) for p in (tmp_path / "eda").iterdir()}
    scope["write_json"](tmp_path / "artifact_publications.json", {"results/final": {"status": "PENDING", "files_sha256": files}})
    assert scope["ft_eda"]({}, tmp_path, [], {})["status"] == "PASS"
    (tmp_path / "eda/chart.png").write_bytes(b"unexpected-change")
    with pytest.raises(ValueError, match="berubah"):
        scope["ft_eda"]({}, tmp_path, [], {})


def test_journaled_evaluation_manifest_is_never_overwritten(tmp_path):
    scope = postprocess_scope()
    path = tmp_path / "evaluation_manifest.json"
    scope["write_json"](path, {"original": True})
    digest = scope["sha256_file"](path)
    scope["write_json"](tmp_path / "artifact_publications.json", {"results/final": {"files_sha256": {path.name: digest}}})
    scope["write_json"](path, {"new_status": "PASS"})
    assert scope["sha256_file"](path) == digest
    path.write_text("{}", encoding="utf-8")
    with pytest.raises(ValueError, match="berubah"):
        scope["write_json"](path, {"original": True})
