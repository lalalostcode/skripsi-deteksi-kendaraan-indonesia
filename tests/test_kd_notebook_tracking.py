"""Offline validation of experiment labels and the new CSAKD s-to-n configuration."""
# ruff: noqa: S102 -- Execute only local config and an extracted, mocked init call.
from __future__ import annotations

import ast
import json
import uuid
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import nbformat
import pytest

from scripts.notebook_paths import notebook_path, notebook_paths

FOLDER = Path(__file__).resolve().parents[1] / "notebooks/kaggle"
# Legacy experiments 01–12 only: later notebook variants use different configuration schemas.
# PKD and AMD/Roboflow have dedicated protocol/source tests; this suite expects Kaggle DATA.
EXPERIMENTS = sorted(path for path in notebook_paths("kaggle")
                     if path.parent.name != "pruning" and path.name[:2].isdigit() and 1 <= int(path.name[:2]) <= 12
                     and path.name not in {"12_pkd_yolo26n.ipynb", "03_baseline_yolo26n_amd_roboflow.ipynb"})
NEW_NOTEBOOK = notebook_path("kaggle", "12_csakd_yolo26n_teacher_yolo26s.ipynb")


def configuration(notebook, overrides=None):
    source = next("".join(cell["source"]) for cell in notebook["cells"]
                  if cell["cell_type"] == "code" and "EXPERIMENT_ID =" in "".join(cell["source"]))
    tree = ast.parse(source)
    for node in tree.body:
        if isinstance(node, ast.Assign) and isinstance(node.targets[0], ast.Name):
            key = node.targets[0].id
            if key in (overrides or {}):
                node.value = ast.Constant(overrides[key])
    env = {}
    exec(compile(ast.fix_missing_locations(tree), "<tracking config>", "exec"), env)
    return env


@pytest.mark.parametrize("path", EXPERIMENTS, ids=lambda path: path.stem)
def test_json_python_and_group_tags_reach_wandb_init(path):
    notebook = json.loads(path.read_bytes())
    nbformat.validate(nbformat.from_dict(notebook))
    nodes = []
    for cell in notebook["cells"]:
        if cell["cell_type"] == "code":
            source = "".join(cell["source"])
            compile(source, str(path), "exec")
            nodes.extend(ast.walk(ast.parse(source)))
    env = configuration(notebook)
    number = int(path.name[:2])
    size = env["MODEL_WEIGHTS"][6]
    expected_group = "supervised" if number <= 3 else f"T26{'s' if number >= 10 else 'm'}_S26{size}"
    assert env["WANDB_GROUP"] == expected_group
    assert env["DATA"] == "/kaggle/input/datasets/ilhamrafiqin/poltrada-v3-yolo26-clean/data.yaml"
    assert env["WANDB_PROJECT"] == "yolom26-kd-indotraffic"
    assert env["WANDB_ENTITY"] == "ilhamrafiqin-brawijaya-university"
    init = next(node for node in nodes if isinstance(node, ast.Call) and ast.unparse(node.func) == "wandb.init")
    if number == 1:
        expected_name = f"base_teacher_yolo26{size}_seed{env['SEED']}"
    elif number <= 3:
        variant = "_scratch" if "scratch" in path.stem else ""
        expected_name = f"base_yolo26{size}{variant}_seed{env['SEED']}"
    else:
        weight = f"_dis{env['KD_WEIGHT']:g}" if env["METHOD"] == "native" else ""
        expected_name = f"kd_{env['METHOD']}_{expected_group}{weight}_seed{env['SEED']}"
    mock = Mock()
    env.update(wandb=SimpleNamespace(init=mock, Settings=Mock(),
                                    util=SimpleNamespace(generate_id=lambda: uuid.uuid4().hex)), RUN_NAME="test-run",
               KD_CONFIG={}, TEACHER_PROVENANCE={}, ENVIRONMENT_INFO={}, RUNTIME_VERSIONS={}, diagnostic=False)

    serial_setup = [n for n in nodes if
                    (isinstance(n, ast.FunctionDef) and n.name == "seed_run_name") or
                    (isinstance(n, ast.Assign) and (
                        ast.unparse(n.targets[0]) == "NEW_WANDB_RUN_ID" or
                        (ast.unparse(n.targets[0]) == "WANDB_RUN_NAME" and "seed_run_name" in ast.unparse(n.value))))]
    serial_setup.sort(key=lambda n: 0 if isinstance(n, ast.FunctionDef) else 1)

    def invoke():
        env["seed"] = env["SEED"]
        exec(compile(ast.Module(serial_setup, type_ignores=[]), "<W&B serial setup>", "exec"), env)
        eval(compile(ast.Expression(init), "<mocked wandb init>", "eval"), env)

    invoke()
    kwargs = mock.call_args.kwargs
    invoke()
    assert mock.call_args.kwargs["name"] == kwargs["name"] == expected_name
    assert mock.call_args.kwargs["id"] != kwargs["id"]
    assert kwargs["resume"] == "never"
    assert kwargs["project"] == env["WANDB_PROJECT"] and kwargs["entity"] == env["WANDB_ENTITY"]
    assert kwargs["group"] == expected_group
    tags = kwargs["tags"]
    assert {f"seed{env['SEED']}", "poltrada-v3", "research", env["EXPERIMENT_ID"]} <= set(tags)
    assert len(tags) == len(set(tags))
    if number <= 3:
        assert ("teacher" if number == 1 else "baseline") in tags
        assert "model-" + size in tags
        assert not any(tag.startswith("student-") for tag in tags)
    else:
        assert {env["METHOD"], "teacher-" + ("s" if number >= 10 else "m"), "student-" + size} <= set(tags)
        if env["METHOD"] == "native":
            assert "dis1" in tags
        elif env["METHOD"] == "crosskd":
            assert {"cls1", "box1"} <= set(tags)
        else:
            assert {"agka1", "cross1"} <= set(tags)
    if "DIAGNOSTIC_ONLY" in env:
        env["DIAGNOSTIC_ONLY"] = True
    env["DATA"] = "coco8.yaml"
    env["diagnostic"] = True
    invoke()
    assert mock.call_args.kwargs["name"] == expected_name + "_smoke"
    assert mock.call_args.kwargs["group"] == "diagnostic-" + expected_group
    assert "diagnostic" in mock.call_args.kwargs["tags"]
    assert "research" not in mock.call_args.kwargs["tags"]


@pytest.mark.parametrize("filename,overrides,expected", [
    ("04_native_yolo26s.ipynb", {"SEED": 7, "KD_WEIGHT": 0.25}, {"seed7", "dis0.25"}),
    ("11_crosskd_yolo26n_teacher_yolo26s.ipynb", {"SEED": 7, "KD_CLS_WEIGHT": 0.5}, {"seed7", "cls0.5"}),
    (NEW_NOTEBOOK.name, {"SEED": 7, "KD_AGKA_WEIGHT": 0.5}, {"seed7", "agka0.5"}),
])
def test_tracking_tags_follow_configured_seed_and_weights(filename, overrides, expected):
    notebook = json.loads((notebook_path("kaggle", filename)).read_bytes())
    env = configuration(notebook, overrides)
    assert expected <= set(env["WANDB_TAGS"])
    assert "seed42" not in env["WANDB_TAGS"]


@pytest.mark.parametrize("filename", [
    "12_pkd_yolo26n.ipynb",
    "13_head_yolo26n_teacher_yolo26s.ipynb",
    "14_rm_yolo26n_teacher_yolo26s.ipynb",
    "15_mgd_yolo26n_teacher_yolo26s.ipynb",
])
def test_teacher_s_special_methods_use_distinct_diagnostic_labels(filename):
    notebook = json.loads(notebook_path("kaggle", filename).read_bytes())
    env = configuration(notebook)
    calls = Mock()
    env.update(
        wandb=SimpleNamespace(init=calls, Settings=Mock(),
                              util=SimpleNamespace(generate_id=lambda: uuid.uuid4().hex)),
        uuid=uuid, seed=env["SEED"], config={}, DATASET_INFO={}, TEACHER_INFO={},
        SESSION_DIR=FOLDER, TEACHER_PROVENANCE={}, ENVIRONMENT_INFO={}, NUMERICAL_RESULT={},
    )
    tree = ast.parse("\n".join("".join(cell["source"]) for cell in notebook["cells"] if cell["cell_type"] == "code"))
    init = next(node for node in ast.walk(tree)
                if isinstance(node, ast.Call) and ast.unparse(node.func) == "wandb.init")
    function = next(node for node in ast.walk(tree)
                    if isinstance(node, ast.FunctionDef) and init in ast.walk(node))
    assignments = [node for node in ast.walk(function) if isinstance(node, ast.Assign)
                   and node.lineno < init.lineno and ast.unparse(node.targets[0]) in {"name", "display_name"}]
    assignments.sort(key=lambda node: node.lineno)
    for diagnostic in (False, True):
        mode = "diagnostic" if diagnostic else "training"
        env.update(diagnostic=diagnostic, mode=mode, token="local-path-token")
        exec(compile(ast.Module(assignments, type_ignores=[]), "<W&B display names>", "exec"), env)
        eval(compile(ast.Expression(init), "<mocked wandb init>", "eval"), env)
        kwargs = calls.call_args.kwargs
        suffix = "_smoke" if diagnostic else ""
        assert kwargs["name"] == f"kd_{env['METHOD']}_T26s_S26n_seed{env['SEED']}{suffix}"
        assert kwargs["group"] == ("diagnostic-T26s_S26n" if diagnostic else "T26s_S26n")
        assert kwargs["project"] == "yolom26-kd-indotraffic"
        assert kwargs["entity"] == "ilhamrafiqin-brawijaya-university"
        assert kwargs["resume"] == "never"
        assert {env["METHOD"], "teacher-s", "student-n", "seed42"} <= set(kwargs["tags"])
    assert calls.call_args_list[0].kwargs["id"] != calls.call_args_list[1].kwargs["id"]


def test_csakd_s_to_n_matches_existing_csakd_protocol_and_uses_baseline_v62():
    new = json.loads(NEW_NOTEBOOK.read_bytes())
    original = json.loads((notebook_path("kaggle", "09_csakd_yolo26n.ipynb")).read_bytes())
    current, reference = configuration(new), configuration(original)
    keys = ["DATA", "MODEL_WEIGHTS", "EPOCHS", "IMGSZ", "BATCH", "SEED", "DEVICE", "WORKERS",
            "OPTIMIZER", "LR", "PATIENCE", "AMP", "WEIGHT_DECAY", "WARMUP_EPOCHS", "CLOSE_MOSAIC",
            "EVAL_SPLIT", "KD_AGKA_WEIGHT", "KD_CROSS_WEIGHT", "KD_CHECKPOINT", "KD_CHUNK_SIZE", "EXPECTED_NC"]
    assert {key: current[key] for key in keys} == {key: reference[key] for key in keys}
    assert current["TEACHER_ARTIFACT"] == (
        "ilhamrafiqin-brawijaya-university/yolom26-kd-indotraffic/"
        "02_baseline_yolo26s-q5kp9k9d-checkpoints:v62"
    )
    assert current["TEACHER_RUN_ID"] == "q5kp9k9d"
    assert current["DIAGNOSTIC_ONLY"] is False
    assert current["WANDB_RUN_NAME"] == "kd_csakd_T26s_S26n_seed42"
    for index in (11, 12, 16, 18):
        assert new["cells"][index]["source"] == original["cells"][index]["source"]
    for notebook, env in ((new, current), (original, reference)):
        env.update(teacher_path=Path("/mock/teacher/best.pt"), RUN_NAME="same-comparison-name")
        for index in (12, 14):
            for node in ast.parse("".join(notebook["cells"][index]["source"])).body:
                if isinstance(node, ast.Assign) and node.targets[0].id in {"KD_TRAIN_ARGS", "train_args"}:
                    exec(compile(ast.Module(body=[node], type_ignores=[]), "<train arguments>", "exec"), env)
        env["train_args"].update(env["KD_TRAIN_ARGS"])
    assert current["train_args"] == reference["train_args"]
    assert current["train_args"]["distill_model"] == str(Path("/mock/teacher/best.pt"))
    assert current["train_args"]["dis"] == 1.0
    for cell in new["cells"]:
        if cell["cell_type"] == "code":
            assert cell["execution_count"] is None and cell["outputs"] == []
