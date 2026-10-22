"""Offline layout, source-link and generator checks; never execute experiment cells."""
from __future__ import annotations

import ast
import hashlib
import json
import re
import runpy
from urllib.parse import unquote

import nbformat
import pytest
from IPython.core.inputtransformer2 import TransformerManager

from scripts.notebook_paths import (
    NOTEBOOK_CATEGORIES,
    NOTEBOOK_LOCATIONS,
    NOTEBOOK_ROOT,
    ROOT,
    notebook_path,
    notebook_paths,
)


def test_complete_inventory_without_duplicates_or_old_copies():
    expected = {path for platform in NOTEBOOK_LOCATIONS for path in notebook_paths(platform)}
    actual = {path for path in NOTEBOOK_ROOT.rglob("*.ipynb") if "archieve" not in path.parts}
    assert actual == expected
    assert len(expected) == 39
    assert {platform: len(notebook_paths(platform)) for platform in NOTEBOOK_LOCATIONS} == {
        "kaggle": 19, "colab": 17, "amd": 1, "local": 2,
    }
    assert len(list((NOTEBOOK_ROOT / "archieve").rglob("*.ipynb"))) == 10
    assert not (NOTEBOOK_ROOT / "yolo26-kd-indotraffic").exists()
    assert not (NOTEBOOK_ROOT / "pruning").exists()
    assert not list(NOTEBOOK_ROOT.glob("*.ipynb"))
    for platform, categories in NOTEBOOK_CATEGORIES.items():
        filenames = [name for names in categories.values() for name in names]
        assert len(filenames) == len(set(filenames)) == len(notebook_paths(platform))


def test_unknown_notebook_is_not_guessed():
    with pytest.raises(ValueError, match="Unknown notebook"):
        notebook_path("kaggle", "../baseline/03_baseline_yolo26n.ipynb")
    with pytest.raises(ValueError, match="Unknown notebook"):
        notebook_path("unknown", "03_baseline_yolo26n.ipynb")


@pytest.mark.parametrize("path", [path for platform in NOTEBOOK_LOCATIONS for path in notebook_paths(platform)],
                         ids=lambda path: path.relative_to(NOTEBOOK_ROOT).as_posix())
def test_notebook_schema_and_independence_of_repository_path_helper(path):
    book = nbformat.read(path, as_version=4)
    nbformat.validate(book)
    for cell in book.cells:
        if cell.cell_type == "code":
            assert "import notebook_paths" not in cell.source
            assert "scripts.notebook_paths" not in cell.source


@pytest.mark.parametrize("path", [path for platform in ("kaggle", "colab", "amd") for path in notebook_paths(platform)],
                         ids=lambda path: path.relative_to(NOTEBOOK_ROOT).as_posix())
def test_standalone_notebook_code_syntax(path):
    # Local legacy notebooks contain historical reference cells, including an
    # already-invalid Python cell. They retain their exact original bytes.
    book = nbformat.read(path, as_version=4)
    for cell in book.cells:
        if cell.cell_type == "code":
            compile(TransformerManager().transform_cell(cell.source), str(path), "exec")


@pytest.mark.parametrize("role", ["teacher_m", "teacher_s"])
def test_kaggle_kd_teacher_matches_configuration(role):
    names = NOTEBOOK_CATEGORIES["kaggle"]["kd/" + role]
    assert len(names) == 6
    for name in names:
        path = notebook_path("kaggle", name)
        book = json.loads(path.read_text(encoding="utf-8"))
        groups = []
        for cell in book["cells"]:
            if cell["cell_type"] != "code":
                continue
            for node in ast.parse("".join(cell["source"])).body:
                if isinstance(node, ast.Assign) and any(isinstance(target, ast.Name) and target.id == "WANDB_GROUP" for target in node.targets):
                    groups.append(ast.literal_eval(node.value))
        expected = "T26m_" if role == "teacher_m" else "T26s_"
        assert len(groups) == 1 and (groups[0].startswith(expected) or groups[0].startswith(role + "_")), (name, groups)
    assert notebook_path("kaggle", "12_pkd_yolo26n.ipynb").parent.name == "teacher_s"


def test_supervised_sources_and_platform_specific_workflows_are_not_duplicated():
    assert notebook_path("kaggle", "02_baseline_yolo26s.ipynb").parent.name == "baseline"
    assert NOTEBOOK_CATEGORIES["kaggle"]["teacher"] == ("01_teacher_yolo26m.ipynb",)
    assert notebook_path("amd", "03_baseline_yolo26n_amd_roboflow.ipynb").parent.name == "baseline"
    assert notebook_path("kaggle", "01_lamp_yolo26n_no_recovery.ipynb").parent.name == "pruning"
    assert "pruning" not in NOTEBOOK_CATEGORIES["colab"]


def test_colab_manifest_source_metadata_and_generator_dry_run():
    generator = runpy.run_path(str(ROOT / "scripts/generate_colab_notebooks.py"))
    manifest = json.loads((NOTEBOOK_ROOT / "colab/manifest.json").read_text(encoding="utf-8"))
    before = {path: hashlib.sha256(path.read_bytes()).hexdigest() for platform in NOTEBOOK_LOCATIONS for path in notebook_paths(platform)}
    assert len(manifest) == 17
    for row, workflow in zip(manifest, generator["WORKFLOWS"], strict=True):
        source = NOTEBOOK_ROOT / "kaggle" / row["source"]
        destination = NOTEBOOK_ROOT / "colab" / row["colab"]
        assert source == notebook_path("kaggle", source.name)
        assert destination == notebook_path("colab", destination.name)
        assert source.relative_to(NOTEBOOK_ROOT / "kaggle").parent == destination.relative_to(NOTEBOOK_ROOT / "colab").parent
        book = json.loads(destination.read_text(encoding="utf-8"))
        metadata = book["metadata"]["colab_source"]
        assert (destination.parent / metadata["notebook"]).resolve() == source
        assert hashlib.sha256(source.read_bytes()).hexdigest() == metadata["sha256"] == row["source_sha256"]
        rebuilt, record = generator["build_notebook"](workflow)
        assert rebuilt == book and record == row
    assert before == {path: hashlib.sha256(path.read_bytes()).hexdigest() for path in before}
    assert generator["readme"](manifest) == (NOTEBOOK_ROOT / "colab/README.md").read_text(encoding="utf-8")


@pytest.mark.parametrize("relative", ["README.md", "notebooks/README.md", "notebooks/kaggle/README.md", "notebooks/colab/README.md"])
def test_active_readme_links_resolve(relative):
    path = ROOT / relative
    text = path.read_text(encoding="utf-8")
    for target in re.findall(r"\]\(([^)]+)\)", text):
        target = unquote(target.strip("<>").split("#", 1)[0])
        if target and not re.match(r"^[a-zA-Z]+:", target):
            assert (path.parent / target).resolve().exists(), (relative, target)


def test_archive_generator_targets_the_existing_archive():
    generator = runpy.run_path(str(ROOT / "scripts/generate_modular_notebooks.py"))
    assert generator["DESTINATION"] == NOTEBOOK_ROOT / "archieve/modular"
