"""Verify generated recovery entry points and the common YOLO26s study settings."""
from __future__ import annotations

import json
import runpy
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
GENERATOR = runpy.run_path(str(ROOT / "scripts/generate_standard_finetuning_notebooks.py"))


@pytest.mark.parametrize("platform", ["filkom", "kaggle"])
@pytest.mark.parametrize("method,variant", GENERATOR["DESTINATIONS"])
def test_generated_recovery_uses_fixed_teacher_and_explicit_raw_input(platform, method, variant):
    filename = GENERATOR["DESTINATIONS"][method, variant]
    path = ROOT / "notebooks" / platform / "recovery" / filename
    book = json.loads(path.read_text(encoding="utf-8"))
    assert book == GENERATOR["build_notebook"](platform, method, variant)
    configuration = next("".join(cell["source"]) for cell in book["cells"] if "config" in cell["metadata"].get("tags", []))
    if variant != "n":
        assert "STUDENT_INPUTS = [" in configuration and "RESUME_RUN_DIR = None" in configuration
    else:
        assert ("RAW_ARTIFACT_REFS = ()" if variant == "n" else "RAW_ARTIFACT_REF = None") in configuration
    expected_variant = "n" if variant == "n" else "s"
    assert f"NOTEBOOK_VARIANT = {expected_variant!r}" in configuration or f'NOTEBOOK_VARIANT = "{expected_variant}"' in configuration
    assert ('Path("/kaggle/' in configuration) is (platform == "kaggle")
    if method != "finetune":
        assert "01_teacher_yolo26m-fux5wdcb-checkpoints:v50" in configuration
        assert "d749ef0baa0d6c3ae94e83ded1d9999d83c804b76ad5a841fc2ebec953af1428" in configuration
