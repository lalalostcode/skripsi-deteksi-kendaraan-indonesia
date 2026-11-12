"""Offline report migration integrity and navigation checks; no experiment execution."""
from __future__ import annotations

import hashlib
import json
import re
import runpy
from pathlib import Path
from urllib.parse import unquote

import pytest

ROOT = Path(__file__).resolve().parents[1]
REPORTS = ROOT / "reports"
MANIFEST_PATH = REPORTS / "engineering/report_reorganization_20261022.json"
MANIFEST = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
RECORDS = MANIFEST["files"]


def digest(data):
    return hashlib.sha256(data).hexdigest()


def without_hrefs(data):
    return re.sub(rb"\]\(([^)]+)\)", b"](__NAVIGATION_TARGET__)", data)


def test_historical_inventory_is_complete_and_destinations_are_unique():
    assert MANIFEST["format_version"] == 1
    assert MANIFEST["source_file_count"] == MANIFEST["migrated_file_count"] == len(RECORDS) == 65
    assert len({row["old_path"] for row in RECORDS}) == len(RECORDS)
    assert len({row["new_path"] for row in RECORDS}) == len(RECORDS)
    assert MANIFEST["unchanged_bytes_count"] == 62
    assert MANIFEST["markdown_navigation_only_changes_count"] == 3
    assert MANIFEST["experiment_output_directories_moved"] is False
    assert {path.name for path in REPORTS.iterdir()} == {
        "README.md", "baseline", "eda", "kd", "pruning", "experiments", "engineering", "log-penelitian.md"
    }
    assert not (REPORTS / "hasil_percobaan").exists()
    assert not (REPORTS / "baseline_seed42_evaluation").exists()
    # New reports are deliberately excluded from this frozen historical inventory.
    assert all((ROOT / row["new_path"]).exists() for row in RECORDS)


@pytest.mark.parametrize("row", RECORDS, ids=lambda row: row["old_path"])
def test_each_moved_file_preserves_bytes_or_only_recorded_markdown_hrefs(row):
    old = ROOT / row["old_path"]
    current = ROOT / row["new_path"]
    assert old.resolve().is_relative_to(REPORTS.resolve())
    assert current.resolve().is_relative_to(REPORTS.resolve())
    assert old.name == current.name
    assert not old.exists(), row["old_path"]
    data = current.read_bytes()
    assert len(data) == row["size_after"]
    assert digest(data) == row["sha256_after"]
    assert digest(without_hrefs(data)) == row["content_without_link_targets_sha256"]
    original = data
    for change in reversed(row["markdown_link_changes"]):
        assert current.suffix == ".md"
        after = ("](" + change["after"] + ")").encode()
        before = ("](" + change["before"] + ")").encode()
        assert original.count(after) == change["occurrences"]
        original = original.replace(after, before)
    assert len(original) == row["size_before"]
    assert digest(original) == row["sha256_before"]
    if current.suffix != ".md" or not row["markdown_link_changes"]:
        assert row["sha256_before"] == row["sha256_after"]


def test_baseline_bundle_and_historical_evidence_remain_together():
    rows = [row for row in RECORDS if row["old_path"].startswith("reports/baseline_seed42_evaluation/")]
    assert len(rows) == 48
    for row in rows:
        assert row["new_path"] == row["old_path"].replace(
            "reports/baseline_seed42_evaluation/", "reports/baseline/seed42_evaluation/", 1
        )
        assert row["sha256_before"] == row["sha256_after"]
    assert (REPORTS / "baseline/seed42_evaluation/overall_validation/confusion_matrix.png").is_file()
    assert (REPORTS / "baseline/seed42_evaluation/tabel_3_5_baseline_seed42.csv").is_file()


def test_historical_notebook_inventory_is_available_and_archives_are_unchanged():
    protected = MANIFEST["protected_existing_notebooks"]
    assert len(protected) == 50
    assert len([name for name in protected if "/archieve/" in name]) == 10
    for relative, expected in protected.items():
        path = ROOT / relative
        assert path.is_file(), relative
        assert re.fullmatch(r"[a-f0-9]{64}", expected), relative
        # Active notebooks can evolve after the historical migration.
        if "/archieve/" in relative:
            assert digest(path.read_bytes()) == expected, relative


def test_report_and_active_document_navigation_resolves():
    historical = {(row["path"], row["target"]) for row in MANIFEST["historical_absolute_links"]}
    observed_historical = set()
    documents = [*REPORTS.rglob("*.md"),
                 ROOT / "docs/lamp_yolo26n_no_recovery.md",
                 ROOT / "docs/lamp_yolo26s_no_recovery.md",
                 ROOT / "docs/pkd_yolo26n_implementation_audit.md",
                 ROOT / "docs/wandb_naming.md",
                 ROOT / "notebooks/colab/README.md"]
    for document in documents:
        markdown = document.read_text(encoding="utf-8")
        markdown = re.sub(r"^```[^\n]*\n.*?^```\s*$", "", markdown, flags=re.MULTILINE | re.DOTALL)
        markdown = re.sub(r"`[^`\r\n]*`", "", markdown)
        for target in re.findall(r"\]\(([^)]+)\)", markdown):
            identifier = (document.relative_to(ROOT).as_posix(), target)
            if identifier in historical:
                observed_historical.add(identifier)
                continue
            target = unquote(target.strip("<>").split("#", 1)[0])
            if not target or target.startswith(("http://", "https://", "mailto:")):
                continue
            # Absolute execution paths are allowed only by the explicit historical manifest.
            assert not re.match(r"^[A-Za-z]:", target), identifier
            resolved = (document.parent / target).resolve()
            assert resolved.is_relative_to(ROOT.resolve()), identifier
            assert resolved.exists(), identifier
    assert observed_historical == historical


def test_colab_generator_retains_current_report_links_without_rewriting_notebooks():
    generator = runpy.run_path(str(ROOT / "scripts/generate_colab_notebooks.py"))
    rows = json.loads((ROOT / "notebooks/colab/manifest.json").read_text(encoding="utf-8"))
    text = generator["readme"](rows)
    assert text == (ROOT / "notebooks/colab/README.md").read_text(encoding="utf-8")
    assert "../../reports/experiments/README.md" in text
    assert "../../reports/experiments/colab/colab_collection_validation_20261022.md" in text
    assert "reports/hasil_percobaan" not in text
