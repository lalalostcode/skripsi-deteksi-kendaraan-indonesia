"""W&B credential setup precedes downloads and never prints or rewrites secrets."""
# ruff: noqa: S102 -- Execute the trusted auth snippet against temporary fixtures.
import ast
import os
from pathlib import Path

import pytest

from scripts.generate_selected_kd_notebooks import build_notebook

ROOT = Path(__file__).resolve().parents[1]
AUTH_SOURCE = (ROOT / "scripts/filkom_mgd_auth_source.py").read_text(encoding="utf-8")


@pytest.mark.parametrize("assignment", [
    "WANDB_API_KEY=fixture-key",
    'WANDB_API_KEY="fixture-key"',
    "WANDB_API_KEY='fixture-key' # comment",
    "export WANDB_API_KEY = fixture-key",
    "\ufeffWANDB_API_KEY=fixture-key",
])
def test_env_file_key_is_loaded_without_rewriting_file_or_printing_value(tmp_path, monkeypatch, capsys, assignment):
    monkeypatch.delenv("WANDB_API_KEY", raising=False)
    env_file = tmp_path / ".env"
    env_file.write_text("UNRELATED_VALUE=untouched\n" + assignment.lstrip("\ufeff") + "\n", encoding="utf-8-sig" if assignment.startswith("\ufeff") else "utf-8")
    before = env_file.read_bytes()
    scope = {"os": os, "Path": Path, "REPO_ROOT": tmp_path, "WANDB_TRACKING": True}
    exec(AUTH_SOURCE, scope)
    assert os.environ["WANDB_API_KEY"] == "fixture-key"
    assert scope["FILKOM_MGD_AUTH_SOURCE"] == str(env_file)
    assert env_file.read_bytes() == before and "UNRELATED_VALUE" not in os.environ
    assert capsys.readouterr().out.strip() == "W&B credential: ready"
    monkeypatch.delenv("WANDB_API_KEY")


def test_existing_environment_key_has_priority_over_env_file(tmp_path, monkeypatch, capsys):
    monkeypatch.setenv("WANDB_API_KEY", "existing-fixture-key")
    env_file = tmp_path / ".env"
    env_file.write_text("WANDB_API_KEY=other-fixture-key\n", encoding="utf-8")
    before = env_file.read_bytes()
    scope = {"os": os, "Path": Path, "REPO_ROOT": tmp_path, "WANDB_TRACKING": True}
    exec(AUTH_SOURCE, scope)
    assert os.environ["WANDB_API_KEY"] == "existing-fixture-key"
    assert scope["FILKOM_MGD_AUTH_SOURCE"] == "environment" and env_file.read_bytes() == before
    assert "fixture-key" not in capsys.readouterr().out


@pytest.mark.parametrize("tracking", [True, False])
def test_missing_key_fails_early_only_when_tracking_is_enabled(tmp_path, monkeypatch, tracking):
    monkeypatch.delenv("WANDB_API_KEY", raising=False)
    scope = {"os": os, "Path": Path, "REPO_ROOT": tmp_path, "WANDB_TRACKING": tracking}
    if tracking:
        with pytest.raises(RuntimeError, match=r"WANDB_API_KEY belum tersedia.*REPO_ROOT"):
            exec(AUTH_SOURCE, scope)
    else:
        exec(AUTH_SOURCE, scope)
        assert scope["FILKOM_MGD_AUTH_SOURCE"] is None
    assert not (tmp_path / ".env").exists()


def test_malformed_key_error_does_not_expose_value(tmp_path, monkeypatch):
    monkeypatch.delenv("WANDB_API_KEY", raising=False)
    (tmp_path / ".env").write_text('WANDB_API_KEY="private-fixture\n', encoding="utf-8")
    with pytest.raises(RuntimeError, match="Format WANDB_API_KEY") as error:
        exec(AUTH_SOURCE, {"os": os, "Path": Path, "REPO_ROOT": tmp_path, "WANDB_TRACKING": True})
    assert "private-fixture" not in str(error.value)


def test_generated_auth_runs_after_python_guard_and_before_dependency_checks_or_output_creation():
    for platform in ("filkom", "kaggle"):
        for method in ("mgd", "native", "crosskd"):
            book = build_notebook(platform, method)
            source = next("".join(c["source"]) for c in book["cells"]
                          if "bootstrap" in c.get("metadata", {}).get("tags", []))
            if (platform, method) == ("filkom", "mgd"):
                assert source.index("sys.version_info") < source.index("filkom_mgd_load_wandb_key")
                assert source.index("FILKOM_MGD_AUTH_SOURCE =") < source.index("RUN_DIR =") < source.index("PRESERVED_TORCH =")
                ast.parse(source, feature_version=(3, 8))
            else:
                assert "filkom_mgd_load_wandb_key" not in source
