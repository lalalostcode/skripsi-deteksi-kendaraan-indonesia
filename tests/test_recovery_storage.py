"""Verify best/latest retention and diagnostic cleanup without deleting research inputs."""
import ast
import hashlib
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from scripts.cleanup_smoke_artifacts import read, save, sha
from scripts.generate_standard_finetuning_notebooks import DESTINATIONS, build_notebook

ROOT = Path(__file__).resolve().parents[1]
PROJECT = "entity/project"


@pytest.fixture
def storage():
    scope = {"ceda_read": read, "ceda_write": save, "ceda_hash": sha, "WANDB_ENTITY": "entity", "WANDB_PROJECT": "project"}
    exec(compile((ROOT / "scripts/recovery_storage_source.py").read_text(), "storage", "exec"), scope)
    return scope


def put_index(folder, epoch):
    best, latest = "checkpoints/best-001-a0.pt", f"checkpoints/resume-{epoch:03d}-b0.pt"
    for name in (best, latest):
        path = folder / name
        path.parent.mkdir(exist_ok=True, parents=True)
        path.write_bytes(name.encode())
    value = {"relative_path": latest, "sha256": sha(folder / latest), "epoch": epoch,
             "best": {"relative_path": best, "sha256": sha(folder / best), "epoch": 1}}
    save(folder / "latest_state.json", value)
    return value


def test_local_retention_keeps_best_and_latest_and_historical_evidence(storage, tmp_path):
    index = put_index(tmp_path, 2)
    old = tmp_path / "checkpoints/resume-001-cafe.pt"
    old.write_bytes(b"previous-state")
    (tmp_path / "checkpoints/research.pt").write_bytes(b"not-a-generated-snapshot")
    save(tmp_path / "epoch_metrics.json", [{"epoch": 1}, {"epoch": 2}])
    removed = storage["storage_prune_local"](tmp_path)
    assert [r["path"] for r in removed] == ["checkpoints/resume-001-cafe.pt"]
    assert (tmp_path / index["relative_path"]).is_file() and (tmp_path / index["best"]["relative_path"]).is_file()
    assert (tmp_path / "checkpoints/research.pt").is_file()
    assert len(read(tmp_path / "epoch_metrics.json")) == 2


def test_corrupt_latest_blocks_all_local_retention(storage, tmp_path):
    index = put_index(tmp_path, 2)
    (tmp_path / index["relative_path"]).write_bytes(b"changed")
    old = tmp_path / "checkpoints/resume-001-cafe.pt"
    old.write_bytes(b"keep-on-failure")
    with pytest.raises(ValueError, match="tidak valid"):
        storage["storage_prune_local"](tmp_path)
    assert old.is_file()


class Artifact:
    def __init__(self, reference, *, producer="current", users=(), kind="model"):
        self.qualified_name = reference
        self.id = reference
        self.type = kind
        self.aliases = ["latest"]
        self.metadata = {"files_sha256": {"best.pt": "test-sha"}}
        self.size = 100
        self.producer = producer
        self.users = users
        self.deleted = False
        self.save = lambda: None

    def logged_by(self):
        return SimpleNamespace(id=self.producer)

    def used_by(self):
        return [SimpleNamespace(id=user, name="full-experiment", group="research", config={}) for user in self.users]

    def delete(self, *, delete_aliases):
        assert delete_aliases
        self.deleted = True


def fake_api(monkeypatch, artifacts, remote=None):
    def collection(**kwargs):
        name = kwargs["name"]
        return SimpleNamespace(artifacts=lambda **kw: [a for a in artifacts.values()
            if not a.deleted and a.qualified_name.rsplit(":", 1)[0] == name], delete=lambda: None)
    api = SimpleNamespace(artifact=lambda ref: artifacts[ref], artifact_collection=collection,
                          run=lambda ref: remote)
    monkeypatch.setitem(sys.modules, "wandb", SimpleNamespace(Api=lambda **kw: api))
    return api


def test_wandb_nested_summary_wrapper_is_detached_before_deleting_anything(storage):
    class SummaryLike:
        def __init__(self, values):
            self.values = values
        def keys(self):
            return self.values.keys()
        def __getitem__(self, key):
            value = self.values[key]
            return SummaryLike(value) if isinstance(value, dict) else value
    summary = SummaryLike({"artifact_hashes": {"best.pt": "sha"}, "rows": [{"epoch": 1}], "status": "PASS"})
    assert json.loads(json.dumps(storage["storage_plain"](summary))) == {
        "artifact_hashes": {"best.pt": "sha"}, "rows": [{"epoch": 1}], "status": "PASS"}


def test_remote_retention_only_retires_own_versions_after_verified_replacement(storage, tmp_path, monkeypatch):
    put_index(tmp_path, 3)
    refs = [f"{PROJECT}/student-ft-resume:v{i}" for i in range(4)]
    artifacts = {ref: Artifact(ref, producer="other" if i == 2 else "current", users=("research",) if i == 1 else ())
                 for i, ref in enumerate(refs)}
    fake_api(monkeypatch, artifacts)
    journal = {f"resume/epoch-{i:03d}": {"status": "CONFIRMED", "reference": ref, "files_sha256": {"best.pt": "test-sha"}}
               for i, ref in enumerate(refs)}
    save(tmp_path / "artifact_publications.json", journal)
    run = SimpleNamespace(id="current", summary={})
    storage["storage_retire"](run, tmp_path, "resume", journal["resume/epoch-003"])
    assert artifacts[refs[0]].deleted
    assert not any(artifacts[ref].deleted for ref in refs[1:])
    receipt = read(tmp_path / "checkpoint_retention.json")["resume"]
    assert receipt["best_ref"] == receipt["latest_ref"] == refs[3]
    assert len(receipt["protected"]) == 2
    assert {"latest", "best", "last"} <= set(artifacts[refs[3]].aliases)
    assert read(tmp_path / "artifact_publications.json")["resume/epoch-000"]["availability_status"] == "DELETED_FOR_RETENTION"


def test_unconfirmed_replacement_never_removes_previous_checkpoint(storage, tmp_path, monkeypatch):
    monkeypatch.setitem(sys.modules, "wandb", SimpleNamespace(Api=lambda **kw: pytest.fail("No API mutation before SHA confirmation")))
    storage["storage_retire"](None, tmp_path, "resume", {"status": "UPLOADED", "reference": "entity/project/test:v1"})
    assert not (tmp_path / "checkpoint_retention.json").exists()


def test_smoke_cleanup_protects_inputs_other_consumers_and_history(storage, tmp_path, monkeypatch):
    refs = [f"{PROJECT}/student-ft-smoke-checkpoints:v0", f"{PROJECT}/yolo26s-lamp-su1.20x-seed2024-raw:v1",
            f"{PROJECT}/01_teacher_yolo26m-fux5wdcb-checkpoints:v50", f"{PROJECT}/shared-smoke:v0",
            f"{PROJECT}/run-current-history:v0", f"{PROJECT}/run-current-tablesbaselinematrix:v0"]
    arts = {ref: Artifact(ref, kind="wandb-history" if i == 4 else "run_table" if i == 5 else "model",
                          users=("research",) if i == 3 else ()) for i, ref in enumerate(refs)}
    remote = SimpleNamespace(name="student-ft-smoke", group="test-smoke", config={"mode": "smoke"}, state="finished",
                             summary={"status": "PASS"}, logged_artifacts=lambda: list(arts.values()))
    fake_api(monkeypatch, arts, remote)
    receipt = storage["storage_cleanup_smoke"](tmp_path, {"status": "PASS", "logging_verification": {"status": "PASS"}}, "current")
    assert arts[refs[0]].deleted and arts[refs[5]].deleted
    assert not any(arts[refs[i]].deleted for i in (1, 2, 3, 4))
    assert len(receipt["deleted"]) == 2 and len(receipt["protected"]) == 4
    assert read(tmp_path / "smoke_verification_before_cleanup.json")["summary_before_cleanup"]["status"] == "PASS"


@pytest.mark.parametrize("mode,state", [("full_val", "finished"), ("smoke", "running")])
def test_cleanup_cannot_remove_full_research_or_active_smoke(storage, tmp_path, monkeypatch, mode, state):
    remote = SimpleNamespace(name="full-run", group="research", config={"mode": mode}, state=state)
    fake_api(monkeypatch, {}, remote)
    if mode == "full_val":
        with pytest.raises(ValueError, match="bukan smoke"):
            storage["storage_cleanup_smoke"](tmp_path, {}, "run")
    else:
        assert storage["storage_cleanup_smoke"](tmp_path, {}, "run")["status"] == "DEFERRED_ACTIVE"


def test_finished_smoke_without_verified_logging_cannot_delete_artifacts(storage, tmp_path, monkeypatch):
    remote = SimpleNamespace(name="run-smoke", group="research-smoke", config={"mode": "smoke"}, state="finished")
    fake_api(monkeypatch, {}, remote)
    with pytest.raises(ValueError, match="belum lulus"):
        storage["storage_cleanup_smoke"](tmp_path, {"status": "PASS"}, "run")


def test_local_smoke_cleanup_keeps_canonical_source_and_all_small_evidence(storage, tmp_path):
    source = tmp_path / "canonical_raw.pt"
    source.write_bytes(b"research-input")
    folder = tmp_path / "smoke"
    put_index(folder, 1)
    save(folder / "epoch_metrics.json", [{"epoch": 1}])
    (folder / "map.svg").write_text("plot")
    (folder / "raw_copy.pt").write_bytes(source.read_bytes())
    receipt = storage["storage_cleanup_local_smoke"](folder, {"status": "PASS", "mode": "smoke"})
    assert receipt["status"] == "PASS" and len(receipt["removed"]) == 3
    assert source.read_bytes() == b"research-input"
    assert (folder / "epoch_metrics.json").is_file() and (folder / "map.svg").is_file()
    assert not list(folder.rglob("*.pt"))
    with pytest.raises(ValueError):
        storage["storage_cleanup_local_smoke"](tmp_path, {"mode": "full_val", "status": "PASS"})


@pytest.mark.parametrize("platform", ["kaggle", "filkom"])
@pytest.mark.parametrize("method,variant", DESTINATIONS)
def test_notebook_hash_contract_headings_and_complete_eda_are_preserved(platform, method, variant):
    import nbformat
    path = ROOT / "notebooks" / platform / "recovery" / DESTINATIONS[method, variant]
    book = build_notebook(platform, method, variant)
    assert book == json.loads(path.read_text(encoding="utf-8"))
    nbformat.validate(nbformat.from_dict(book))
    before = read(ROOT / "outputs/recovery_retention_contracts_before_20261108.json")[path.relative_to(ROOT).as_posix()]
    headings = [line for c in book["cells"] if c["cell_type"] == "markdown" for line in "".join(c["source"]).splitlines() if line.startswith("#")]
    assert headings == before["headings"]
    core = None
    for cell in book["cells"]:
        if cell["cell_type"] != "code":
            continue
        source = "".join(cell["source"])
        compile(source, str(path), "exec")
        for node in ast.parse(source).body:
            if isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id == "IMPLEMENTATION_SHA256" for t in node.targets):
                core = ast.literal_eval(node.value)
    assert core == before["implementation"]
    assert len(book["metadata"]["research_workflow"]["storage_sha256"]) == 64
    text = json.dumps(book)
    for token in ("ceda_display", "ceda_log", "AUTO_DELETE_SMOKE_ARTIFACTS = True", "AUTO_DELETE_LOCAL_SMOKE_WEIGHTS = True"):
        assert token in text


def test_other_sixty_two_notebooks_are_byte_identical():
    allowed = {f"notebooks/{p}/recovery/{f}" for p in ("kaggle", "filkom") for f in DESTINATIONS.values()}
    for name, digest in read(ROOT / "outputs/recovery_retention_before_20261108.json").items():
        if name not in allowed:
            assert hashlib.sha256((ROOT / name).read_bytes()).hexdigest() == digest, name


def test_legacy_cleanup_requires_explicit_smoke_mode_for_new_cpu_roots(tmp_path, monkeypatch):
    import scripts.cleanup_smoke_artifacts as cleanup
    monkeypatch.setattr(cleanup, "ROOT", tmp_path)
    parent = tmp_path / "outputs/pruning"
    for name, mode in (("lamp_short_cpu", "smoke"), ("lamp_research_cpu", "full_val"), ("lamp_unknown_cpu", None)):
        folder = parent / name
        folder.mkdir(parents=True)
        save(folder / "suite_manifest.json", {"mode": mode})
        (folder / "checkpoint.pt").write_bytes(b"model")
    raw = parent / "raw_per_seed_verified" / "pruned_fp32.pt"
    raw.parent.mkdir()
    raw.write_bytes(b"canonical")
    unrelated = tmp_path / "outputs/test_mgd/untagged.pt"
    unrelated.parent.mkdir()
    unrelated.write_bytes(b"not-confirmed-smoke")
    receipt = cleanup.cleanup_local(tmp_path / "receipt.json", execute=True)
    assert receipt["status"] == "PASS" and len(receipt["removed"]) == 1
    assert not (parent / "lamp_short_cpu/checkpoint.pt").exists()
    assert (parent / "lamp_research_cpu/checkpoint.pt").is_file()
    assert (parent / "lamp_unknown_cpu/checkpoint.pt").is_file()
    assert raw.is_file() and unrelated.is_file()


def test_legacy_cleanup_does_not_follow_an_external_checkpoint_symlink(tmp_path, monkeypatch):
    import scripts.cleanup_smoke_artifacts as cleanup
    monkeypatch.setattr(cleanup, "ROOT", tmp_path)
    folder = tmp_path / "outputs/pruning/lamp_short_cpu"
    folder.mkdir(parents=True)
    save(folder / "suite_manifest.json", {"mode": "smoke"})
    outside = tmp_path / "protected_raw.pt"
    outside.write_bytes(b"canonical")
    try:
        (folder / "alias.pt").symlink_to(outside)
    except OSError:
        pytest.skip("Symlink creation is unavailable on this Windows account")
    with pytest.raises(AssertionError):
        cleanup.cleanup_local(tmp_path / "receipt.json", execute=True)
    assert outside.is_file()
