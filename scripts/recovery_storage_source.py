"""Best/latest checkpoint retention and verified-smoke artifact cleanup (I/O only)."""
# ruff: noqa: F821 -- Embedded after notebook persistence and evaluator declarations.
import re as _storage_re
from datetime import datetime as _StorageDateTime
from datetime import timedelta as _StorageDelta
from datetime import timezone as _StorageTimezone
from pathlib import Path as _StoragePath


def storage_time():
    return _StorageDateTime.now(_StorageTimezone(_StorageDelta(hours=7))).isoformat()


def storage_smoke(value):
    return bool(_storage_re.search(r"(?:^|[-_])smoke(?:[-_]|$)", value or "", _storage_re.I))


def storage_protected(name):
    name = name.rsplit("/", 1)[-1].split(":", 1)[0]
    if name.startswith("run-"):
        return False  # Baseline confusion-matrix Tables are diagnostics, not input models.
    return bool(_storage_re.match(r"^yolo26s-lamp-su\d+\.\d+x-seed\d+-raw$", name) or
                (("baseline" in name.lower() or "teacher" in name.lower()) and not storage_smoke(name)))


def storage_collection(reference):
    return reference.rsplit(":", 1)[0]


def storage_plain(value):
    """Detach nested W&B SummarySubDict wrappers before writing an audit JSON."""
    if value is None or isinstance(value, (str, bool, int, float)):
        return value
    if isinstance(value, (list, tuple)):
        return [storage_plain(item) for item in value]
    return {str(key): storage_plain(value[key]) for key in value.keys()}


def storage_prune_local(folder):
    """Remove only unreferenced generated snapshots after an atomic index commit."""
    folder = _StoragePath(folder)
    index = ceda_read(folder / "latest_state.json")
    if not index:
        return []
    keep = {index["relative_path"], index["best"]["relative_path"]}
    for name, digest in ((index["relative_path"], index["sha256"]),
                         (index["best"]["relative_path"], index["best"]["sha256"])):
        target = (folder / name).resolve()
        if not target.is_relative_to(folder.resolve()) or ceda_hash(target) != digest:
            raise ValueError("Best/latest tidak valid; snapshot lama tidak dihapus.")
    removed = []
    for path in (folder / "checkpoints").glob("*.pt"):
        if (path.relative_to(folder).as_posix() in keep
                or not _storage_re.fullmatch(r"(?:best|resume)-\d+-[a-f0-9]+\.pt", path.name)):
            continue
        if not path.resolve().is_relative_to((folder / "checkpoints").resolve()):
            raise ValueError("Snapshot berada di luar folder recovery.")
        removed.append({"path": path.relative_to(folder).as_posix(), "sha256": ceda_hash(path), "bytes": path.stat().st_size})
        path.unlink()
    if removed:
        history = ceda_read(folder / "local_retention_history.json", [])
        ceda_write(folder / "local_retention_history.json", history + [{"at": storage_time(), "removed": removed, "kept": sorted(keep)}])
    return removed


def storage_retire(run, folder, role, current, journal_name="artifact_publications.json"):
    """Keep one confirmed package containing both best and latest; protect other runs."""
    import wandb
    folder = _StoragePath(folder)
    reference = current.get("reference")
    if not reference or current.get("status") not in {"CONFIRMED", "PASS"}:
        return current
    if storage_protected(reference):
        return current
    api = wandb.Api(timeout=60)
    index = ceda_read(folder / "latest_state.json", {})
    best_epoch = index.get("best", {}).get("epoch")
    state = ceda_read(folder / "checkpoint_retention.json", {})
    previous = state.get(role, {})
    # Every resume package already includes the best FP32 model and latest full
    # training state. A second historical package would duplicate those weights.
    keep = {reference}
    remote = api.artifact(reference)
    aliases = set(remote.aliases) | {"latest"}
    aliases.update({"best", "last"})
    remote.aliases = sorted(aliases)
    remote.metadata.update(checkpoint_retention="best-and-latest-v1", best_model_epoch=best_epoch,
                           snapshot_epoch=index.get("epoch"), retention_scope="run")
    remote.save()
    journal = ceda_read(folder / journal_name, {})
    removed, protected, errors = [], [], []
    for event, item in journal.items():
        old_ref = item.get("reference")
        if (not old_ref or old_ref in keep or storage_collection(old_ref) != storage_collection(reference)
                or item.get("availability_status") in {"DELETED_FOR_RETENTION", "DELETED_AFTER_SMOKE"}
                or item.get("status") != "CONFIRMED"):
            continue
        try:
            old = api.artifact(old_ref)
            producer = old.logged_by()
            if producer is None or producer.id != run.id:
                protected.append({"reference": old_ref, "reason": "different producer"})
                continue
            other_users = [user.id for user in old.used_by() if user.id != run.id]
            if other_users:
                protected.append({"reference": old_ref, "reason": "used by another run", "users": other_users})
                continue
            old.delete(delete_aliases=True)
            item.update(availability_status="DELETED_FOR_RETENTION", deleted_at=storage_time())
            removed.append({"reference": old_ref, "event": event, "files_sha256": item.get("files_sha256", {})})
        except Exception as error:
            errors.append({"reference": old_ref, "error": str(error)})
    ceda_write(folder / journal_name, journal)
    state[role] = {"latest_ref": reference, "best_ref": reference, "best_epoch": best_epoch,
                   "policy": "best-and-latest-v1", "removed": previous.get("removed", []) + removed,
                   "protected": protected, "errors": errors, "updated_at": storage_time()}
    ceda_write(folder / "checkpoint_retention.json", state)
    run.summary.update({"checkpoint_retention": "best-and-latest-v1", "checkpoint_retention_status": "PARTIAL" if errors else "PASS"})
    return current


def storage_verified_retention(run, folder, role, publication):
    """Storage errors are visible and retryable without interrupting an epoch commit."""
    try:
        return storage_retire(run, folder, role, publication)
    except Exception as error:
        path = _StoragePath(folder) / "checkpoint_retention.json"
        state = ceda_read(path, {})
        state[role] = {**state.get(role, {}), "errors": [{"error": str(error)}], "updated_at": storage_time()}
        ceda_write(path, state)
        run.summary.update({"checkpoint_retention_status": "PARTIAL", "checkpoint_retention_error": str(error)})
        return publication


def storage_verify_package(reference, hashes, folder):
    import wandb
    from tempfile import TemporaryDirectory
    remote = wandb.Api(timeout=60).artifact(reference)
    if remote.qualified_name != reference:
        raise ValueError("Versi checkpoint tidak tetap.")
    with TemporaryDirectory(prefix="verify_retention_", dir=folder) as cache:
        downloaded = _StoragePath(remote.download(root=cache))
        for name, digest in hashes.items():
            path = (downloaded / name).resolve()
            if not path.is_relative_to(downloaded.resolve()) or ceda_hash(path) != digest:
                raise ValueError(f"SHA-256 paket best/latest berbeda: {name}")


def storage_cleanup_local_smoke(folder, result):
    """Remove only generated binary copies inside this completed diagnostic run."""
    folder = _StoragePath(folder).resolve()
    if result.get("mode") != "smoke" or result.get("status") != "PASS":
        raise ValueError("Cleanup lokal memerlukan smoke yang telah lulus verifikasi.")
    if not (folder / "wandb_identity.json").is_file() and not (folder / "latest_state.json").is_file():
        raise ValueError("Folder smoke tidak memiliki identitas atau commit training.")
    receipt = ceda_read(folder / "smoke_local_cleanup.json", {"removed": [], "errors": []})
    for path in sorted(folder.rglob("*")):
        if not path.is_file() or path.suffix.lower() not in {".pt", ".pth", ".ckpt", ".safetensors", ".onnx", ".engine"}:
            continue
        if not path.resolve().is_relative_to(folder):
            raise ValueError("File berada di luar folder smoke; cleanup ditolak.")
        row = {"relative_path": path.relative_to(folder).as_posix(), "sha256": ceda_hash(path), "bytes": path.stat().st_size}
        try:
            path.unlink()
            receipt["removed"].append(row)
        except OSError as error:
            receipt["errors"].append({**row, "error": str(error)})
        ceda_write(folder / "smoke_local_cleanup.json", receipt)
    receipt.update(status="PASS" if not receipt["errors"] else "PARTIAL", completed_at=storage_time(),
                   bytes_removed=sum(row["bytes"] for row in receipt["removed"]), audit_files_preserved=True)
    ceda_write(folder / "smoke_local_cleanup.json", receipt)
    return receipt


def storage_prepare_retry(folder):
    """An intentionally deleted smoke publication must be re-uploaded on explicit local resume."""
    folder = _StoragePath(folder)
    cleanup = ceda_read(folder / "smoke_artifact_cleanup.json", {})
    deleted = {item["reference"] for item in cleanup.get("deleted", [])}
    if not deleted:
        return
    journal = ceda_read(folder / "artifact_publications.json", {})
    archived = {key: row for key, row in journal.items() if row.get("reference") in deleted}
    if archived:
        history = ceda_read(folder / "deleted_smoke_publication_history.json", [])
        ceda_write(folder / "deleted_smoke_publication_history.json", history + [{"at": storage_time(), "publications": archived}])
        ceda_write(folder / "artifact_publications.json", {k: v for k, v in journal.items() if k not in archived})
    custom = ceda_read(folder / "custom_eda_publication.json", {})
    if custom.get("reference") in deleted:
        custom["status"] = "DELETED_AFTER_SMOKE"
        ceda_write(folder / "custom_eda_publication.json", custom)


def storage_cleanup_smoke(folder, result, run_id):
    """Archive the API proof, then remove test artifacts while preserving run metrics and research inputs."""
    import wandb
    folder = _StoragePath(folder)
    api = wandb.Api(timeout=60)
    remote = api.run(f"{WANDB_ENTITY}/{WANDB_PROJECT}/{run_id}")
    if not (storage_smoke(remote.name) or storage_smoke(remote.group) or remote.config.get("mode") == "smoke"):
        raise ValueError("Cleanup ditolak: run bukan smoke.")
    if remote.state in {"running", "pending", "preempting"}:
        return {"status": "DEFERRED_ACTIVE", "run_id": run_id}
    if result.get("status") != "PASS" or result.get("logging_verification", {}).get("status") != "PASS":
        raise ValueError("Smoke belum lulus evaluasi dan verifikasi logging; artefak tidak dihapus.")
    artifacts = list(remote.logged_artifacts())
    proof = {"at": storage_time(), "run_id": run_id, "run_name": remote.name,
             "summary_before_cleanup": storage_plain(remote.summary), "result_status": result.get("status"),
             "logging_verification": result.get("logging_verification"),
             "artifacts": [{"reference": a.qualified_name, "id": a.id, "size": a.size,
                            "files_sha256": a.metadata.get("files_sha256", {})} for a in artifacts]}
    proof_path = folder / "smoke_verification_before_cleanup.json"
    if not proof_path.is_file():
        ceda_write(proof_path, proof)
    else:
        history = ceda_read(folder / "smoke_cleanup_retry_history.json", [])
        ceda_write(folder / "smoke_cleanup_retry_history.json", history + [proof])
    prior = ceda_read(folder / "smoke_artifact_cleanup.json", {})
    receipt = {"status": "RUNNING", "run_id": run_id, "deleted": prior.get("deleted", []),
               "protected": [], "errors": [], "at": storage_time(),
               "previous_errors": prior.get("previous_errors", []) + prior.get("errors", []) +
                   ([{"error": prior["error"]}] if prior.get("error") else [])}
    ceda_write(folder / "smoke_artifact_cleanup.json", receipt)
    collections = {}
    for artifact in artifacts:
        reference = artifact.qualified_name
        if reference in {row["reference"] for row in receipt["deleted"]}:
            continue
        if artifact.type.startswith("wandb-"):
            receipt["protected"].append({"reference": reference, "reason": "system managed; retained with run history"})
            continue
        if storage_protected(reference):
            receipt["protected"].append(reference)
            continue
        try:
            non_smoke = [user.id for user in artifact.used_by() if user.id != run_id and
                         not (storage_smoke(user.name) or storage_smoke(user.group) or user.config.get("mode") == "smoke")]
            if non_smoke:
                receipt["protected"].append({"reference": reference, "users": non_smoke})
                continue
            artifact.delete(delete_aliases=True)
            receipt["deleted"].append({"reference": reference, "id": artifact.id, "size": artifact.size, "type": artifact.type,
                                       "files_sha256": artifact.metadata.get("files_sha256", {})})
            collections[(artifact.type, storage_collection(reference))] = True
        except Exception as error:
            receipt["errors"].append({"reference": reference, "error": str(error)})
        ceda_write(folder / "smoke_artifact_cleanup.json", receipt)
    receipt["empty_collections_removed"] = []
    for kind, collection_name in collections:
        try:
            collection = api.artifact_collection(type_name=kind, name=collection_name)
            if not list(collection.artifacts(per_page=100)):
                collection.delete()
                receipt["empty_collections_removed"].append(collection_name)
        except Exception as error:
            receipt["errors"].append({"collection": collection_name, "error": str(error)})
    receipt.update(status="PASS" if not receipt["errors"] else "PARTIAL", completed_at=storage_time(),
                   artifacts_available=False, run_metrics_preserved=True,
                   logical_bytes_deleted=sum(row["size"] or 0 for row in receipt["deleted"]))
    ceda_write(folder / "smoke_artifact_cleanup.json", receipt)
    remote.summary.update({"smoke_artifact_cleanup": receipt["status"], "artifact_availability": "DELETED_AFTER_SMOKE",
                           "checkpoint_retention": "best-and-latest-v1"})
    for journal_name in ("artifact_publications.json",):
        journal = ceda_read(folder / journal_name, {})
        deleted = {row["reference"] for row in receipt["deleted"]}
        for item in journal.values():
            if item.get("reference") in deleted:
                item["availability_status"] = "DELETED_AFTER_SMOKE"
        ceda_write(folder / journal_name, journal)
    return receipt
