"""I/O hooks shared by selected raw FT, Native KD, CrossKD and MGD."""
# ruff: noqa: F821 -- Notebook declarations provide these functions.
_storage_ft_publish = ft_publish
_storage_ft_execute = ft_execute
_storage_ft_fit = ft_fit
_storage_ft_registry = ft_registry


def ft_publish(run, folder, label, files, metadata, event="final"):
    publication = _storage_ft_publish(run, folder, label, files,
        {**metadata, "storage_sha256": STORAGE_SHA256, "checkpoint_retention": CHECKPOINT_RETENTION}, event=event)
    if label in {"resume", "checkpoints"}:
        storage_verified_retention(run, folder, label, publication)
    return publication


def ft_fit(raw, raw_path, manifest, dataset, record, train, data_yaml, folder, run, contract):
    state = _storage_ft_fit(raw, raw_path, manifest, dataset, record, train, data_yaml, folder, run, contract)
    storage_prune_local(folder)
    return state


def ft_registry(result, folder):
    row = _storage_ft_registry(result, folder)
    row.update(checkpoint_retention=CHECKPOINT_RETENTION, storage_sha256=STORAGE_SHA256)
    if result.get("artifact_availability"):
        row["artifact_availability"] = result["artifact_availability"]
        row["publication_status"] = result["artifact_availability"]
    if result.get("local_checkpoint_availability"):
        row["local_checkpoint_availability"] = result["local_checkpoint_availability"]
    stem = "kd_registry" if RECOVERY_METHOD != "finetune" else "finetuning_registry"
    path = _StoragePath(folder) / f"{stem}.json"
    value = ceda_read(path)
    value["rows"][0].update(row)
    ceda_write(path, value)
    write_csv(_StoragePath(folder) / f"{stem}.csv", [row])
    return row


def ft_execute(folder):
    folder = _StoragePath(folder)
    if ceda_read(folder / "smoke_local_cleanup.json", {}).get("removed"):
        raise ValueError("Checkpoint smoke ini sudah dibersihkan. Buat run smoke baru; resume penelitian tidak terpengaruh.")
    storage_prepare_retry(folder)
    result = _storage_ft_execute(folder)
    if MODE != "smoke" or result.get("status") != "PASS":
        return result
    # The inner execution has finished the run and verified history/media and
    # every uploaded package. Reacquire its folder lock before deleting copies.
    with FileLock(str(folder / ".run.lock"), timeout=0):
        if WANDB_TRACKING and AUTO_DELETE_SMOKE_ARTIFACTS:
            run_id = ceda_read(folder / "wandb_identity.json")["id"]
            try:
                cleanup = storage_cleanup_smoke(folder, result, run_id)
            except Exception as error:
                cleanup = ceda_read(folder / "smoke_artifact_cleanup.json", {})
                cleanup.update(status="PARTIAL", error=str(error), run_id=run_id)
                ceda_write(folder / "smoke_artifact_cleanup.json", cleanup)
            result["smoke_artifact_cleanup"] = cleanup
            if cleanup.get("status") == "PASS":
                result["artifact_availability"] = "DELETED_AFTER_SMOKE"
        if AUTO_DELETE_LOCAL_SMOKE_WEIGHTS:
            result["smoke_local_cleanup"] = storage_cleanup_local_smoke(folder, {**result, "mode": MODE})
            result["local_checkpoint_availability"] = "DELETED_AFTER_SMOKE"
        name = "kd_manifest.json" if RECOVERY_METHOD != "finetune" else "fine_tuned_manifest.json"
        ceda_write(folder / name, result)
        ft_registry(result, folder)
    return result
