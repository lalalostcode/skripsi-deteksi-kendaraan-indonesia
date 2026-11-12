"""Keep the YOLO26n recovery workflow on the same best/latest storage policy."""
# ruff: noqa: F821 -- Embedded after the suite runtime.
_storage_workflow_publish = workflow_publish
_storage_suite_save_model = ResumableStructuralTrainer.save_model
_storage_recovery_run_job = recovery_run_job


def workflow_publish(run, folder, label, files, *, metadata_info=None, aliases=None, event=None, kind="model"):
    files = dict(files)
    if run is not None and label == "recovered":
        # Preserve the old recovered_fp32.pt entry and add explicit best/last
        # names without reconstructing a standard, unpruned model.
        index = ceda_read(_StoragePath(folder) / "latest_state.json")
        state = torch.load(_StoragePath(folder) / index["relative_path"], map_location="cpu", weights_only=False)
        last = _StoragePath(folder) / "last.pt"
        if not last.is_file():
            recovery_atomic_torch_save({"model": _recovery_native_snapshot(state["ema"]), "ema": None,
                "optimizer": None, "train_args": {}, "epoch": state["epoch"] - 1, "recovery_enabled": True}, last)
        files["best.pt"] = _StoragePath(folder) / index["best"]["relative_path"]
        files["last.pt"] = last
    record = _storage_workflow_publish(run, folder, label, files, metadata_info={**(metadata_info or {}),
        "storage_sha256": STORAGE_SHA256, "checkpoint_retention": CHECKPOINT_RETENTION},
        aliases=aliases, event=event, kind=kind)
    if run is not None and record.get("status") == "CONFIRMED":
        storage_verify_package(record["reference"], record["files_sha256"], folder)
        if label in {"recovered", "training-state"}:
            storage_verified_retention(run, folder, label, record)
    return record


def _storage_suite_save(self):
    result = _storage_suite_save_model(self)
    storage_prune_local(self.recovery_folder)
    return result


ResumableStructuralTrainer.save_model = _storage_suite_save


def _storage_verify_suite_logging(folder, result, run_id):
    import wandb
    remote = wandb.Api(timeout=60).run(f"{WANDB_ENTITY}/{WANDB_PROJECT}/{run_id}")
    epochs = remote.history(keys=["epoch", "fitness"], samples=max(500, remote.lastHistoryStep + 1), pandas=False)
    if sorted(int(row["epoch"]) for row in epochs) != list(range(1, result["completed_epochs"] + 1)):
        raise ValueError("History smoke belum lengkap atau memiliki epoch ganda.")
    custom = ceda_read(folder / "custom_eda_logging.json", {})
    images = ceda_read(folder / "wandb_media.json", {}).get("keys", []) + custom.get("image_keys", [])
    tables = ["recovery/three_stage_metrics", "recovery/three_stage_per_class"] + custom.get("table_keys", [])
    for key in images + tables:
        if not remote.summary.get(key, {}).get("_type"):
            raise ValueError(f"EDA smoke belum tercatat di W&B: {key}")
    proof = {"status": "PASS", "run_id": run_id, "epoch_count": len(epochs),
             "image_count": len(images), "table_count": len(tables)}
    ceda_write(folder / "wandb_logging_verification.json", proof)
    return proof


def recovery_run_job(folder, *args, **kwargs):
    if ceda_read(_StoragePath(folder) / "smoke_local_cleanup.json", {}).get("removed"):
        raise ValueError("Checkpoint smoke sudah dibersihkan; gunakan suite diagnostik baru.")
    storage_prepare_retry(folder)
    result = _storage_recovery_run_job(folder, *args, **kwargs)
    folder = _StoragePath(folder)
    if MODE == "smoke" and result.get("status") == "PASS":
        if WANDB_TRACKING and AUTO_DELETE_SMOKE_ARTIFACTS:
            identity = ceda_read(folder / "wandb_identity.json")
            result["logging_verification"] = _storage_verify_suite_logging(folder, result, identity["id"])
            cleanup = storage_cleanup_smoke(folder, result, identity["id"])
            result["smoke_artifact_cleanup"] = cleanup
            if cleanup["status"] == "PASS":
                result["artifact_availability"] = "DELETED_AFTER_SMOKE"
        if AUTO_DELETE_LOCAL_SMOKE_WEIGHTS:
            result["smoke_local_cleanup"] = storage_cleanup_local_smoke(folder, {**result, "mode": MODE})
            result["local_checkpoint_availability"] = "DELETED_AFTER_SMOKE"
        ceda_write(folder / "recovery_manifest.json", result)
    return result
