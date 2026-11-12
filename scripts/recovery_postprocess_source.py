"""Resume immutable evaluation packages without regenerating their EDA or rewriting payloads."""
# ruff: noqa: F821 -- Embedded after the existing training runtime and input binding.
_post_eda = ft_eda
_post_publish = ft_publish
_post_tracking = ft_tracking
_post_write_json = write_json


def recovery_media_staging():
    from wandb.sdk.data_types._private import MEDIA_TMP
    for directory in (MEDIA_TMP.name, wandb.Artifact._TMP_DIR.name):
        Path(directory).mkdir(parents=True, exist_ok=True)


@contextmanager
def ft_tracking(folder, manifest, record):
    original_log = None
    run = None
    if WANDB_TRACKING:
        recovery_media_staging()
    try:
        with _post_tracking(folder, manifest, record) as run:
            if run is not None:
                original_log = run.log

                def log(*args, **kwargs):
                    recovery_media_staging()
                    return original_log(*args, **kwargs)

                run.log = log
                run.config.update({"postprocessing_sha256": POSTPROCESSING_SHA256})
            yield run
    finally:
        if original_log is not None:
            run.log = original_log


def write_json(path, value):
    path = Path(path)
    if path.name == "evaluation_manifest.json":
        publication = ft_read(path.parent / "artifact_publications.json", {}).get("results/final", {})
        expected = publication.get("files_sha256", {}).get(path.name)
        if expected:
            if not path.is_file() or sha256_file(path) != expected:
                raise ValueError("Payload evaluasi yang sudah dijurnal berubah atau hilang.")
            return
    return _post_write_json(path, value)


def ft_eda(result, folder, epochs, files):
    cache = ft_read(folder / "eda/eda_status.json", {})
    previous_metadata = globals()["_recovery_eda_metadata"]
    globals()["_recovery_eda_metadata"] = lambda row: {**previous_metadata(row), **ft_identity()}
    try:
        expected_rows = fine_tuned_comparison(result)[0]
    finally:
        globals()["_recovery_eda_metadata"] = previous_metadata
    stored = ft_read(folder / "eda/stage_metrics.json")
    journal = ft_read(folder / "artifact_publications.json", {}).get("results/final", {})
    if cache.get("status") == "PASS" and stored == expected_rows:
        hashes = {path.relative_to(folder).as_posix(): sha256_file(path)
                  for path in (folder / "eda").rglob("*") if path.is_file()}
        receipt = ft_read(folder / "eda_cache_receipt.json")
        expected_hashes = receipt["files_sha256"] if receipt else {
            name: digest for name, digest in journal.get("files_sha256", {}).items() if name.startswith("eda/")}
        if expected_hashes and expected_hashes != hashes:
            raise ValueError("File EDA tersimpan berubah; payload publikasi tetap immutable.")
        if hashes and any(name.endswith(".png") for name in hashes):
            audit = ft_read(folder / "kd_training_audit.json")
            if audit:
                result.update(method=audit["method"], teacher=audit["teacher"], kd_audit=audit)
            binding = ft_read(folder / "input_artifact_resolution.json")
            if binding:
                result["input_artifact_resolution"] = binding
            write_json(folder / "eda_cache_receipt.json", {"status": "PASS", "files_sha256": hashes,
                       "postprocessing_sha256": POSTPROCESSING_SHA256})
            return cache
    if journal:
        raise ValueError("Input EDA berbeda dari paket hasil yang sudah dijurnal.")
    report = _post_eda(result, folder, epochs, files)
    if report["status"] == "PASS":
        write_json(folder / "eda_cache_receipt.json", {"status": "PASS", "postprocessing_sha256": POSTPROCESSING_SHA256,
                   "files_sha256": {path.relative_to(folder).as_posix(): sha256_file(path)
                                    for path in (folder / "eda").rglob("*") if path.is_file()}})
    return report


def ft_publish(run, folder, label, files, metadata, event="final"):
    recovery_media_staging()
    return _post_publish(run, folder, label, files, {**metadata, "postprocessing_sha256": POSTPROCESSING_SHA256}, event=event)
