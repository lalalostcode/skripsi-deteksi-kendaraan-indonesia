"""Resolve restored raw artifact versions while preserving the original training contract."""
# ruff: noqa: F821 -- Embedded after the unchanged fine-tuning runtime.
_binding_source = ft_source
_binding_restore = ft_restore_snapshot
_binding_registry = ft_registry
_binding_eda = ft_eda
_binding_publish = ft_publish


def ft_resolved_raw_ref():
    historical = ft_identity()["raw_artifact_ref"]
    return RAW_ARTIFACT_ALIASES.get(historical, historical)


def ft_binding_select_resume(reference, digest):
    global SEED
    name = reference.rsplit("/", 1)[-1]
    selected = STUDENT_INPUTS[0]
    if (name not in STUDENT_PINS or name.rsplit(":", 1)[0] != selected.rsplit(":", 1)[0]
            or STUDENT_PINS[name]["raw_sha256"] != digest
            or STUDENT_PINS[selected]["raw_sha256"] != digest):
        raise ValueError("Input snapshot berbeda dari raw pilihan.")
    STUDENT_INPUTS[:] = [name]
    SEED = STUDENT_PINS[name]["seed"]


def ft_source(folder):
    identity = ft_identity()
    name = STUDENT_INPUTS[0]
    live = ft_resolved_raw_ref()
    for other, pin in STUDENT_PINS.items():
        if other in LOCAL_RAW_PACKAGES and pin["raw_sha256"] == identity["raw_sha256"]:
            LOCAL_RAW_PACKAGES.setdefault(name, LOCAL_RAW_PACKAGES[other])
            break
    cache = folder / "raw_artifact"
    path = cache / "raw_pruned_manifest.json"
    if path.exists() and sha256_file(path) != STUDENT_PINS[name]["manifest_sha256"]:
        raise ValueError("SHA-256 manifest raw berubah dari pin suite pruning.")
    cached = ft_read(path)
    required = {"pruned_fp32.pt", "baseline_fp32.pt", *(cached or {}).get("audit_files", {})}
    if (live != identity["raw_artifact_ref"] and name not in LOCAL_RAW_PACKAGES
            and (cached is None or any(not (cache / file).is_file() for file in required))):
        workflow_login()
        artifact = wandb.Api(timeout=30).artifact(live, type="model")
        if artifact.qualified_name != live:
            raise ValueError("Versi raw hasil pemulihan tidak sesuai binding.")
        artifact.download(root=str(cache))
    manifest, files = _binding_source(folder)
    write_json(folder / "input_artifact_resolution.json", {
        "status": "PASS", "contract_raw_artifact_ref": identity["raw_artifact_ref"],
        "resolved_raw_artifact_ref": live, "raw_sha256": identity["raw_sha256"],
        "manifest_sha256": sha256_file(files["raw_pruned_manifest.json"]),
        "artifact_binding_sha256": ARTIFACT_BINDING_SHA256,
        "training_implementation_sha256": IMPLEMENTATION_SHA256})
    return manifest, files


def ft_restore_snapshot(folder):
    if RESUME_SNAPSHOT_REF and not (folder / "resume_contract.json").exists():
        workflow_login()
        remote = wandb.Api(timeout=30).artifact(workflow_version_ref(RESUME_SNAPSHOT_REF), type="model")
        ft_binding_select_resume(remote.metadata["raw_artifact_ref"], remote.metadata["raw_sha256"])
    return _binding_restore(folder)


def ft_registry(result, folder):
    row = _binding_registry(result, folder)
    row.update(resolved_raw_artifact_ref=ft_resolved_raw_ref(), artifact_binding_sha256=ARTIFACT_BINDING_SHA256)
    write_json(folder / "finetuning_registry.json", {"format_version": 1, "workflow": "selected-raw-finetune-v2", "rows": [row]})
    write_csv(folder / "finetuning_registry.csv", [row])
    return row


def ft_eda(result, folder, epochs, files):
    result["input_artifact_resolution"] = ft_read(folder / "input_artifact_resolution.json")
    report = _binding_eda(result, folder, epochs, files)
    write_json(folder / "eda/input_artifact_resolution.json", result["input_artifact_resolution"])
    return report


def ft_publish(run, folder, label, files, metadata, event="final"):
    return _binding_publish(run, folder, label, files, {
        **metadata, "resolved_raw_artifact_ref": ft_resolved_raw_ref(),
        "artifact_binding_sha256": ARTIFACT_BINDING_SHA256}, event=event)
