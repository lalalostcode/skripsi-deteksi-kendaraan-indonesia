"""Run independent standard fine-tuning jobs from verified raw artifacts."""
# ruff: noqa: F821 -- Standalone notebook declarations share one namespace.
# ruff: noqa: BLE001 -- Independent stages record failures and keep completed configurations.
import shutil


def recovery_read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def recovery_dataset(folder):
    folder = Path(folder)
    expected = None
    portable = None
    if DATASET_MANIFEST_PATH:
        portable = recovery_read_json(DATASET_MANIFEST_PATH)
        if portable["dataset_id"] != DATASET_ID or portable["provenance_status"] != "VERIFIED":
            raise ValueError("Gunakan manifest Poltrada V3 sumber yang VERIFIED.")
        expected = portable["val_fingerprint"]
    dataset = gflops_prepare_dataset(MODE, DATA_YAML, DATASET_ROOT, folder, seed=SEED,
                                    expected_fingerprint=expected, dataset_id=DATASET_ID)
    if dataset is None:
        raise FileNotFoundError("Isi DATASET_ROOT dan DATA_YAML dari Poltrada V3 asli, dengan train dan val.")
    record, train, _ = workflow_dataset_manifest(dataset, DATA_YAML, DATASET_ROOT, folder)
    if portable and record["train_fingerprint"] != portable["train_fingerprint"]:
        raise ValueError("Fingerprint train berbeda dari sumber dataset.")
    if MODE == "smoke":
        root = Path(DATASET_ROOT).resolve()
        train = sorted(sorted(train, key=lambda image: hashlib.sha256(
            f"{SEED}:{image.relative_to(root).as_posix()}".encode()).hexdigest())[:128])
    train_fp, files = _gflops_fingerprint(Path(DATASET_ROOT).resolve(), train, "train", record["class_names"])
    train_file = folder / "train_images.txt"
    train_file.write_text("".join(f"{path.as_posix()}\n" for path in train), encoding="utf-8")
    train_yaml = folder / "train_val_data.yaml"
    train_yaml.write_text(yaml.safe_dump({**dataset["info"], "train": str(train_file.resolve())},
                                         sort_keys=False, allow_unicode=True), encoding="utf-8")
    write_json(folder / "train_manifest.json", {"image_count": len(train), "fingerprint": train_fp, "files": files})
    return dataset, record, train_yaml, train_fp


def recovery_discover_inputs(folder):
    """Accept portable pruning suites, explicit manifests and pinned W&B versions."""
    sources = {}

    def add(path, reference=None):
        path = Path(path).expanduser().resolve()
        manifest = recovery_read_json(path)
        target = manifest.get("target_speedup")
        if manifest.get("model_variant") != NOTEBOOK_VARIANT or target not in GFLOPS_TARGETS:
            return
        if manifest["configuration_id"] != f"target-{target:.2f}x":
            raise ValueError("ID manifest raw tidak sesuai target GFLOPs.")
        existing = sources.get(target)
        if existing and existing["manifest"]["raw_checkpoint"]["sha256"] != manifest["raw_checkpoint"]["sha256"]:
            raise ValueError(f"Dua raw berbeda diajukan untuk target {target:.2f}×.")
        if not existing or reference:
            sources[target] = {"path": path, "manifest": manifest, "reference": reference}

    errors = []
    if PRUNING_SUITE_DIR:
        root = Path(PRUNING_SUITE_DIR).expanduser().resolve()
        try:
            rows = recovery_read_json(root / "student_candidates.json")["rows"]
        except Exception as error:
            rows = []
            errors.append({"source": str(root), "error": safe_error(error)})
        for row in rows:
            relative = row.get("checkpoint_relative_path")
            if row["model_variant"] == NOTEBOOK_VARIANT and row.get("target_speedup") in GFLOPS_TARGETS and relative:
                try:
                    # Resume a partial pruning suite from whichever manifest passed structural checks.
                    add((root / relative.replace("\\", "/")).parent / "raw_pruned_manifest.json")
                    target = row["target_speedup"]
                    # An aliased target reuses the exact saved structure and raw version.
                    if target not in sources and row.get("configuration_id") == f"target-{target:.2f}x":
                        canonical = next(value for value in sources.values()
                                         if value["manifest"]["raw_checkpoint"]["sha256"] == row["checkpoint_sha256"])
                        sources[target] = {**canonical, "alias": True}
                except Exception as error:
                    errors.append({"source": relative, "target_speedup": row.get("target_speedup"), "error": safe_error(error)})
    for filename in RAW_MANIFEST_PATHS:
        try:
            add(filename)
        except Exception as error:
            errors.append({"source": str(filename), "error": safe_error(error)})
    if RAW_ARTIFACT_REFS:
        import wandb
        workflow_login()
        api = wandb.Api()
        for ref in RAW_ARTIFACT_REFS:
            try:
                ref = workflow_version_ref(ref)
                artifact = api.artifact(ref, type="model")
                location = Path(artifact.download(root=str(folder / "inputs" / workflow_digest(ref)[:16])))
                add(location / "raw_pruned_manifest.json", ref)
            except Exception as error:
                errors.append({"source": str(ref), "error": safe_error(error)})
    selected = sorted(sources if not RECOVERY_TARGETS else set(sources) & set(RECOVERY_TARGETS))
    for target in set(RECOVERY_TARGETS) - set(selected):
        errors.append({"target_speedup": target, "error": "Manifest raw target belum tersedia."})
    return [(target, sources[target]) for target in selected], errors


def recovery_input(source, dataset, data_record):
    manifest, path = source["manifest"], source["path"]
    variant = NOTEBOOK_VARIANT
    if (manifest.get("format_version") != 2 or manifest.get("stage") != "raw_pruned"
            or manifest["model_variant"] != variant or manifest["source_baseline"]["sha256"] != SOURCE_CHECKPOINT_SHA256[variant]
            or any(manifest.get(key) != "PASS" for key in ("structural_status", "oracle_status", "reload_status"))
            or manifest["target_status"] not in {"REACHED", "TARGET_NOT_REACHED"}):
        raise ValueError("Input bukan raw pruned yang lulus struktur, oracle, dan reload dari baseline terkunci.")
    identity = manifest["dataset"]
    if (identity["dataset_id"] != data_record["dataset_id"] or identity["class_names"] != data_record["class_names"]
            or identity.get("train_fingerprint") != data_record["train_fingerprint"] or manifest["seed"] != SEED):
        raise ValueError("Dataset train, urutan kelas, atau seed berbeda dari manifest raw.")
    if manifest["evaluation_mode"] == "full_val":
        if identity["fingerprint"] != data_record["val_fingerprint"]:
            raise ValueError("Fingerprint full val berbeda dari manifest raw.")
    elif MODE != "smoke" or identity["fingerprint"] != dataset["fingerprint"]:
        raise ValueError("Recovery full_val memerlukan raw full_val; subset smoke juga harus cocok.")
    raw_meta, baseline_meta = manifest["raw_checkpoint"], manifest["baseline_checkpoint"]
    raw_path = (path.parent / raw_meta["relative_path"]).resolve(strict=True)
    local_baseline = path.parent / baseline_meta["relative_path"].replace("\\", "/")
    packaged_baseline = path.parent / baseline_meta["artifact_path"]
    baseline_path = (packaged_baseline if packaged_baseline.is_file() else local_baseline).resolve(strict=True)
    if sha256_file(raw_path) != raw_meta["sha256"] or sha256_file(baseline_path) != baseline_meta["sha256"]:
        raise ValueError("SHA-256 input raw/baseline tidak cocok; training ditolak.")
    payload = torch.load(raw_path, map_location="cpu", weights_only=False)
    raw = payload.get("model")
    if (type(raw) is not _RecoveryDetectionModel or payload.get("recovery_enabled") is not False
            or not raw.end2end or any(p.dtype != torch.float32 for p in raw.parameters()) or not _recovery_finite_state(raw)):
        raise ValueError("Raw wajib DetectionModel FP32 native dual-head dari notebook pruning.")
    signature = recovery_structure_signature(raw)
    serialized_signature = {"parameters": signature["parameters"], "conv_bn": {
        name: [layer["in"], layer["out"], layer["groups"]] if "in" in layer else [layer["features"]]
        for name, layer in signature["layers"].items()}}
    if (model_state_sha256(raw) != manifest["state_sha256"] or serialized_signature != manifest["structure_signature"]
            or recovery_ordered_names(raw.names) != data_record["class_names"]):
        raise ValueError("Tensor, signature struktur, atau kelas tidak identik dengan manifest raw.")
    files = {"pruned_fp32.pt": raw_path, "baseline_fp32.pt": baseline_path, "raw_pruned_manifest.json": path}
    for name, digest in manifest.get("audit_files", {}).items():
        file = path.parent / name
        if sha256_file(file) != digest:
            raise ValueError(f"SHA-256 bukti pruning tidak cocok: {name}")
        files[name] = file
    return raw, raw_path, baseline_path, files


def recovery_raw_reference(source, files, folder, target):
    if not WANDB_TRACKING:
        return None
    if source["reference"]:
        return workflow_version_ref(source["reference"])
    journal = source["path"].parent / "artifact_publications.json"
    publication = recovery_read_json(journal).get("raw-pruned") if journal.is_file() else None
    hashes = {name: sha256_file(value) for name, value in files.items()}
    if publication and publication.get("status") == "CONFIRMED" and publication.get("files_sha256") == hashes:
        return workflow_version_ref(publication["reference"])
    # The registration run is independent of the recovery run and is resumable itself.
    registry = folder / "register_raw"
    workflow_bind(registry, "input_contract.json", {"files_sha256": hashes, "platform": PLATFORM,
                  "variant": NOTEBOOK_VARIANT, "target": target, "mode": MODE, "seed": SEED})
    with workflow_tracking(registry, NOTEBOOK_VARIANT, target, registration=True) as run:
        record = workflow_publish(run, registry, "raw-pruned", files, aliases=["raw-pruned", "latest"],
                                  metadata_info={"stage": "raw_pruned", "raw_sha256": hashes["pruned_fp32.pt"]})
    return workflow_version_ref(record["reference"])


def recovery_epoch_publication(run, folder, index, row, raw_ref):
    if run is None:
        return
    epoch = index["epoch"]
    label = f"epoch-{epoch:03d}"
    # The upload package uses the same relative filenames as latest_state.json.
    state = folder / index["relative_path"]
    best = folder / index["best"]["relative_path"]
    if sha256_file(state) != index["sha256"] or sha256_file(best) != index["best"]["sha256"]:
        raise ValueError("File training yang akan diunggah berbeda dari commit epoch.")
    aliases = [label, "latest"] + (["best"] if index["best"]["epoch"] == epoch else [])
    stage_log = folder / "epoch_publications"
    stage_log.mkdir(parents=True, exist_ok=True)
    epoch_record = stage_log / (label + ".json")
    if not epoch_record.is_file():
        write_json(epoch_record, {**row, "raw_artifact_ref": raw_ref})
    files = {index["relative_path"]: state, index["best"]["relative_path"]: best,
             "epoch.json": epoch_record, "latest_state.json": folder / "latest_state.json",
             "resume_contract.json": folder / "resume_contract.json", "wandb_identity.json": folder / "wandb_identity.json",
             "training_config.json": folder / "training_config.json", "epoch_metrics.json": folder / "epoch_metrics.json"}
    marker = stage_log / (label + "-logged.json")
    if not marker.is_file():
        try:
            run.log({"epoch": epoch, "optimizer_steps": row["optimizer_steps"], "fitness": row["fitness"],
                     **{f"train/{name.removeprefix('train/')}": value for name, value in row["losses"].items()},
                     **row["metrics"], **{f"lr/pg{idx}": value for idx, value in enumerate(row["learning_rates"])}}, step=epoch)
            write_json(marker, {"status": "LOGGED", "epoch": epoch})
        except Exception as error:
            raise WorkflowUploadPending(safe_error(error)) from error
    workflow_publish(run, folder, "training-state", files, aliases=aliases, event=label,
                     metadata_info={"epoch": epoch, "raw_artifact_ref": raw_ref, "state_sha256": index["sha256"]},
                     kind="training-state")


def recovery_measure(path, dataset, folder, image):
    folder.mkdir(parents=True, exist_ok=True)
    measurement_file = folder / "measurement.json"
    digest = sha256_file(path)
    row = recovery_read_json(measurement_file) if measurement_file.is_file() else {}
    if row and row["checkpoint_sha256"] != digest:
        raise ValueError("Checkpoint berubah sesudah pengukuran.")
    if not row:
        core = YOLO(str(path), task="detect").model.to(TARGET_DEVICE).float().eval()
        core.end2end = True
        forward_snapshot(core, image)
        row = {**model_measurements(core, IMGSZ), "checkpoint_sha256": digest, "checkpoint_bytes": Path(path).stat().st_size,
               "conv_output_channels": sum(m.out_channels for m in core.modules() if isinstance(m, torch.nn.Conv2d)),
               "validation_status": "NOT_RUN", "errors": [], "latency": gflops_latency(core, image)}
    elif row["latency"]["status"] not in {"PASS", "MEASURED"} and RUN_LATENCY_BENCHMARK:
        core = YOLO(str(path), task="detect").model.to(TARGET_DEVICE).float().eval()
        core.end2end = True
        row["latency"] = gflops_latency(core, image)
    if row["validation_status"] != "PASS":
        gflops_validation(row, path, dataset, folder)
    write_json(measurement_file, row)
    return row


def recovery_run_job(folder, target, source, dataset, data_record, train_yaml, train_fp, settings, image, method):
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / "recovery_manifest.json"
    previous = recovery_read_json(path) if path.is_file() else {}
    manifest = source["manifest"]
    trainer_class, trainer_kwargs, method_contract = method
    result = previous or {"format_version": 2, "model_variant": NOTEBOOK_VARIANT, "platform": PLATFORM, "mode": MODE,
        "method": RECOVERY_METHOD,
        "configuration_id": f"target-{target:.2f}x", "target_speedup": target, "target_status": manifest["target_status"],
        "actual_gflops_speedup": manifest["actual_gflops_speedup"], "errors": [], "status": "NOT_RUN",
        **{key + "_status": "NOT_RUN" for key in ("input", "training", "structural", "reload", "validation", "latency", "eda", "upload")}}
    stage = "input"
    committed = False
    try:
        raw, raw_path, baseline_path, files = recovery_input(source, dataset, data_record)
        raw_sha = manifest["raw_checkpoint"]["sha256"]
        if target in EXPECTED_RAW_SHA256 and EXPECTED_RAW_SHA256[target].lower() != raw_sha:
            raise ValueError("SHA raw berbeda dari kondisi awal KD/fine-tuning yang dikunci.")
        if previous and previous.get("raw_sha256") not in {None, raw_sha}:
            raise ValueError("Input raw berbeda dari konfigurasi recovery yang sudah tersimpan.")
        stage = "upload"
        raw_ref = recovery_raw_reference(source, files, folder, target)
        if WANDB_TRACKING and not raw_ref:
            raise WorkflowUploadPending("Artefak raw belum dikonfirmasi.")
        contract = workflow_bind(folder, "resume_contract.json", {"implementation": IMPLEMENTATION_SHA256,
            "platform": PLATFORM, "mode": MODE, "variant": NOTEBOOK_VARIANT, "target": target,
            "target_status": manifest["target_status"], "raw_sha256": raw_sha, "raw_artifact_ref": raw_ref,
            "baseline_sha256": manifest["baseline_checkpoint"]["sha256"], "settings": settings, "validation_batch": BATCH,
            "val_fingerprint": dataset["fingerprint"], "train_fingerprint": train_fp,
            "gpu": ENVIRONMENT.get("gpu"), "cuda_build": ENVIRONMENT.get("cuda_build"), "packages": ENVIRONMENT.get("packages"),
            "validation_workers": WORKERS, "latency_warmup": LATENCY_WARMUP, "latency_repeats": LATENCY_REPEATS,
            "run_latency": RUN_LATENCY_BENCHMARK, "method": method_contract})
        committed = True
        result.pop("interrupted", None)
        result.update(input_status="PASS", raw_sha256=raw_sha, raw_artifact_ref=raw_ref,
                      baseline_sha256=manifest["baseline_checkpoint"]["sha256"], dataset=data_record,
                      validation_fingerprint=dataset["fingerprint"], train_fingerprint=train_fp,
                      raw_initial_state_sha256=model_state_sha256(raw), contract_sha256=contract["sha256"], method_config=method_contract)
        write_json(path, result)
        # All three inference evaluations are remeasured with this exact device and protocol.
        stage = "validation"
        baseline = recovery_measure(baseline_path, dataset, folder.parent.parent / "baselines" / result["baseline_sha256"][:16], image)
        raw_metrics = recovery_measure(raw_path, dataset, folder / "raw_validation", image)
        if raw_metrics["parameters"] >= baseline["parameters"]:
            raise ValueError("Raw tidak lebih sempit dari baseline; training tidak dimulai.")
        result["stages"] = {**result.get("stages", {}), "baseline": baseline, "raw_pruned": raw_metrics}
        write_json(path, result)
        stage = "upload"
        with workflow_tracking(folder, NOTEBOOK_VARIANT, target, recovery=True, source_ref=raw_ref) as run:
            teacher_ref = method_contract.get("teacher_artifact_ref")
            if run is not None and teacher_ref:
                run.use_artifact(workflow_version_ref(teacher_ref), type="model")
            if run is not None:
                run.config.update({"recovery": method_contract, "training": settings, "raw_artifact_ref": raw_ref,
                                   "raw_sha256": raw_sha, "target_status": result["target_status"],
                                   "actual_gflops_speedup": result["actual_gflops_speedup"],
                                   "val_fingerprint": dataset["fingerprint"], "train_fingerprint": train_fp}, allow_val_change=False)
            stage = "training"
            state, index = recovery_fit(raw, raw_path, raw_sha, train_yaml, dataset["class_names"], folder, settings,
                contract["sha256"], val_batch=BATCH, trainer_class=trainer_class, trainer_kwargs=trainer_kwargs,
                committed_callback=lambda idx, row: recovery_epoch_publication(run, folder, idx, row, raw_ref))
            result.update(training_status="PASS", completed_epochs=state["epoch"], optimizer_steps=state["optimizer_steps"],
                          best_epoch=state["best"]["epoch"], best_fitness=state["best"]["fitness"], resume_checkpoint=index,
                          final_live_state_sha256=model_state_sha256(state["live_model"]),
                          final_ema_state_sha256=model_state_sha256(state["ema"]))
            write_csv(folder / "epoch_metrics.csv", recovery_epoch_rows(state["epoch_rows"]))
            write_json(path, result)
            stage = "structural"
            best = folder / state["best"]["relative_path"]
            recovered = folder / "recovered_fp32.pt"
            if not recovered.is_file():
                temp = recovered.with_name(f"recovered-{uuid.uuid4().hex}.tmp")
                shutil.copyfile(best, temp)
                os.replace(temp, recovered)
            recovered_sha = sha256_file(recovered)
            if recovered_sha != state["best"]["sha256"] or sha256_file(raw_path) != raw_sha:
                raise ValueError("Checkpoint recovered atau raw berubah di luar commit.")
            core = YOLO(str(recovered), task="detect").model.to(TARGET_DEVICE).float().eval()
            core.end2end = True
            signature = recovery_structure_signature(raw)
            if (recovery_structure_signature(core) != signature
                    or recovery_state_sha256(core, parameters_only=True) == recovery_state_sha256(raw, parameters_only=True)):
                raise ValueError("Checkpoint recovered berbeda struktur atau tidak memperbarui bobot raw.")
            result.update(structural_status="PASS", recovered_checkpoint={"relative_path": recovered.name, "sha256": recovered_sha},
                          structure_signature=signature, recovered_state_sha256=model_state_sha256(core), raw_unchanged=True)
            stage = "reload"
            if result["reload_status"] != "PASS":
                (folder / "reload").mkdir(exist_ok=True)
                result["reload"] = verify_fresh_reload(recovered, core, image, folder / "reload", timeout=RELOAD_TIMEOUT_SECONDS)
                result["reload_status"] = "PASS"
            write_json(path, result)
            stage = "validation"
            recovered_metrics = recovery_measure(recovered, dataset, folder / "recovered_validation", image)
            if (recovered_metrics["parameters"] != raw_metrics["parameters"]
                    or not math.isclose(recovered_metrics["gflops"], raw_metrics["gflops"], rel_tol=1e-10)):
                raise ValueError("Parameter/GFLOPs berubah sesudah fine-tuning struktur raw.")
            result["stages"]["recovered"] = recovered_metrics
            result["validation_status"] = workflow_status(row["validation_status"] for row in result["stages"].values())
            latency = [row["latency"]["status"] for row in result["stages"].values()]
            result["latency_status"] = workflow_status("PASS" if status == "MEASURED" else status for status in latency)
            stage = "eda"
            eda_metrics = folder / "eda/stage_metrics.json"
            if (result["eda_status"] != "PASS" or not eda_metrics.is_file()
                    or recovery_read_json(eda_metrics) != recovery_comparison_rows(result)[0]):
                try:
                    result["eda"] = recovery_build_eda(result, folder, state["epoch_rows"], files)
                    result["eda_status"] = result["eda"]["status"]
                except Exception as error:
                    result["eda_status"] = "FAIL"
                    result["errors"].append({"stage": "eda", "error": safe_error(error)})
            write_json(path, result)
            stage = "upload"
            # Keep the immutable model manifest independent of report/upload state.
            model_manifest = folder / "recovered_model.json"
            write_json(model_manifest, {"format_version": 2, "stage": "recovered", "model_variant": NOTEBOOK_VARIANT,
                    "configuration_id": result["configuration_id"], "target_speedup": target, "target_status": result["target_status"],
                    "checkpoint": result["recovered_checkpoint"], "state_sha256": result["recovered_state_sha256"],
                    "structure_signature": signature, "raw_artifact_ref": raw_ref, "raw_sha256": raw_sha,
                    "method": method_contract, "epochs": state["epoch"], "best_epoch": state["best"]["epoch"], "dataset": data_record})
            if run is not None:
                result["recovered_publication"] = workflow_publish(run, folder, "recovered", {
                    recovered.name: recovered, model_manifest.name: model_manifest}, aliases=["recovered", "latest"],
                    metadata_info={"raw_artifact_ref": raw_ref, "raw_sha256": raw_sha, "recovered_sha256": recovered_sha})
                recovery_log_comparison(run, result, folder)
                write_json(path, result)
                report_files = workflow_report_files(folder)
                report_files.pop(path.name, None)  # Mutable run status is not part of the immutable report version.
                report_files["protocol_environment.json"] = folder.parent.parent / "environment.json"
                result["results_publication"] = workflow_publish(run, folder, "results", report_files, kind="results")
                result["upload_status"] = "PASS"
    except (Exception, KeyboardInterrupt) as error:
        if isinstance(error, KeyboardInterrupt):
            result["interrupted"] = True
        if previous and not committed and stage != "upload":
            return {**previous, "resume_rejected": safe_error(error)}
        if isinstance(error, WorkflowUploadPending):
            stage = "upload"
        result[stage + "_status"] = "FAIL"
        result["errors"].append({"stage": stage, "error": safe_error(error) or "Interrupted"})
        if (folder / "latest_state.json").is_file():
            result["completed_epochs"] = recovery_read_json(folder / "latest_state.json")["epoch"]
            if result["training_status"] != "PASS":
                result["training_status"] = "PARTIAL"
    statuses = [result[key + "_status"] for key in ("input", "training", "structural", "reload", "validation", "eda")]
    if RUN_LATENCY_BENCHMARK:
        statuses.append(result["latency_status"])
    if WANDB_TRACKING:
        statuses.append(result["upload_status"])
    result["status"] = workflow_status(statuses, useful=bool(result.get("completed_epochs") or result.get("stages")))
    write_json(path, result)
    print(result["configuration_id"], result["status"], "| epoch", result.get("completed_epochs", 0), flush=True)
    return result


def recovery_execute(folder):
    report = {"format_version": 2, "model_variant": NOTEBOOK_VARIANT, "mode": MODE, "platform": PLATFORM,
              "method": RECOVERY_METHOD, "status": "NOT_RUN", "jobs": [], "errors": []}
    folder = Path(folder)
    path = folder / "suite_manifest.json"
    previous = recovery_read_json(path) if path.is_file() else None
    bound = False
    try:
        if MODE not in {"smoke", "full_val"} or not set(RECOVERY_TARGETS) <= set(GFLOPS_TARGETS) or NOTEBOOK_VARIANT not in {"n", "s"}:
            raise ValueError("Recovery memerlukan smoke/full_val dan target dari notebook pruning.")
        if not (PRUNING_SUITE_DIR or RAW_MANIFEST_PATHS or RAW_ARTIFACT_REFS):
            report["reason"] = "Isi PRUNING_SUITE_DIR, RAW_MANIFEST_PATHS atau RAW_ARTIFACT_REFS."
            return report
        if RECOVERY_METHOD != "finetune" and not (TEACHER_ARTIFACT_REF or TEACHER_LOCAL_PATH):
            report["reason"] = "Isi TEACHER_ARTIFACT_REF atau TEACHER_LOCAL_PATH beserta TEACHER_SHA256."
            return report
        if MODE == "full_val" and (RECOVERY_EPOCHS != 100 or BATCH != 16 or IMGSZ != 640 or SEED != 42
                                    or not WANDB_TRACKING or TARGET_DEVICE.type != "cuda"):
            raise ValueError("Recovery full_val memakai 100 epoch, batch 16, 640 px, seed 42, CUDA dan W&B aktif.")
        dataset, data_record, train_yaml, train_fp = recovery_dataset(folder / "dataset_checks" / uuid.uuid4().hex[:8])
        settings = recovery_settings()
        method = ((ResumableStructuralTrainer, {}, {"method": "finetune", "loss": "native detection"})
                  if RECOVERY_METHOD == "finetune" else recovery_prepare_method(folder))
        workflow_bind(folder, "suite_contract.json", {"implementation": IMPLEMENTATION_SHA256, "platform": PLATFORM,
            "variant": NOTEBOOK_VARIANT, "mode": MODE, "settings": settings, "val_fingerprint": dataset["fingerprint"],
            "train_fingerprint": train_fp, "dataset_provenance": data_record["provenance_status"],
            "validation_batch": BATCH, "validation_workers": WORKERS, "latency_warmup": LATENCY_WARMUP,
            "latency_repeats": LATENCY_REPEATS, "run_latency": RUN_LATENCY_BENCHMARK, "wandb": WANDB_TRACKING,
            "method": method[2],
            **{key: ENVIRONMENT.get(key) for key in ("packages", "gpu", "cuda_build")}})
        bound = True
        write_json(folder / "dataset_manifest.json", data_record)
        write_json(folder / "environment.json", ENVIRONMENT)
        sources, report["errors"] = recovery_discover_inputs(folder)
        completed = {job["target_speedup"]: job for job in (previous or {}).get("jobs", [])}
        by_raw = {}
        image = torch.rand(1, 3, IMGSZ, IMGSZ, generator=torch.Generator().manual_seed(SEED)).to(TARGET_DEVICE)
        for target, source in sources:
            raw_sha = source["manifest"]["raw_checkpoint"]["sha256"]
            configuration = f"target-{target:.2f}x"
            job_folder = folder / "jobs" / configuration
            if target in EXPECTED_RAW_SHA256 and EXPECTED_RAW_SHA256[target].lower() != raw_sha:
                report["errors"].append({"target_speedup": target, "error": "SHA raw berbeda dari kondisi awal yang dikunci."})
                continue
            if target in completed and completed[target].get("raw_sha256") not in {None, raw_sha}:
                report["errors"].append({"target_speedup": target, "error": "SHA raw berbeda dari konfigurasi recovery tersimpan."})
                continue
            if raw_sha in by_raw and not (job_folder / "latest_state.json").is_file():
                canonical = by_raw[raw_sha]
                status = ("REACHED" if source["manifest"]["actual_gflops_speedup"] >= target - 1e-12 else "TARGET_NOT_REACHED")
                job = {**canonical, "configuration_id": configuration, "target_speedup": target, "target_status": status,
                       "alias_of": canonical["configuration_id"]}
                if canonical.get("recovered_checkpoint"):
                    job["recovered_checkpoint"] = {**canonical["recovered_checkpoint"],
                        "relative_path": f"../{canonical['configuration_id']}/recovered_fp32.pt"}
                recovery_build_alias_eda(job, job_folder)
                write_json(job_folder / "recovery_manifest.json", job)
            else:
                job = recovery_run_job(job_folder, target, source, dataset, data_record, train_yaml, train_fp, settings, image, method)
                if job.get("training_status") == "PASS":
                    by_raw[raw_sha] = job
            completed[target] = job
            report["jobs"] = [completed[key] for key in sorted(completed)]
            write_json(path, report)
            if job.get("interrupted"):
                break
        report["jobs"] = [completed[key] for key in sorted(completed)]
        report["status"] = workflow_status([job["status"] for job in report["jobs"]] + ["FAIL"] * bool(report["errors"]))
        recovery_suite_tables(report, folder)
    except (Exception, KeyboardInterrupt) as error:
        if previous and not bound:
            return {**previous, "resume_rejected": safe_error(error) or "Interrupted"}
        report["status"] = "PARTIAL" if report["jobs"] else "FAIL"
        report["errors"].append({"error": safe_error(error) or "Interrupted"})
    finally:
        if bound or previous is None:
            write_json(path, report)
    return report
