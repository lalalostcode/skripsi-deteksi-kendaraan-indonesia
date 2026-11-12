"""Notebook 05 runtime, embedded after the shared LAMP helpers."""
# ruff: noqa: F821 -- Runtime globals are supplied by preceding notebook cells.
# ruff: noqa: BLE001 -- Independent jobs persist stage-specific failures before continuing.


def gflops_load_source(path, provenance, variant):
    if sha256_file(path) != SOURCE_CHECKPOINT_SHA256[variant] or provenance["sha256"] != SOURCE_CHECKPOINT_SHA256[variant]:
        raise ValueError(f"Checkpoint sumber {variant} tidak cocok dengan SHA-256 yang dikunci.")
    core = YOLO(str(path), task="detect").model.to(TARGET_DEVICE).float().eval()
    core.end2end = True
    architecture_audit(core, variant)
    return core


def gflops_config():
    if MODE not in {"smoke", "full_val", "test"}:
        raise ValueError("MODE harus smoke, full_val, atau test.")
    if len(MODEL_VARIANTS) != len(set(MODEL_VARIANTS)) or not set(MODEL_VARIANTS) <= {"n", "s"} or not MODEL_VARIANTS:
        raise ValueError("MODEL_VARIANTS harus memuat n dan/atau s tanpa duplikasi.")
    if NOTEBOOK_VARIANT not in {"n", "s"} or MODEL_VARIANTS != [NOTEBOOK_VARIANT]:
        raise ValueError("Setiap notebook hanya menjalankan varian yang sesuai judulnya.")
    if MODE == "test" and STUDENT_LOCK_PATH is None:
        raise ValueError("MODE test memerlukan STUDENT_LOCK_PATH.")
    return {**configuration(), "mode": MODE, "targets": GFLOPS_TARGETS, "dataset_id": DATASET_ID,
            "expected_dataset_fingerprint": EXPECTED_DATASET_FINGERPRINT, "source_sha256": SOURCE_CHECKPOINT_SHA256,
            "eda_configurations": list(EDA_CONFIGURATIONS), "eda_detail_configurations": list(EDA_DETAIL_CONFIGURATIONS),
            "notebook_variant": NOTEBOOK_VARIANT,
            "comparison_candidate_paths": [str(path) for path in COMPARISON_CANDIDATE_PATHS],
            "implementation_sha256": IMPLEMENTATION_SHA256,
            "platform": PLATFORM, "run_diagnostic_controls": RUN_DIAGNOSTIC_CONTROLS,
            "trace_mode": "unfused dual-head FP32", "validation_mode": "one-to-many NMS FP32",
            "latency_mode": "unfused dual-head raw forward FP32, batch 1", "training_epochs": 0}


def gflops_latency(core, image):
    if not RUN_LATENCY_BENCHMARK:
        return {"status": "NOT_RUN", "reason": "Benchmark dimatikan."}
    try:
        return cuda_latency(core, image, LATENCY_WARMUP, LATENCY_REPEATS)
    except Exception as error:
        return {"status": "FAIL", "error": safe_error(error)}


def gflops_validation(row, checkpoint, dataset, folder):
    if dataset is None:
        return
    try:
        row["validation"] = evaluate_checkpoint(checkpoint, dataset, folder, "validation")
        row["validation_status"] = "PASS"
    except Exception as error:
        row["validation_status"] = "FAIL"
        row["errors"].append({"stage": "validation", "error": safe_error(error)})


def gflops_baseline(path, provenance, variant, image, folder, dataset):
    folder.mkdir(parents=True, exist_ok=True)
    previous_path = folder / "baseline_report.json"
    previous = json.loads(previous_path.read_text(encoding="utf-8")) if previous_path.is_file() else {}
    core = gflops_load_source(path, provenance, variant)
    forward_snapshot(core, image)
    measurements = model_measurements(core, IMGSZ)
    latency = previous.get("reserialized", {}).get("latency", {})
    measurements["latency"] = latency if latency.get("status") in {"PASS", "MEASURED"} else gflops_latency(core, image)
    source = previous.get("source") or {**measurements, "checkpoint_bytes": Path(path).stat().st_size,
                                       "validation_status": "NOT_RUN", "errors": []}
    if source["validation_status"] != "PASS":
        gflops_validation(source, path, dataset, folder / "source")
    saved = workflow_save_checkpoint(core, folder / "baseline_fp32.pt", provenance)
    reload_report = previous.get("reload")
    if not reload_report:
        reload_report = verify_fresh_reload(saved["path"], core, image, folder, timeout=RELOAD_TIMEOUT_SECONDS)
    clone = gflops_reload(saved["path"], core)
    if model_state_sha256(core) != model_state_sha256(clone):
        raise ValueError("Baseline hasil reload mengubah state sumber.")
    baseline = previous.get("reserialized") or {**measurements, "checkpoint_bytes": saved["file_bytes"],
                                                "validation_status": "NOT_RUN", "errors": []}
    source["latency"] = baseline["latency"] = measurements["latency"]
    if baseline["validation_status"] != "PASS":
        gflops_validation(baseline, saved["path"], dataset, folder / "reserialized")
    parity = "NOT_RUN"
    if dataset is not None:
        fields = ("map50_95", "map50", "precision", "recall")
        parity = ("PASS" if source["validation_status"] == baseline["validation_status"] == "PASS"
                  and all(abs(source["validation"][key] - baseline["validation"][key]) <= 1e-6 for key in fields)
                  else "FAIL")
    result = {"source": source, "reserialized": baseline, "validation_parity_status": parity,
              "state_sha256": model_state_sha256(core), "checkpoint": saved, "reload": reload_report,
              "architecture": architecture_audit(core, variant)["report"]}
    write_json(previous_path, result)
    return result


def gflops_reload(path, reference):
    model = YOLO(str(path), task="detect").model.to(TARGET_DEVICE).float().eval()
    model.end2end = reference.end2end
    return model


def gflops_make_graph(core, image, variant, ratio):
    return build_policy_pruner(core, image, "extended", ratio, variant)


def gflops_plan_from_history(history):
    return {item["root"]: sorted(item["removed_original_indices"]) for item in history}


def gflops_result_row(result, baseline, dataset):
    after = result["pruned"]
    before = baseline["reserialized"]
    latency = after.get("latency", {})
    before_latency = before["latency"].get("median_ms")
    median = latency.get("median_ms")
    checkpoint = result.get("saved_pruned", {})
    validation = after.get("validation", {})
    return {
        "model": result["model_name"], "model_variant": result["model_variant"],
        "configuration_id": result["configuration_id"], "policy": result["policy"], "ratio": result["ratio"],
        "actual_eligible_ratio": result.get("actual_eligible_channel_ratio"),
        "target_speedup": result.get("target_speedup"), "target_status": result.get("target_status"),
        "gflops_speedup": before["gflops"] / after["gflops"] if after.get("gflops") else None,
        "latency_speedup": before_latency / median if before_latency and median else None,
        **{key: after.get(key) for key in ("parameters", "gflops", "checkpoint_bytes")},
        **{key: validation.get(key) for key in ("map50_95", "map50", "precision", "recall")},
        "baseline_map50_95": (before.get("validation") or {}).get("map50_95"),
        "status": result.get("status", "PARTIAL"), "upload_status": result.get("upload_status", "NOT_RUN"),
        "raw_artifact_ref": result.get("raw_publication", {}).get("reference"),
        "raw_manifest_path": str(Path(checkpoint["path"]).parent / "raw_pruned_manifest.json") if checkpoint.get("path") else None,
        "latency_median_ms": median, "latency_p95_ms": latency.get("p95_ms"),
        "structural_status": result["structural_smoke_status"], "reload_status": result["reload_status"],
        "validation_status": result["validation_status"], "latency_status": latency.get("status", "NOT_RUN"),
        "eda_status": result["eda_status"], "baseline_validation_parity_status": baseline["validation_parity_status"],
        "checkpoint_path": checkpoint.get("path"), "checkpoint_sha256": checkpoint.get("sha256"),
        "baseline_checkpoint_path": baseline["checkpoint"]["path"],
        "baseline_checkpoint_sha256": baseline["checkpoint"]["sha256"],
        "evaluation_mode": MODE, "dataset_provenance_status": (dataset or {}).get("provenance_status"),
        "dataset_fingerprint": (dataset or {}).get("fingerprint"), **result["comparison"],
    }


def gflops_experiment(path, provenance, variant, image, baseline, dataset, folder, job):
    folder.mkdir(parents=True, exist_ok=True)
    workflow_bind(folder, "job_contract.json", {"id": job["id"], "policy": job["policy"],
        "plan": job.get("plan"), "caps": job.get("caps"), "ratio": job.get("ratio"),
        "source": provenance["sha256"], "baseline_state": baseline["state_sha256"]})
    report_path = folder / "pruning_manifest.json"
    ratio = job.get("ratio", 0.10)
    result = json.loads(report_path.read_text(encoding="utf-8")) if report_path.is_file() else {
        "format_version": 2, "stage": "raw_pruned", "configuration_id": job["id"],
        "model_name": f"YOLO26{variant}", "model_variant": variant,
        "policy": job["policy"], "ratio": ratio, "target_speedup": job.get("target"),
        "target_status": job.get("status"), "tier": job.get("tier"), "provenance": provenance,
        "baseline": baseline["reserialized"], "pruned": {}, "comparison": {}, "errors": [],
        "structural_smoke_status": "NOT_RUN", "reload_status": "NOT_RUN",
        "validation_status": "NOT_RUN", "eda_status": "NOT_RUN", "upload_status": "NOT_RUN"}
    write_json(report_path, result)
    stage = "structural_smoke"
    try:
        saved = result.get("saved_pruned")
        ready = result["structural_smoke_status"] == "PASS" and saved is not None
        if not ready:
            seed_environment(SEED)
            core = gflops_load_source(path, provenance, variant)
            if model_state_sha256(core) != baseline["state_sha256"]:
                raise ValueError("State kandidat sebelum pruning berbeda dari baseline.")
            original = deepcopy(core)
            protected = protected_policy_snapshot(core, variant)
            pruner, audit = gflops_make_graph(core, image, variant, ratio)
            inventory_before = measure_layer_inventory(core, image, audit, "extended")
            if job["policy"] == "extended":
                history = apply_policy_pruning(pruner, audit)
            else:
                history = apply_exact_gflops_plan(pruner, audit, job["plan"], caps=job.get("caps"))
            assert_layer_consistency(core)
            assert_policy_protected_unchanged(core, protected)
            forward_snapshot(core, image)
            result["oracle"] = verify_gflops_oracle(original, core, image, audit)
            del original
            selected = gflops_plan_from_history(history)
            repeat = gflops_load_source(path, provenance, variant)
            repeat_pruner, repeat_audit = gflops_make_graph(repeat, image, variant, ratio)
            if job["policy"] == "extended":
                replay = apply_policy_pruning(repeat_pruner, repeat_audit)
            else:
                replay = apply_exact_gflops_plan(repeat_pruner, repeat_audit, job["plan"], caps=job.get("caps"))
            if (gflops_plan_from_history(replay) != selected or model_state_sha256(core) != model_state_sha256(repeat)
                    or repeat_audit["report"]["channel_mapping"] != audit["report"]["channel_mapping"]):
                raise ValueError("Pengulangan tidak menghasilkan ledger dan state yang sama.")
            del repeat, repeat_pruner, repeat_audit
            result["state_sha256"] = model_state_sha256(core)
            result["reproducibility_status"] = "PASS"
            result["audit"] = audit["report"]
            result["groups"] = history
            result["pruned_group_count"] = len(history)
            result["selected_plan"] = selected
            result["actual_eligible_channel_ratio"] = sum(map(len, selected.values())) / audit["report"].get(
                "target_policy_eligible_channels", audit["independent_channels_before"])
            if job["policy"] != "extended":
                result["ratio"] = result["actual_eligible_channel_ratio"]
            write_json(folder / "channel_mapping.json", audit["report"]["channel_mapping"])
            write_json(folder / "channel_masks.json", audit["report"]["channel_masks"])
            write_json(folder / "dependency_audit.json", audit["report"])
            write_csv(folder / "lamp_scores.csv", audit["report"]["lamp_scores"])
            inventory_after = measure_layer_inventory(core, image, audit, "extended")
            write_json(folder / "inventories.json", {"before": inventory_before, "after": inventory_after})
            result["pruned"] = model_measurements(core, IMGSZ)
            if result["pruned"]["parameters"] >= result["baseline"]["parameters"]:
                raise ValueError("Parameter tidak berkurang.")
            if job.get("gflops") is not None and not math.isclose(result["pruned"]["gflops"], job["gflops"], rel_tol=1e-10):
                raise ValueError("GFLOPs kandidat berbeda dari profil pencarian.")
            result["structural_smoke_status"] = "PASS"
            stage = "save"
            saved = workflow_save_checkpoint(core, folder / "pruned_fp32.pt", provenance)
            result["saved_pruned"] = saved
            result["pruned"]["checkpoint_bytes"] = saved["file_bytes"]
            result["structure_signature"] = {
                "parameters": {name: list(value.shape) for name, value in core.named_parameters()},
                "conv_bn": {name: [module.in_channels, module.out_channels, module.groups]
                            if isinstance(module, torch.nn.Conv2d) else [module.num_features]
                            for name, module in core.named_modules()
                            if isinstance(module, (torch.nn.Conv2d, torch.nn.BatchNorm2d))}}
            write_json(report_path, result)
        else:
            checkpoint = folder / "pruned_fp32.pt"
            if sha256_file(checkpoint) != saved["sha256"]:
                raise ValueError("Raw checkpoint berubah; resume tidak boleh menimpanya.")
            saved["path"] = str(checkpoint.resolve())
            core = YOLO(str(checkpoint), task="detect").model.to(TARGET_DEVICE).float().eval()
            core.end2end = True
            if model_state_sha256(core) != result["state_sha256"]:
                raise ValueError("State raw resume berbeda dari ledger.")
            inventories = json.loads((folder / "inventories.json").read_text(encoding="utf-8"))
            inventory_before, inventory_after = inventories["before"], inventories["after"]
        stage = "reload"
        if result["reload_status"] != "PASS":
            result["reload"] = verify_fresh_reload(saved["path"], core, image, folder, timeout=RELOAD_TIMEOUT_SECONDS)
            reloaded = gflops_reload(saved["path"], core)
            if model_state_sha256(reloaded) != result["state_sha256"]:
                raise ValueError("State berubah setelah reload.")
            result["reload_status"] = "PASS"
            write_json(report_path, result)
        if result["validation_status"] != "PASS":
            stage = "validation"
            evaluated = {"validation_status": "NOT_RUN", "errors": result["errors"]}
            gflops_validation(evaluated, saved["path"], dataset, folder)
            result["validation_status"] = evaluated["validation_status"]
            if "validation" in evaluated:
                result["pruned"]["validation"] = evaluated["validation"]
            write_json(report_path, result)
        stage = "latency"
        if result["pruned"].get("latency", {}).get("status") not in {"PASS", "MEASURED"}:
            result["pruned"]["latency"] = gflops_latency(core, image)
        result["comparison"] = metric_comparison(result["baseline"], result["pruned"])
        write_json(report_path, result)
        stage = "eda"
        if result["eda_status"] != "PASS":
            try:
                result["eda"] = build_run_eda(result, folder, inventory_before, inventory_after)
                eda_status = result["eda"].get("status", "NOT_RUN")
                result["eda_status"] = "PASS" if eda_status == "MEASURED" else eda_status
            except Exception as error:
                result["eda_status"] = "FAIL"
                result["errors"].append({"stage": "eda", "error": safe_error(error)})
        audit_files = {name: sha256_file(folder / name) for name in ("dependency_audit.json", "inventories.json")
                       if (folder / name).is_file()}
        raw_manifest = {"format_version": 2, "stage": "raw_pruned", "model_variant": variant,
            "configuration_id": job["id"], "target_speedup": job.get("target"), "target_status": job.get("status"),
            "actual_gflops_speedup": baseline["reserialized"]["gflops"] / result["pruned"]["gflops"],
            "raw_checkpoint": {"relative_path": "pruned_fp32.pt", "sha256": saved["sha256"]},
            "baseline_checkpoint": {"relative_path": Path(os.path.relpath(baseline["checkpoint"]["path"], folder)).as_posix(),
                                    "artifact_path": "baseline_fp32.pt", "sha256": baseline["checkpoint"]["sha256"]},
            "state_sha256": result["state_sha256"], "structure_signature": result["structure_signature"],
            "source_baseline": provenance, "structural_status": result["structural_smoke_status"],
            "oracle_status": result.get("oracle", {}).get("status"), "reload_status": result["reload_status"],
            "evaluation_mode": MODE, "dataset": {key: (dataset or {}).get(key) for key in (
                "dataset_id", "fingerprint", "full_val_fingerprint", "train_fingerprint", "provenance_status",
                "class_names", "image_count", "split", "seed")},
            "baseline": baseline["reserialized"], "raw_pruned": result["pruned"], "audit_files": audit_files,
            "validation_status": result["validation_status"], "baseline_validation_parity_status": baseline["validation_parity_status"],
            "seed": SEED, "imgsz": IMGSZ, "platform": PLATFORM, "implementation_sha256": IMPLEMENTATION_SHA256}
        write_json(folder / "raw_pruned_manifest.json", raw_manifest)
        (folder / "RESTORE.md").write_text("Load native DetectionModel FP32 from pruned_fp32.pt; verify raw_pruned_manifest.json SHA-256. Do not rebuild from baseline YAML. Raw is the shared KD/fine-tuning initial state.", encoding="utf-8")
        stage = "upload"
        if WANDB_TRACKING:
            with workflow_tracking(folder, variant, job.get("target") or 1.0, source_ref=model_spec(variant)["artifact"]) as run:
                publication = workflow_publish(run, folder, "raw-pruned", {
                    "pruned_fp32.pt": saved["path"], "baseline_fp32.pt": baseline["checkpoint"]["path"],
                    "raw_pruned_manifest.json": folder / "raw_pruned_manifest.json",
                    **{name: folder / name for name in audit_files}},
                    metadata_info={"stage": "raw_pruned", "target_status": result["target_status"],
                                   "raw_sha256": saved["sha256"]}, aliases=["raw-pruned", "latest"])
                result["raw_publication"] = publication
                result["upload_status"] = "PASS" if publication["status"] == "CONFIRMED" else "NOT_RUN"
                write_json(report_path, result)
                row = gflops_result_row(result, baseline, dataset)
                run.log({"configuration": wandb.Table(dataframe=pd.DataFrame([row]))})
                workflow_publish(run, folder, "results", workflow_report_files(folder), kind="results")
    except Exception as error:
        result[stage + "_status"] = "FAIL"
        result["errors"].append({"stage": stage, "error": safe_error(error), "traceback": safe_error(traceback.format_exc())})
    statuses = [result.get(key, "NOT_RUN") for key in ("structural_smoke_status", "reload_status", "validation_status", "eda_status")]
    if job.get("status") == "TARGET_NOT_REACHED":
        statuses.append("PARTIAL")
    if RUN_LATENCY_BENCHMARK:
        statuses.append("PASS" if result["pruned"].get("latency", {}).get("status") in {"PASS", "MEASURED"} else "NOT_RUN")
    if WANDB_TRACKING:
        statuses.append(result["upload_status"])
    result["status"] = workflow_status(statuses, useful=bool(result.get("saved_pruned")))
    write_json(report_path, result)
    row = gflops_result_row(result, baseline, dataset)
    write_json(folder / "metrics.json", row)
    print(variant, job["id"], result["status"], row["structural_status"], row["validation_status"], flush=True)
    return result, row


def gflops_p5_comparison(results, folder):
    by_id = {result["configuration_id"]: result for result in results}
    summaries, classes = [], []
    for label, reference in (("p5-only", None), ("p5-protected", "extended-pr0.1")):
        if label not in by_id:
            continue
        result = by_id[label]
        before = by_id[reference]["pruned"] if reference else result["baseline"]
        after = result["pruned"]
        summaries.append({"contrast": label, "reference": reference or "baseline",
                          "reference_gflops": before.get("gflops"), "gflops": after.get("gflops"),
                          "validation_status": result["validation_status"],
                          **(metric_comparison(before, after) if before.get("checkpoint_bytes")
                             and after.get("checkpoint_bytes") else {})})
        base_classes = {row["class_id"]: row for row in before.get("validation", {}).get("per_class", [])}
        for row in after.get("validation", {}).get("per_class", []):
            prior = base_classes.get(row["class_id"], {})
            before_ap, after_ap = prior.get("ap50_95"), row.get("ap50_95")
            classes.append({"contrast": label, "reference": reference or "baseline", **row,
                            "reference_ap50_95": before_ap,
                            "delta_ap50_95_pp": 100 * (after_ap - before_ap)
                            if after_ap is not None and before_ap is not None else None})
    write_csv(folder / "p5_contrasts.csv", summaries)
    write_csv(folder / "p5_per_class.csv", classes)


def gflops_run_model(variant, folder, dataset):
    folder.mkdir(parents=True, exist_ok=True)
    manifest = folder / "model_manifest.json"
    suite = {"model_name": f"YOLO26{variant}", "model_variant": variant, "source": {},
             "baseline": {}, "baseline_report": {}, "results": [], "rows": [], "target_search": {},
             "status": "NOT_RUN", "errors": []}
    rows, results = suite["rows"], suite["results"]
    try:
        path, provenance = obtain_baseline(folder, variant)
        suite["source"] = provenance
        image = torch.rand(1, 3, IMGSZ, IMGSZ, generator=torch.Generator().manual_seed(SEED)).to(TARGET_DEVICE)
        baseline = gflops_baseline(path, provenance, variant, image, folder / "baseline", dataset)
        suite.update(baseline=baseline["reserialized"], baseline_report=baseline)
        for kind in ("source", "reserialized"):
            metrics = baseline[kind]
            rows.append({"model": f"YOLO26{variant}", "model_variant": variant,
                         "configuration_id": f"baseline-{kind}", "policy": "baseline", "ratio": 0,
                         "gflops_speedup": 1.0, "latency_speedup": 1.0 if metrics["latency"]["status"] == "PASS" else None,
                         **{key: metrics[key] for key in ("parameters", "gflops", "checkpoint_bytes")},
                         **{key: metrics.get("validation", {}).get(key) for key in ("map50_95", "map50", "precision", "recall")},
                         "latency_median_ms": metrics["latency"].get("median_ms"),
                         "latency_p95_ms": metrics["latency"].get("p95_ms"),
                         "structural_status": "PASS", "reload_status": "PASS" if kind == "reserialized" else "NOT_RUN",
                         "validation_status": metrics["validation_status"], "latency_status": metrics["latency"]["status"],
                         "eda_status": "NOT_RUN"})
        write_json(manifest, suite)

        def run(job):
            try:
                result, row = gflops_experiment(path, provenance, variant, image, baseline, dataset, folder / job["id"], job)
            except Exception as error:
                result = {"configuration_id": job["id"], "model_name": f"YOLO26{variant}", "model_variant": variant,
                          "policy": job["policy"], "ratio": job.get("ratio"), "target_speedup": job.get("target"),
                          "target_status": job.get("status"), "status": "FAIL", "baseline": baseline["reserialized"],
                          "pruned": {}, "comparison": {}, "structural_smoke_status": "FAIL", "reload_status": "NOT_RUN",
                          "validation_status": "NOT_RUN", "eda_status": "NOT_RUN", "errors": [{"error": safe_error(error)}]}
                row = gflops_result_row(result, baseline, dataset)
            results.append(result)
            rows.append(row)
            suite["status"] = "PARTIAL"
            write_json(manifest, suite)
            write_csv(folder / "metrics.csv", rows)
            import gc
            gc.collect()
            if TARGET_DEVICE.type == "cuda":
                torch.cuda.empty_cache()
            return result

        if RUN_DIAGNOSTIC_CONTROLS:
            controls = {ratio: run({"id": f"extended-pr{ratio:g}", "policy": "extended", "ratio": ratio})
                        for ratio in PRUNING_RATIOS}
        else:
            controls = {}
        graph_source = gflops_load_source(path, provenance, variant)
        _, graph_audit = gflops_make_graph(graph_source, image, variant, 0.10)
        if controls.get(0.10, {}).get("structural_smoke_status") == "PASS":
            try:
                for name, plan in p5_control_plans(graph_audit, controls[0.10]["groups"]).items():
                    run({"id": name, "policy": name, "plan": plan})
                gflops_p5_comparison(results, folder)
            except Exception as error:
                suite["errors"].append({"stage": "p5", "error": safe_error(error)})

        def profile(plan):
            probe = gflops_load_source(path, provenance, variant)
            pruner, audit = gflops_make_graph(probe, image, variant, 0.10)
            apply_exact_gflops_plan(pruner, audit, plan)
            return model_measurements(probe, IMGSZ)["gflops"]

        cache_path = folder / "profile_cache.json"
        cache = json.loads(cache_path.read_text(encoding="utf-8")) if cache_path.is_file() else {}
        def progress(values, points):
            write_json(cache_path, values)
            write_json(folder / "profile_progress.json", points)
        search = search_gflops_targets(graph_audit, baseline["reserialized"]["gflops"], profile,
                                      tuple(GFLOPS_TARGETS), profile_cache=cache, progress_callback=progress)
        suite["target_search"] = search
        write_json(folder / "target_search.json", search)
        write_json(manifest, suite)
        unique = {}
        for target in search["targets"]:
            label = f"target-{target['target']:.2f}x"
            if target["status"] == "FAIL":
                failed = {"configuration_id": label, "model_name": f"YOLO26{variant}", "model_variant": variant,
                    "policy": "gflops-target", "ratio": None, "target_speedup": target["target"],
                    "target_status": "FAIL", "baseline": baseline["reserialized"], "pruned": {}, "comparison": {},
                    "structural_smoke_status": "FAIL", "reload_status": "NOT_RUN", "validation_status": "NOT_RUN",
                    "eda_status": "NOT_RUN", "status": "FAIL", "errors": [{"stage": "search", "error": target["error"]}]}
                results.append(failed)
                rows.append(gflops_result_row(failed, baseline, dataset))
                write_json(manifest, suite)
                continue
            key = gflops_plan_key(target["plan"])
            if key in unique:
                original = unique[key]
                alias = deepcopy(original)
                alias.update(configuration_id=label, target_speedup=target["target"], target_status=target["status"],
                             artifact_configuration_id=original["configuration_id"])
                if target["status"] != "REACHED":
                    alias["status"] = "PARTIAL"
                results.append(alias)
                rows.append(gflops_result_row(alias, baseline, dataset))
                write_json(manifest, suite)
                continue
            result = run({**target, "id": label, "policy": "gflops-target", "ratio": .10})
            unique[key] = result
        suite["status"] = workflow_status([result.get("status", "FAIL") for result in results], useful=bool(baseline))
    except Exception as error:
        suite["errors"].append({"stage": "model", "error": safe_error(error), "traceback": safe_error(traceback.format_exc())})
        suite["status"] = "PARTIAL" if suite["baseline"] or results else "FAIL"
        for target in GFLOPS_TARGETS:
            label = f"target-{target:.2f}x"
            if not any(row.get("configuration_id") == label for row in rows):
                rows.append({"model": f"YOLO26{variant}", "model_variant": variant, "configuration_id": label,
                             "policy": "gflops-target", "target_speedup": target, "target_status": "NOT_RUN",
                             "status": "NOT_RUN", "structural_status": "NOT_RUN", "reload_status": "NOT_RUN",
                             "validation_status": "NOT_RUN", "latency_status": "NOT_RUN", "eda_status": "NOT_RUN"})
    finally:
        write_json(manifest, suite)
        write_csv(folder / "metrics.csv", rows)
    return suite


def gflops_execute(folder):
    report = {"format_version": 2, "status": "NOT_RUN", "mode": MODE, "models": [], "errors": []}
    manifest = folder / "suite_manifest.json"
    prior = json.loads(manifest.read_text(encoding="utf-8")) if manifest.is_file() else None
    bound = False
    dataset = None
    try:
        expected = EXPECTED_DATASET_FINGERPRINT
        portable = None
        if DATASET_MANIFEST_PATH is not None:
            portable = json.loads(Path(DATASET_MANIFEST_PATH).read_text(encoding="utf-8"))
            if portable.get("dataset_id") != DATASET_ID or portable.get("provenance_status") != "VERIFIED":
                raise ValueError("Manifest dataset sumber belum VERIFIED atau ID berbeda.")
            expected = portable["val_fingerprint"]
        dataset = gflops_prepare_dataset(MODE, DATA_YAML, DATASET_ROOT, folder / "dataset_checks" / uuid.uuid4().hex[:8], seed=SEED,
                                         expected_fingerprint=expected, dataset_id=DATASET_ID)
        dataset_record = None
        if dataset is not None and (EXPORT_DATASET_MANIFEST or DATASET_MANIFEST_ONLY):
            dataset_record, _, _ = workflow_dataset_manifest(dataset, DATA_YAML, DATASET_ROOT, Path(dataset["manifest_path"]).parent)
            if portable is not None and dataset_record["train_fingerprint"] != portable["train_fingerprint"]:
                raise ValueError("Fingerprint train Filkom berbeda dari sumber.")
            dataset["train_fingerprint"] = dataset_record["train_fingerprint"]
        workflow_bind(folder, "resume_contract.json", {
            "implementation": IMPLEMENTATION_SHA256, "variant": NOTEBOOK_VARIANT, "platform": PLATFORM,
            "targets": list(GFLOPS_TARGETS), "imgsz": IMGSZ, "batch": BATCH, "seed": SEED,
            "diagnostics": RUN_DIAGNOSTIC_CONTROLS, "diagnostic_ratios": list(PRUNING_RATIOS) if RUN_DIAGNOSTIC_CONTROLS else [],
            "dataset_fingerprint": (dataset or {}).get("fingerprint"), "mode": MODE,
            "train_fingerprint": (dataset or {}).get("train_fingerprint"),
            "dataset_provenance_status": (dataset or {}).get("provenance_status"),
            "device": str(TARGET_DEVICE), "gpu": ENVIRONMENT.get("gpu"), "cuda_build": ENVIRONMENT.get("cuda_build"),
            "packages": ENVIRONMENT.get("packages"), "validation_workers": WORKERS,
            "run_latency": RUN_LATENCY_BENCHMARK, "latency_warmup": LATENCY_WARMUP, "latency_repeats": LATENCY_REPEATS})
        bound = True
        write_json(folder / "config.json", gflops_config())
        if dataset_record is not None:
            write_json(folder / "dataset_manifest.json", dataset_record)
        if DATASET_MANIFEST_ONLY:
            report["reason"] = "Preflight dataset saja; pruning tidak dijalankan."
            return report
        if MODE == "test":
            report["test"] = gflops_run_locked_test(Path(STUDENT_LOCK_PATH), dataset, folder / "test",
                                                    checkpoint_path=TEST_STUDENT_CHECKPOINT_PATH,
                                                    baseline_checkpoint_path=TEST_BASELINE_CHECKPOINT_PATH)
            report["status"] = report["test"]["status"]
            return report
        for variant in MODEL_VARIANTS:
            model = gflops_run_model(variant, folder / f"yolo26{variant}", dataset)
            report["models"].append(model)
            write_json(manifest, report)
        rows = [row for model in report["models"] for row in model["rows"]]
        write_csv(folder / "metrics.csv", rows)
        if any(model["results"] for model in report["models"]):
            plot_model_comparison(report["models"], folder / "comparison")
        report["status"] = workflow_status([model["status"] for model in report["models"]],
                                            useful=any(model["baseline"] for model in report["models"]))
        report["evaluation_status"] = workflow_status([row.get("validation_status", "NOT_RUN") for row in rows])
        report["selection"] = {"status": "NOT_RUN", "selected": None,
                                "reason": "Gabungkan full validation N/S; hanya raw dengan metrik lengkap yang eligible."}
        gflops_export_candidates(report, dataset, folder)
    except (Exception, KeyboardInterrupt) as error:
        if prior is not None and not bound:
            write_json(folder / "resume_rejection.json", {"status": "FAIL", "error": safe_error(error)})
            return {**prior, "status": "PARTIAL", "resume_rejected": safe_error(error)}
        if isinstance(error, KeyboardInterrupt) and not report["models"]:
            for variant in MODEL_VARIANTS:
                partial = folder / f"yolo26{variant}" / "model_manifest.json"
                if partial.is_file():
                    report["models"].append(json.loads(partial.read_text(encoding="utf-8")))
        report["status"] = "PARTIAL" if report["models"] else "FAIL"
        report["errors"].append({"error": safe_error(error), "traceback": safe_error(traceback.format_exc())})
    finally:
        if bound or prior is None:
            write_json(manifest, report)
    return report


def gflops_export_candidates(report, dataset, folder):
    """Export the measured rows and a cross-session comparison contract."""
    folder = Path(folder).resolve()
    rows = []
    for model in report["models"]:
        for source in model["rows"]:
            row = dict(source)
            for key in ("checkpoint", "baseline_checkpoint"):
                if row.get(key + "_path"):
                    row[key + "_relative_path"] = Path(row[key + "_path"]).resolve().relative_to(folder).as_posix()
            rows.append(row)
    contract = {"implementation_sha256": IMPLEMENTATION_SHA256, "source_sha256": SOURCE_CHECKPOINT_SHA256,
                "platform": PLATFORM,
                "seed": SEED, "imgsz": IMGSZ, "validation_batch": BATCH,
                "validation_mode": "one-to-many NMS FP32", "latency_mode": "unfused dual-head FP32 batch 1",
                "latency_warmup": LATENCY_WARMUP, "latency_repeats": LATENCY_REPEATS,
                **{key: ENVIRONMENT.get(key) for key in ("gpu", "cuda_build", "packages", "flops_method")}}
    record = {"format_version": 2, "mode": MODE, "status": report["status"],
              "evaluation_status": report["evaluation_status"],
              "model_variants": [model["model_variant"] for model in report["models"]],
              "dataset_provenance_status": (dataset or {}).get("provenance_status"),
              "dataset_fingerprint": (dataset or {}).get("fingerprint"),
              "comparison_contract": contract, "rows": rows}
    write_json(folder / "student_candidates.json", record)


def gflops_finalize_student(folder, comparison_paths):
    """Can run again after attaching the other notebook's complete output folder."""
    if MODE != "full_val" or not comparison_paths:
        return {"status": "NOT_RUN", "selected": None,
                "reason": "Isi COMPARISON_CANDIDATE_PATHS dengan ekspor full_val varian lainnya."}
    selection = gflops_select_candidate_files([Path(folder) / "student_candidates.json", *comparison_paths])
    if selection.get("selected"):
        if ":v" not in (selection["selected"].get("raw_artifact_ref") or ""):
            return {"status": "NOT_RUN", "selected": None, "provisional": selection["selected"],
                    "reason": "Publikasi raw terpilih belum dikonfirmasi; retry upload sebelum mengunci student."}
        gflops_lock_student(selection, Path(folder) / "student_lock.json")
    write_json(Path(folder) / "comparison/final_selection.json", selection)
    return selection
