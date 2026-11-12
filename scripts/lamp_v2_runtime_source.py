"""Authoring source embedded verbatim in the standalone Kaggle notebook.

Shared scientific helpers are embedded from the preserved V1 notebook by the generator.
This file is never imported by a notebook at runtime.
"""
# ruff: noqa: F821, SIM905 -- Runtime globals and public config whitelist are embedded in notebook cells.

# %% configuration_helpers
# Backward-compatible N helper aliases; public inputs are the per-model mappings.
WANDB_ARTIFACT = MODEL_ARTIFACTS["n"]
LOCAL_CHECKPOINT_PATH = None
LOCAL_EXPECTED_SHA256 = None


def model_spec(model_variant="n"):
    if model_variant not in {"n", "s"}:
        raise ValueError("Model variant harus n atau s.")
    local = LOCAL_CHECKPOINT_PATHS.get(model_variant)
    expected_hash = LOCAL_EXPECTED_SHA256S.get(model_variant)
    if model_variant == "n" and LOCAL_CHECKPOINT_PATH is not None:
        local, expected_hash = LOCAL_CHECKPOINT_PATH, LOCAL_EXPECTED_SHA256
    return {"model_variant": model_variant, "model_name": f"YOLO26{model_variant}",
            "artifact": MODEL_ARTIFACTS[model_variant], "local_checkpoint": local,
            "expected_sha256": expected_hash}


def policies_to_run(policy):
    if policy not in {"conservative", "extended", "both"}:
        raise ValueError("PRUNING_POLICY harus conservative, extended, atau both.")
    return ["conservative", "extended"] if policy == "both" else [policy]


def configuration(policy=None, ratio=None, model_variant=None):
    keys = ("MODEL_VARIANTS MODEL_ARTIFACTS LOCAL_CHECKPOINT_PATHS LOCAL_EXPECTED_SHA256S CHECKPOINT_SELECTOR "
            "DATASET_ROOT DATA_YAML OUTPUT_DIR PRUNING_POLICY PRUNING_RATIOS GLOBAL_PRUNING "
            "ITERATIVE_STEPS MAX_PRUNING_RATIO ROUND_TO IMPORTANCE_P SEED IMGSZ DEVICE "
            "RUN_FULL_VALIDATION RUN_LATENCY_BENCHMARK BATCH WORKERS INSTALL_DEPENDENCIES "
            "WANDB_TRACKING UPLOAD_WANDB_ARTIFACTS LATENCY_WARMUP LATENCY_REPEATS "
            "RELOAD_TIMEOUT_SECONDS RECOVERY_ENABLED FINETUNING_ENABLED KD_ENABLED "
            "EDA_DPI EDA_MAX_LAYERS_PER_PAGE EDA_DISPLAY_RATIO").split()
    def serializable(value):
        if isinstance(value, Path):
            return str(value)
        if isinstance(value, dict):
            return {key: serializable(item) for key, item in value.items()}
        if isinstance(value, (tuple, list)):
            return [serializable(item) for item in value]
        return value
    result = {key: serializable(globals()[key]) for key in keys}
    if policy is not None:
        result["policy"] = policy
    if ratio is not None:
        result["ratio"] = ratio
    if model_variant is not None:
        result["selected_model"] = serializable(model_spec(model_variant))
    return result


def validate_configuration():
    if not MODEL_VARIANTS or len(set(MODEL_VARIANTS)) != len(MODEL_VARIANTS):
        raise ValueError("MODEL_VARIANTS harus nonkosong dan tanpa duplikat.")
    for model_variant in MODEL_VARIANTS:
        artifact = model_spec(model_variant)["artifact"]
        if artifact.count("/") != 2 or ":" not in artifact:
            raise ValueError("Artifact wajib entity/project/artifact:version.")
    policies_to_run(PRUNING_POLICY)
    if not PRUNING_RATIOS or len(set(PRUNING_RATIOS)) != len(PRUNING_RATIOS):
        raise ValueError("PRUNING_RATIOS harus nonkosong dan tanpa duplikat.")
    if any(isinstance(r, bool) or not isinstance(r, (int, float)) or not math.isfinite(r)
           or not 0 < r <= MAX_PRUNING_RATIO for r in PRUNING_RATIOS):
        raise ValueError("Setiap ratio wajib finite dan 0 < ratio <= MAX_PRUNING_RATIO.")
    if MAX_PRUNING_RATIO != 0.50 or ROUND_TO is not None or not GLOBAL_PRUNING:
        raise ValueError("V2 memakai global ranking, cap 0.50, dan ROUND_TO=None.")
    if ITERATIVE_STEPS != 1 or IMPORTANCE_P != 2:
        raise ValueError("V2 hanya mendukung one-shot dengan p=2.")
    if any((RECOVERY_ENABLED, FINETUNING_ENABLED, KD_ENABLED)):
        raise ValueError("Recovery, fine-tuning, dan KD wajib disabled.")
    if IMGSZ < 32 or IMGSZ % 32 or BATCH < 1 or WORKERS < 0:
        raise ValueError("IMGSZ kelipatan 32, BATCH positif, WORKERS nonnegatif.")
    if EDA_DPI < 72 or EDA_MAX_LAYERS_PER_PAGE < 1 or not 0 < EDA_DISPLAY_RATIO <= 0.5:
        raise ValueError("Konfigurasi EDA DPI/page/ratio invalid.")
    if LATENCY_WARMUP < 1 or LATENCY_REPEATS < 1 or RELOAD_TIMEOUT_SECONDS < 1:
        raise ValueError("Jumlah pengukuran dan timeout wajib positif.")


def write_csv(path, rows):
    if not rows:
        return
    keys = list(dict.fromkeys(key for row in rows for key in row))
    with Path(path).open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=keys)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: json.dumps(value, ensure_ascii=False) if isinstance(value, (list, dict))
                             else value for key, value in row.items()})


def model_state_sha256(core):
    """Buktikan setiap konfigurasi dimulai dari bobot/buffer FP32 yang identik."""
    digest = hashlib.sha256()
    for name, value in sorted(core.state_dict().items()):
        tensor = value.detach().cpu().contiguous()
        digest.update(name.encode())
        digest.update(str((str(tensor.dtype), tuple(tensor.shape))).encode())
        digest.update(tensor.numpy().tobytes())
    return digest.hexdigest()


def metric_comparison(before, after):
    result = {}
    for key, label in (("parameters", "parameter"), ("gflops", "gflops"),
                       ("checkpoint_bytes", "checkpoint")):
        base, pruned = before.get(key), after.get(key)
        result[label + "_reduction_pct"] = 100 * (base - pruned) / base if base else None
    base = (before.get("validation") or {}).get("map50_95")
    pruned = (after.get("validation") or {}).get("map50_95")
    delta = pruned - base if base is not None and pruned is not None else None
    result.update({"delta_map50_95": delta, "delta_map50_95_pp": 100 * delta if delta is not None else None,
                   "relative_map50_95_pct": 100 * delta / base if delta is not None and base else None})
    return result


def lamp_v2_display_name(policy, ratio_tag, model_variant, recovery_epochs=None):
    model_spec(model_variant)
    policy_suffix = "_c" if policy == "conservative" else ""
    recovery_suffix = f"_ft{recovery_epochs}" if recovery_epochs is not None else ""
    return f"prun_lamp2_yolo26{model_variant}{policy_suffix}{recovery_suffix}_pruned-{ratio_tag}"


def run_identity(policy, ratio, suite_id, model_variant="n"):
    tag = f"pr{ratio:g}"
    return {"name": lamp_v2_display_name(policy, tag, model_variant),
            "id": uuid.uuid4().hex, "resume": "never", "group": f"lamp-v2-{suite_id}",
            "job_type": "lamp-structured-pruning-v2",
            "tags": [f"yolo26{model_variant}", "lamp", "v2", "structured", "pruned", policy,
                     "no-recovery", tag, f"seed{SEED}"]}


def safe_error(error):
    message = str(error)
    secret = os.environ.get("WANDB_API_KEY")
    return message.replace(secret, "[REDACTED]") if secret else message


# %% acquisition_helpers
def wandb_session_login():
    if "WANDB_API_KEY" not in os.environ and Path("/kaggle").exists() and not wandb.api.api_key:
        from kaggle_secrets import UserSecretsClient
        key = UserSecretsClient().get_secret("WANDB_API_KEY")
        wandb.login(key=key, relogin=False)
        del key


def obtain_baseline(suite_dir, model_variant="n"):
    """Ambil sumber satu kali via API; setiap run eksperimen dibuat terpisah."""
    spec = model_spec(model_variant)
    if spec["local_checkpoint"] is not None:
        path = select_checkpoint(spec["local_checkpoint"], CHECKPOINT_SELECTOR)
        return path, {**checkpoint_provenance(path, spec["artifact"], expected_hash=spec["expected_sha256"]),
                      "model_variant": model_variant}
    if not WANDB_TRACKING:
        raise ValueError("Isi LOCAL_CHECKPOINT_PATH atau aktifkan WANDB_TRACKING untuk sumber W&B.")
    wandb_session_login()
    artifact = wandb.Api(timeout=30).artifact(spec["artifact"], type="model")
    root = artifact.download(root=str(Path(suite_dir) / "baseline" / "source_artifact"))
    path = select_checkpoint(root, CHECKPOINT_SELECTOR)
    provenance = checkpoint_provenance(path, spec["artifact"], resolved_artifact=artifact.qualified_name)
    provenance.update({"artifact_digest": artifact.digest, "artifact_id": artifact.id,
                       "artifact_version": artifact.version,
                       "artifact_relative_path": path.relative_to(Path(root).resolve()).as_posix(),
                       "model_variant": model_variant})
    return path, provenance


# %% tracking_helpers
def _init_tracking(result, run_dir):
    wandb_session_login()
    variant = result.get("model_variant", "n")
    config = configuration(result["policy"], result["ratio"], variant)
    config.update({"source_checkpoint_sha256": result["provenance"]["sha256"],
                   "dataset_yaml_sha256": result.get("dataset", {}).get("source_yaml_sha256"),
                   "versions": result["environment"]["packages"],
                   "validation_protocol": validation_arguments(DATA_YAML, run_dir, "protocol")})
    return wandb.init(entity=WANDB_ENTITY, project=WANDB_PROJECT, dir=str(run_dir), reinit="create_new",
                      config=config, **run_identity(result["policy"], result["ratio"], result["suite_id"], variant))


def tracking_metrics(result):
    before, after = result.get("baseline", {}), result.get("pruned", {})
    metric = {"pruning/target_ratio": result["ratio"],
              "smoke/structural_pass": int(result["structural_smoke_status"] == "PASS"),
              "smoke/validation_pass": int(result["validation_status"] == "PASS")}
    for label, values in (("baseline", before), ("pruned", after)):
        for key in ("parameters", "gflops", "checkpoint_bytes"):
            if values.get(key) is not None:
                metric[f"{label}/{key}"] = values[key]
        for key in ("map50_95", "map50", "precision", "recall"):
            value = (values.get("validation") or {}).get(key)
            if value is not None:
                metric[f"{label}/{key}"] = value
    for key in ("parameter_reduction_pct", "gflops_reduction_pct"):
        value = result.get("comparison", {}).get(key)
        if value is not None:
            metric["pruning/" + key] = value
    if result.get("actual_eligible_channel_ratio") is not None:
        metric["pruning/actual_eligible_ratio"] = result["actual_eligible_channel_ratio"]
    delta = result.get("comparison", {}).get("delta_map50_95")
    if delta is not None:
        metric["delta/map50_95"] = delta
        metric["delta/map50_95_pp"] = 100 * delta
    return metric


def track_result(result, run_dir):
    """Kegagalan jaringan tidak menghapus bukti lokal atau mengubah validitas struktur."""
    folder = Path(run_dir)
    report = {"status": "NOT_RUN", "uploads": [], "errors": []}
    run = None
    try:
        if not WANDB_TRACKING:
            report["reason"] = "WANDB_TRACKING=False"
            return report
        run = _init_tracking(result, folder)
        report["run_id"] = run.id
        report["run_url"] = run.url
        # Catat lineage sumber; hanya menggunakannya, tanpa mengubah artifact baseline.
        variant = result.get("model_variant", "n")
        run.use_artifact(model_spec(variant)["artifact"], type="model")
        run.log(tracking_metrics(result))
        run.summary.update({"policy": result["policy"], "ratio": result["ratio"],
                            "structural_smoke_status": result["structural_smoke_status"],
                            "validation_status": result["validation_status"],
                            "source_checkpoint_sha256": result["provenance"]["sha256"]})
        if UPLOAD_WANDB_ARTIFACTS:
            ratio_name = f"{result['ratio']:g}".replace(".", "p")
            stem = f"yolo26{variant}-lamp-v2-{result['policy']}-pr{ratio_name}-{result['suite_id']}"
            metadata_info = {"model_variant": variant, "policy": result["policy"], "ratio": result["ratio"], "seed": SEED,
                             "source": result["provenance"], "structural_smoke_status": result["structural_smoke_status"],
                             "validation_status": result["validation_status"]}
            for kind in ("model", "report"):
                artifact = wandb.Artifact(stem + "-" + kind, type=kind, metadata=metadata_info)
                if kind == "model":
                    if not (folder / "pruned_fp32.pt").is_file():
                        continue
                    artifact.add_file(str(folder / "pruned_fp32.pt"))
                    artifact.add_file(str(folder / "RESTORE.md"))
                else:
                    for path in sorted(folder.rglob("*")):
                        relative = path.relative_to(folder)
                        if (path.is_file() and relative.parts[0] != "wandb"
                                and path.suffix in {".json", ".csv", ".md", ".png", ".svg"}):
                            artifact.add_file(str(path), name=relative.as_posix())
                logged = run.log_artifact(artifact)
                logged.wait(timeout=300)
                report["uploads"].append({"type": kind, "status": "CONFIRMED", "name": logged.qualified_name})
        report["status"] = "PASS"
    except Exception as error:  # noqa: BLE001 -- Preserve local evidence for any SDK/network failure.
        report["status"] = "FAIL"
        report["errors"].append({"type": type(error).__name__, "message": safe_error(error)})
        print("W&B tracking/upload gagal; hasil lokal dipertahankan:", safe_error(error))
    finally:
        if run is not None:
            try:
                run.finish(exit_code=0 if report["status"] == "PASS" else 1)
            except Exception as error:  # noqa: BLE001 -- Finish failures must also be persisted locally.
                report["status"] = "FAIL"
                report["errors"].append({"type": type(error).__name__, "message": safe_error(error)})
        write_json(folder / "tracking_report.json", report)
    return report


# %% experiment_helpers
def _load_source_core(source_path, expected_hash):
    if sha256_file(source_path) != expected_hash:
        raise ValueError("Checkpoint sumber berubah antar konfigurasi.")
    return YOLO(str(source_path), task="detect").model.to(TARGET_DEVICE).float().eval()


def _prepare_baseline(source_path, provenance, suite_dir, model_variant="n"):
    folder = Path(suite_dir) / "baseline"
    core = _load_source_core(source_path, provenance["sha256"])
    audit = architecture_audit(core, model_variant)
    image = torch.zeros(1, 3, IMGSZ, IMGSZ, device=TARGET_DEVICE, dtype=torch.float32)
    forward_snapshot(core, image)
    measurements = model_measurements(core, IMGSZ)
    metadata_info = {"model_variant": model_variant, "model_name": f"YOLO26{model_variant}",
                     "state_sha256": model_state_sha256(core), "names": ordered_names(core.names),
                     "architecture": audit["report"]}
    payload = torch.load(source_path, map_location="cpu", weights_only=False)
    metadata_info["source_checkpoint_metadata"] = {
        key: payload.get(key) for key in ("epoch", "date", "version", "best_fitness")
        if isinstance(payload.get(key), (str, int, float, bool, type(None)))
    }
    source_args = payload.get("train_args", {}) or {}
    metadata_info["source_checkpoint_metadata"]["training"] = {
        key: source_args.get(key) for key in ("data", "model", "seed", "epochs", "imgsz", "task")
        if isinstance(source_args.get(key), (str, int, float, bool, type(None)))
    }
    del payload
    saved = save_fp32_checkpoint(core, folder / "baseline_fp32.pt", provenance)
    measurements["checkpoint_bytes"] = saved["file_bytes"]
    validation_report = {"status": "NOT_RUN", "reason": "dataset unavailable or validation disabled"}
    write_json(folder / "validation_report.json", validation_report)
    try:
        dataset = prepare_validation_dataset(DATA_YAML, DATASET_ROOT, folder) if RUN_FULL_VALIDATION else None
        if dataset is not None:
            assert_class_match(core.names, dataset["class_names"])
            measurements["validation"] = evaluate_checkpoint(folder / "baseline_fp32.pt", dataset, folder, "validation")
            validation_report = {"status": "PASS", "source_yaml_sha256": dataset["source_yaml_sha256"]}
    except Exception as error:
        write_json(folder / "validation_report.json", {"status": "FAIL", "type": type(error).__name__, "error": safe_error(error)})
        raise
    write_json(folder / "validation_report.json", validation_report)
    measurements["latency"] = (cuda_latency(core, image, LATENCY_WARMUP, LATENCY_REPEATS)
                                if RUN_LATENCY_BENCHMARK else {"status": "NOT_RUN", "reason": "disabled"})
    metadata_info.update({"saved_checkpoint": saved, "validation_status": "PASS" if dataset else "NOT_RUN",
                          "validation_reason": None if dataset else "dataset unavailable or validation disabled"})
    write_json(folder / "architecture_audit.json", audit["report"])
    write_json(folder / "metrics.json", measurements)
    write_json(folder / "metadata.json", metadata_info)
    return measurements, dataset, metadata_info


def _write_run_reports(folder, result):
    write_json(folder / "pruning_manifest.json", result)
    write_json(folder / "before_after_metrics.json", {
        "baseline": result["baseline"], "pruned": result["pruned"], "comparison": result["comparison"]})
    write_json(folder / "smoke_test_report.json", {key: result[key] for key in
               ("policy", "ratio", "checks", "structural_smoke_status", "validation_status", "stages", "errors", "warnings")})


def _run_configuration(source_path, provenance, folder, policy, ratio, baseline, dataset, baseline_metadata, environment, suite_id, model_variant="n"):
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=False)
    result = {"model_variant": model_variant, "model_name": f"YOLO26{model_variant}",
              "policy": policy, "ratio": ratio, "suite_id": suite_id,
              "structural_smoke_status": "NOT_RUN", "validation_status": "NOT_RUN",
              "checks": {key: False for key in REQUIRED_CHECKS}, "stages": {}, "errors": [], "warnings": [],
              "baseline": deepcopy(baseline), "pruned": {}, "comparison": {}, "provenance": provenance,
              "environment": environment, "dataset": {key: value for key, value in (dataset or {}).items() if key != "info"}}
    write_json(folder / "config.json", configuration(policy, ratio, model_variant))
    write_json(folder / "environment.json", environment)
    write_json(folder / "checkpoint_provenance.json", provenance)
    _write_run_reports(folder, result)
    stage = "baseline_reload"
    try:
        seed_environment(SEED)
        core = _load_source_core(source_path, provenance["sha256"])
        if model_state_sha256(core) != baseline_metadata["state_sha256"]:
            raise ValueError("Baseline awal berbeda: cumulative pruning atau sumber berubah.")
        result["initial_state_sha256"] = baseline_metadata["state_sha256"]
        assert_class_match(core.names, baseline_metadata["names"])
        result["checks"]["baseline_loaded"] = True
        image = torch.zeros(1, 3, IMGSZ, IMGSZ, device=TARGET_DEVICE, dtype=torch.float32)
        forward_snapshot(core, image)
        result["checks"]["baseline_forward"] = True
        protected = protected_policy_snapshot(core, model_variant)
        stage = "dependency_graph"
        pruner, audit = build_policy_pruner(core, image, policy, ratio, model_variant)
        result["checks"]["dependency_graph"] = True
        result["audit"] = audit["report"]
        write_json(folder / "dependency_audit.json", audit["report"])
        inventory_before = measure_layer_inventory(core, image, audit, policy)
        write_csv(folder / "layer_audit_before.csv", inventory_before)
        stage = "structural_pruning"
        history = apply_policy_pruning(pruner, audit)
        write_json(folder / "channel_mapping.json", audit["report"]["channel_mapping"])
        write_json(folder / "channel_masks.json", audit["report"]["channel_masks"])
        write_csv(folder / "lamp_scores.csv", audit["report"]["lamp_scores"])
        write_json(folder / "dependency_audit.json", audit["report"])
        result["groups"] = history
        result["pruned_group_count"] = len(history)
        result["checks"]["structural_channels_removed"] = bool(history)
        assert_layer_consistency(core)
        assert_policy_protected_unchanged(core, protected)
        result["checks"]["protected_head_unchanged"] = True
        forward_snapshot(core, image)
        result["checks"]["pruned_forward"] = True
        after = model_measurements(core, IMGSZ)
        if after["parameters"] >= baseline["parameters"]:
            raise ValueError("Parameter tidak berkurang setelah structural pruning.")
        result["checks"]["parameters_reduced"] = True
        initial = audit["independent_channels_before"]
        removed = sum(change["channels_before"] - change["channels_after"] for change in history)
        result["eligible_channels_before"] = initial
        result["eligible_channels_after"] = initial - removed
        result["actual_eligible_channel_ratio"] = removed / initial
        if abs(removed / initial - ratio) > 1e-12:
            result["warnings"].append("Rasio aktual berbeda karena integer channel atau threshold ranking; denominator tercatat eksplisit.")
        inventory_after = measure_layer_inventory(core, image, audit, policy)
        write_csv(folder / "layer_audit_after.csv", inventory_after)
        before_by_name = {row["layer_name"]: row for row in inventory_before}
        layer_changes = []
        for row in inventory_after:
            previous = before_by_name[row["layer_name"]]
            layer_changes.append({**row, "in_channels_before": previous["in_channels"],
                                  "out_channels_before": previous["out_channels"],
                                  "parameters_before": previous["parameters"],
                                  "estimated_flops_before": previous["estimated_flops"]})
        write_csv(folder / "layer_pruning_report.csv", layer_changes)
        result["physical_conv_output_channels_removed"] = sum(
            row["out_channels_before"] - row["out_channels"] for row in layer_changes if row["module_type"] == "Conv2d")
        stage = "save_and_reload"
        result["saved_pruned"] = save_fp32_checkpoint(core, folder / "pruned_fp32.pt", provenance)
        result["checks"]["checkpoint_saved"] = True
        (folder / "RESTORE.md").write_text(
            "# Restore LAMP V2 FP32\n\nGunakan versi runtime pada environment.json. "
            "Checkpoint berisi objek DetectionModel struktural; jangan muat state dict ke YAML standar.\n\n"
            "```python\nfrom ultralytics import YOLO\nmodel = YOLO('pruned_fp32.pt', task='detect')\n"
            "results = model.predict('image.jpg', imgsz=640, nms=None, quantize=None)\n```\n\n"
            "Tidak memerlukan import repository. Recovery belum dijalankan. "
            "Gunakan checkpoint dan budget identik untuk perbandingan recovery di masa depan.\n", encoding="utf-8")
        reload_report = verify_fresh_reload(folder / "pruned_fp32.pt", core, image, folder, timeout=RELOAD_TIMEOUT_SECONDS)
        reloaded = YOLO(str(folder / "pruned_fp32.pt"), task="detect").model.to(TARGET_DEVICE).float().eval()
        assert_class_match(reloaded.names, baseline_metadata["names"])
        assert_layer_consistency(reloaded)
        assert_policy_protected_unchanged(reloaded, protected)
        forward_snapshot(reloaded, image)
        result["reload"] = reload_report
        write_json(folder / "reload_report.json", reload_report)
        for key in ("checkpoint_reloaded", "reload_consistent", "detection_prediction"):
            result["checks"][key] = True
        if not all(result["checks"].values()):
            raise RuntimeError("Kontrak structural smoke belum terpenuhi.")
        result["structural_smoke_status"] = "PASS"
        after["checkpoint_bytes"] = result["saved_pruned"]["file_bytes"]
        result["pruned"] = after
        stage = "validation"
        if dataset is not None:
            after["validation"] = evaluate_checkpoint(folder / "pruned_fp32.pt", dataset, folder, "validation")
            result["validation_status"] = "PASS"
        else:
            result["warnings"].append("Validasi V3 NOT_RUN: dataset tidak tersedia atau dinonaktifkan.")
        stage = "latency"
        after["latency"] = (cuda_latency(reloaded, image, LATENCY_WARMUP, LATENCY_REPEATS)
                             if RUN_LATENCY_BENCHMARK else {"status": "NOT_RUN", "reason": "disabled"})
        result["comparison"] = metric_comparison(baseline, after)
        stage = "eda"
        result["eda"] = build_run_eda(result, folder, inventory_before, inventory_after)
        result["eda_status"] = "PASS"
        result["tradeoff_status"] = "MEASURED_NO_AUTOMATIC_PASS" if dataset else "NOT_ASSESSED"
        result["stages"]["experiment"] = "PASS"
    except Exception as error:
        result["stages"][stage] = "FAIL"
        result["errors"].append({"stage": stage, "type": type(error).__name__, "message": safe_error(error),
                                  "traceback": safe_error(traceback.format_exc())})
        if stage == "validation":
            result["validation_status"] = "FAIL"
        elif result["structural_smoke_status"] != "PASS":
            result["structural_smoke_status"] = "FAIL"
        else:
            result[stage + "_status"] = "FAIL"
        _write_run_reports(folder, result)
        result["tracking"] = track_result(result, folder)
        _write_run_reports(folder, result)
        raise
    _write_run_reports(folder, result)
    result["tracking"] = track_result(result, folder)
    _write_run_reports(folder, result)
    print(f"YOLO26{model_variant} {policy} pr{ratio:g}: structural={result['structural_smoke_status']}, "
          f"validation={result['validation_status']}, params={after['parameters']:,}, GFLOPs={after['gflops']:.6f}")
    return result


def summary_rows(suite):
    before = suite.get("baseline")
    if not before:
        return []
    entries = [("baseline", 0, before, {}, None)]
    entries.extend((r["policy"], r["ratio"], r["pruned"], r["comparison"], r) for r in suite.get("results", []))
    rows = []
    for policy, ratio, metric, comparison, result in entries:
        val = metric.get("validation") or {}
        variant = suite.get("model_variant", "n")
        rows.append({"model": f"YOLO26{variant}", "model_variant": variant, "policy": policy, "ratio": ratio,
                     "parameters": metric.get("parameters"), "gflops": metric.get("gflops"),
                     "checkpoint_bytes": metric.get("checkpoint_bytes"),
                     "map50_95": val.get("map50_95"), "map50": val.get("map50"),
                     "precision": val.get("precision"), "recall": val.get("recall"),
                     "parameter_reduction_pct": comparison.get("parameter_reduction_pct", 0),
                     "gflops_reduction_pct": comparison.get("gflops_reduction_pct", 0),
                     "delta_map50_95_pp": comparison.get("delta_map50_95_pp", 0 if val else None),
                     "relative_map50_95_pct": comparison.get("relative_map50_95_pct", 0 if val else None),
                     "actual_eligible_ratio": result.get("actual_eligible_channel_ratio") if result else 0,
                     "pruned_group_count": result.get("pruned_group_count") if result else 0,
                     "latency_median_ms": metric.get("latency", {}).get("median_ms"),
                     "structural_status": result["structural_smoke_status"] if result else "BASELINE",
                     "validation_status": result["validation_status"] if result else suite.get("baseline_validation_status")})
    return rows


def execute_suite(source_path, provenance, suite_dir, model_variant="n", suite_id=None):
    """Baseline dievaluasi sekali; setiap konfigurasi memuat sumber asli secara independen."""
    suite_dir = Path(suite_dir)
    suite_dir.mkdir(parents=True, exist_ok=True)
    (suite_dir / "baseline").mkdir(exist_ok=True)
    (suite_dir / "summary").mkdir(exist_ok=True)
    jobs = [{"policy": p, "ratio": r, "status": "NOT_RUN"} for p in policies_to_run(PRUNING_POLICY) for r in PRUNING_RATIOS]
    suite = {"format_version": 3, "model_variant": model_variant, "model_name": f"YOLO26{model_variant}",
             "suite_id": suite_id or suite_dir.name, "source": provenance,
             "configuration": configuration(model_variant=model_variant), "jobs": jobs, "results": [], "status": "RUNNING",
             "errors": [], "baseline_validation_status": "NOT_RUN",
             "v1_reference": {"source": "user_provided_brief", "verification": "UNVERIFIED",
                 "ratio": "UNKNOWN", "baseline_map50_95": 0.526105, "pruned_map50_95": 0.346850,
                 "baseline_gflops": 6.025830, "pruned_gflops": 5.840742,
                 "included_in_primary_comparison": False}}
    if model_variant != "n":
        suite["v1_reference"] = {"verification": "UNAVAILABLE", "reason": "No verified S V1 dataset metrics supplied",
                                  "included_in_primary_comparison": False, "ratio": "UNKNOWN"}
    write_json(suite_dir / "config.json", configuration(model_variant=model_variant))
    write_json(suite_dir / "checkpoint_provenance.json", provenance)
    write_json(suite_dir / "suite_manifest.json", suite)
    try:
        environment = environment_report()
        suite["environment"] = environment
        write_json(suite_dir / "environment.json", environment)
        baseline, dataset, base_info = _prepare_baseline(source_path, provenance, suite_dir, model_variant)
        suite.update({"baseline": baseline, "baseline_metadata": base_info,
                      "baseline_validation_status": base_info["validation_status"]})
        for job in jobs:
            job["status"] = "RUNNING"
            write_json(suite_dir / "suite_manifest.json", suite)
            folder = suite_dir / job["policy"] / f"pr{job['ratio']:g}"
            result = _run_configuration(source_path, provenance, folder, job["policy"], job["ratio"],
                                        baseline, dataset, base_info, environment, suite["suite_id"], model_variant)
            suite["results"].append(result)
            job["status"] = "PASS"
            write_json(suite_dir / "suite_manifest.json", suite)
            import gc
            gc.collect()
            if TARGET_DEVICE.type == "cuda":
                torch.cuda.empty_cache()
        suite["status"] = "COMPLETE"
    except Exception as error:
        suite["status"] = "FAIL"
        validation_report_path = suite_dir / "baseline" / "validation_report.json"
        if validation_report_path.is_file():
            suite["baseline_validation_status"] = json.loads(validation_report_path.read_text(encoding="utf-8"))["status"]
        for job in jobs:
            if job["status"] == "RUNNING":
                job["status"] = "FAIL"
        suite["errors"].append({"type": type(error).__name__, "message": safe_error(error)})
        raise
    finally:
        write_json(suite_dir / "suite_manifest.json", suite)
        rows = summary_rows(suite)
        write_json(suite_dir / "summary" / "metrics.json", rows)
        write_csv(suite_dir / "summary" / "metrics.csv", rows)
    return suite


def execute_models(suite_dir):
    """Orkestrasi N/S dengan baseline, provenance, hasil, dan failure report terpisah."""
    root = Path(suite_dir)
    root.mkdir(parents=True, exist_ok=True)
    (root / "comparison").mkdir(exist_ok=True)
    bundle = {"format_version": 3, "suite_id": root.name, "model_variants": list(MODEL_VARIANTS),
              "models": [], "jobs": [{"model_variant": variant, "status": "NOT_RUN"} for variant in MODEL_VARIANTS],
              "status": "RUNNING", "errors": [], "configuration": configuration()}
    write_json(root / "suite_manifest.json", bundle)
    try:
        baseline_names = None
        for job in bundle["jobs"]:
            variant = job["model_variant"]
            folder = root / f"yolo26{variant}"
            (folder / "baseline").mkdir(parents=True, exist_ok=False)
            job["status"] = "RUNNING"
            write_json(root / "suite_manifest.json", bundle)
            source, provenance = obtain_baseline(folder, variant)
            child = execute_suite(source, provenance, folder, variant, root.name)
            names = child["baseline_metadata"]["names"]
            if baseline_names is not None:
                assert_class_match(names, baseline_names)
            baseline_names = names
            plot_suite(child, folder / "summary")
            bundle["models"].append(child)
            job["status"] = "PASS"
            write_json(root / "suite_manifest.json", bundle)
        bundle["comparison_plots"] = plot_model_comparison(bundle["models"], root / "comparison")
        bundle["status"] = "COMPLETE"
    except Exception as error:
        bundle["status"] = "FAIL"
        for job in bundle["jobs"]:
            if job["status"] == "RUNNING":
                job["status"] = "FAIL"
        bundle["errors"].append({"type": type(error).__name__, "message": safe_error(error)})
        raise
    finally:
        write_json(root / "suite_manifest.json", bundle)
        rows = [row for child in bundle["models"] for row in summary_rows(child)]
        write_json(root / "comparison" / "metrics.json", rows)
        write_csv(root / "comparison" / "metrics.csv", rows)
    return bundle


# %% plot_helpers
def plot_suite(suite, output_dir):
    """Hanya hasil aktual; panel akurasi kosong diberi NOT_RUN, tanpa angka pengganti."""
    import matplotlib.pyplot as plt
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    rows = summary_rows(suite)
    plots = [
        ("ratio_vs_map50_95", "ratio", "map50_95", "Target eligible-channel ratio", "mAP50–95"),
        ("ratio_vs_parameter_reduction", "ratio", "parameter_reduction_pct", "Target eligible-channel ratio", "Parameter reduction (%)"),
        ("ratio_vs_gflops_reduction", "ratio", "gflops_reduction_pct", "Target eligible-channel ratio", "GFLOPs reduction (%)"),
        ("map_vs_gflops", "gflops", "map50_95", "GFLOPs", "mAP50–95"),
        ("map_vs_parameters", "parameters", "map50_95", "Parameters", "mAP50–95"),
    ]
    status = []
    for name, x, y, xlabel, ylabel in plots:
        fig, ax = plt.subplots(figsize=(7, 4.5))
        count = 0
        for policy in ["baseline", *policies_to_run(PRUNING_POLICY)]:
            values = [row for row in rows if row["policy"] == policy and row.get(x) is not None and row.get(y) is not None]
            if values:
                ax.plot([row[x] for row in values], [row[y] for row in values], "o-", label=policy)
                count += len(values)
        if not count:
            ax.text(0.5, 0.5, "NOT_RUN: validation V3 belum tersedia", ha="center", transform=ax.transAxes)
        else:
            ax.legend()
        ax.set(xlabel=xlabel, ylabel=ylabel)
        ax.grid(alpha=0.2)
        fig.tight_layout()
        fig.savefig(output / (name + ".png"), dpi=160)
        plt.close(fig)
        status.append({"plot": name, "status": "MEASURED" if count else "NOT_RUN"})
    fig, ax = plt.subplots(figsize=(10, 5))
    base = (suite.get("baseline", {}).get("validation") or {}).get("per_class", [])
    if base:
        ax.plot([r["name"] for r in base], [r["ap50_95"] for r in base], "o-", label="baseline")
        for result in suite["results"]:
            per_class = (result["pruned"].get("validation") or {}).get("per_class", [])
            if per_class:
                ax.plot([r["name"] for r in per_class], [r["ap50_95"] for r in per_class], "o-",
                        label=f"{result['policy']} pr{result['ratio']:g}")
        ax.legend(fontsize="small")
    else:
        ax.text(0.5, 0.5, "NOT_RUN: AP per kelas belum tersedia", ha="center", transform=ax.transAxes)
    ax.set(ylabel="AP50–95")
    ax.tick_params(axis="x", rotation=20)
    fig.tight_layout()
    fig.savefig(output / "per_class_ap.png", dpi=160)
    plt.close(fig)
    status.append({"plot": "per_class_ap", "status": "MEASURED" if base else "NOT_RUN"})
    write_json(output / "plot_status.json", status)
    return status
