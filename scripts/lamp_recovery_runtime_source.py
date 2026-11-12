"""Ordinary recovery orchestration embedded after the preserved pruning declarations."""
# ruff: noqa: F821 -- The standalone notebook supplies configuration and V2 helpers.

# %% configuration_helpers
_pruning_configuration = configuration
_validate_pruning_configuration = validate_configuration


def recovery_settings():
    return {"epochs": RECOVERY_EPOCHS, "optimizer": RECOVERY_OPTIMIZER, "lr0": RECOVERY_LR0,
            "lrf": 0.01, "weight_decay": RECOVERY_WEIGHT_DECAY, "momentum": 0.937,
            "warmup_epochs": RECOVERY_WARMUP_EPOCHS, "warmup_bias_lr": 0.0,
            "close_mosaic": RECOVERY_CLOSE_MOSAIC, "patience": RECOVERY_PATIENCE,
            "batch": RECOVERY_BATCH, "nbs": RECOVERY_NBS, "workers": RECOVERY_WORKERS, "imgsz": IMGSZ,
            "device": str(TARGET_DEVICE) if "TARGET_DEVICE" in globals() else DEVICE,
            "seed": SEED, "deterministic": True, "amp": RECOVERY_AMP,
            "resume": False, "freeze": None, "cache": False, "save": True, "val": True,
            "plots": True, "cos_lr": False, "mosaic": 1.0, "mixup": 0.0,
            "hsv_h": 0.015, "hsv_s": 0.7, "hsv_v": 0.4, "degrees": 0.0,
            "translate": 0.1, "scale": 0.5, "shear": 0.0, "perspective": 0.0,
            "flipud": 0.0, "fliplr": 0.5, "multi_scale": False}


def configuration(policy=None, ratio=None, model_variant=None):
    result = _pruning_configuration(policy, ratio, model_variant)
    result.update({"experiment_type": "lamp_structured_pruning_then_ordinary_finetune",
                   "recovery": recovery_settings(), "checkpoint_selection": "best_validation_fitness_ema_fp32"})
    return result


def validate_configuration():
    if RECOVERY_ENABLED is not True or FINETUNING_ENABLED is not True or KD_ENABLED is not False:
        raise ValueError("Notebook recovery memerlukan fine-tuning aktif dan KD disabled.")
    # Reuse V2 pruning validation without changing its no-recovery contract or source.
    flags = {key: globals()[key] for key in ("RECOVERY_ENABLED", "FINETUNING_ENABLED", "KD_ENABLED")}
    try:
        globals().update(dict.fromkeys(flags, False))
        _validate_pruning_configuration()
    finally:
        globals().update(flags)
    if isinstance(RECOVERY_EPOCHS, bool) or not isinstance(RECOVERY_EPOCHS, int) or RECOVERY_EPOCHS < 1:
        raise ValueError("RECOVERY_EPOCHS wajib integer positif.")
    if RECOVERY_OPTIMIZER not in {"AdamW", "SGD"}:
        raise ValueError("Optimizer recovery wajib eksplisit AdamW atau SGD; auto tidak digunakan.")
    for value in (RECOVERY_LR0, RECOVERY_WEIGHT_DECAY, RECOVERY_WARMUP_EPOCHS):
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0:
            raise ValueError("Hyperparameter recovery wajib finite dan nonnegatif.")
    if RECOVERY_LR0 <= 0 or RECOVERY_BATCH < 1 or RECOVERY_WORKERS < 0 or RECOVERY_AMP is not False:
        raise ValueError("Recovery memakai LR/batch positif, workers nonnegatif dan FP32 amp=False.")
    if isinstance(RECOVERY_NBS, bool) or not isinstance(RECOVERY_NBS, int) or RECOVERY_NBS < 1:
        raise ValueError("RECOVERY_NBS wajib integer positif untuk batch efektif.")
    if RECOVERY_PATIENCE != 0 or not 0 <= RECOVERY_CLOSE_MOSAIC <= RECOVERY_EPOCHS:
        raise ValueError("Budget tetap memakai patience=0 dan close_mosaic <= recovery epochs.")


# %% recovery_helpers
_run_pruning_configuration = _run_configuration
_pruning_summary_rows = summary_rows
_pruning_tracking_metrics = tracking_metrics


def run_identity(policy, ratio, suite_id, model_variant="n"):
    tag = f"pr{ratio:g}"
    return {"name": lamp_v2_display_name(policy, tag, model_variant, RECOVERY_EPOCHS),
            "id": uuid.uuid4().hex, "resume": "never", "group": f"lamp-finetune-{suite_id}",
            "job_type": "lamp-ordinary-finetune-recovery",
            "tags": [f"yolo26{model_variant}", "lamp", "structured", "pruned", policy, tag,
                     "finetune", "recovery", f"epochs{RECOVERY_EPOCHS}", f"seed{SEED}"]}


def track_result(result, run_dir):
    """The reused pruning stage defers tracking to the complete recovery experiment."""
    return {"status": "NOT_RUN", "uploads": [], "errors": [],
            "reason": "Tracking is owned by the pruning plus recovery experiment."}


def tracking_metrics(result):
    values = _pruning_tracking_metrics(result)
    values["recovery/completed"] = int(result.get("recovery_status") == "PASS")
    for key in ("parameters", "gflops", "checkpoint_bytes"):
        value = result.get("recovered", {}).get(key)
        if value is not None:
            values["recovered/" + key] = value
    for key, value in (result.get("recovered", {}).get("validation") or {}).items():
        if key in {"map50_95", "map50", "precision", "recall"} and value is not None:
            values["recovered/" + key] = value
    for comparison in ("recovered_vs_pruned", "recovered_vs_baseline"):
        for key in ("delta_map50_95_pp", "relative_map50_95_pct"):
            value = result.get(comparison, {}).get(key)
            if value is not None:
                values[comparison + "/" + key] = value
    return values


def _start_recovery_tracking(result, folder):
    report = {"status": "NOT_RUN", "uploads": [], "errors": []}
    run = None
    if not WANDB_TRACKING:
        report["reason"] = "WANDB_TRACKING=False"
        return None, report
    try:
        run = _init_tracking(result, folder)
        run.use_artifact(model_spec(result["model_variant"])["artifact"], type="model")
        report.update({"status": "RUNNING", "run_id": run.id, "run_url": run.url})
        return run, report
    except Exception as error:  # noqa: BLE001 -- Tracking is independent of local scientific results.
        report.update({"status": "FAIL", "errors": [{"message": safe_error(error)}]})
        if run is not None:
            try:
                run.finish(exit_code=1)
            except Exception as finish_error:  # noqa: BLE001 -- Do not leak a failed SDK run.
                report["errors"].append({"message": safe_error(finish_error)})
        return None, report


def _finish_recovery_tracking(result, folder, run, report):
    if run is None:
        write_json(folder / "tracking_report.json", report)
        return report
    try:
        run.log(tracking_metrics(result))
        run.summary.update({"recovery_status": result["recovery_status"],
                            "structural_smoke_status": result["structural_smoke_status"],
                            "recovery_validation_status": result["recovery_validation_status"]})
        if UPLOAD_WANDB_ARTIFACTS:
            tag = f"{result['ratio']:g}".replace(".", "p")
            stem = f"yolo26{result['model_variant']}-lamp-v2-{result['policy']}-finetune-pr{tag}-{result['suite_id']}"
            for label, kind, checkpoint in (("pruned", "model", folder / "pruned_fp32.pt"),
                                             ("recovered", "model", folder / "recovery/recovered_fp32.pt"),
                                             ("report", "report", None)):
                if checkpoint is not None and not checkpoint.is_file():
                    continue
                artifact = wandb.Artifact(stem + "-" + label, type=kind,
                    metadata={"source": result["provenance"], "stage": label, "seed": SEED,
                              "ratio": result["ratio"], "recovery_status": result["recovery_status"]})
                if checkpoint is not None:
                    artifact.add_file(str(checkpoint), name=checkpoint.relative_to(folder).as_posix())
                else:
                    for path in sorted(folder.rglob("*")):
                        relative = path.relative_to(folder)
                        if (path.is_file() and "wandb" not in relative.parts
                                and path.suffix in {".json", ".csv", ".md", ".png", ".svg", ".yaml"}):
                            artifact.add_file(str(path), name=relative.as_posix())
                logged = run.log_artifact(artifact)
                logged.wait(timeout=300)
                report["uploads"].append({"stage": label, "status": "CONFIRMED", "name": logged.qualified_name})
        if not report["errors"]:
            report["status"] = "PASS"
    except Exception as error:  # noqa: BLE001 -- Upload failures must preserve local outputs.
        report["status"] = "FAIL"
        report["errors"].append({"message": safe_error(error)})
    finally:
        try:
            run.finish(exit_code=0 if report["status"] == "PASS" and result["recovery_status"] != "FAIL" else 1)
        except Exception as error:  # noqa: BLE001 -- Persist SDK finish failures too.
            report["status"] = "FAIL"
            report["errors"].append({"message": safe_error(error)})
        write_json(folder / "tracking_report.json", report)
    return report


def _write_recovery_reports(folder, result):
    _write_run_reports(folder, result)
    write_json(folder / "before_after_metrics.json", {key: result.get(key) for key in (
        "baseline", "pruned", "recovered", "comparison", "recovered_vs_pruned", "recovered_vs_baseline")})
    write_json(folder / "recovery_manifest.json", result)
    write_json(folder / "recovery_status.json", {key: result.get(key) for key in (
        "recovery_status", "recovery_validation_status", "recovery_reload", "recovery_training", "errors")})


def save_recovered_fp32_checkpoint(core, path, provenance, training_report):
    """Use the same FP32 object format with explicit post-recovery metadata."""
    save_fp32_checkpoint(core, path, provenance)
    payload = torch.load(path, map_location="cpu", weights_only=False)
    payload.update({"recovery_enabled": True, "recovery_method": "ordinary_supervised_finetune",
                    "recovery_settings": recovery_settings(),
                    "recovery_epochs_completed": training_report["epochs_completed"],
                    "epoch": training_report["best_epoch"] - 1,
                    "best_fitness": training_report["best_fitness"],
                    "checkpoint_selection": "best_validation_fitness_ema_fp32"})
    torch.save(payload, path)
    return {"path": str(Path(path).resolve()), "sha256": sha256_file(path), "file_bytes": Path(path).stat().st_size}


def _run_configuration(source_path, provenance, folder, policy, ratio, baseline, dataset,
                       baseline_metadata, environment, suite_id, model_variant="n"):
    folder = Path(folder)
    run, tracking = None, {"status": "NOT_RUN", "uploads": [], "errors": []}
    result = None
    stage = "pruning"
    try:
        result = _run_pruning_configuration(source_path, provenance, folder, policy, ratio,
            baseline, dataset, baseline_metadata, environment, suite_id, model_variant)
        result.update({"recovery_status": "NOT_RUN", "recovery_validation_status": "NOT_RUN", "recovered": {}})
        result["stages"]["experiment"] = "RUNNING"
        recovery = folder / "recovery"
        recovery.mkdir(exist_ok=False)
        run, tracking = _start_recovery_tracking(result, folder)

        def log_epoch(_trainer, metrics):
            if run is None or tracking["status"] == "FAIL":
                return
            try:
                values = {"recovery_epoch/" + key: metrics[key] for key in ("epoch", "fitness", "optimizer_steps")}
                for key in ("losses", "metrics"):
                    values.update({f"recovery_epoch/{key}/{name}": value for name, value in metrics.get(key, {}).items()})
                run.log(values)
            except Exception as error:  # noqa: BLE001 -- A logger failure cannot abort fine-tuning.
                tracking["status"] = "FAIL"
                tracking["errors"].append({"message": safe_error(error)})

        if dataset is None:
            result["warnings"].append("Recovery NOT_RUN: dataset tidak tersedia; tidak ada training pengganti.")
            result["stages"]["experiment"] = "PARTIAL_NOT_RUN"
            write_json(recovery / "eda_status.json", {"status": "NOT_RUN", "reason": "dataset unavailable"})
            return result
        stage = "recovery_training"
        seed_environment(SEED)
        pruned_path = folder / "pruned_fp32.pt"
        pruned = YOLO(str(pruned_path), task="detect").model.float().eval()
        shapes = {name: list(value.shape) for name, value in pruned.state_dict().items()}
        core, training_report = fine_tune_pruned_checkpoint(
            pruned_path, dataset["yaml"], recovery / "training", dataset["class_names"],
            recovery_settings(), epoch_callback=log_epoch)
        result["recovery_training"] = training_report
        core = core.to(TARGET_DEVICE).float().eval()
        if {name: list(value.shape) for name, value in core.state_dict().items()} != shapes:
            raise ValueError("Struktur model hasil pruning berubah selama fine-tuning.")
        assert_class_match(core.names, baseline_metadata["names"])
        assert_layer_consistency(core)
        image = torch.zeros(1, 3, IMGSZ, IMGSZ, device=TARGET_DEVICE, dtype=torch.float32)
        forward_snapshot(core, image)
        measures = model_measurements(core, IMGSZ)
        if measures["parameters"] != result["pruned"]["parameters"]:
            raise ValueError("Jumlah parameter recovery tidak sama dengan model pruned.")
        stage = "recovery_save_reload"
        saved = save_recovered_fp32_checkpoint(core, recovery / "recovered_fp32.pt", {
            **provenance, "pruned_checkpoint_sha256": sha256_file(pruned_path),
            "recovery_method": "ordinary_supervised_finetune", "checkpoint_selection": "best_validation_fitness_ema_fp32"},
            training_report)
        result["saved_recovered"] = saved
        measures["checkpoint_bytes"] = saved["file_bytes"]
        result["recovery_reload"] = verify_fresh_reload(recovery / "recovered_fp32.pt", core, image, recovery,
                                                       timeout=RELOAD_TIMEOUT_SECONDS)
        write_json(recovery / "reload_report.json", result["recovery_reload"])
        result["recovered"] = measures
        stage = "recovery_validation"
        measures["validation"] = evaluate_checkpoint(recovery / "recovered_fp32.pt", dataset, recovery, "validation")
        result["recovery_validation_status"] = "PASS"
        stage = "recovery_latency"
        measures["latency"] = (cuda_latency(core, image, LATENCY_WARMUP, LATENCY_REPEATS)
            if RUN_LATENCY_BENCHMARK else {"status": "NOT_RUN", "reason": "disabled"})
        result["recovered_vs_baseline"] = metric_comparison(baseline, measures)
        result["recovered_vs_pruned"] = metric_comparison(result["pruned"], measures)
        result["recovery_status"] = "PASS"
        result["stages"]["recovery"] = "PASS"
        result["stages"]["experiment"] = "PASS"
        write_json(recovery / "eda_status.json", {
            "status": "MEASURED", "channel_structure": "UNCHANGED_FROM_PRUNED",
            "channel_eda": "../eda", "validation_exports": measures["validation"].get("exports"),
            "meaning": "Shape preservation and training completion do not guarantee recovered accuracy."})
        (recovery / "RESTORE.md").write_text(
            "# Model setelah ordinary fine-tuning\n\n"
            "```python\nfrom ultralytics import YOLO\nmodel = YOLO('recovered_fp32.pt', task='detect')\n"
            "model.predict('image.jpg', imgsz=640, nms=None, quantize=None)\n```\n\n"
            "Native DetectionModel struktural FP32; ema=None, optimizer=None. "
            "Subclass trainer tidak diperlukan untuk inference. Lihat environment.json pada direktori konfigurasi.\n",
            encoding="utf-8")
        return result
    except Exception as error:
        if result is None and (folder / "pruning_manifest.json").is_file():
            result = json.loads((folder / "pruning_manifest.json").read_text(encoding="utf-8"))
        if result is not None:
            result.setdefault("recovered", {})
            result["recovery_status"] = "FAIL" if stage != "pruning" else "NOT_RUN"
            result.setdefault("recovery_validation_status", "NOT_RUN")
            if stage == "recovery_validation":
                result["recovery_validation_status"] = "FAIL"
            result["stages"][stage] = "FAIL"
            result["stages"]["experiment"] = "FAIL"
            result["errors"].append({"stage": stage, "type": type(error).__name__, "message": safe_error(error)})
        raise
    finally:
        if result is not None:
            result["tracking"] = tracking
            _write_recovery_reports(folder, result)
            result["tracking"] = _finish_recovery_tracking(result, folder, run, tracking)
            _write_recovery_reports(folder, result)


def summary_rows(suite):
    rows = _pruning_summary_rows(suite)
    for row in rows:
        row["stage"] = "baseline" if row["policy"] == "baseline" else "pruned"
    for result in suite.get("results", []):
        if result.get("recovered"):
            entry = {**result, "pruned": result["recovered"], "comparison": result["recovered_vs_baseline"],
                     "validation_status": result["recovery_validation_status"]}
            row = _pruning_summary_rows({**suite, "results": [entry]})[-1]
            row.update({"stage": "recovered", "recovery_status": result["recovery_status"],
                        "recovery_gain_map50_95_pp": result["recovered_vs_pruned"].get("delta_map50_95_pp")})
            rows.append(row)
    return rows


def plot_suite(suite, output_dir):
    return plot_recovery_comparison([suite], output_dir)


def plot_model_comparison(suites, output_dir):
    return plot_recovery_comparison(suites, output_dir)


def plot_recovery_comparison(suites, output_dir):
    import matplotlib.pyplot as plt
    folder = Path(output_dir)
    folder.mkdir(parents=True, exist_ok=True)
    rows = [row for suite in suites for row in summary_rows(suite)]
    write_csv(folder / "three_stage_metrics.csv", rows)
    files = []
    for x, label in (("parameters", "Parameters"), ("gflops", "GFLOPs"), ("latency_median_ms", "Raw forward latency (ms)")):
        fig, ax = plt.subplots(figsize=(11, 7))
        count = 0
        for stage, marker in (("baseline", "s"), ("pruned", "x"), ("recovered", "o")):
            for row in rows:
                if row["stage"] != stage or row.get(x) is None or row.get("map50_95") is None:
                    continue
                ax.scatter(row[x], row["map50_95"], marker=marker,
                           color="tab:blue" if row["model_variant"] == "n" else "tab:orange")
                ax.annotate(f"{row['model']} {row['policy']} {stage} pr{row['ratio']:g}",
                            (row[x], row["map50_95"]), xytext=(4, 4), textcoords="offset points", fontsize=7)
                count += 1
        if not count:
            ax.text(.5, .5, "NOT_RUN: measured accuracy/latency unavailable", ha="center", transform=ax.transAxes)
        ax.set(xlabel=label, ylabel="mAP50–95", title="Baseline / pruned / ordinary fine-tuned")
        fig.tight_layout()
        files.append(_eda_save_figure(fig, folder / ("recovery_map_vs_" + x)))
    return files
