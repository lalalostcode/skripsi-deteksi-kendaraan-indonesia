"""Notebook-embedded MGD callbacks and independent serial seed lifecycle."""

MGD_CALLBACKS_SOURCE = '''def prune_uploaded_snapshots(context):
    """Bound local disk use only AFTER upload confirmation; retain all failed/unknown snapshots."""
    base = (context["directory"] / "epoch_snapshots").resolve()
    uploaded = sorted(context["uploaded_epochs"])
    for epoch in uploaded[:-KEEP_LAST_LOCAL_EPOCHS]:
        candidate = base / f"epoch-{epoch:03d}"
        resolved = candidate.resolve()
        if resolved.parent != base or not resolved.is_relative_to(context["directory"].resolve()):
            raise RuntimeError("Snapshot cleanup path keluar dari direktori run.")
        confirmation = candidate / "upload_status.json"
        if candidate.is_dir() and confirmation.is_file():
            metadata = json.loads(confirmation.read_text())
            if metadata.get("status") == "PASS" and metadata.get("epoch") == epoch:
                shutil.rmtree(candidate)


def gradient_norm(parameters):
    values = [p.grad.detach().float().norm() for p in parameters if p.grad is not None]
    norm = float(torch.stack(values).norm()) if values else 0.0
    if not math.isfinite(norm) or norm <= 0:
        raise RuntimeError("Gradien komponen trainable harus finite dan nonzero.")
    return norm


def make_callbacks(context):
    seed, run = SEED, context["run"]

    def guard():
        if SEED != seed or context["run"] is not run:
            raise RuntimeError("Callback berasal dari seed/run lain.")

    def start(trainer):
        guard()
        model = unwrap_model(trainer.model)
        audit_optimizer(model, trainer.optimizer)
        if trainer.batch_size != BATCH or trainer.args.imgsz != IMGSZ or bool(trainer.amp) != AMP:
            raise RuntimeError("Batch/imgsz/AMP efektif berbeda dari protokol.")
        context.update(teacher_before=state_digest(model.teacher_model), feature_layout=model.feature_layout,
                       optimizer_steps=0, amp_skipped_steps=0, gradient=None, started=time.perf_counter())
        run.config.update({"effective_training_args": vars(trainer.args), "feature_taps": model.feature_layout,
                           "mgd_trainable_parameters": sum(p.numel() for p in model.projector.parameters()),
                           "dataset_yaml_sha256": file_sha256(Path(DATA)), "resume_status": RESUME_STATUS})
        if trainer.device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(trainer.device)

        def before_step(optimizer, *_):
            found_inf = getattr(optimizer, "found_inf", None)
            if found_inf is not None and float(found_inf) != 0:
                context["amp_skipped_steps"] += 1
                return
            context["gradient"] = {
                "student": gradient_norm(model.student_model.parameters()),
                "alignment": gradient_norm(p for block in model.projector if block.align is not None for p in block.align.parameters()),
                "generation": gradient_norm(p for block in model.projector for p in block.generation.parameters()),
            }
            context["optimizer_steps"] += 1
        context["optimizer_hook"] = trainer.optimizer.register_step_pre_hook(before_step)

    def batch_end(trainer):
        guard()
        context["batches"] += 1
        model = unwrap_model(trainer.model)
        if model.teacher_model.training or any(p.grad is not None for p in model.teacher_model.parameters()):
            raise RuntimeError("Teacher keluar dari eval atau menerima gradien.")
        if model._teacher_feats or model._student_feats:
            raise RuntimeError("Cache fitur belum dibersihkan setelah loss.")
        if not all(math.isfinite(float(v)) for v in trainer.loss_items.values()):
            raise RuntimeError("Loss/metrik MGD non-finite.")

    def save(trainer):
        guard()
        epoch = int(trainer.epoch) + 1
        if epoch in context["uploaded_epochs"]:
            return
        LAST_PHASE["value"] = "checkpoint"
        record = snapshot_epoch(trainer, context)
        artifact = wandb.Artifact(f"{EXPERIMENT_ID}-{run.id}-checkpoints", type="model",
            metadata={"epoch": epoch, "best_epoch_so_far": context["best_epoch"], "seed": seed,
                      "method": METHOD, "mgd_alpha": MGD_ALPHA, "mgd_mask_ratio": MGD_MASK_RATIO,
                      "teacher_provenance": TEACHER_PROVENANCE, "resume_status": RESUME_STATUS,
                      "diagnostic_only": context["diagnostic"]})
        for path in sorted(record["directory"].iterdir()):
            # Snapshots are immutable until .wait() completes; avoid another persistent cache copy.
            artifact.add_file(str(path), name=path.name, policy="immutable", skip_cache=True)
        try:
            run.log_artifact(artifact, aliases=["latest", f"epoch-{epoch:03d}"] + (["best"] if record["is_best"] else [])).wait()
        except BaseException:
            (record["directory"] / "upload_status.json").write_text(json.dumps({"status": "FAIL", "epoch": epoch}))
            raise
        (record["directory"] / "upload_status.json").write_text(json.dumps({"status": "PASS", "epoch": epoch}))
        context["uploaded_epochs"].add(epoch)
        if epoch not in context["logged_epochs"]:
            values = {f"train/{key}": float(value) for key, value in trainer.tloss.items()}
            values.update({key: float(value) for key, value in trainer.metrics.items()})
            values.update(trainer.lr)
            values.update(epoch=epoch, optimizer_steps=context["optimizer_steps"], amp_skipped_steps=context["amp_skipped_steps"],
                          elapsed_seconds=time.perf_counter() - context["started"], best_epoch=context["best_epoch"])
            if context["gradient"]:
                values.update({"gradient/" + key: value for key, value in context["gradient"].items()})
            if trainer.device.type == "cuda":
                values.update(peak_allocated_gib=torch.cuda.max_memory_allocated(trainer.device) / 2**30,
                              peak_reserved_gib=torch.cuda.max_memory_reserved(trainer.device) / 2**30)
            run.log(values, step=epoch)
            context["logged_epochs"].add(epoch)
        prune_uploaded_snapshots(context)

    return {"on_train_start": start, "on_train_batch_end": batch_end, "on_model_save": save}
'''

MGD_RUNTIME_SOURCE = '''def validate_research_settings():
    # Boundary values remain available to the loss unit oracle, not the research gradient gate.
    if not math.isfinite(MGD_ALPHA) or MGD_ALPHA <= 0:
        raise ValueError("Training MGD memerlukan MGD_ALPHA positif dan finite.")
    if not math.isfinite(MGD_MASK_RATIO) or not 0 <= MGD_MASK_RATIO < 1:
        raise ValueError("Training MGD memerlukan 0 <= MGD_MASK_RATIO < 1 agar alignment menerima gradien.")
    if isinstance(KEEP_LAST_LOCAL_EPOCHS, bool) or not isinstance(KEEP_LAST_LOCAL_EPOCHS, int) or KEEP_LAST_LOCAL_EPOCHS < 1:
        raise ValueError("KEEP_LAST_LOCAL_EPOCHS harus integer positif.")


def validate_seed_list(seeds):
    if not isinstance(seeds, (list, tuple)) or not seeds:
        raise ValueError("SEEDS_TO_RUN harus berisi minimal satu seed.")
    if any(isinstance(seed, bool) or not isinstance(seed, int) or not 0 <= seed < 2**32 for seed in seeds):
        raise ValueError("Seed harus integer 0 sampai 2**32-1.")
    if len(seeds) != len(set(seeds)):
        raise ValueError("Seed duplikat tidak diperbolehkan.")
    return list(seeds)


def run_one_seed(seed, queue_dir, *, diagnostic=False):
    global SEED, TEACHER_CHECKPOINT, TEACHER_PROVENANCE
    validate_research_settings()
    require_cuda()
    if wandb.run is not None:
        raise RuntimeError("Selesaikan run W&B aktif sebelum menjalankan seed baru.")
    if EVAL_SPLIT != "val" or not 1 <= EPOCHS <= 100:
        raise ValueError("Gunakan val dan 1 sampai 100 epoch.")
    SEED = seed
    init_seeds(seed, deterministic=True)
    directory = Path(queue_dir) / (("smoke-" if diagnostic else "seed-") + str(seed))
    directory.mkdir(parents=True, exist_ok=False)
    context = {"directory": directory, "diagnostic": diagnostic, "batch": BATCH, "batches": 0,
               "best_epoch": None, "uploaded_epochs": set(), "logged_epochs": set()}
    run = student = trainer = None
    started = time.perf_counter()
    try:
        # Re-audit the same fixed teacher for every independent seed; never inherit prior student weights.
        TEACHER_CHECKPOINT, TEACHER_PROVENANCE = download_teacher()
        if file_sha256(Path(TEACHER_CHECKPOINT)) != TEACHER_SHA256:
            raise RuntimeError("SHA-256 teacher berbeda dari teacher Native/CrossKD.")
        name = f"kd_mgd_T26s_S26n_seed{seed}" + ("_smoke" if diagnostic else "")
        run = wandb.init(entity=WANDB_ENTITY, project=WANDB_PROJECT,
            group=f"diagnostic-{WANDB_GROUP}" if diagnostic else WANDB_GROUP,
            name=name, id=wandb.util.generate_id(), resume="never", job_type="diagnostic" if diagnostic else "training",
            config={"experiment": EXPERIMENT_ID, "method": METHOD, "teacher": "yolo26s", "model": MODEL_WEIGHTS,
                    "seed": seed, "seeds_to_run": list(SEEDS_TO_RUN), "teacher_artifact": TEACHER_ARTIFACT,
                    "teacher_provenance": TEACHER_PROVENANCE, "dataset": DATA,
                    "mgd_alpha": MGD_ALPHA, "mgd_mask_ratio": MGD_MASK_RATIO,
                    "reference_commit": MGD_REFERENCE_COMMIT, "reference_sha256": MGD_REFERENCE_SHA256,
                    "environment": ENVIRONMENT_INFO, "resume_status": RESUME_STATUS, "diagnostic_only": diagnostic},
            tags=[METHOD, "teacher-s", "student-n", f"seed{seed}", "poltrada-v3", "diagnostic" if diagnostic else "research"],
            settings=wandb.Settings(init_timeout=90))
        context["run"] = run
        linked = run.use_artifact(TEACHER_ARTIFACT, type="model")
        if linked.version != "v62":
            raise RuntimeError("Run harus menggunakan teacher artifact v62.")
        # Reseed after audits/probes so their allocations cannot change student/module initialization.
        init_seeds(seed, deterministic=True)
        student = YOLO(MODEL_WEIGHTS)
        for event, callback in make_callbacks(context).items():
            student.add_callback(event, callback)
        student.train(trainer=MGDTrainer, **protocol_args(directory, diagnostic=diagnostic))
        trainer = student.trainer
        model = unwrap_model(trainer.model)
        if state_digest(model.teacher_model) != context["teacher_before"]:
            raise RuntimeError("Teacher parameters atau BN berubah selama training.")
        if not context["uploaded_epochs"] or context["optimizer_steps"] < 1:
            raise RuntimeError("Belum ada epoch terunggah atau optimizer update nyata.")
        snapshot = directory / "epoch_snapshots" / f"epoch-{max(context['uploaded_epochs']):03d}"
        best = snapshot / "best.pt"
        evaluation = evaluate_student(best, directory)
        report = {"status": "PASS", **evaluation, "seed": seed, "diagnostic_only": diagnostic,
                  "completed_epochs": len(context["uploaded_epochs"]), "requested_epochs": 1 if diagnostic else EPOCHS,
                  "best_epoch": context["best_epoch"],
                  "student_checkpoint": str(best), "seed_directory": str(directory), "wandb_run_id": run.id,
                  "optimizer_steps": context["optimizer_steps"], "amp_skipped_steps": context["amp_skipped_steps"],
                  "teacher_parameters_and_bn_unchanged": True, "resume_status": RESUME_STATUS,
                  "duration_seconds": time.perf_counter() - started}
        (directory / "metrics.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
        (directory / "config.json").write_text(json.dumps({"effective_args": vars(trainer.args),
            "feature_taps": context["feature_layout"], "teacher_provenance": TEACHER_PROVENANCE,
            "mgd_alpha": MGD_ALPHA, "mgd_mask_ratio": MGD_MASK_RATIO, "environment": ENVIRONMENT_INFO,
            "reference_sha256": MGD_REFERENCE_SHA256, "resume_status": RESUME_STATUS}, indent=2, default=str), encoding="utf-8")
        final = wandb.Artifact(f"{EXPERIMENT_ID}-{run.id}-results", type="results", metadata={"seed": seed})
        for filename in ("metrics.json", "config.json"):
            final.add_file(str(directory / filename), name=filename)
        run.log_artifact(final, aliases=["final"]).wait()
        run.summary.update({f"final/{key}": value for key, value in report.items() if isinstance(value, (str, bool, int, float))})
        run.finish()
        return report
    except BaseException as error:
        (directory / "failure.json").write_text(json.dumps({"status": "FAIL", "seed": seed,
            "phase": LAST_PHASE["value"], "error_type": type(error).__name__, "message": str(error),
            "protocol_changed_automatically": False}, indent=2), encoding="utf-8")
        if run is not None:
            try:
                run.finish(exit_code=1)
            except Exception:  # noqa: BLE001 -- Preserve the training/upload exception.
                status("W&B", "Penutupan run gagal; antrean berhenti dan file lokal dipertahankan.")
        raise
    finally:
        if context.get("optimizer_hook") is not None:
            context["optimizer_hook"].remove()
        if student is not None and hasattr(student, "trainer"):
            for loader in (getattr(student.trainer, "train_loader", None), getattr(student.trainer, "test_loader", None)):
                if hasattr(loader, "close"):
                    loader.close()
        student = trainer = model = None
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def run_seed_queue():
    validate_research_settings()
    seeds = validate_seed_list(SEEDS_TO_RUN)
    if not RUN_FULL_TRAINING and not RUN_SMOKE_TEST:
        status("TRAIN", "Training dan smoke dinonaktifkan.")
        return []
    require_cuda()
    if not WANDB_ENABLED:
        raise ValueError("Eksperimen ini memerlukan W&B online.")
    if wandb.run is not None:
        raise RuntimeError("Masih ada run W&B aktif.")
    queue_dir = Path(PROJECT) / EXPERIMENT_ID / ("multiseed_" + datetime.now(UTC).strftime("%Y%m%dT%H%M%S_%fZ"))
    queue_dir.mkdir(parents=True, exist_ok=False)
    progress = {"seeds_to_run": seeds, "completed": [], "failed_seed": None, "status": "running"}
    results = []
    progress_path = queue_dir / "multiseed_progress.json"
    progress_path.write_text(json.dumps(progress, indent=2))
    for seed in seeds:
        try:
            if RUN_SMOKE_TEST:
                run_one_seed(seed, queue_dir, diagnostic=True)
            if RUN_FULL_TRAINING:
                result = run_one_seed(seed, queue_dir)
                results.append(result)
                progress["completed"].append({"seed": seed, "wandb_run_id": result["wandb_run_id"],
                                              "seed_directory": result["seed_directory"]})
                (queue_dir / "multiseed_results.json").write_text(json.dumps(results, indent=2))
            progress_path.write_text(json.dumps(progress, indent=2))
        except BaseException:
            progress.update(status="failed", failed_seed=seed)
            progress_path.write_text(json.dumps(progress, indent=2))
            raise
    progress["status"] = "completed"
    progress_path.write_text(json.dumps(progress, indent=2))
    return results
'''
