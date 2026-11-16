"""Ablation orchestration; embedded after the unchanged selected KD runtime."""
# ruff: noqa: F821 -- Standalone notebook declarations share a namespace.
from ultralytics.cfg import DEFAULT_CFG_DICT

_ablation_identity = ft_identity
_ablation_dataset = ft_dataset
_ablation_fit = ft_fit
_ablation_registry = ft_registry
_ablation_restore = ft_restore_snapshot
_ablation_log = ft_log_results


def ft_identity():
    return {**_ablation_identity(), "ablation_parameter": ABLATION_PARAMETER,
            "ablation_value": ABLATION_VALUE, "method_source_sha256": METHOD_SOURCE_SHA256}


def ft_name():
    return f"yolo26s-su1p20-seed42-{RECOVERY_METHOD}-{PARAMETER_TOKEN}-ablation-{EXECUTION_ID}" + ("-smoke" if MODE == "smoke" else "")


def ft_group():
    return ("crosskd-kdscale-ablation" if RECOVERY_METHOD == "crosskd" else "mgd-alpha-ablation") + ("-smoke" if MODE == "smoke" else "")


def ablation_protocol(options):
    """Resolve SDK defaults before comparing historical requested/effective options."""
    options = {**DEFAULT_CFG_DICT, **options, "task": "detect", "mode": "train"}
    for name in ("data", "model", "project", "name", "save_dir", "device", "distill_model", "resume", "exist_ok"):
        options.pop(name, None)
    return options


def ablation_settings(record):
    options = fine_tuned_train_args("DATA", "PROJECT", "training")
    return {"dataset": {key: record[key] for key in ("train_fingerprint", "val_fingerprint", "class_names")},
            "label_policy": FILKOM_MGD_LABEL_POLICY, "raw_sha256": ft_identity()["raw_sha256"],
            "teacher_sha256": LOCKED_TEACHER_SHA256, "seed": 42,
            "training": ablation_protocol(options),
            "evaluation": {k: v for k, v in fine_tuned_val_args("DATA").items() if k not in {"data", "device"}},
            "environment": {key: ENVIRONMENT.get(key) for key in
                            ("python", "torch", "torchvision", "ultralytics", "wandb", "cuda", "gpu")},
            "mode": MODE}


def ft_dataset(manifest, folder):
    dataset, record, train, yaml_path = _ablation_dataset(manifest, folder)
    global ABLATION_SETTINGS, ABLATION_SIGNATURE
    ABLATION_SETTINGS = ablation_settings(record)
    # Dataset and training conditions are shared by both methods, including after restart.
    lock_root = ABLATION_ROOT / ("smoke" if MODE == "smoke" else "research")
    lock_root.mkdir(parents=True, exist_ok=True)
    with FileLock(str(lock_root / ".protocol.lock"), timeout=30):
        workflow_bind(lock_root, "protocol.json", ABLATION_SETTINGS)
    ABLATION_SIGNATURE = workflow_digest({**ABLATION_SETTINGS, "method": RECOVERY_METHOD,
                                         "method_source_sha256": METHOD_SOURCE_SHA256})
    return dataset, record, train, yaml_path


def ft_restore_snapshot(folder):
    global EXECUTION_ID
    if RESUME_SNAPSHOT_REF and not (folder / "resume_contract.json").exists():
        remote = wandb.Api(timeout=30).artifact(workflow_version_ref(RESUME_SNAPSHOT_REF), type="model")
        execution = remote.metadata.get("ablation_execution_id")
        if not isinstance(execution, str) or not re.fullmatch(r"\d{8}T\d{6}Z-[0-9a-f]{8}", execution):
            raise ValueError("Snapshot bukan eksekusi ablation dengan identitas valid.")
        EXECUTION_ID = execution
    _ablation_restore(folder)
    write_json(folder / "ablation_identity.json", {"method": RECOVERY_METHOD, "value": ABLATION_VALUE,
               "mode": MODE, "execution_id": EXECUTION_ID})


_ablation_publish = ft_publish


def ft_publish(run, folder, label, files, metadata, event="final"):
    return _ablation_publish(run, folder, label, files,
        {**metadata, "ablation_execution_id": EXECUTION_ID, "comparison_signature": ABLATION_SIGNATURE}, event=event)


def ablation_smoke(raw, folder):
    """Probe isolated objects and restore RNG so setup cannot change training initialization."""
    device = torch.device("cpu" if DEVICE == "cpu" else f"cuda:{DEVICE}")
    rng = recovery_rng(device)
    model = teacher = wrapper = None
    try:
        random.seed(42); np.random.seed(42); torch.manual_seed(42)
        if device.type == "cuda":
            torch.cuda.manual_seed_all(42)
        model = deepcopy(raw).to(device).float().train().requires_grad_(True)
        model.criterion = None
        path, _ = source_matched_teacher(folder, raw)
        teacher, _ = load_checkpoint(path, device=device, fuse=False)
        teacher.float().eval().requires_grad_(False)
        teacher.end2end = True
        model.args = SimpleNamespace(**{**DEFAULT_CFG_DICT, **fine_tuned_train_args("DATA", "PROJECT", "probe"),
                                       "imgsz": 64, "dis": 1.0})
        wrapper = (CrossKDModel if RECOVERY_METHOD == "crosskd" else MGDModel)(teacher, model)
        wrapper.train(True)
        parameters = [p for p in wrapper.parameters() if p.requires_grad]
        optimizer = torch.optim.AdamW(parameters, lr=LR, weight_decay=WEIGHT_DECAY)
        batch = {"img": torch.rand(2, 3, 64, 64, device=device),
                 "batch_idx": torch.tensor([0., 1.], device=device),
                 "cls": torch.zeros(2, 1, device=device),
                 "bboxes": torch.tensor([[.5, .5, .3, .3]] * 2, device=device)}
        loss, items = wrapper.loss(batch)
        loss.sum().backward()
        gradients = [p.grad for p in parameters if p.grad is not None]
        if not gradients or not all(torch.isfinite(g).all() for g in gradients):
            raise FloatingPointError("Smoke: gradien student/adapter nonfinite atau kosong.")
        if any(p.grad is not None or p.requires_grad for p in teacher.parameters()) or teacher.training:
            raise RuntimeError("Smoke: teacher tidak frozen/eval.")
        if any(not torch.isfinite(v).all() for v in items.values()) or not torch.isfinite(loss).all():
            raise FloatingPointError("Smoke: loss nonfinite.")
        initial = recovery_state_sha256(model, parameters_only=True)
        teacher_hash = recovery_state_sha256(teacher)
        optimizer.step()
        if recovery_state_sha256(model, parameters_only=True) == initial or recovery_state_sha256(teacher) != teacher_hash:
            raise RuntimeError("Smoke: student tidak berubah atau teacher berubah.")
        proof = {"status": "PASS", "mode": "synthetic-smoke", "seed": 42, "imgsz": 64, "batch": 2,
                 "gradient_tensors": len(gradients), "teacher_unchanged": True,
                 "losses": {k: float(v.detach().cpu()) for k, v in items.items()},
                 "candidate": ft_identity(), "raw_state_sha256": recovery_state_sha256(raw)}
        write_json(folder / "ablation_smoke.json", proof)
        return proof
    finally:
        if wrapper is not None:
            wrapper._remove_feature_hooks()
        del wrapper, teacher, model
        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()
        recovery_restore_rng(rng, device)


def ablation_reuse_check(proof, record):
    """Fail closed: no name/score/seed-only equivalence, no inference from missing metadata."""
    reasons = []
    identity = ft_identity()
    for key in ("seed", "method", "raw_artifact_ref", "raw_sha256", "teacher_artifact_ref", "teacher_sha256"):
        if proof.get(key) != identity.get(key):
            reasons.append(key)
    if proof.get("status") != "PASS" or not proof.get("logging_verified") or proof.get("epochs") != EPOCHS:
        reasons.append("completion")
    if proof.get("hyperparameters") != recovery_kd_settings(RECOVERY_METHOD):
        reasons.append("hyperparameters")
    dataset = proof.get("dataset", {})
    if any(dataset.get(k) != record[k] for k in ("train_fingerprint", "val_fingerprint", "class_names")):
        reasons.append("dataset")
    if dataset.get("label_adapter", {}).get("policy") != FILKOM_MGD_LABEL_POLICY:
        reasons.append("label_policy")
    if proof.get("method_source_sha256") != METHOD_SOURCE_SHA256 and proof.get("implementation") not in HISTORICAL_IMPLEMENTATIONS:
        reasons.append("method_implementation")
    if ablation_protocol(proof.get("training", {})) != ABLATION_SETTINGS["training"]:
        reasons.append("training_protocol")
    if {k: v for k, v in (proof.get("evaluation") or {}).items() if k not in {"data", "device"}} != ABLATION_SETTINGS["evaluation"]:
        reasons.append("evaluation_protocol")
    if proof.get("environment") != ABLATION_SETTINGS["environment"]:
        reasons.append("environment")
    return {"eligible": not reasons, "reasons": reasons}


def ablation_download(artifact, folder):
    ref = workflow_version_ref(artifact.qualified_name)
    hashes = artifact.metadata.get("files_sha256")
    if not hashes:
        raise ValueError("Artifact historis tidak mempunyai daftar SHA-256.")
    artifact.download(root=str(folder))
    for name, digest in hashes.items():
        path = (folder / name).resolve()
        if not path.is_relative_to(folder.resolve()) or sha256_file(path) != digest:
            raise ValueError("SHA-256/path artifact historis tidak valid.")
    return ref


def ablation_find_default(folder, record, raw):
    if MODE != "full_val" or not REUSE_EXISTING_DEFAULT or ABLATION_VALUE != (1.0 if RECOVERY_METHOD == "crosskd" else 2e-5):
        return None
    audit = {"status": "RERUN", "checks": []}
    api = wandb.Api(timeout=30)
    # Only historical main recovery runs; never silently select a new ablation by its score.
    try:
        runs = list(api.runs(f"{WANDB_ENTITY}/{WANDB_PROJECT}", filters={
            "config.seed": 42, "config.method": RECOVERY_METHOD, "state": "finished"}, order="-created_at"))
    except Exception as error:  # noqa: BLE001 -- Audit unavailable means no historical result is accepted.
        write_json(folder / "default_reuse_audit.json", {**audit, "lookup_error": safe_error(error)})
        return None
    for remote in runs:
        if remote.name != f"yolo26s-lamp-su1.20x-seed42-{RECOVERY_METHOD}":
            continue
        entry = {"run_url": remote.url, "eligible": False}
        try:
            artifacts = list(remote.logged_artifacts())
            collections = {label: [a for a in artifacts if a.name.rsplit(":", 1)[0] == remote.name + "-" + label]
                           for label in ("results", "resume")}
            if not all(collections.values()):
                raise ValueError("Paket results/resume historis tidak lengkap.")
            refs = {}
            cache = folder / "historical" / remote.id
            for label, values in collections.items():
                artifact = max(values, key=lambda a: int(a.qualified_name.rsplit(":v", 1)[1]))
                refs[label] = ablation_download(artifact, cache / label)
            result = ft_read(cache / "results/evaluation_manifest.json")
            training = ft_read(cache / "results/training_config.json")
            contract = ft_read(cache / "resume/resume_contract.json")["contract"]
            proof = {**result, "status": remote.summary.get("status"), "logging_verified": remote.summary.get("logging_verified"),
                     "hyperparameters": remote.config.get("method_hyperparameters"),
                     "implementation": contract.get("implementation"),
                     "training": training["effective"],
                     "evaluation": remote.config.get("study_settings", {}).get("evaluation"),
                     "environment": {k: result.get("environment", {}).get(k) for k in ABLATION_SETTINGS["environment"]}}
            entry.update(ablation_reuse_check(proof, record))
            if entry["eligible"]:
                index = ft_read(cache / "resume/latest_state.json")
                state = torch.load(cache / "resume" / index["relative_path"], map_location="cpu", weights_only=False)
                if not state.get("training_finished") or state["epoch"] != EPOCHS or not state.get("optimizer_steps"):
                    raise ValueError("State training historis belum lengkap.")
                if (state["raw_sha256"] != ft_identity()["raw_sha256"]
                        or state["raw_state_sha256"] != recovery_state_sha256(raw)
                        or any(recovery_structure_signature(state[k]) != recovery_structure_signature(raw) for k in ("live_model", "ema"))):
                    raise ValueError("State historis tidak cocok dengan raw terverifikasi.")
                best_path = cache / "resume" / state["best"]["relative_path"]
                if sha256_file(best_path) != state["best"]["sha256"] or sha256_file(best_path) != result["checkpoint_artifact"]["files_sha256"]["best.pt"]:
                    raise ValueError("Best historis tidak cocok dengan hasil evaluasi.")
                target = folder / "historical_best.pt"
                shutil.copyfile(best_path, target)
                state["best"] = {**state["best"], "relative_path": target.name}
                write_json(folder / "kd_training_audit.json", result["kd_audit"])
                write_json(folder / "epoch_metrics.json", state["epoch_rows"])
                write_json(folder / "training_config.json", training)
                audit.update(status="REUSED", source_run_url=remote.url, artifact_refs=refs)
                audit["checks"].append(entry)
                write_json(folder / "default_reuse_audit.json", audit)
                return state
        except Exception as error:  # noqa: BLE001 -- Unverifiable historical runs must be audited then rerun.
            entry.update(eligible=False, error=safe_error(error))
        audit["checks"].append(entry)
    write_json(folder / "default_reuse_audit.json", audit)
    return None


class AblationKDTrainer(SelectedKDTrainer):
    def save_model(self):
        # Attach peaks to the committed epoch and therefore to every resume snapshot.
        self.ablation_memory = ({"allocated_bytes": torch.cuda.max_memory_allocated(self.device),
                                 "reserved_bytes": torch.cuda.max_memory_reserved(self.device)}
                                if self.device.type == "cuda" else None)
        return super().save_model()

    def recovery_extra_state(self):
        memory = getattr(self, "ablation_memory", None)
        previous = (self.recovery_resume_state or {}).get("method_state", {}).get("ablation_memory")
        if memory and previous:
            memory = {key: max(memory[key], previous[key]) for key in memory}
        return {**super().recovery_extra_state(), "ablation_memory": memory}


SelectedFineTuneTrainer = AblationKDTrainer


def ft_fit(raw, raw_path, manifest, dataset, record, train, data_yaml, folder, run, contract):
    # Synthetic probe uses copies; it cannot update raw, BN, optimizer, or the epoch RNG.
    ablation_smoke(raw, folder)
    reused = ablation_find_default(folder, record, raw) if not (folder / "latest_state.json").exists() else None
    if reused is not None:
        if run is not None:
            provenance = ft_read(folder / "default_reuse_audit.json")
            run.config.update({"training_origin": "REUSED", "reuse_provenance": provenance})
            for reference in provenance["artifact_refs"].values():
                run.use_artifact(reference)
            for row in reused["epoch_rows"]:
                run.log({**recovery_epoch_rows([row])[0], "epoch_seconds": row.get("epoch_seconds")})
        return reused
    if DEVICE != "cpu":
        torch.cuda.reset_peak_memory_stats(DEVICE)
    previous = ft_read(folder / "resource_segments.json", [])
    started = time.perf_counter()
    try:
        state = _ablation_fit(raw, raw_path, manifest, dataset, record, train, data_yaml, folder, run, contract)
        write_json(folder / "training_resources.json", state["method_state"].get("ablation_memory"))
        return state
    finally:
        segment = {"wall_seconds": time.perf_counter() - started,
                   "allocated_bytes": torch.cuda.max_memory_allocated(DEVICE) if DEVICE != "cpu" else None,
                   "reserved_bytes": torch.cuda.max_memory_reserved(DEVICE) if DEVICE != "cpu" else None}
        write_json(folder / "resource_segments.json", [*previous, segment])


def ablation_row(result, folder):
    epochs = ft_read(folder / "epoch_metrics.json", [])
    best = next((v for v in epochs if v["epoch"] == result.get("best_epoch")), {})
    losses = best.get("losses", {})
    resources = ft_read(folder / "resource_segments.json", [])
    cumulative = ft_read(folder / "training_resources.json")
    if cumulative:
        resources = [*resources, cumulative]
    peaks = [v for v in resources if v.get("allocated_bytes") is not None]
    metrics = result.get("stages", {}).get("recovered", {}).get("validation", {})
    return {**ft_identity(), "run_dir": str(folder), "run_name": ft_name(), "group": ft_group(), "mode": MODE,
            "run_id": ft_read(folder / "wandb_identity.json", {}).get("id"),
            "wandb_run_url": result.get("logging_verification", {}).get("run_url"),
            "status": result.get("status", "NOT_RUN"), "publication_status": result.get("publication_status", "NOT_RUN"),
            "logging_verified": result.get("logging_verification", {}).get("status") == "PASS",
            "comparison_signature": globals().get("ABLATION_SIGNATURE"),
            "configuration": globals().get("ABLATION_SETTINGS"),
            "best_epoch": result.get("best_epoch"), "epochs_completed": result.get("epochs"),
            "final_validation": {k: metrics.get(k) for k in ("map50_95", "map50", "precision", "recall")},
            "best_epoch_detection_loss": losses.get("det_total_loss"),
            "best_epoch_distillation_loss": losses.get("dis_loss" if RECOVERY_METHOD == "crosskd" else "mgd_weighted_loss"),
            "best_epoch_loss_components": losses,
            "training_seconds": sum(v.get("epoch_seconds", 0) or 0 for v in epochs) if epochs else None,
            "peak_allocated_bytes": max((v["allocated_bytes"] for v in peaks), default=None),
            "peak_reserved_bytes": max((v["reserved_bytes"] for v in peaks), default=None),
            "training_origin": ft_read(folder / "default_reuse_audit.json", {}).get("status", "NEW"),
            "artifacts": {k: v.get("reference") for k, v in ft_read(folder / "artifact_publications.json", {}).items()},
            "best_sha256": result.get("stages", {}).get("recovered", {}).get("checkpoint_sha256")
                or result.get("checkpoint_artifact", {}).get("files_sha256", {}).get("best.pt")}


def ft_registry(result, folder):
    original = _ablation_registry(result, folder)
    row = ablation_row(result, folder)
    root = ABLATION_ROOT / ("smoke" if MODE == "smoke" else "research")
    root.mkdir(parents=True, exist_ok=True)
    with FileLock(str(root / ".registry.lock"), timeout=30):
        path = root / "ablation_registry.json"
        rows = ft_read(path, {"rows": []})["rows"]
        rows = [v for v in rows if v["run_dir"] != str(folder)] + [row]
        write_json(path, {"format_version": 1, "rows": rows})
        write_csv(root / "ablation_registry.csv", [{**v, "final_validation": json.dumps(v["final_validation"]),
                   "configuration": json.dumps(v["configuration"]), "artifacts": json.dumps(v["artifacts"]),
                   "best_epoch_loss_components": json.dumps(v["best_epoch_loss_components"])} for v in rows])
    write_json(folder / "ablation_record.json", row)
    return {**original, **{k: v for k, v in row.items() if isinstance(v, (str, int, float, bool)) or v is None}}


def ft_log_results(run, result, folder):
    _ablation_log(run, result, folder)
    metrics = result["stages"]["recovered"]["validation"]
    run.summary.update({f"final_validation/{k}": metrics[k] for k in ("map50_95", "map50", "precision", "recall")})
    run.config.update({"ablation_settings": ABLATION_SETTINGS, "ablation_comparison_signature": ABLATION_SIGNATURE,
                       "method_source_sha256": METHOD_SOURCE_SHA256})
    for name in ("ablation_smoke.json", "default_reuse_audit.json", "resource_segments.json"):
        value = ft_read(folder / name)
        if value is not None:
            write_json(folder / "eda" / name, value)


def ablation_comparison(rows, signature):
    values = [0.5, 1.0, 2.0] if RECOVERY_METHOD == "crosskd" else [1e-5, 2e-5, 4e-5]
    selected = []
    for value in values:
        candidates = [r for r in rows if r.get("method") == RECOVERY_METHOD and r.get("mode") == MODE
                      and r.get("ablation_value") == value and r.get("comparison_signature") == signature]
        complete = [r for r in candidates if r.get("status") == "PASS" and r.get("publication_status") == "CONFIRMED"
                    and r.get("logging_verified") and r.get("epochs_completed") == EPOCHS
                    and isinstance(r.get("final_validation", {}).get("map50_95"), (int, float))
                    and math.isfinite(r["final_validation"]["map50_95"])]
        # Preserve every attempt in registry; prefer latest verified attempt, never maximum score.
        row = (complete or candidates or [{}])[-1]
        selected.append({**row, "ablation_value": value, "eligible": bool(complete),
                         "status": row.get("status", "NOT_RUN")})
    winners = []
    if all(r["eligible"] for r in selected):
        score = max(r["final_validation"]["map50_95"] for r in selected)
        winners = [r["ablation_value"] for r in selected if r["final_validation"]["map50_95"] == score]
    return selected, {"status": "COMPLETE" if winners else "PENDING", "best_values": winners,
                      "boundary_optimum": any(v in (values[0], values[-1]) for v in winners),
                      "warning": "Optimum global belum terbukti." if any(v in (values[0], values[-1]) for v in winners) else None}


def ablation_report(sync=False):
    root = ABLATION_ROOT / ("smoke" if MODE == "smoke" else "research")
    settings = ft_read(root / "protocol.json", {}).get("contract")
    if settings is None:
        print("Belum ada protokol dataset terverifikasi.")
        return None
    signature = workflow_digest({**settings, "method": RECOVERY_METHOD, "method_source_sha256": METHOD_SOURCE_SHA256})
    globals()["ABLATION_SIGNATURE"] = signature
    rows = ft_read(root / "ablation_registry.json", {"rows": []})["rows"]
    if sync:
        workflow_login()
        for remote in wandb.Api(timeout=30).runs(f"{WANDB_ENTITY}/{WANDB_PROJECT}", filters={"group": ft_group()}):
            if remote.config.get("ablation_comparison_signature") != signature or not remote.summary.get("logging_verified"):
                continue
            artifacts = [a for a in remote.logged_artifacts() if a.name.rsplit(":", 1)[0] == remote.name + "-ablation-record"]
            if artifacts:
                artifact = max(artifacts, key=lambda a: int(a.qualified_name.rsplit(":v", 1)[1]))
                cache = root / "remote_registry" / remote.id
                ablation_download(artifact, cache)
                row = ft_read(cache / "ablation_record.json")
                if row["run_id"] == remote.id and row["comparison_signature"] == signature:
                    rows = [r for r in rows if r.get("run_id") != row["run_id"]] + [row]
    rows.sort(key=lambda r: r.get("run_name", ""))
    selected, decision = ablation_comparison(rows, signature)
    folder = root / "summaries" / RECOVERY_METHOD
    folder.mkdir(parents=True, exist_ok=True)
    flat = [{**{k: r.get(k) for k in ("ablation_value", "status", "eligible", "best_epoch", "best_epoch_detection_loss",
              "best_epoch_distillation_loss", "training_seconds", "peak_allocated_bytes", "peak_reserved_bytes", "wandb_run_url")},
             **r.get("final_validation", {})} for r in selected]
    write_json(folder / "comparison.json", {"rows": selected, "selection": decision})
    write_csv(folder / "comparison.csv", flat)
    display(pd.DataFrame(flat))
    print(json.dumps(decision, ensure_ascii=False))
    fig, ax = _eda_plt.subplots(figsize=(8, 4))
    points = [r for r in selected if r.get("final_validation", {}).get("map50_95") is not None]
    ax.plot([r["ablation_value"] for r in points], [r["final_validation"]["map50_95"] for r in points], marker="o")
    ax.set(xlabel=ABLATION_PARAMETER, ylabel="final validation mAP50–95", xticks=[r["ablation_value"] for r in selected])
    ax.set_ylim(0, 1)
    fig.suptitle(f"{RECOVERY_METHOD.upper()} | {MODE} | seed 42")
    fig.tight_layout(); fig.savefig(folder / "sensitivity.png"); _eda_plt.close(fig)
    fig, axes = _eda_plt.subplots(1, 3, figsize=(16, 4))
    for row in selected:
        epochs = ft_read(Path(row.get("run_dir", "__missing__")) / "epoch_metrics.json", [])
        if not epochs:
            epochs = row.get("epoch_history", [])
        for ax, getter in zip(axes, (lambda e: e.get("losses", {}).get("det_total_loss"),
                                    lambda e: e.get("losses", {}).get("dis_loss" if RECOVERY_METHOD == "crosskd" else "mgd_weighted_loss"),
                                    lambda e: e.get("metrics", {}).get("metrics/mAP50-95(B)")), strict=True):
            points = [(e["epoch"], getter(e)) for e in epochs if getter(e) is not None]
            if points:
                ax.plot(*zip(*points), label=str(row["ablation_value"]), marker="o" if len(points) < 4 else None)
    for ax, title in zip(axes, ("Training detection loss", "Training weighted KD loss", "Ultralytics validation history"), strict=True):
        ax.set(xlabel="epoch", title=title)
        if ax.lines:
            ax.legend()
    axes[2].set_ylim(0, 1)
    fig.suptitle(f"{RECOVERY_METHOD.upper()} | {MODE} | seed 42")
    fig.tight_layout(); fig.savefig(folder / "training_comparison.png"); _eda_plt.close(fig)
    if sync:
        # A separate analysis run keeps finished training runs and immutable results untouched.
        with wandb.init(entity=WANDB_ENTITY, project=WANDB_PROJECT, group=ft_group(), job_type="ablation-analysis",
                        name=ft_group() + "-summary-" + uuid.uuid4().hex[:8], dir=str(folder),
                        config={"comparison_signature": signature}) as run:
            run.log({"ablation/comparison": wandb.Table(dataframe=pd.DataFrame(flat)),
                     "ablation/sensitivity": wandb.Image(str(folder / "sensitivity.png")),
                     "ablation/training_curves": wandb.Image(str(folder / "training_comparison.png"))})
            run.summary.update(decision)
            ft_publish(run, folder, "results", {p.name: p for p in folder.iterdir() if p.suffix in {".csv", ".json", ".png"}
                       and p.name in {"comparison.json", "comparison.csv", "sensitivity.png", "training_comparison.png"}},
                       {"comparison_signature": signature}, event="summary-" + uuid.uuid4().hex[:8])
    return selected, decision


def ablation_execute(folder):
    result = ft_execute(folder)
    if result.get("status") == "PASS" and WANDB_TRACKING:
        # Publish the final registry only after the inherited upload/history verification succeeds.
        row = ablation_row(result, folder)
        row["epoch_history"] = ft_read(folder / "epoch_metrics.json", [])
        # A retry must upload exactly the same bytes even after the publication journal grows.
        payload = folder / "ablation_record_payload.json"
        if not payload.exists():
            write_json(payload, row)
        identity = ft_read(folder / "wandb_identity.json")
        with wandb.init(entity=WANDB_ENTITY, project=WANDB_PROJECT, id=identity["id"], resume="must",
                        name=identity["name"], group=identity["group"], dir=str(folder)) as run:
            ft_publish(run, folder, "ablation-record", {"ablation_record.json": payload}, ft_identity())
    return result
