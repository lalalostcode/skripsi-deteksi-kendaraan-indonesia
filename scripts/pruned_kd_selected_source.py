"""KD adapters for the selected-student runtime; the fine-tuning source stays unchanged."""
# ruff: noqa: F821 -- Standalone notebook cells share declarations.

_selected_identity = ft_identity
_selected_settings = ft_study_settings
_selected_tracking = ft_tracking
_selected_fit = ft_fit
_selected_eda = ft_eda
_selected_logging = ft_log_results
_selected_publish = ft_publish
_SelectedTrainer = SelectedFineTuneTrainer

KD_DEFAULT_SETTINGS = {
    "native": {"NATIVE_DIS_WEIGHT": 1.0},
    "crosskd": {"KD_CLS_WEIGHT": 1.0, "KD_BOX_WEIGHT": 1.0, "KD_CLS_GAMMA": 1.0},
    "mgd": {"MGD_ALPHA": 2e-5, "MGD_MASK_RATIO": 0.65},
}


def ft_identity():
    if (TEACHER_SHA256 != LOCKED_TEACHER_SHA256
            or TEACHER_ARTIFACT_REF not in (None, LOCKED_TEACHER_REF)):
        raise ValueError("Teacher harus YOLO26m v50 dengan SHA-256 terkunci.")
    if RECOVERY_METHOD not in KD_DEFAULT_SETTINGS:
        raise ValueError("Metode KD tidak dikenal.")
    return {**_selected_identity(), "method": RECOVERY_METHOD, "teacher_architecture": "YOLO26m",
            "teacher_artifact_ref": LOCKED_TEACHER_REF, "teacher_sha256": LOCKED_TEACHER_SHA256,
            "method_settings_sha256": workflow_digest(recovery_kd_settings(RECOVERY_METHOD))}


def ft_name():
    return f"yolo26s-lamp-su1.20x-seed{ft_identity()['seed']}-{RECOVERY_METHOD}" + ("-smoke" if MODE == "smoke" else "")


def ft_group():
    return f"yolo26s-lamp-kd-{PRUNING_SUITE_ID}" + ("-smoke" if MODE == "smoke" else "")


def kd_comparison_settings():
    settings = _selected_settings()
    for key in ("distill_model", "dis", "compile", "resume"):
        settings["training"].pop(key, None)
    return {**settings, "teacher_sha256": LOCKED_TEACHER_SHA256,
            "augmentation_epoch_seed": "Python/NumPy seed=(manifest seed + epoch) mod 2**32; fresh loader workers"}


def ft_study_settings():
    return {**_selected_settings(), "method": RECOVERY_METHOD,
            "hyperparameters": recovery_kd_settings(RECOVERY_METHOD), "teacher_sha256": LOCKED_TEACHER_SHA256}


def kd_media_staging():
    # W&B 0.28.1 retains these directory names even when Windows removes temp folders.
    from wandb.sdk.data_types._private import MEDIA_TMP
    for path in (MEDIA_TMP.name, wandb.Artifact._TMP_DIR.name):
        Path(path).mkdir(parents=True, exist_ok=True)


def ft_publish(run, folder, label, files, metadata, event="final"):
    kd_media_staging()
    return _selected_publish(run, folder, label, files, metadata, event=event)


@contextmanager
def ft_tracking(folder, manifest, record):
    original_log = None
    run = None
    if WANDB_TRACKING:
        kd_media_staging()
    try:
        with _selected_tracking(folder, manifest, record) as run:
            if run is not None:
                original_log = run.log

                def log(*args, **kwargs):
                    kd_media_staging()
                    return original_log(*args, **kwargs)

                run.log = log
                run.use_artifact(LOCKED_TEACHER_REF, type="model")
                run.config.update({"comparison_signature": workflow_digest(kd_comparison_settings()),
                                   "method_hyperparameters": recovery_kd_settings(RECOVERY_METHOD)})
                run.tags = tuple(dict.fromkeys((*run.tags, "kd", RECOVERY_METHOD, "teacher-yolo26m")))
            yield run
    finally:
        if original_log is not None:
            run.log = original_log


class SelectedKDTrainer(KDStructuralRecoveryTrainer, _SelectedTrainer):
    def __init__(self, raw, **kwargs):
        teacher_path, teacher = source_matched_teacher(kwargs["folder"], raw)
        self.selected_teacher = teacher
        contract = {"method": RECOVERY_METHOD, "teacher_sha256": teacher["sha256"],
                    "teacher_artifact_ref": LOCKED_TEACHER_REF,
                    "hyperparameters": recovery_kd_settings(RECOVERY_METHOD)}
        # The structural trainer calls the KD wrapper explicitly. Keep the SDK from wrapping twice.
        kwargs["overrides"]["distill_model"] = None
        super().__init__(raw, teacher_checkpoint=teacher_path, method_contract=contract, **kwargs)

    def _forward_recovery_batch(self, batch):
        loss, items = super()._forward_recovery_batch(batch)
        items.setdefault("det_total_loss", sum(value for key, value in items.items()
                                               if key in {"box_loss", "cls_loss", "l1_loss", "dfl_loss"}))
        items.setdefault("total_loss", loss.sum().detach() / batch["img"].shape[0])
        return loss, items

    def _restart_epoch_loader(self, epoch):
        # Isolate augmentation from setup/validation imports and reproduce an epoch after restore.
        epoch_seed = (self.args.seed + epoch) % (1 << 32)
        random.seed(epoch_seed)
        np.random.seed(epoch_seed)
        super()._restart_epoch_loader(epoch)

    def plot_metrics(self):
        # The SDK's fixed detection grid cannot accommodate additional KD loss columns.
        _recovery_epoch_plots(recovery_epoch_rows(self.recovery_epochs), self.save_dir / "kd_curves")

    def recovery_extra_state(self):
        state = super().recovery_extra_state()
        state["teacher_record"] = self.selected_teacher
        return state


SelectedFineTuneTrainer = SelectedKDTrainer
RECOVERY_STAGE_STYLES["recovered"] = ("Setelah KD", "#217854")


def ft_fit(raw, raw_path, manifest, dataset, record, train, data_yaml, folder, run, contract):
    state = _selected_fit(raw, raw_path, manifest, dataset, record, train, data_yaml, folder, run, contract)
    extra = state["method_state"]
    teacher = extra["teacher_record"]
    if teacher["sha256"] != LOCKED_TEACHER_SHA256 or teacher["state_sha256"] != extra["teacher_state_sha256"]:
        raise ValueError("Teacher berubah selama training KD.")
    losses = state["epoch_rows"][-1]["losses"]
    kd_losses = {key: value for key, value in losses.items() if key.startswith(("kd_", "mgd_")) or key == "dis_loss"}
    if not kd_losses or any(not math.isfinite(value) for value in losses.values()):
        raise ValueError("Loss KD tidak tersedia atau nonfinite.")
    audit = {**ft_identity(), "status": "PASS", "teacher": teacher, "teacher_unchanged": True,
             "student_initial_state_sha256": recovery_state_sha256(raw),
             "student_final_state_sha256": recovery_state_sha256(state["live_model"]),
             "projector_tensors": len(extra["projector"]), "pending_projector_gradients": len(extra["projector_gradients"]),
             "losses": losses, "kd_losses": kd_losses, "optimizer_steps": state["optimizer_steps"]}
    write_json(folder / "kd_training_audit.json", audit)
    return state


def ft_eda(result, folder, epochs, files):
    audit = ft_read(folder / "kd_training_audit.json")
    result.update(method=RECOVERY_METHOD, teacher=audit["teacher"], kd_audit=audit)
    report = _selected_eda(result, folder, epochs, files)
    write_json(folder / "eda/kd_training_audit.json", audit)
    return report


def ft_log_results(run, result, folder):
    _selected_logging(run, result, folder)
    audit = result["kd_audit"]
    run.summary.update({"teacher_unchanged": audit["teacher_unchanged"],
                        "teacher_state_sha256": audit["teacher"]["state_sha256"],
                        "projector_tensors": audit["projector_tensors"], "kd_audit_status": audit["status"]})


def ft_group_results():
    workflow_login()
    runs = wandb.Api(timeout=30).runs(f"{WANDB_ENTITY}/{WANDB_PROJECT}", filters={"group": ft_group()})
    chosen = {}
    expected = {**KD_DEFAULT_SETTINGS, RECOVERY_METHOD: recovery_kd_settings(RECOVERY_METHOD)}
    signature = workflow_digest(kd_comparison_settings())
    for run in runs:
        values = {key: value for key, value in dict(run.summary).items()
                  if value is None or isinstance(value, (str, int, float, bool))}
        method = values.get("method")
        name = values.get("raw_artifact_ref", "").rsplit("/", 1)[-1]
        if (values.get("status") != "PASS" or not values.get("logging_verified") or method not in expected
                or run.config.get("protocol") != "selected-raw-kd-v1"
                or run.config.get("comparison_signature") != signature
                or run.config.get("method_hyperparameters") != expected[method]
                or values.get("teacher_sha256") != LOCKED_TEACHER_SHA256
                or name not in STUDENT_PINS or values.get("raw_sha256") != STUDENT_PINS[name]["raw_sha256"]):
            continue
        values["platform"] = values.get("platform") or run.config.get("platform")
        values["gpu"] = values.get("gpu") or run.config.get("environment", {}).get("gpu") or "CPU"
        key = (method, name, values["platform"], values["gpu"])
        if key not in chosen or run.created_at > chosen[key][0]:
            chosen[key] = (run.created_at, {**values, "wandb_run_url": run.url})
    rows = [row for _, row in chosen.values()]
    print("Perlakuan tersedia", len({(r["method"], r["raw_artifact_ref"]) for r in rows}), "/ 15")
    frame = pd.DataFrame(rows)
    if not frame.empty:
        metrics = [key for key in frame if key.startswith(("baseline_", "raw_pruned_", "recovered_"))
                   and key.endswith(("map50_95", "map50", "precision", "recall"))]
        display(frame.groupby(["method", "platform", "gpu"], dropna=False)[metrics].agg(["count", "mean", "std"]))
    return rows
