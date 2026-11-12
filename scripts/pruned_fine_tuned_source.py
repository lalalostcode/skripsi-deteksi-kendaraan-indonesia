"""Standalone baseline-matched fine-tuning of a single pruned YOLO26s student."""
# ruff: noqa: F821 -- Common declarations are embedded in the preceding notebook cells.
# ruff: noqa: BLE001 -- Record failures per stage and retain local results.
from copy import deepcopy


def fine_tuned_source(folder):
    reference = None
    if RAW_ARTIFACT_REF:
        reference = workflow_version_ref(RAW_ARTIFACT_REF)
        workflow_login()
        artifact = wandb.Api().artifact(reference, type="model")
        path = Path(artifact.download(root=str(folder / "raw_artifact"))) / "raw_pruned_manifest.json"
    else:
        path = Path(RAW_MANIFEST_PATH).expanduser().resolve(strict=True)
    manifest = recovery_read_json(path)
    target = manifest.get("target_speedup")
    if (manifest.get("stage") != "raw_pruned" or manifest.get("format_version") != 2
            or manifest.get("model_variant") != "s" or type(target) not in {int, float}
            or not 1 < target <= 1.70 or manifest.get("configuration_id") != f"target-{target:.2f}x"
            or manifest.get("target_status") not in {"REACHED", "TARGET_NOT_REACHED"}
            or any(manifest.get(key) != "PASS" for key in ("structural_status", "oracle_status", "reload_status"))):
        raise ValueError("Input harus manifest raw pruned YOLO26s notebook 06 dengan audit struktur, oracle, dan reload PASS.")
    baseline = manifest["source_baseline"]
    if baseline["sha256"] != BASELINE_SHA256 or baseline["requested_artifact"] != BASELINE_REF:
        raise ValueError("Baseline sumber bukan YOLO26s v62 yang dikunci untuk pruning.")

    raw = manifest["raw_checkpoint"]
    saved_baseline = manifest["baseline_checkpoint"]
    raw_path = path.parent / raw["relative_path"]
    baseline_path = path.parent / saved_baseline["artifact_path"]
    if not baseline_path.is_file() and not reference:
        baseline_path = path.parent / saved_baseline["relative_path"].replace("\\", "/")
    files = {"raw_pruned_manifest.json": path, "pruned_fp32.pt": raw_path.resolve(strict=True),
             "baseline_fp32.pt": baseline_path.resolve(strict=True)}
    if EXPECTED_RAW_SHA256 is not None and raw["sha256"] != EXPECTED_RAW_SHA256.lower():
        raise ValueError("SHA-256 raw berbeda dari student lock.")
    expected_files = {"pruned_fp32.pt": raw["sha256"], "baseline_fp32.pt": saved_baseline["sha256"]}
    for name, expected in manifest.get("audit_files", {}).items():
        if Path(name).name != name or name in files:
            raise ValueError(f"Nama file audit raw tidak valid: {name}")
        files[name] = path.parent / name
        expected_files[name] = expected
    for name, expected in expected_files.items():
        if sha256_file(files[name]) != expected:
            raise ValueError(f"SHA-256 input tidak cocok: {name}")
    return {"path": path, "manifest": manifest, "reference": reference}, files


def fine_tuned_dataset(manifest, folder):
    identity = manifest["dataset"]
    if MODE == "full_val" and (manifest["evaluation_mode"] != "full_val" or manifest["validation_status"] != "PASS"
                               or identity["provenance_status"] != "VERIFIED" or identity["dataset_id"] != POLTRADA_V3_ID):
        raise ValueError("Run penelitian membutuhkan raw hasil full validation Poltrada V3 VERIFIED.")
    expected_val = identity["fingerprint"] if manifest["evaluation_mode"] == "full_val" else None
    dataset = gflops_prepare_dataset(MODE, DATA_YAML, DATASET_ROOT, folder, seed=manifest["seed"],
                                    expected_fingerprint=expected_val, dataset_id=DATASET_ID)
    if dataset is None:
        raise FileNotFoundError("Atur DATASET_ROOT dan DATA_YAML Poltrada V3 dengan train dan val asli.")
    record, train, _ = workflow_dataset_manifest(dataset, DATA_YAML, DATASET_ROOT, folder)
    if (record["dataset_id"] != identity["dataset_id"] or record["class_names"] != identity["class_names"]
            or record["train_fingerprint"] != identity["train_fingerprint"]
            or (expected_val is None and dataset["fingerprint"] != identity["fingerprint"])):
        raise ValueError("Fingerprint data train/val atau urutan kelas berbeda dari manifest raw.")
    train_list = folder / "train_images.txt"
    train_list.write_text("".join(f"{image.as_posix()}\n" for image in train), encoding="utf-8")
    data_yaml = folder / "train_val_data.yaml"
    data_yaml.write_text(yaml.safe_dump({**dataset["info"], "train": str(train_list.resolve())},
                                       allow_unicode=True, sort_keys=False), encoding="utf-8")
    return dataset, record, train, data_yaml


def fine_tuned_check_heads(core, names):
    if type(core) is not _RecoveryDetectionModel:
        raise ValueError("Diperlukan DetectionModel YOLO26 native dual-head.")
    head = core.model[-1]
    # Native validation sets the EMA inference flag to one-to-many while preserving both heads.
    if (type(head) is not Detect or recovery_ordered_names(core.names) != names or head.nc != len(names)
            or getattr(head, "one2one_cv2", None) is None or getattr(head, "one2one_cv3", None) is None
            or any(not isinstance(branch, torch.nn.ModuleList) or len(branch) != head.nl
                   for branches in (head.one2many, head.one2one) for branch in branches.values())):
        raise ValueError("Cabang Detect one-to-many/one-to-one atau urutan kelas tidak valid.")


def fine_tuned_load_raw(path, manifest):
    # Only deserialize the trusted input package, after all file hashes match.
    payload = torch.load(path, map_location="cpu", weights_only=False)
    core = payload.get("model")
    fine_tuned_check_heads(core, manifest["dataset"]["class_names"])
    signature = recovery_structure_signature(core)
    serialized = {"parameters": signature["parameters"], "conv_bn": {
        name: [layer["in"], layer["out"], layer["groups"]] if "in" in layer else [layer["features"]]
        for name, layer in signature["layers"].items()}}
    if (payload.get("recovery_enabled") is not False or not core.end2end or serialized != manifest["structure_signature"]
            or model_state_sha256(core) != manifest["state_sha256"] or not _recovery_finite_state(core)
            or any(p.dtype != torch.float32 for p in core.parameters())):
        raise ValueError("Dimensi atau tensor checkpoint raw tidak cocok dengan manifest FP32.")
    return core


class BaselinePrunedTrainer(DetectionTrainer):
    """Inject a verified pruned model and leave the native training loop intact."""

    def __init__(self, raw, *, overrides):
        self.raw = raw
        super().__init__(overrides=overrides)

    def setup_model(self):
        if self.resume:
            raise ValueError("Recovery harus mulai dari student raw dengan optimizer baru.")
        self.model = deepcopy(self.raw)
        self.model.criterion = None
        self.model.requires_grad_(True)

    def get_model(self, cfg=None, weights=None, verbose=True):
        raise RuntimeError("Rekonstruksi model dari YAML dilarang untuk student pruned.")

    def run_callbacks(self, event):
        for callback in self.callbacks.get(event, []):
            module = getattr(callback, "__module__", None) or ""
            if not module.startswith("ultralytics.utils.callbacks.") or module.endswith(".base"):
                callback(self)


def fine_tuned_check_images(loader, expected):
    observed = [Path(path).resolve() for path in loader.dataset.im_files]
    if len(observed) != len(expected) or set(observed) != {Path(path).resolve() for path in expected}:
        raise ValueError("Citra aktual dataloader berbeda dari split yang diverifikasi.")


def fine_tuned_fit(raw, raw_path, data_yaml, train, dataset, folder, run, errors, teacher_path, teacher_contract):
    options = fine_tuned_train_args(str(data_yaml), str(folder), "training", teacher_path=teacher_path)
    trainer_class = SourceMatchedKDTrainer if teacher_path else BaselinePrunedTrainer
    trainer = trainer_class(raw, overrides={**options, "model": str(raw_path), "task": "detect", "mode": "train"})
    initial = recovery_state_sha256(raw)
    initial_parameters = recovery_state_sha256(raw, parameters_only=True)
    signature = recovery_structure_signature(raw)
    epochs = []

    def check_protocol(trainer):
        if trainer.batch_size != options["batch"] or trainer.amp != options["amp"]:
            raise RuntimeError("Batch atau AMP efektif berubah dari protokol yang diminta.")

    def check_start(trainer):
        check_protocol(trainer)
        student = getattr(trainer.model, "student_model", trainer.model)
        fine_tuned_check_heads(student, dataset["class_names"])
        if recovery_state_sha256(student) != initial or recovery_structure_signature(student) != signature:
            raise ValueError("Bobot awal training harus identik dengan raw pruned.")
        if teacher_contract:
            source_matched_check_teacher(trainer, teacher_contract, check_digest=True)
        fine_tuned_check_images(trainer.train_loader, train)
        fine_tuned_check_images(trainer.test_loader, dataset["image_files"])
        write_json(folder / "training_config.json", {
            "requested": options, "effective": vars(trainer.args), "amp_effective": trainer.amp,
            "training_batch": trainer.train_loader.batch_size,
            "training_validation_batch": trainer.test_loader.batch_size, "initial_state_sha256": initial,
            "optimizer_class": type(trainer.optimizer).__name__, "scheduler_class": type(trainer.scheduler).__name__,
        })

    def log_epoch(trainer):
        check_protocol(trainer)
        epoch = int(trainer.epoch) + 1
        if teacher_contract:
            source_matched_check_teacher(trainer, teacher_contract)
        row = {"epoch": epoch, "fitness": _validation_number(trainer.fitness),
               "losses": {name: _validation_number(value) for name, value in trainer.tloss.items()},
               "metrics": {name: _validation_number(value) for name, value in trainer.metrics.items()},
               "optimizer_steps": trainer.ema.updates,
               "learning_rates": [float(group["lr"]) for group in trainer.optimizer.param_groups]}
        epochs.append(row)
        write_json(folder / "epoch_metrics.json", epochs)
        if run is not None:
            try:
                run.log(recovery_epoch_rows([row])[0], step=epoch)
            except Exception as error:
                errors.append({"stage": "upload", "epoch": epoch, "error": safe_error(error)})

    trainer.add_callback("on_train_start", check_start)
    trainer.add_callback("on_train_epoch_start", check_protocol)
    trainer.add_callback("on_model_save", log_epoch)
    with gflops_disable_label_cache():
        trainer.train()
    if len(epochs) != options["epochs"] or not trainer.best.is_file():
        raise RuntimeError("Training belum menyelesaikan budget epoch atau best.pt tidak tersedia.")
    student = getattr(trainer.model, "student_model", trainer.model)
    final_parameters = recovery_state_sha256(student, parameters_only=True)
    if final_parameters == initial_parameters or recovery_structure_signature(student) != signature:
        raise RuntimeError("Bobot student tidak berubah selama training.")
    best = YOLO(str(trainer.best)).model
    fine_tuned_check_heads(best, dataset["class_names"])
    if recovery_structure_signature(best) != signature or not _recovery_finite_state(best):
        raise RuntimeError("Reload best.pt gagal mempertahankan struktur raw atau mengandung tensor nonfinite.")
    if teacher_contract:
        source_matched_check_teacher(trainer, teacher_contract, check_digest=True)
        if sha256_file(teacher_path) != teacher_contract["sha256"]:
            raise RuntimeError("Checkpoint teacher berubah selama training.")
        trainer.model._remove_feature_hooks()
    return trainer.best, epochs, {"initial_state_sha256": initial, "structure_signature": signature,
                                  "initial_parameters_sha256": initial_parameters,
                                  "trained_parameters_sha256": final_parameters,
                                  "teacher_frozen": True if teacher_contract else None}


def fine_tuned_measure(path, data_yaml, dataset, folder, stage):
    model = YOLO(str(path))
    fine_tuned_check_heads(model.model, dataset["class_names"])
    core = model.model
    core.end2end = True  # Match notebook 06 for profiling. Native val selects its own head mode.
    complexity = model_measurements(core, IMGSZ)
    channels = sum(module.out_channels for module in core.modules() if isinstance(module, torch.nn.Conv2d))
    effective = {}
    diagnostics = {}

    def capture(validator):
        fine_tuned_check_images(validator.dataloader, dataset["image_files"])
        fields = ("split", "imgsz", "batch", "conf", "iou", "max_det", "rect", "quantize", "augment", "workers", "nms")
        effective.update({key: getattr(validator.args, key, None) for key in fields}, device=str(validator.device),
                         end2end=bool(validator.end2end), dataloader_batch=validator.dataloader.batch_size)
        diagnostics["confusion_matrix"] = _validation_confusion_matrix(validator, dataset["class_names"], effective)

    model.add_callback("on_val_end", capture)
    arguments = fine_tuned_val_args(str(data_yaml))
    with gflops_disable_label_cache(), gflops_require_complete_nms():
        metrics = model.val(**arguments, project=str(folder), name=stage)
    overall = {name: _validation_number(getattr(metrics.box, attr)) for name, attr in
               {"map50_95": "map", "map50": "map50", "precision": "mp", "recall": "mr"}.items()}
    if not effective or any(value is None for value in overall.values()):
        raise RuntimeError("Metrik validator baseline belum lengkap.")
    classes, class_ids = _validation_per_class(metrics, dataset["class_names"])
    validation = {**overall, **diagnostics, "validation_arguments_requested": arguments, "validation_arguments_effective": effective,
                  "validation_image_count": len(dataset["image_files"]), "dataset_fingerprint": dataset["fingerprint"],
                  "per_class": classes, "curves": _validation_curves(metrics, dataset["class_names"], class_ids),
                  "inference_ms_per_image": _validation_number(metrics.speed.get("inference"))}
    latency = {"status": "NOT_RUN"}
    if RUN_LATENCY_BENCHMARK and DEVICE != "cpu":
        try:
            device = torch.device(effective["device"])
            image = torch.zeros((1, 3, IMGSZ, IMGSZ), device=device)
            # Use the untouched checkpoint, since the validator may fuse its inference model.
            latency_core = YOLO(str(path)).model.float().to(device).eval()
            latency_core.end2end = True
            latency = cuda_latency(latency_core, image)
        except (RuntimeError, ValueError) as error:
            latency = {"status": "FAIL", "error": safe_error(error)}
    return {"validation_status": "PASS", "validation": validation, **complexity,
            "gflops_protocol": "notebook-06 / FP32 unfused dual-head", "conv_output_channels": channels, "latency": latency,
            "checkpoint_bytes": Path(path).stat().st_size}


def fine_tuned_comparison(result):
    rows, classes = recovery_comparison_rows(result)
    for row in rows:
        stage = result["stages"].get(row["stage"], {})
        row["inference_ms_per_image"] = stage.get("validation", {}).get("inference_ms_per_image")
    return rows, classes


def fine_tuned_eda(result, folder, epochs, files):
    from matplotlib.patches import Patch

    folder = folder / "eda"
    rows, classes = fine_tuned_comparison(result)
    panels = [
        ("accuracy", "Metrik deteksi", "Proporsi", tuple(RECOVERY_ACCURACY_FIELDS.items()), True),
        ("drop", "Penurunan terhadap baseline", "Poin persentase", (("map50_95_drop_pp", "mAP50–95"),), False),
        ("gain", "Gain terhadap raw pruned", "Poin persentase", (("map50_95_recovery_gain_pp", "mAP50–95"),), False),
        ("parameters", "Parameter", "Jumlah parameter", (("parameters", "Parameter"),), False),
        ("gflops", "Komputasi", "GFLOPs", (("gflops", "Komputasi"),), False),
        ("channels", "Lebar model", "Jumlah output channel Conv2d", (("conv_output_channels", "Channel"),), False),
        ("bytes", "Ukuran berkas checkpoint", "Byte", (("checkpoint_bytes", "Checkpoint"),), False),
        ("latency", "Raw forward CUDA FP32 · batch 1", "Milidetik", (("latency_median_ms", "p50"), ("latency_p95_ms", "p95")), False),
        ("speedup", "Speed-up terhadap baseline", "Rasio (×)", (("gflops_speedup", "GFLOPs"), ("latency_speedup", "Latency p50")), False),
        ("inference", "Waktu inference dari validator", "Milidetik per citra",
         (("inference_ms_per_image", "Inference"),), False),
    ]
    plots = [(name, title, unit, bounded,
              [{"group": key, "metric": key, "group_label": label, "stage": row["stage"], "value": row[key], "unit": unit}
               for key, label in fields for row in rows]) for name, title, unit, fields, bounded in panels]
    for metric, label in (("ap50_95", "AP50–95"), ("ap50", "AP50")):
        plots.append((metric, f"{label} per kelas", "Proporsi", True,
                      [{"group": row["class_id"], "metric": metric, "group_label": row["name"], "stage": row["stage"],
                        "value": row[metric], "unit": "Proporsi"} for row in classes]))
    plot_data = []
    for name, title, unit, bounded, values in plots:
        plot_data.extend({"figure": name, **value} for value in values)
        fig, ax = _eda_plt.subplots(figsize=(8, 4.5))
        _recovery_draw_bar_panel(ax, values, title=title, unit=unit, bounded=bounded)
        ax.legend(handles=[Patch(color=color, label=label) for label, color in RECOVERY_STAGE_STYLES.values()])
        fig.tight_layout()
        _eda_save_figure(fig, folder / f"comparison_{name}")
    _recovery_epoch_plots(recovery_epoch_rows(epochs), folder / "training")
    write_json(folder / "pruning_diagnostics.json", _recovery_pruning_diagnostics(result, files, folder / "pruning"))
    write_json(folder / "validation_diagnostics.json", _recovery_stage_validation_plots(result, folder / "validation"))
    for name, values in (("stage_metrics", rows), ("per_class_metrics", classes), ("plot_data", plot_data)):
        write_json(folder / f"{name}.json", values)
        write_csv(folder / f"{name}.csv", values)


def fine_tuned_execute(folder):
    result = {"status": "NOT_RUN", "output": str(folder), "platform": PLATFORM, "model_variant": "s",
              "method": RECOVERY_METHOD, "protocol": "source-matched-v1", "source_notebook": SOURCE_NOTEBOOK,
              "resume_status": "RESUME_NOT_VERIFIED", "mode": MODE, "seed": SEED, "stages": {}, "errors": []}
    result.update({f"{stage}_status": "NOT_RUN" for stage in ("input", "training", "reload", "validation", "eda", "upload")})
    manifest_path = folder / "fine_tuned_manifest.json"
    stage = "input"
    try:
        source, files = fine_tuned_source(folder)
        manifest = source["manifest"]
        target = manifest["target_speedup"]
        dataset, record, train, data_yaml = fine_tuned_dataset(manifest, folder / "data")
        raw_path = files["pruned_fp32.pt"]
        raw = fine_tuned_load_raw(raw_path, manifest)
        teacher_path, teacher_contract = (source_matched_teacher(folder, raw) if RECOVERY_METHOD != "finetune" else (None, None))
        raw_sha = manifest["raw_checkpoint"]["sha256"]
        result.update(raw_sha256=raw_sha, raw_pruning_seed=manifest["seed"], target_speedup=target,
                      target_status=manifest["target_status"], configuration_id=manifest["configuration_id"],
                      raw_gflops_speedup=manifest["actual_gflops_speedup"], dataset=record,
                      baseline_artifact_ref=BASELINE_REF, baseline_sha256=manifest["baseline_checkpoint"]["sha256"],
                      baseline_source_sha256=BASELINE_SHA256, environment=ENVIRONMENT, input_status="PASS", teacher=teacher_contract)
        stage = "upload"
        raw_ref = source["reference"] or recovery_raw_reference(source, files, folder, target)
        result["raw_artifact_ref"] = raw_ref
        write_json(manifest_path, result)
        with workflow_tracking(folder, "s", target, recovery=True, source_ref=raw_ref,
                               group=f"lamp-source-matched-yolo26s-{PLATFORM}-{MODE}") as run:
            if run is not None:
                if teacher_contract and teacher_contract["reference"]:
                    run.use_artifact(teacher_contract["reference"], type="model")
                run.config.update({"protocol": result["protocol"], "source_notebook": SOURCE_NOTEBOOK,
                                   "teacher": teacher_contract, "raw_artifact_ref": raw_ref, "raw_sha256": raw_sha,
                                   "raw_pruning_seed": manifest["seed"], "target_status": manifest["target_status"],
                                   "actual_gflops_speedup": manifest["actual_gflops_speedup"], "dataset": record,
                                   "training": fine_tuned_train_args(str(data_yaml), str(folder), "training", teacher_path)})
            stage = "training"
            best, epochs, audit = fine_tuned_fit(raw, raw_path, data_yaml, train, dataset, folder, run, result["errors"],
                                                teacher_path, teacher_contract)
            result.update(training_status="PASS", reload_status="PASS", epochs=len(epochs), **audit)
            result["checkpoint"] = {"relative_path": best.relative_to(folder).as_posix(), "sha256": sha256_file(best)}
            result["raw_unchanged"] = sha256_file(raw_path) == raw_sha
            if not result["raw_unchanged"]:
                raise RuntimeError("SHA-256 raw berubah sesudah training.")
            model_manifest = folder / "fine_tuned_model.json"
            model_fields = ("model_variant", "configuration_id", "target_status", "raw_sha256", "raw_artifact_ref",
                            "seed", "structure_signature", "checkpoint", "epochs", "teacher", "protocol", "source_notebook")
            write_json(model_manifest, {key: result[key] for key in model_fields})
            stage = "validation"
            for name, checkpoint in (("baseline", files["baseline_fp32.pt"]), ("raw_pruned", raw_path), ("recovered", best)):
                result["stages"][name] = fine_tuned_measure(checkpoint, data_yaml, dataset, folder / "validation", name)
                write_json(manifest_path, result)
            baseline, pruned, recovered = result["stages"].values()
            if any(row["validation"]["validation_arguments_effective"] != baseline["validation"]["validation_arguments_effective"]
                   for row in (pruned, recovered)) or any(pruned[key] != recovered[key] for key in ("parameters", "gflops")):
                raise RuntimeError("Protokol evaluasi atau kompleksitas model raw/recovered berbeda.")
            result["validation_status"] = "PASS"
            latency_states = {row["latency"]["status"] for row in result["stages"].values()}
            result["latency_status"] = ("PASS" if latency_states == {"PASS"} else
                                        "NOT_RUN" if latency_states == {"NOT_RUN"} else "PARTIAL")
            stage = "eda"
            fine_tuned_eda(result, folder, epochs, files)
            result["eda_status"] = "PASS"
            if run is not None:
                stage = "upload"
                result["checkpoint_artifact"] = workflow_publish(run, folder, "fine-tuned", {
                    result["checkpoint"]["relative_path"]: best, model_manifest.name: model_manifest}, aliases=["best", "latest"],
                    metadata_info={"raw_sha256": raw_sha, "raw_artifact_ref": raw_ref})
                recovery_log_comparison(run, result, folder)
                reports = {path.relative_to(folder).as_posix(): path for sub in ("eda", "validation")
                           for path in (folder / sub).rglob("*") if path.suffix in {".png", ".svg", ".csv", ".json"}}
                reports["training_config.json"] = folder / "training_config.json"
                reports["epoch_metrics.json"] = folder / "epoch_metrics.json"
                result["results_artifact"] = workflow_publish(run, folder, "fine-tuned-results", reports, kind="results")
                result["upload_status"] = "PARTIAL" if result["errors"] else "PASS"
    except (Exception, KeyboardInterrupt) as error:
        result[f"{stage}_status"] = "FAIL"
        result["reason"] = safe_error(error) or "Interrupted"
        result["errors"].append({"stage": stage, "error": result["reason"]})
    required = ["input", "training", "reload", "validation", "eda"] + (["upload"] if WANDB_TRACKING else [])
    if RUN_LATENCY_BENCHMARK and DEVICE != "cpu":
        required.append("latency")
    result["status"] = ("PASS" if all(result[f"{key}_status"] == "PASS" for key in required) else
                        "PARTIAL" if result["training_status"] == "PASS" or result["stages"] else "FAIL")
    write_json(manifest_path, result)
    return result
