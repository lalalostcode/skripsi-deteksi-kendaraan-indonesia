"""Single selected raw student, native training, resumable epochs and verified W&B output."""
# ruff: noqa: F821 -- Declarations are embedded in the standalone notebook.
import gc
import shutil
from copy import deepcopy
from tempfile import TemporaryDirectory

from filelock import FileLock


def ft_read(path, default=None):
    path = Path(path)
    return json.loads(path.read_text(encoding="utf-8")) if path.is_file() else default


recovery_read_json = ft_read


def ft_login():
    if not os.environ.get("WANDB_API_KEY") and PLATFORM == "kaggle":
        from kaggle_secrets import UserSecretsClient

        os.environ["WANDB_API_KEY"] = UserSecretsClient().get_secret("WANDB_API_KEY")
    if not os.environ.get("WANDB_API_KEY"):
        raise RuntimeError("Set WANDB_API_KEY pada environment atau Kaggle Secret.")
    if os.environ.get("WANDB_MODE", "online") not in {"online", "shared"}:
        raise ValueError("Publikasi membutuhkan mode W&B online.")


workflow_login = ft_login


def ft_select():
    if len(STUDENT_INPUTS) != 1 or STUDENT_INPUTS[0] not in STUDENT_PINS:
        raise ValueError("Aktifkan tepat satu student raw yang tercantum pada STUDENT_INPUTS.")
    return STUDENT_INPUTS[0], STUDENT_PINS[STUDENT_INPUTS[0]]


def ft_identity():
    name, pin = ft_select()
    if SEED != pin["seed"]:
        raise ValueError("SEED harus mengikuti identitas raw terpilih.")
    return {"raw_artifact_ref": f"{WANDB_ENTITY}/{WANDB_PROJECT}/{name}", "raw_sha256": pin["raw_sha256"],
            "seed": pin["seed"], "target_speedup": pin["target"], "source_suite_id": PRUNING_SUITE_ID,
            "baseline_artifact_ref": pin["baseline_ref"], "baseline_source_sha256": pin["baseline_source_sha256"],
            "baseline_fp32_sha256": pin["baseline_fp32_sha256"]}


def ft_name():
    identity = ft_identity()
    return f"yolo26s-lamp-su{identity['target_speedup']:.2f}x-seed{identity['seed']}-ft" + ("-smoke" if MODE == "smoke" else "")


def ft_group():
    return f"yolo26s-lamp-ft-{PRUNING_SUITE_ID}" + ("-smoke" if MODE == "smoke" else "")


def ft_epoch_history(remote):
    # Avoid the pinned SDK's parquet schema error and request every recorded step.
    # The caller checks the exact epoch range and rejects missing or duplicate rows.
    return remote.history(keys=["epoch", "fitness"], samples=max(500, remote.lastHistoryStep + 1), pandas=False)


def ft_study_settings():
    training = fine_tuned_train_args("DATA", "PROJECT", "training")
    training["epochs"] = SMOKE_EPOCHS if MODE == "smoke" else EPOCHS
    evaluation = fine_tuned_val_args("DATA")
    for key in ("data", "project", "name", "seed", "device"):
        training.pop(key, None)
    for key in ("data", "device"):
        evaluation.pop(key, None)
    return {"training": training, "evaluation": evaluation,
            "epoch_loader_policy": "fresh workers and augmentation buffer at each epoch boundary"}


def ft_source(folder):
    identity = ft_identity()
    cache = folder / "raw_artifact"
    pin = STUDENT_PINS[STUDENT_INPUTS[0]]
    manifest_path = cache / "raw_pruned_manifest.json"
    if manifest_path.is_file() and sha256_file(manifest_path) != pin["manifest_sha256"]:
        raise ValueError("SHA-256 manifest raw berubah dari pin suite pruning.")
    cached = ft_read(manifest_path)
    required = {"pruned_fp32.pt", "baseline_fp32.pt", *(cached or {}).get("audit_files", {})}
    if cached is None or any(not (cache / name).is_file() for name in required):
        local = LOCAL_RAW_PACKAGES.get(STUDENT_INPUTS[0])
        if local:
            shutil.copytree(Path(local), cache, dirs_exist_ok=True)
        else:
            workflow_login()
            remote = wandb.Api(timeout=30).artifact(identity["raw_artifact_ref"], type="model")
            if remote.qualified_name != identity["raw_artifact_ref"]:
                raise ValueError("Versi raw W&B berubah.")
            remote.download(root=str(cache))
    manifest = ft_read(cache / "raw_pruned_manifest.json")
    if sha256_file(cache / "raw_pruned_manifest.json") != pin["manifest_sha256"]:
        raise ValueError("SHA-256 manifest raw berubah dari pin suite pruning.")
    baseline = manifest["source_baseline"]
    if (manifest.get("suite_id") != PRUNING_SUITE_ID or manifest.get("seed") != pin["seed"]
            or manifest.get("baseline_seed") != pin["seed"] or manifest.get("target_speedup") != pin["target"]
            or manifest.get("stage") != "raw_pruned" or manifest.get("format_version") != 2
            or manifest.get("model_variant") != "s" or baseline["requested_artifact"] != pin["baseline_ref"]
            or baseline["sha256"] != pin["baseline_source_sha256"]
            or manifest["raw_checkpoint"]["sha256"] != pin["raw_sha256"]
            or manifest["baseline_checkpoint"]["sha256"] != pin["baseline_fp32_sha256"]
            or any(manifest.get(key) != "PASS" for key in ("structural_status", "oracle_status", "reload_status"))):
        raise ValueError("Identitas raw, seed, atau baseline berbeda dari perlakuan yang dipilih.")
    hashes = {"pruned_fp32.pt": pin["raw_sha256"], "baseline_fp32.pt": pin["baseline_fp32_sha256"],
              **manifest["audit_files"]}
    files = {"raw_pruned_manifest.json": cache / "raw_pruned_manifest.json"}
    for name, digest in hashes.items():
        if Path(name).name != name or sha256_file(cache / name) != digest:
            raise ValueError(f"SHA-256 paket raw tidak cocok: {name}")
        files[name] = cache / name
    return manifest, files


def ft_dataset(manifest, folder):
    original = manifest["dataset"]
    diagnostic_fixture = MODE == "smoke" and SMOKE_DATASET_ROOT is not None
    root = Path(SMOKE_DATASET_ROOT) if diagnostic_fixture else DATASET_ROOT
    source_yaml = root / "data.yaml" if diagnostic_fixture else DATA_YAML
    if MODE == "full_val" and (original["provenance_status"] != "VERIFIED" or manifest["validation_status"] != "PASS"):
        raise ValueError("Full validation membutuhkan raw dengan dataset terverifikasi.")
    dataset = gflops_prepare_dataset(MODE, source_yaml, root, folder, seed=manifest["seed"],
                                    expected_fingerprint=None if diagnostic_fixture else original["full_val_fingerprint"],
                                    dataset_id=DATASET_ID)
    if dataset is None:
        raise FileNotFoundError("Atur DATASET_ROOT dan DATA_YAML.")
    record, train, _ = workflow_dataset_manifest(dataset, source_yaml, root, folder)
    if (record["class_names"] != original["class_names"] or (not diagnostic_fixture and
            (record["train_fingerprint"] != original["train_fingerprint"]
             or record["val_fingerprint"] != original["full_val_fingerprint"]))):
        raise ValueError("Dataset atau urutan kelas berbeda dari raw.")
    record["diagnostic_fixture"] = diagnostic_fixture
    data_yaml = folder / "train_val_data.yaml"
    train_list = folder / "train_images.txt"
    train_list.write_text("".join(f"{p.as_posix()}\n" for p in train), encoding="utf-8")
    data_yaml.write_text(yaml.safe_dump({**dataset["info"], "train": str(train_list.resolve())}, sort_keys=False), encoding="utf-8")
    return dataset, record, train, data_yaml


def ft_verify_raw(raw, manifest):
    fine_tuned_check_heads(raw, manifest["dataset"]["class_names"])
    if model_state_sha256(raw) != manifest["state_sha256"]:
        raise ValueError("State awal berbeda dari raw terpilih.")


def ft_publish(run, folder, label, files, metadata, event="final"):
    """Keep immutable upload bytes and recover committed versions after network failures."""
    journal = folder / "artifact_publications.json"
    records = ft_read(journal, {})
    key = f"{label}/{event}"
    files = {name: Path(path) for name, path in files.items()}
    hashes = {name: sha256_file(path) for name, path in files.items()}
    record = records.get(key, {"name": f"{run.name}-{label}", "status": "PENDING", "files_sha256": hashes})
    if record["files_sha256"] != hashes:
        raise ValueError("Berkas publikasi yang sudah dijurnal berubah.")
    if record["status"] == "CONFIRMED":
        return record
    records[key] = record
    write_json(journal, records)
    api = wandb.Api(timeout=30)
    if not record.get("reference"):
        if record.get("upload_started"):
            for artifact in api.run(f"{WANDB_ENTITY}/{WANDB_PROJECT}/{run.id}").logged_artifacts():
                if artifact.name.rsplit(":", 1)[0] == record["name"] and artifact.metadata.get("files_sha256") == hashes:
                    record["reference"] = workflow_version_ref(artifact.qualified_name)
                    break
        if not record.get("reference"):
            artifact = wandb.Artifact(record["name"], type="model" if label != "results" else "results",
                                      metadata={**metadata, "event": event, "files_sha256": hashes})
            for name, path in files.items():
                artifact.add_file(str(path), name=name)
            record["upload_started"] = True
            write_json(journal, records)
            aliases = ["latest", "best", "last"] if label == "checkpoints" else ["latest", event]
            uploaded = run.log_artifact(artifact, aliases=aliases)
            uploaded.wait(timeout=600)
            record["reference"] = workflow_version_ref(uploaded.qualified_name)
            write_json(journal, records)
    reference = workflow_version_ref(record["reference"])
    remote = api.artifact(reference)
    if remote.qualified_name != reference or reference.rsplit(":", 1)[0] != f"{WANDB_ENTITY}/{WANDB_PROJECT}/{record['name']}":
        raise ValueError("Koleksi atau versi hasil upload tidak sesuai run.")
    with TemporaryDirectory(prefix="verify_", dir=folder) as cache:
        downloaded_root = Path(remote.download(root=cache))
        for name, digest in hashes.items():
            if sha256_file(downloaded_root / name) != digest:
                raise ValueError(f"SHA-256 unduhan W&B tidak cocok: {name}")
    record.update(status="CONFIRMED", url=remote.url)
    write_json(journal, records)
    return record


class SelectedFineTuneTrainer(ResumableStructuralTrainer):
    """Use the baseline training protocol with the audited native epoch resume implementation."""

    def _build_train_pipeline(self):
        if self.batch_size != self.recovery_fixed_batch:
            raise ValueError("Batch training berubah.")
        DetectionTrainer._build_train_pipeline(self)

    def validate(self):
        with gflops_require_complete_nms():
            return DetectionTrainer.validate(self)

    def _setup_train(self):
        close = self.args.close_mosaic
        if close > self.args.epochs:
            self.args.close_mosaic = 0
        try:
            super()._setup_train()
        finally:
            self.args.close_mosaic = close
        if self.recovery_resume_state:
            self.stopper = deepcopy(self.recovery_resume_state["stopper"])

    def save_model(self):
        previous = ft_read(self.recovery_folder / "latest_state.json")
        # Native best.pt selects the current EMA when fitness ties the best fitness.
        if self.recovery_best and self.fitness == self.best_fitness:
            self.recovery_best = None
        callback = self.recovery_committed_callback
        self.recovery_committed_callback = None
        try:
            super().save_model()
        finally:
            self.recovery_committed_callback = callback
        index = ft_read(self.recovery_folder / "latest_state.json")
        path = self.recovery_folder / index["relative_path"]
        state = torch.load(path, map_location="cpu", weights_only=False)
        state.update(stopper=deepcopy(self.stopper), training_finished=bool(self.stop),
                     epoch_seconds=time.time() - self.epoch_time_start)
        state["epoch_rows"][-1]["epoch_seconds"] = state["epoch_seconds"]
        self.recovery_epochs = state["epoch_rows"]
        recovery_atomic_torch_save(state, path)
        index["sha256"] = sha256_file(path)
        index["training_finished"] = bool(self.stop)
        write_json(self.recovery_folder / "epoch_metrics.json", self.recovery_epochs)
        write_json(self.recovery_folder / "latest_state.json", index)
        if callback:
            callback(index, state["epoch_rows"][-1])
        if previous and previous["relative_path"] != index["relative_path"]:
            (self.recovery_folder / previous["relative_path"]).unlink(missing_ok=True)
            if previous["best"]["relative_path"] != index["best"]["relative_path"]:
                (self.recovery_folder / previous["best"]["relative_path"]).unlink(missing_ok=True)
        recovery_restore_rng(state["rng"], self.device)
        return True


def ft_fit(raw, raw_path, manifest, dataset, record, train, data_yaml, folder, run, contract):
    options = fine_tuned_train_args(str(data_yaml), str(folder), "training")
    options["epochs"] = SMOKE_EPOCHS if MODE == "smoke" else EPOCHS
    options.update(project=str(folder / "training_attempts"), name=uuid.uuid4().hex[:10])
    state, _ = recovery_committed_state(folder, contract["sha256"], manifest["raw_checkpoint"]["sha256"])

    def committed(index, row):
        if run is not None:
            epoch = row["epoch"]
            payload = recovery_epoch_rows([row])[0]
            payload.update(epoch=epoch, epoch_seconds=row.get("epoch_seconds"))
            logged = ft_read(folder / "logged_epochs.json", [])
            if epoch not in logged:
                run.log(payload)
                write_json(folder / "logged_epochs.json", sorted({*logged, epoch}))
            run.summary.update({"epochs_completed": epoch, "best_epoch": index["best"]["epoch"],
                                "best_fitness": index["best"]["fitness"]})
            if epoch % RESUME_UPLOAD_INTERVAL == 0 or epoch == options["epochs"] or index.get("training_finished"):
                files = {index["relative_path"]: folder / index["relative_path"],
                         index["best"]["relative_path"]: folder / index["best"]["relative_path"]}
                for name in ("latest_state.json", "resume_contract.json", "wandb_identity.json", "epoch_metrics.json",
                             "logged_epochs.json", "training_config.json"):
                    files[name] = folder / name
                ft_publish(run, folder, "resume", files, ft_identity(), event=f"epoch-{epoch:03d}")

    if state and run is not None:
        logged = ft_read(folder / "logged_epochs.json", [])
        for row in state["epoch_rows"]:
            if row["epoch"] not in logged:
                run.log({**recovery_epoch_rows([row])[0], "epoch_seconds": row.get("epoch_seconds")})
                logged.append(row["epoch"])
                write_json(folder / "logged_epochs.json", logged)
        pending = ft_read(folder / "artifact_publications.json", {})
        if state.get("training_finished") or any(r["status"] != "CONFIRMED" for r in pending.values()):
            index = ft_read(folder / "latest_state.json")
            committed(index, state["epoch_rows"][-1])
    if state and state.get("training_finished"):
        return state
    trainer = SelectedFineTuneTrainer(raw, overrides={**options, "model": str(raw_path), "task": "detect", "mode": "train"},
                                     expected_names=dataset["class_names"], raw_sha=manifest["raw_checkpoint"]["sha256"],
                                     folder=folder, contract_sha=contract["sha256"], val_batch=2 * BATCH,
                                     resume_state=state, committed_callback=committed)

    def check_start(trainer):
        if trainer.batch_size != BATCH or bool(trainer.amp) != bool(options["amp"]):
            raise RuntimeError("Batch atau AMP efektif berbeda dari protokol training.")
        expected = state["live_model"] if state else raw
        if recovery_state_sha256(trainer.model) != recovery_state_sha256(expected):
            raise ValueError("Model awal training berbeda dari raw atau snapshot resume.")
        fine_tuned_check_images(trainer.train_loader, train)
        fine_tuned_check_images(trainer.test_loader, dataset["image_files"])
        write_json(folder / "training_config.json", {"requested": options, "effective": vars(trainer.args),
                   "initial_state_sha256": recovery_state_sha256(raw), "resume_epoch": trainer.start_epoch,
                   "amp_effective": trainer.amp, "dataset": record,
                   "epoch_loader_policy": "fresh workers and augmentation buffer at each epoch boundary"})

    trainer.add_callback("on_train_start", check_start)
    with gflops_disable_label_cache():
        trainer.train()
    state, _ = recovery_committed_state(folder, contract["sha256"], manifest["raw_checkpoint"]["sha256"])
    if not state or not state["training_finished"] or state["optimizer_steps"] < 1:
        raise RuntimeError("Training belum menyelesaikan budget atau early stopping.")
    if recovery_state_sha256(state["live_model"], parameters_only=True) == recovery_state_sha256(raw, parameters_only=True):
        raise RuntimeError("Bobot tidak berubah selama fine-tuning.")
    return state


def ft_restore_snapshot(folder):
    if not RESUME_SNAPSHOT_REF:
        return
    workflow_login()
    ref = workflow_version_ref(RESUME_SNAPSHOT_REF)
    remote = wandb.Api(timeout=30).artifact(ref, type="model")
    expected = ft_identity()
    if any(remote.metadata.get(key) != value for key, value in expected.items()):
        raise ValueError("Snapshot W&B berasal dari input lain.")
    hashes = remote.metadata["files_sha256"]
    if (folder / "resume_contract.json").exists():
        return
    with TemporaryDirectory(prefix="resume_download_", dir=folder) as cache:
        remote.download(root=cache)
        for name, digest in hashes.items():
            path = Path(cache) / name
            if path.resolve().is_relative_to(Path(cache).resolve()) is False or sha256_file(path) != digest:
                raise ValueError("Berkas snapshot W&B tidak valid.")
        for name in hashes:
            target = folder / name
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(Path(cache) / name, target)
    event = remote.metadata.get("event")
    if not isinstance(event, str) or not re.fullmatch(r"epoch-\d+", event):
        raise ValueError("Artefak resume belum memiliki nomor epoch yang valid.")
    write_json(folder / "artifact_publications.json", {f"resume/{event}": {
        "name": remote.name.rsplit(":", 1)[0], "status": "CONFIRMED", "files_sha256": hashes,
        "reference": ref, "url": remote.url}})


def ft_contract(folder, record):
    options = fine_tuned_train_args("DATA", "PROJECT", "training")
    options["epochs"] = SMOKE_EPOCHS if MODE == "smoke" else EPOCHS
    options.pop("device", None)
    return workflow_bind(folder, "resume_contract.json", {
        "protocol": "selected-raw-finetune-v2", "implementation": IMPLEMENTATION_SHA256,
        **ft_identity(), "dataset": record, "training": options, "mode": MODE, "platform": PLATFORM,
        "gpu": ENVIRONMENT["gpu"], "precision": "AMP" if AMP and MODE != "smoke" else "FP32",
        "packages": {name: ENVIRONMENT[name] for name in ("torch", "torchvision", "ultralytics", "wandb")},
        "latency": RUN_LATENCY_BENCHMARK, "resume_upload_interval": RESUME_UPLOAD_INTERVAL,
        "wandb_entity": WANDB_ENTITY, "wandb_project": WANDB_PROJECT,
        "wandb_base_url": os.environ.get("WANDB_BASE_URL", "https://api.wandb.ai"),
    })


@contextmanager
def ft_tracking(folder, manifest, record):
    if not WANDB_TRACKING:
        if MODE == "full_val":
            raise ValueError("Full validation mewajibkan publikasi W&B.")
        yield None
        return
    workflow_login()
    path = folder / "wandb_identity.json"
    previous = ft_read(path)
    identity = previous or {"id": uuid.uuid4().hex, "name": ft_name(), "group": ft_group()}
    if identity["name"] != ft_name() or identity["group"] != ft_group():
        raise ValueError("Identitas run W&B berbeda dari input atau mode.")
    write_json(path, identity)
    if previous:
        remote = wandb.Api(timeout=30).run(f"{WANDB_ENTITY}/{WANDB_PROJECT}/{identity['id']}")
        logged = {int(row["epoch"]) for row in ft_epoch_history(remote) if row.get("epoch") is not None}
        write_json(folder / "logged_epochs.json", sorted(logged))
    metadata = {**ft_identity(), "platform": PLATFORM, "mode": MODE, "protocol": "selected-raw-finetune-v2",
                "target_status": manifest["target_status"], "actual_gflops_speedup": manifest["actual_gflops_speedup"],
                "dataset": record, "environment": ENVIRONMENT, "epochs": SMOKE_EPOCHS if MODE == "smoke" else EPOCHS,
                "study_settings": ft_study_settings(), "study_signature": workflow_digest(ft_study_settings())}
    with wandb.init(entity=WANDB_ENTITY, project=WANDB_PROJECT, id=identity["id"], name=identity["name"],
                    group=identity["group"], resume="allow" if previous else "never", dir=str(folder),
                    job_type="finetune", config=metadata,
                    tags=["yolo26s", "lamp", "finetune", PLATFORM, MODE, f"seed{SEED}"]) as run:
        run.define_metric("epoch")
        for pattern in ("train/*", "val/*", "metrics/*", "lr/*", "fitness", "optimizer_steps", "epoch_seconds"):
            run.define_metric(pattern, step_metric="epoch")
        run.use_artifact(ft_identity()["raw_artifact_ref"], type="model")
        run.use_artifact(ft_identity()["baseline_artifact_ref"], type="model")
        run.summary.update({"status": "RUNNING", "logging_verified": False, **ft_identity()})
        try:
            yield run
        except (Exception, KeyboardInterrupt):
            run.summary["status"] = "PARTIAL"
            raise


def ft_eda(result, folder, epochs, files):
    original_save = globals()["_eda_save_figure"]
    original_metadata = globals()["_recovery_eda_metadata"]
    label = f"YOLO26s · seed {SEED} · su{result['target_speedup']:.2f}x · aktual {result['raw_gflops_speedup']:.6f}x"

    def save(fig, path):
        fig.suptitle(label, fontsize=10)
        if not fig.legends:
            fig.tight_layout(rect=(0, 0, 1, .94))
        return original_save(fig, path)

    globals()["_eda_save_figure"] = save
    globals()["_recovery_eda_metadata"] = lambda value: {**original_metadata(value), **ft_identity()}
    try:
        report = recovery_build_eda(result, folder, epochs, files)
        rows, classes = fine_tuned_comparison(result)
        for metric, title in (("precision", "Precision"), ("recall", "Recall"), ("f1", "F1"), ("support", "Support")):
            fig, ax = _eda_plt.subplots(figsize=(11, 5))
            values = [{"group": row["class_id"], "group_label": row["name"], "stage": row["stage"], "value": row[metric]}
                      for row in classes]
            _recovery_draw_bar_panel(ax, values, title=f"{title} per kelas", unit="Objek" if metric == "support" else "Proporsi",
                                     bounded=metric != "support")
            ax.legend([label for label, _color in RECOVERY_STAGE_STYLES.values()])
            save(fig, folder / "eda" / f"comparison_class_{metric}")
        inventory = ft_read(files["inventories.json"])
        layers = [row for row in layer_eda_rows(inventory["before"], inventory["after"]) if row["module_type"] == "Conv2d"]
        physical = []
        for offset in range(0, len(layers), EDA_MAX_LAYERS_PER_PAGE):
            page = layers[offset:offset + EDA_MAX_LAYERS_PER_PAGE]
            values = [{"group": row["layer_name"], "group_label": row["layer_name"], "stage": stage,
                       "value": row["out_channels_before" if stage == "baseline" else "out_channels_after"]}
                      for row in page for stage in RECOVERY_STAGE_STYLES]
            physical.extend(values)
            fig, ax = _eda_plt.subplots(figsize=(16, 6))
            _recovery_draw_bar_panel(ax, values, title="Channel per layer · tiga kondisi", unit="Channel")
            ax.tick_params(axis="x", labelsize=6)
            ax.legend([name for name, _color in RECOVERY_STAGE_STYLES.values()])
            save(fig, folder / "eda" / f"channels_three_stages_{offset // EDA_MAX_LAYERS_PER_PAGE:02d}")
        write_csv(folder / "eda/channels_three_stages.csv", [{**ft_identity(), **row} for row in physical])
        for metric in ("parameters", "gflops", "latency_median_ms"):
            fig, ax = _eda_plt.subplots(figsize=(7, 5))
            for row in rows:
                if row.get(metric) is not None and row.get("map50_95") is not None:
                    stage_label, color = RECOVERY_STAGE_STYLES[row["stage"]]
                    ax.scatter(row[metric], row["map50_95"], color=color, label=stage_label)
            ax.set(xlabel=metric, ylabel="mAP50–95", title="Trade-off tiga kondisi")
            if ax.collections:
                ax.legend()
            save(fig, folder / "eda" / f"tradeoff_{metric}")
        write_json(folder / "eda/stage_metrics.json", rows)
        write_csv(folder / "eda/stage_metrics.csv", rows)
        write_json(folder / "eda/per_class_metrics.json", classes)
        write_csv(folder / "eda/per_class_metrics.csv", classes)
        return report
    finally:
        globals()["_eda_save_figure"] = original_save
        globals()["_recovery_eda_metadata"] = original_metadata


def ft_log_eda_tables(run, result, folder):
    """Expose every custom EDA CSV as an interactive W&B Table."""
    tables = {}
    identity = {**ft_identity(), "actual_gflops_speedup": result["raw_gflops_speedup"]}
    for path in sorted((folder / "eda").rglob("*.csv")):
        frame = pd.read_csv(path)
        for key, value in identity.items():
            if key not in frame:
                frame[key] = value
        key = "tables/eda/" + path.relative_to(folder / "eda").with_suffix("").as_posix()
        run.log({key: wandb.Table(dataframe=frame)})
        tables[key] = {"rows": len(frame), "source": path.relative_to(folder).as_posix(),
                       "source_sha256": sha256_file(path)}
    write_json(folder / "wandb_tables.json", tables)
    return tables


def ft_log_results(run, result, folder):
    rows = ft_read(folder / "eda/stage_metrics.json")
    classes = ft_read(folder / "eda/per_class_metrics.json")
    for row in rows:
        stage = row["stage"]
        run.summary.update({f"{stage}/{key}": value for key, value in row.items()
                            if isinstance(value, (int, float)) and not isinstance(value, bool)})
    for name, values in (("three_stage_metrics", rows), ("three_stage_per_class", classes),
                         ("epoch_history", recovery_epoch_rows(ft_read(folder / "epoch_metrics.json")))):
        columns = list(dict.fromkeys(key for row in values for key in row))
        run.log({f"comparison/{name}": wandb.Table(columns=columns, data=[[row.get(k) for k in columns] for row in values])})
    ft_log_eda_tables(run, result, folder)
    media = sorted(path for base in (folder / "eda", folder / "validation", folder / "training_attempts")
                   for path in base.rglob("*") if path.suffix.lower() in {".png", ".jpg", ".jpeg"})
    keys = []
    for path in media:
        key = "visuals/" + path.relative_to(folder).with_suffix("").as_posix()
        run.log({key: wandb.Image(str(path))})
        keys.append(key)
    write_json(folder / "wandb_media.json", {"keys": keys, "image_count": len(keys)})
    return keys


def ft_registry(result, folder):
    publications = ft_read(folder / "artifact_publications.json", {})
    checkpoint = publications.get("checkpoints/final", {})
    reports = publications.get("results/final", {})
    row = {**ft_identity(), "platform": PLATFORM, "mode": MODE, "status": result["status"],
           "run_id": ft_read(folder / "wandb_identity.json", {}).get("id"), "run_name": ft_name(), "group": ft_group(),
           "target_status": result.get("target_status"), "actual_gflops_speedup": result.get("raw_gflops_speedup"),
           "best_epoch": result.get("best_epoch"), "epochs_completed": result.get("epochs"),
           "checkpoint_artifact_ref": checkpoint.get("reference"), "results_artifact_ref": reports.get("reference"),
           "publication_status": "CONFIRMED" if checkpoint.get("status") == reports.get("status") == "CONFIRMED" else "NOT_RUN",
           "checkpoint_hashes": checkpoint.get("files_sha256"), "gpu": ENVIRONMENT["gpu"]}
    for stage, values in result.get("stages", {}).items():
        row.update({f"{stage}_{metric}": values["validation"].get(metric) for metric in RECOVERY_ACCURACY_FIELDS})
    write_json(folder / "finetuning_registry.json", {"format_version": 1, "workflow": "selected-raw-finetune-v2", "rows": [row]})
    write_csv(folder / "finetuning_registry.csv", [row])
    return row


def ft_verify_logging(result, folder):
    """Read the finished run back from W&B before accepting complete logging."""
    identity = ft_read(folder / "wandb_identity.json")
    remote = wandb.Api(timeout=30).run(f"{WANDB_ENTITY}/{WANDB_PROJECT}/{identity['id']}")
    summary = remote.summary
    epochs = ft_epoch_history(remote)
    if sorted(int(row["epoch"]) for row in epochs) != list(range(1, result["epochs"] + 1)):
        raise ValueError("History epoch W&B belum lengkap atau mengandung duplikasi.")
    media = ft_read(folder / "wandb_media.json")["keys"]
    files = {entry.name for entry in remote.files(per_page=200)}
    for key in media:
        value = summary.get(key)
        if value is None or value.get("_type") != "image-file" or value.get("path") not in files:
            raise ValueError(f"Media W&B belum terkonfirmasi: {key}")
    tables = {"comparison/three_stage_metrics": 3,
              "comparison/three_stage_per_class": len(result["dataset"]["class_names"]) * 3,
              "comparison/epoch_history": result["epochs"]}
    tables.update({key: value["rows"] for key, value in ft_read(folder / "wandb_tables.json", {}).items()})
    for key, count in tables.items():
        value = summary.get(key)
        if value is None or value.get("nrows") != count or value.get("path") not in files:
            raise ValueError(f"Tabel W&B belum lengkap: {key}")
    for stage in ("baseline", "raw_pruned", "recovered"):
        for metric in ("map50_95", "map50", "precision", "recall", "gflops", "parameters"):
            if summary.get(f"{stage}/{metric}") is None:
                raise ValueError(f"Scalar W&B belum tersedia: {stage}/{metric}")
    report = {"status": "PASS", "run_id": identity["id"], "epoch_count": len(epochs),
              "image_count": len(media), "table_rows": tables, "run_url": remote.url}
    write_json(folder / "wandb_logging_verification.json", report)
    remote.summary.update({"logging_verified": True, "status": "PASS"})
    return report


def ft_execute(folder):
    folder = Path(folder)
    with FileLock(str(folder / ".run.lock"), timeout=0):
        prior_contract = ft_read(folder / "resume_contract.json")
        if prior_contract and (prior_contract["contract"].get("implementation") != IMPLEMENTATION_SHA256
                or any(prior_contract["contract"].get(key) != value for key, value in ft_identity().items())):
            raise ValueError("Identitas input atau implementasi berbeda dari kontrak resume.")
        stage = "input"
        manifest_path = folder / "fine_tuned_manifest.json"
        result = ft_read(manifest_path, {"status": "NOT_RUN", "stages": {}, "errors": []})
        result.update(**ft_identity(), platform=PLATFORM, mode=MODE, model_variant="s", method="finetune",
                      protocol="selected-raw-finetune-v2", run_name=ft_name())
        try:
            ft_restore_snapshot(folder)
            manifest, files = ft_source(folder)
            raw = fine_tuned_load_raw(files["pruned_fp32.pt"], manifest)
            ft_verify_raw(raw, manifest)
            dataset, record, train, data_yaml = ft_dataset(manifest, folder / "data")
            contract = ft_contract(folder, record)
            result.update(target_status=manifest["target_status"], configuration_id=manifest["experiment_id"],
                          raw_gflops_speedup=manifest["actual_gflops_speedup"], dataset=record, environment=ENVIRONMENT)
            write_json(manifest_path, result)
            with ft_tracking(folder, manifest, record) as run:
                stage = "training"
                state = ft_fit(raw, files["pruned_fp32.pt"], manifest, dataset, record, train, data_yaml, folder, run, contract)
                result.update(epochs=state["epoch"], best_epoch=state["best"]["epoch"], training_status="PASS",
                              initial_state_sha256=recovery_state_sha256(raw),
                              trained_state_sha256=recovery_state_sha256(state["live_model"]))
                checkpoints = folder / "final_checkpoints"
                checkpoints.mkdir(exist_ok=True)
                best, last = checkpoints / "best.pt", checkpoints / "last.pt"
                if not best.exists():
                    shutil.copyfile(folder / state["best"]["relative_path"], best)
                if sha256_file(best) != state["best"]["sha256"]:
                    raise ValueError("SHA-256 best hasil training berubah.")
                if not last.exists():
                    recovery_atomic_torch_save({"model": _recovery_native_snapshot(state["ema"]), "ema": None,
                                               "optimizer": None, "train_args": {}, "epoch": state["epoch"] - 1,
                                               "recovery_enabled": True}, last)
                stage = "reload"
                core = YOLO(str(best)).model.cpu().float().eval()
                fine_tuned_check_heads(core, dataset["class_names"])
                if recovery_structure_signature(core) != recovery_structure_signature(raw):
                    raise ValueError("Arsitektur best berbeda dari raw.")
                if not result.get("reload_report"):
                    core.end2end = True
                    result["reload_report"] = verify_fresh_reload(best, core, torch.zeros(1, 3, IMGSZ, IMGSZ), checkpoints)
                result["reload_status"] = "PASS"
                write_json(manifest_path, result)
                stage = "validation"
                for name, checkpoint in (("baseline", files["baseline_fp32.pt"]), ("raw_pruned", files["pruned_fp32.pt"]), ("recovered", best)):
                    cached = folder / "validation" / f"{name}_measurements.json"
                    metrics = ft_read(cached)
                    digest = sha256_file(checkpoint)
                    if metrics and metrics["checkpoint_sha256"] != digest:
                        raise ValueError("Checkpoint evaluasi tersimpan berubah.")
                    if not metrics:
                        metrics = fine_tuned_measure(checkpoint, data_yaml, dataset, folder / "validation", name)
                        metrics["checkpoint_sha256"] = digest
                        write_json(cached, metrics)
                    result["stages"][name] = metrics
                    write_json(manifest_path, result)
                baseline, pruned, recovered = [result["stages"][name] for name in ("baseline", "raw_pruned", "recovered")]
                if (any(pruned[key] != recovered[key] for key in ("parameters", "gflops", "conv_output_channels"))
                        or any(value["validation"]["validation_arguments_effective"] != baseline["validation"]["validation_arguments_effective"]
                               for value in (pruned, recovered))):
                    raise ValueError("Protokol evaluasi atau struktur raw/recovered berbeda.")
                result["validation_status"] = "PASS"
                result["latency_status"] = ("PASS" if all(row["latency"]["status"] == "PASS" for row in result["stages"].values())
                                            else "NOT_RUN" if DEVICE == "cpu" or not RUN_LATENCY_BENCHMARK else "FAIL")
                if MODE == "full_val" and RUN_LATENCY_BENCHMARK and result["latency_status"] != "PASS":
                    raise ValueError("Benchmark latency CUDA belum lengkap.")
                stage = "eda"
                report = ft_eda(result, folder, state["epoch_rows"], files)
                if report["status"] != "PASS":
                    raise ValueError("Panel EDA belum lengkap.")
                result["eda_status"] = "PASS"
                if sha256_file(files["pruned_fp32.pt"]) != result["raw_sha256"]:
                    raise ValueError("File input raw berubah selama training.")
                result["raw_unchanged"] = True
                stage = "publication"
                if run is not None:
                    ft_log_results(run, result, folder)
                    write_json(checkpoints / "model_manifest.json", {**ft_identity(), "best_sha256": sha256_file(best),
                               "last_sha256": sha256_file(last), "best_epoch": result["best_epoch"],
                               "epochs": result["epochs"], "structure_signature": recovery_structure_signature(raw)})
                    result["checkpoint_artifact"] = ft_publish(run, folder, "checkpoints", {
                        "best.pt": best, "last.pt": last, "model_manifest.json": checkpoints / "model_manifest.json"}, ft_identity())
                    result["status"] = "MEASURED"
                    write_json(manifest_path, result)
                    write_json(folder / "evaluation_manifest.json", result)
                    reports = {path.relative_to(folder).as_posix(): path for base in (
                               folder / "eda", folder / "validation", folder / "training_attempts", folder / "raw_artifact")
                               for path in base.rglob("*") if path.suffix in {".png", ".jpg", ".svg", ".json", ".csv"}}
                    for name in ("training_config.json", "epoch_metrics.json", "resume_contract.json", "wandb_media.json", "wandb_tables.json", "evaluation_manifest.json"):
                        reports[name] = folder / name
                    result["results_artifact"] = ft_publish(run, folder, "results", reports, ft_identity())
                    result["publication_status"] = "CONFIRMED"
                result.update(status="PASS", reason=None)
                write_json(manifest_path, result)
                registry = ft_registry(result, folder)
                if run is not None:
                    run.summary.update({**registry, "status": "PASS", "checkpoint_sha256": sha256_file(best),
                                        "media_count": len(ft_read(folder / "wandb_media.json")["keys"])})
                    run.log({"comparison/registry": wandb.Table(dataframe=pd.DataFrame([registry]))})
            if WANDB_TRACKING:
                stage = "logging_verification"
                result["logging_verification"] = ft_verify_logging(result, folder)
                write_json(manifest_path, result)
                ft_registry(result, folder)
        except (Exception, KeyboardInterrupt) as error:
            if prior_contract and stage == "input":
                write_json(folder / "last_error.json", {"stage": stage, "error": safe_error(error)})
                raise
            result.update(status="PARTIAL" if result.get("training_status") == "PASS" or (folder / "latest_state.json").exists() else "FAIL",
                          reason=safe_error(error) or "Interrupted")
            result["errors"].append({"stage": stage, "error": result["reason"]})
            write_json(manifest_path, result)
            ft_registry(result, folder)
        finally:
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        return result


def ft_group_results():
    workflow_login()
    runs = wandb.Api(timeout=30).runs(f"{WANDB_ENTITY}/{WANDB_PROJECT}", filters={"group": ft_group()})
    chosen = {}
    for run in runs:
        values = dict(run.summary)
        if (values.get("status") != "PASS" or not values.get("logging_verified")
                or run.config.get("protocol") != "selected-raw-finetune-v2"
                or run.config.get("study_signature") != workflow_digest(ft_study_settings())):
            continue
        raw_ref = values.get("raw_artifact_ref", "")
        name = raw_ref.rsplit("/", 1)[-1]
        if name not in STUDENT_PINS or values.get("raw_sha256") != STUDENT_PINS[name]["raw_sha256"]:
            continue
        values["platform"] = values.get("platform") or run.config.get("platform")
        values["gpu"] = values.get("gpu") or run.config.get("environment", {}).get("gpu") or "CPU"
        key = (name, values["platform"], values["gpu"])
        if key not in chosen or run.created_at > chosen[key][0]:
            chosen[key] = (run.created_at, {**values, "wandb_run_url": run.url})
    rows = [row for _created, row in chosen.values()]
    print("Perlakuan tersedia", len({row["raw_artifact_ref"] for row in rows}), "/ 7")
    frame = pd.DataFrame(rows)
    if not frame.empty:
        primary = frame.loc[frame["target_speedup"].eq(1.20)]
        metrics = [key for key in frame if key.startswith(("baseline_", "raw_pruned_", "recovered_")) and key.endswith(("map50_95", "map50", "precision", "recall"))]
        if not primary.empty:
            display(primary.groupby(["platform", "gpu"], dropna=False)[metrics].agg(["count", "mean", "std"]))
    return rows
