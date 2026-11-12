"""Standalone orchestration for the YOLO26s LAMP multi-seed notebook."""
# ruff: noqa: F821, BLE001 -- Notebook stages share declarations and record independent failures.
import gc
import shutil
from copy import deepcopy
from pathlib import Path
from tempfile import TemporaryDirectory

from filelock import FileLock

PRIMARY_TARGET = 1.20
PRUNING_SEEDS = (42, 0, 2024, 7, 3407)
PRUNING_QUEUE = ((1.10, 42), (PRIMARY_TARGET, 42), (1.70, 42)) + tuple(
    (PRIMARY_TARGET, seed) for seed in PRUNING_SEEDS[1:]
)
EVAL_SEED = 42


def ms_read(path, default=None):
    return json.loads(path.read_text(encoding="utf-8")) if path.is_file() else default


def ms_id(target, seed):
    return f"target-{target:.2f}x-seed{seed}"


def ms_run_name(target, seed):
    return f"yolo26s-lamp-su{target:.2f}x-seed{seed}"


def ms_model_name(seed):
    return f"YOLO26s-seed{seed}"


def ms_run_group(suite_id):
    return f"yolo26s-lamp-multiseed-{suite_id}"


def ms_wandb_environment():
    if not os.environ.get("WANDB_API_KEY") and PLATFORM == "kaggle":
        try:
            from kaggle_secrets import UserSecretsClient
            key = UserSecretsClient().get_secret("WANDB_API_KEY")
            if key:
                os.environ["WANDB_API_KEY"] = key
        except Exception as error:
            raise RuntimeError("Lampirkan Kaggle Secret WANDB_API_KEY atau setel variabel environment.") from error
    if not os.environ.get("WANDB_API_KEY"):
        raise RuntimeError("Setel WANDB_API_KEY pada environment atau Kaggle Secret.")
    if os.environ.get("WANDB_MODE", "online") not in {"online", "shared"}:
        raise RuntimeError("Aktifkan mode online W&B untuk mengunduh atau mempublikasikan model.")


def ms_dataset(folder):
    portable = ms_read(Path(DATASET_MANIFEST_PATH).expanduser()) if DATASET_MANIFEST_PATH else None
    if DATASET_MANIFEST_PATH and (not portable or portable.get("dataset_id") != DATASET_ID
                                  or portable.get("provenance_status") != "VERIFIED"):
        raise ValueError("Gunakan dataset_manifest.json Poltrada V3 dari sumber terverifikasi.")
    check_dir = folder / "dataset_checks" / uuid.uuid4().hex[:8]
    data = gflops_prepare_dataset(MODE, DATA_YAML, DATASET_ROOT, check_dir, seed=EVAL_SEED,
                                  expected_fingerprint=portable["val_fingerprint"] if portable else None,
                                  dataset_id=DATASET_ID)
    if data is None:
        raise FileNotFoundError("Atur DATASET_ROOT dan DATA_YAML ke Poltrada V3 YOLO26 Clean.")
    if MODE == "full_val" and data["provenance_status"] != "VERIFIED":
        raise ValueError("Full validation memerlukan mount Kaggle resmi atau manifest Poltrada V3 terverifikasi.")
    record, _, _ = workflow_dataset_manifest(data, DATA_YAML, DATASET_ROOT, check_dir)
    if portable and (record["train_fingerprint"] != portable["train_fingerprint"]
                     or record["val_fingerprint"] != portable["val_fingerprint"]):
        raise ValueError("Fingerprint train/val berbeda dari manifest sumber.")
    data["train_fingerprint"] = record["train_fingerprint"]
    return data, record


def ms_bind(folder, dataset):
    path = folder / "resume_contract.json"
    previous = ms_read(path, {}).get("contract", {})
    if RESUME_SUITE_DIR and "suite_id" not in previous:
        raise ValueError("RESUME_SUITE_DIR harus menunjuk ke suite per-seed yang sudah terikat.")
    return workflow_bind(folder, "resume_contract.json", {
        "suite_id": previous.get("suite_id", folder.name),
        "implementation": IMPLEMENTATION_SHA256, "variant": "s", "mode": MODE,
        "queue": PRUNING_QUEUE, "baselines": BASELINE_SOURCES,
        "dataset_id": dataset["dataset_id"], "val_fingerprint": dataset["fingerprint"],
        "train_fingerprint": dataset["train_fingerprint"], "provenance": dataset["provenance_status"],
        "imgsz": IMGSZ, "batch": BATCH, "workers": WORKERS, "eval_seed": EVAL_SEED,
        "run_latency": RUN_LATENCY_BENCHMARK, "latency_warmup": LATENCY_WARMUP, "latency_repeats": LATENCY_REPEATS,
        "platform": PLATFORM, "device": str(TARGET_DEVICE), "environment": ENVIRONMENT,
        "wandb_entity": WANDB_ENTITY, "wandb_project": WANDB_PROJECT,
        "wandb_base_url": os.environ.get("WANDB_BASE_URL", "https://api.wandb.ai").rstrip("/"),
    })


def ms_check_source(path, record, seed):
    pinned = BASELINE_SOURCES[seed]
    if (record["baseline_seed"] != seed or record["requested_artifact"] != pinned["ref"]
            or record["sha256"] != pinned["sha256"] or sha256_file(path) != pinned["sha256"]):
        raise ValueError(f"Identitas atau SHA-256 best.pt baseline seed {seed} berbeda dari pin.")


def ms_source(folder, seed):
    cache = folder / "source"
    cache.mkdir(parents=True, exist_ok=True)
    journal = cache / "provenance.json"
    prior = ms_read(journal)
    reference = workflow_version_ref(BASELINE_SOURCES[seed]["ref"])
    expected = BASELINE_SOURCES[seed]["sha256"]
    if prior:
        path = cache / prior["relative_path"]
        ms_check_source(path, prior, seed)
        return path, prior

    if seed in LOCAL_BASELINE_PATHS:
        local = select_checkpoint(LOCAL_BASELINE_PATHS[seed], "best.pt")
        record = checkpoint_provenance(local, reference, expected_hash=expected)
        path = cache / "best.pt"
        if not path.is_file():
            temporary = path.with_suffix(".pt.tmp")
            shutil.copyfile(local, temporary)
            os.replace(temporary, path)
    else:
        ms_wandb_environment()
        artifact = wandb.Api(timeout=30).artifact(reference, type="model")
        if artifact.qualified_name != reference or artifact.metadata.get("seed") != seed:
            raise ValueError(f"Versi atau metadata seed baseline W&B tidak cocok untuk seed {seed}.")
        path = Path(artifact.get_path("best.pt").download(root=str(cache))).resolve()
        record = checkpoint_provenance(path, reference, expected_hash=expected, resolved_artifact=artifact.qualified_name)
    record.pop("checkpoint_path")
    record.update(relative_path=path.resolve().relative_to(cache.resolve()).as_posix(), model_variant="s", baseline_seed=seed)
    ms_check_source(path, record, seed)
    write_json(journal, record)
    return path, record


def gflops_load_source(path, provenance, variant):
    ms_check_source(path, provenance, provenance["baseline_seed"])
    core = YOLO(str(path), task="detect").model.to(TARGET_DEVICE).float().eval()
    core.end2end = True
    architecture_audit(core, variant)
    return core


def ms_probe(seed):
    return torch.rand(1, 3, IMGSZ, IMGSZ, generator=torch.Generator().manual_seed(seed)).to(TARGET_DEVICE)


def ms_baseline(folder, path, source, dataset):
    previous = ms_read(folder / "baseline_report.json")
    if previous:
        checkpoint = folder / "baseline_fp32.pt"
        if sha256_file(checkpoint) != previous["checkpoint"]["sha256"]:
            raise ValueError("SHA-256 baseline FP32 tersimpan berubah.")
        if (previous["validation_parity_status"] == "PASS"
                and (MODE == "smoke" or previous["reserialized"]["latency"]["status"] == "PASS")):
            previous["checkpoint"]["path"] = str(checkpoint.resolve())
            return previous
    seed_environment(EVAL_SEED)
    return gflops_baseline(path, source, "s", ms_probe(EVAL_SEED), folder, dataset)


def ms_search(folder, path, source, baseline, seed, targets):
    if source["baseline_seed"] != seed:
        raise ValueError("Seed pencarian berbeda dari seed baseline.")
    seed_environment(seed)
    image = ms_probe(seed)
    graph = gflops_load_source(path, source, "s")
    _, audit = gflops_make_graph(graph, image, "s", 0.10)

    def measure(plan):
        probe = gflops_load_source(path, source, "s")
        pruner, candidate_audit = gflops_make_graph(probe, image, "s", 0.10)
        apply_exact_gflops_plan(pruner, candidate_audit, plan)
        return model_measurements(probe, IMGSZ)["gflops"]

    cache_path = folder / "profile_cache.json"

    def progress(values, points):
        write_json(cache_path, values)
        write_json(folder / "profile_progress.json", points)

    search = search_gflops_targets(audit, baseline["reserialized"]["gflops"], measure, targets,
                                   profile_cache=ms_read(cache_path, {}), progress_callback=progress)
    write_json(folder / "target_search.json", search)
    return search


def ms_signature(core):
    return {
        "parameters": {name: list(value.shape) for name, value in core.named_parameters()},
        "conv_bn": {name: [module.in_channels, module.out_channels, module.groups]
                    if isinstance(module, torch.nn.Conv2d) else [module.num_features]
                    for name, module in core.named_modules()
                    if isinstance(module, (torch.nn.Conv2d, torch.nn.BatchNorm2d))},
    }


def ms_audit(path, source, baseline, seed, target_row):
    if source["baseline_seed"] != seed:
        raise ValueError("Seed pruning berbeda dari seed baseline.")
    seed_environment(seed)
    image = ms_probe(seed)
    core = gflops_load_source(path, source, "s")
    if model_state_sha256(core) != baseline["state_sha256"]:
        raise ValueError("State awal kandidat berbeda dari baseline seed yang sama.")
    original = deepcopy(core)
    protected = protected_policy_snapshot(core, "s")
    pruner, audit = gflops_make_graph(core, image, "s", 0.10)
    before = measure_layer_inventory(core, image, audit, "extended")
    plan, caps = target_row["plan"], target_row["caps"]
    history = apply_exact_gflops_plan(pruner, audit, plan, caps=caps)
    assert_layer_consistency(core)
    assert_policy_protected_unchanged(core, protected)
    forward_snapshot(core, image)
    oracle = verify_gflops_oracle(original, core, image, audit)
    del original

    seed_environment(seed)
    repeat = gflops_load_source(path, source, "s")
    repeat_pruner, repeat_audit = gflops_make_graph(repeat, image, "s", 0.10)
    replay = apply_exact_gflops_plan(repeat_pruner, repeat_audit, plan, caps=caps)
    report = audit["report"]
    ledger = report["channel_mapping"]
    state = model_state_sha256(core)
    signature = ms_signature(core)
    if (gflops_plan_key(gflops_plan_from_history(history)) != gflops_plan_key(plan)
            or history != replay or ledger != repeat_audit["report"]["channel_mapping"]
            or model_state_sha256(repeat) != state or ms_signature(repeat) != signature):
        raise ValueError("Replay mengubah plan, ledger, struktur, atau tensor.")
    del repeat, repeat_pruner, repeat_audit

    after = measure_layer_inventory(core, image, audit, "extended")
    measured = model_measurements(core, IMGSZ)
    if (measured["parameters"] >= baseline["reserialized"]["parameters"]
            or not math.isclose(measured["gflops"], target_row["gflops"], rel_tol=1e-10)):
        raise ValueError("Parameter atau GFLOPs kandidat berbeda dari profil pencarian.")
    identity = {"plan_sha256": workflow_digest(gflops_plan_key(plan)), "ledger_sha256": workflow_digest(ledger),
                "structure_sha256": workflow_digest(signature), "state_sha256": state}
    return core, {**identity, "identity_sha256": workflow_digest({**identity, "source": source["sha256"]}),
                  "structure_signature": signature, "oracle": oracle, "audit": report, "groups": history,
                  "selected_plan": plan, "pruned": measured}, before, after


def ms_verify_files(folder, candidate):
    saved = candidate["saved_pruned"]
    if sha256_file(folder / "pruned_fp32.pt") != saved["sha256"]:
        raise ValueError("SHA-256 raw checkpoint tersimpan berubah.")
    for name, digest in candidate["audit_files"].items():
        if sha256_file(folder / name) != digest:
            raise ValueError(f"SHA-256 audit berubah: {name}")


def ms_commit_raw(folder, core, proof, source, before, after):
    key = proof["identity_sha256"]
    journal = folder / "pruning_manifest.json"
    candidate = ms_read(journal)
    if candidate:
        if candidate["identity_sha256"] != key:
            raise ValueError("Direktori perlakuan sudah berisi kandidat berbeda.")
        ms_verify_files(folder, candidate)
    else:
        saved = workflow_save_checkpoint(core, folder / "pruned_fp32.pt", source)
        proof["pruned"]["checkpoint_bytes"] = saved["file_bytes"]
        write_json(folder / "dependency_audit.json", proof["audit"])
        write_json(folder / "inventories.json", {"before": before, "after": after})
        candidate = {**proof, "artifact_configuration_id": folder.name, "model_variant": "s",
                     "model_name": ms_model_name(source["baseline_seed"]),
                     "provenance": source,
                     "policy": "gflops-target", "ratio": sum(map(len, proof["selected_plan"].values())) /
                     proof["audit"]["target_policy_eligible_channels"],
                     "pruned_group_count": len(proof["groups"]), "saved_pruned": {"sha256": saved["sha256"],
                     "file_bytes": saved["file_bytes"]}, "audit_files": {
                         name: sha256_file(folder / name) for name in ("dependency_audit.json", "inventories.json")},
                     "structural_smoke_status": "PASS", "reproducibility_status": "PASS", "reload_status": "NOT_RUN",
                     "validation_status": "NOT_RUN", "eda_status": "NOT_RUN", "errors": []}
        candidate["actual_eligible_channel_ratio"] = candidate["ratio"]
        write_json(journal, candidate)
    if candidate["reload_status"] != "PASS":
        checkpoint = folder / "pruned_fp32.pt"
        image = ms_probe(EVAL_SEED)
        try:
            candidate["reload"] = verify_fresh_reload(checkpoint, core, image, folder, timeout=RELOAD_TIMEOUT_SECONDS)
            restored = gflops_reload(checkpoint, core)
            if model_state_sha256(restored) != proof["state_sha256"] or ms_signature(restored) != proof["structure_signature"]:
                raise ValueError("State atau signature arsitektur berubah setelah reload.")
        except Exception as error:
            candidate["reload_status"] = "FAIL"
            candidate["errors"].append({"stage": "reload", "error": safe_error(error)})
            write_json(journal, candidate)
            raise
        candidate["reload_status"] = "PASS"
        write_json(journal, candidate)
    return candidate


def ms_attempts(folder):
    return [ms_read(folder / "attempts" / f"{ms_id(target, seed)}.json", {
        "configuration_id": ms_id(target, seed), "target_speedup": target, "pruning_seed": seed,
        "recovery_seed": seed, "target_status": "NOT_RUN", "search_status": "NOT_RUN", "audit_status": "NOT_RUN",
    }) for target, seed in PRUNING_QUEUE]


def ms_find_targets(folder, path, source, baseline, seed, attempts):
    search = ms_read(folder / "target_search.json", {})
    ready = {item["target"] for item in search.get("targets", []) if item["status"] != "FAIL"}
    if any(row["audit_status"] != "PASS" and row["target_speedup"] not in ready for row in attempts):
        try:
            search = ms_search(folder, path, source, baseline, seed, tuple(row["target_speedup"] for row in attempts))
        except Exception as error:
            search = {"targets": [{"target": row["target_speedup"], "status": "FAIL", "error": safe_error(error)}
                                  for row in attempts]}
            write_json(folder / "target_search.json", search)
    return {item["target"]: item for item in search.get("targets", [])}


def ms_prune(folder, row, item, path, source, baseline):
    row.update(target_status=item["status"], search_status="FAIL" if item["status"] == "FAIL" else "PASS")
    if item["status"] == "FAIL":
        row.update(error=item["error"], audit_status="NOT_RUN")
        return
    row["actual_gflops_speedup"] = item["speed_up"]
    try:
        core, proof, before, after = ms_audit(path, source, baseline, row["pruning_seed"], item)
        ms_commit_raw(folder, core, proof, source, before, after)
        row.update({name: proof[name] for name in (
            "identity_sha256", "state_sha256", "structure_sha256", "ledger_sha256", "plan_sha256")})
        row.update(audit_status="PASS", oracle_status=proof["oracle"]["status"], reproducibility_status="PASS", error=None)
    except Exception as error:
        row.update(audit_status="FAIL", error=safe_error(error))


def ms_raw_manifest(candidate, attempt, baseline, source, dataset, suite_id):
    target = attempt["target_speedup"]
    return {
        "format_version": 2, "stage": "raw_pruned", "model_variant": "s",
        "configuration_id": f"target-{target:.2f}x", "target_speedup": target,
        "experiment_id": attempt["configuration_id"], "suite_id": suite_id,
        "target_status": attempt["target_status"], "actual_gflops_speedup": attempt["actual_gflops_speedup"],
        "raw_checkpoint": {"relative_path": "pruned_fp32.pt", "sha256": candidate["saved_pruned"]["sha256"]},
        "baseline_checkpoint": {"relative_path": "../baseline/baseline_fp32.pt", "artifact_path": "baseline_fp32.pt",
                                "sha256": baseline["checkpoint"]["sha256"]},
        "state_sha256": candidate["state_sha256"], "structure_signature": candidate["structure_signature"],
        "source_baseline": source, "structural_status": candidate["structural_smoke_status"],
        "oracle_status": candidate["oracle"]["status"], "reload_status": candidate["reload_status"],
        "evaluation_mode": MODE, "dataset": {key: dataset[key] for key in (
            "dataset_id", "fingerprint", "full_val_fingerprint", "train_fingerprint", "provenance_status",
            "class_names", "image_count", "split", "seed")},
        "baseline": baseline["reserialized"], "raw_pruned": candidate["pruned"], "audit_files": candidate["audit_files"],
        "validation_status": candidate["validation_status"],
        "baseline_validation_parity_status": baseline["validation_parity_status"],
        "seed": attempt["pruning_seed"], "baseline_seed": source["baseline_seed"], "imgsz": IMGSZ, "platform": PLATFORM,
        "implementation_sha256": IMPLEMENTATION_SHA256,
    }


def ms_metrics(metrics, prefix=""):
    validation = metrics.get("validation", {})
    latency = metrics.get("latency", {})
    return {prefix + name: value for name, value in {
        "gflops": metrics.get("gflops"), "parameters": metrics.get("parameters"),
        "map50_95": validation.get("map50_95"), "map50": validation.get("map50"),
        "latency_p50_ms": latency.get("median_ms"), "latency_p95_ms": latency.get("p95_ms"),
    }.items()}


def ms_published_ref(reference, name):
    if workflow_version_ref(reference).rsplit(":", 1)[0] != f"{WANDB_ENTITY}/{WANDB_PROJECT}/{name}":
        raise ValueError("Referensi W&B tidak menunjuk ke koleksi perlakuan ini.")
    return reference


def ms_publish(folder, manifest):
    name = ms_run_name(manifest["target_speedup"], manifest["seed"])
    collection = f"{name}-raw"
    group = ms_run_group(manifest["suite_id"])
    checkpoint = folder / "pruned_fp32.pt"
    files = {"pruned_fp32.pt": checkpoint, "baseline_fp32.pt": folder.parent / "baseline/baseline_fp32.pt",
             "raw_pruned_manifest.json": folder / "raw_pruned_manifest.json",
             **{key: folder / key for key in manifest["audit_files"]}}
    hashes = {key: sha256_file(path) for key, path in files.items()}
    if (hashes["pruned_fp32.pt"] != manifest["raw_checkpoint"]["sha256"]
            or hashes["baseline_fp32.pt"] != manifest["baseline_checkpoint"]["sha256"]
            or any(hashes[key] != digest for key, digest in manifest["audit_files"].items())):
        raise ValueError("Hash paket artefak berbeda dari manifest raw.")
    journal = folder / "artifact_publications.json"
    record = ms_read(journal, {}).get("raw-pruned")
    if record and (record["files_sha256"] != hashes or record["name"] != collection or record["group"] != group):
        raise ValueError("Paket publikasi tersimpan berubah. Gunakan suite baru.")
    if record and record["status"] == "CONFIRMED":
        ms_published_ref(record["reference"], collection)
        return record
    existed = record is not None
    record = record or {"name": collection, "run_id": uuid.uuid4().hex, "group": group, "files_sha256": hashes}
    record["status"] = "PENDING"
    write_json(journal, {"raw-pruned": record})
    try:
        ms_wandb_environment()
        api = wandb.Api(timeout=30)
        raw_metrics = ms_metrics(manifest["raw_pruned"])
        baseline_metrics = ms_metrics(manifest["baseline"], "baseline_")
        meta = {"stage": "raw_pruned", "model_variant": "s", "platform": PLATFORM, "seed": manifest["seed"],
                "suite_id": manifest["suite_id"], "experiment_id": manifest["experiment_id"],
                "baseline_seed": manifest["baseline_seed"],
                "target_speedup": manifest["target_speedup"], "target_status": manifest["target_status"],
                "actual_gflops_speedup": manifest["actual_gflops_speedup"], **raw_metrics, **baseline_metrics,
                "raw_sha256": hashes["pruned_fp32.pt"], "baseline_fp32_sha256": hashes["baseline_fp32.pt"],
                "baseline_source_sha256": manifest["source_baseline"]["sha256"],
                "state_sha256": manifest["state_sha256"], "dataset_fingerprint": manifest["dataset"]["fingerprint"],
                "dataset_id": manifest["dataset"]["dataset_id"], "train_fingerprint": manifest["dataset"]["train_fingerprint"],
                "validation_image_count": manifest["dataset"]["image_count"], "imgsz": manifest["imgsz"],
                "gpu": ENVIRONMENT["gpu"], "baseline_artifact_ref": manifest["source_baseline"]["requested_artifact"],
                "files_sha256": hashes}
        if not record.get("reference"):
            # A wait timeout can occur after the server has committed the model. Recover that version by its run and hashes.
            if record.get("upload_started"):
                logged = api.run(f"{WANDB_ENTITY}/{WANDB_PROJECT}/{record['run_id']}").logged_artifacts()
                for artifact in logged:
                    if (artifact.type == "model" and artifact.name.rsplit(":", 1)[0] == record["name"]
                            and artifact.metadata.get("files_sha256") == hashes):
                        record["reference"] = ms_published_ref(artifact.qualified_name, collection)
                        break
            if not record.get("reference"):
                with wandb.init(entity=WANDB_ENTITY, project=WANDB_PROJECT, name=name, id=record["run_id"], dir=str(folder),
                                resume="allow" if existed else "never", job_type="pruning-no-recovery",
                                group=group,
                                tags=["yolo26s", "lamp", "depgraph", "poltrada-v3", PLATFORM], config=meta) as run:
                    run.use_artifact(manifest["source_baseline"]["requested_artifact"], type="model")
                    run.summary.update({**raw_metrics, **baseline_metrics, "target_status": manifest["target_status"],
                                        "actual_gflops_speedup": manifest["actual_gflops_speedup"]})
                    artifact = wandb.Artifact(record["name"], type="model", metadata=meta)
                    for entry, path in files.items():
                        artifact.add_file(str(path), name=entry)
                    record["upload_started"] = True
                    write_json(journal, {"raw-pruned": record})
                    uploaded = run.log_artifact(artifact).wait(timeout=300)
                    record["reference"] = ms_published_ref(uploaded.qualified_name, collection)
                    write_json(journal, {"raw-pruned": record})
            write_json(journal, {"raw-pruned": record})
        reference = ms_published_ref(record["reference"], collection)
        remote = api.artifact(reference, type="model")
        if remote.qualified_name != reference:
            raise ValueError("Versi W&B yang diunduh berbeda dari versi upload.")
        with TemporaryDirectory(prefix="wandb_check_", dir=folder) as verify_dir:
            for entry, digest in hashes.items():
                downloaded = remote.get_path(entry).download(root=verify_dir)
                if sha256_file(downloaded) != digest:
                    raise ValueError(f"SHA-256 unduhan berbeda dari paket raw lokal: {entry}.")
        record.update(status="CONFIRMED", artifact_url=remote.url, error=None)
    except Exception as error:
        record.update(status="PENDING", error=safe_error(error))
    write_json(journal, {"raw-pruned": record})
    return record


def ms_complete_raw(folder, candidate, attempt, baseline, source, dataset, suite_id):
    journal = folder / "pruning_manifest.json"
    stage = "integrity"
    try:
        if (candidate["identity_sha256"] != attempt["identity_sha256"] or
                candidate["provenance"] != source or source["baseline_seed"] != attempt["pruning_seed"]):
            raise ValueError("Jurnal raw dan baseline tidak cocok dengan perlakuan.")
        ms_verify_files(folder, candidate)
        candidate["integrity_status"] = "PASS"
        stage = "validation"
        if candidate["validation_status"] != "PASS":
            seed_environment(EVAL_SEED)
            gflops_validation(candidate, folder / "pruned_fp32.pt", dataset, folder)
            if "validation" in candidate:
                candidate["pruned"]["validation"] = candidate.pop("validation")
            write_json(journal, candidate)
        stage = "latency"
        latency_status = candidate["pruned"].get("latency", {}).get("status")
        if latency_status != "PASS" and (latency_status != "NOT_RUN" or TARGET_DEVICE.type == "cuda"):
            core = YOLO(str(folder / "pruned_fp32.pt"), task="detect").model.to(TARGET_DEVICE).float().eval()
            core.end2end = True
            candidate["pruned"]["latency"] = gflops_latency(core, ms_probe(EVAL_SEED))
            del core
            write_json(journal, candidate)
        candidate["baseline"] = baseline["reserialized"]
        candidate["comparison"] = metric_comparison(candidate["baseline"], candidate["pruned"])
        stage = "eda"
        eda_key = workflow_digest({"baseline": candidate["baseline"], "pruned": candidate["pruned"]})
        if candidate["eda_status"] != "PASS" or candidate.get("eda_input_sha256") != eda_key:
            inventory = ms_read(folder / "inventories.json")
            candidate["eda"] = build_run_eda(candidate, folder, inventory["before"], inventory["after"])
            candidate["eda_status"] = "PASS" if candidate["eda"]["status"] == "MEASURED" else candidate["eda"]["status"]
            candidate["eda_input_sha256"] = eda_key
            write_json(journal, candidate)
    except Exception as error:
        candidate[stage + "_status"] = "FAIL"
        candidate["errors"].append({"stage": stage, "error": safe_error(error)})

    if candidate.get("integrity_status") == "PASS":
        manifest_path = folder / "raw_pruned_manifest.json"
        existing = ms_read(folder / "artifact_publications.json", {}).get("raw-pruned")
        # Keep package bytes fixed after the first publication attempt, even on a relocated suite.
        manifest = (ms_read(manifest_path) if existing else
                    ms_raw_manifest(candidate, attempt, baseline, source, dataset, suite_id))
        if not existing:
            write_json(manifest_path, manifest)
        if (MODE == "full_val" and candidate["validation_status"] == "PASS"
                and candidate["pruned"].get("latency", {}).get("status") == "PASS"
                and baseline["validation_parity_status"] == "PASS" and baseline["reserialized"]["latency"]["status"] == "PASS"):
            try:
                candidate["publication"] = ms_publish(folder, manifest)
            except Exception as error:
                candidate["publication"] = {"status": "FAIL", "error": safe_error(error)}
    write_json(journal, candidate)


def ms_report(folder, attempts, dataset, seeds, suite_id):
    rows, models = [], {}
    for attempt in attempts:
        seed = attempt["pruning_seed"]
        identity = BASELINE_SOURCES[seed]
        state = seeds.get(seed, {})
        baseline = state.get("baseline", {})
        source = state.get("source", {})
        baseline_parity = baseline.get("validation_parity_status", "NOT_RUN")
        baseline_metrics = baseline.get("reserialized", {})
        model_name = ms_model_name(seed)
        model_dir = folder / model_name.lower()
        raw_path = model_dir / attempt["configuration_id"] / "pruned_fp32.pt"
        audit_passed = attempt["audit_status"] == "PASS"
        candidate = ms_read(raw_path.with_name("pruning_manifest.json"), {}) if audit_passed else {}
        metrics = candidate.get("pruned", {})
        validation = metrics.get("validation", {}) if candidate.get("validation_status") == "PASS" else {}
        latency = metrics.get("latency", {})
        publication = candidate.get("publication", {})
        confirmed = publication.get("status") == "CONFIRMED"
        usable = (candidate.get("integrity_status") == "PASS" and candidate.get("reload_status") == "PASS"
                  and source == candidate.get("provenance") and bool(baseline))
        row = {**attempt, "model": model_name, "model_variant": "s", "evaluation_mode": MODE, "suite_id": suite_id,
               "actual_gflops_speedup": attempt.get("actual_gflops_speedup"),
               "dataset_id": dataset["dataset_id"], "dataset_fingerprint": dataset["fingerprint"],
               "train_fingerprint": dataset["train_fingerprint"],
               "dataset_provenance_status": dataset["provenance_status"], "baseline_seed": seed,
               "source_baseline_artifact_ref": identity["ref"], "source_baseline_sha256": identity["sha256"],
               "baseline_fp32_sha256": baseline.get("checkpoint", {}).get("sha256"),
               "source_status": state.get("source_status", "NOT_RUN"),
               "baseline_validation_parity_status": baseline_parity, "baseline_error": state.get("error"),
               **ms_metrics(baseline_metrics, "baseline_"),
               "checkpoint_relative_path": raw_path.relative_to(folder).as_posix() if candidate else None,
               "raw_sha256": candidate.get("saved_pruned", {}).get("sha256"),
               "artifact_ref": publication.get("reference") if confirmed else None,
               "artifact_url": publication.get("artifact_url") if confirmed else None,
               "wandb_run_id": publication.get("run_id"), "wandb_group": ms_run_group(suite_id),
               "publication_status": publication.get("status", "NOT_RUN"), "publication_error": publication.get("error"),
               "structural_status": candidate.get("structural_smoke_status", attempt["audit_status"]),
               "integrity_status": candidate.get("integrity_status", "FAIL" if audit_passed and not candidate else "NOT_RUN"),
               "reload_status": candidate.get("reload_status", "NOT_RUN"), "validation_status": candidate.get("validation_status", "NOT_RUN"),
               "eda_status": candidate.get("eda_status", "NOT_RUN"), "latency_status": latency.get("status", "NOT_RUN"),
               **ms_metrics(metrics),
               "validation_image_count": validation.get("validation_image_count"),
               "delta_map50_95_pp": candidate.get("comparison", {}).get("delta_map50_95_pp"),
               "recovery_ready": bool(usable and confirmed and validation and baseline_parity == "PASS" and MODE == "full_val"),
        }
        rows.append(row)
        if seed not in models:
            models[seed] = {"model_name": model_name, "model_variant": "s", "baseline_seed": seed,
                            "baseline_report": baseline, "source": source, "baseline": baseline_metrics,
                            "results": [], "rows": [], "target_search": ms_read(model_dir / "search/target_search.json", {})}
        models[seed]["rows"].append(row)
        if usable:
            models[seed]["results"].append({**candidate, "configuration_id": row["configuration_id"],
                                            "target_speedup": row["target_speedup"], "target_status": row["target_status"],
                                            "baseline": baseline_metrics, "status": "PASS" if validation else "PARTIAL"})
    counts = {"pruned_checkpoint_count": sum(bool(row["raw_sha256"]) for row in rows),
              "confirmed_artifact_count": sum(row["publication_status"] == "CONFIRMED" for row in rows)}
    registry = {"format_version": 2, "workflow": "lamp-yolo26s-multiseed", "mode": MODE, "suite_id": suite_id,
                **counts, "rows": rows}
    write_json(folder / "checkpoint_registry.json", registry)
    write_csv(folder / "checkpoint_registry.csv", rows)
    status = workflow_status(("PASS" if row["validation_status"] == row["eda_status"] == "PASS"
                               and row["integrity_status"] == row["baseline_validation_parity_status"] == "PASS"
                               and (MODE == "smoke" or row["recovery_ready"])
                               else "PARTIAL" if row["audit_status"] == "PASS" else
                               "FAIL" if row["baseline_error"] or row.get("error") else "NOT_RUN"
                               for row in rows), useful=any(model["baseline"] for model in models.values()))
    bundle = {"mode": MODE, "status": status, "registry": registry, "models": list(models.values())}
    write_json(folder / "suite_manifest.json", {"mode": MODE, "status": status, "suite_id": suite_id,
                                               "registry": "checkpoint_registry.json", **counts})
    return bundle


def ms_execute(folder):
    with FileLock(str(folder / ".suite.lock"), timeout=0):
        persist = None
        try:
            if MODE not in {"full_val", "smoke"}:
                raise ValueError("Pilih full_val atau smoke.")
            if not GLOBAL_PRUNING or any((RECOVERY_ENABLED, FINETUNING_ENABLED, KD_ENABLED)):
                raise ValueError("Workflow mengunci global pruning dan menonaktifkan training, recovery, serta KD.")
            if MODE == "full_val" and not DATASET_MANIFEST_ONLY and (
                    TARGET_DEVICE.type != "cuda" or (IMGSZ, BATCH) != (640, 16) or not RUN_LATENCY_BENCHMARK):
                raise ValueError("Full validation penelitian memerlukan CUDA, latency, 640 px, dan batch 16.")
            dataset, data_record = ms_dataset(folder)
            if DATASET_MANIFEST_ONLY:
                if (folder / "resume_contract.json").exists():
                    raise ValueError("Gunakan suite baru untuk ekspor manifest dataset.")
                write_json(folder / "dataset_manifest.json", data_record)
                return {"mode": MODE, "status": "NOT_RUN", "reason": "Manifest dataset tersedia.", "models": []}
            suite_id = ms_bind(folder, dataset)["contract"]["suite_id"]
            attempts = ms_attempts(folder)
            seeds = {}

            def persist():
                return ms_report(folder, attempts, dataset, seeds, suite_id)

            write_json(folder / "dataset_manifest.json", data_record)
            write_json(folder / "environment.json", ENVIRONMENT)
            persist()
            for seed in dict.fromkeys(row["pruning_seed"] for row in attempts):
                jobs = [row for row in attempts if row["pruning_seed"] == seed]
                model_dir = folder / ms_model_name(seed).lower()
                state = seeds[seed] = {}
                try:
                    path, source = ms_source(model_dir, seed)
                    state.update(source=source, source_status="PASS")
                    baseline = ms_baseline(model_dir / "baseline", path, source, dataset)
                    state["baseline"] = baseline
                except Exception as error:
                    state.setdefault("source_status", "FAIL")
                    state["error"] = safe_error(error)
                    print(f"Baseline seed {seed}: {state['error']}", flush=True)
                    persist()
                    continue
                persist()
                targets = ms_find_targets(model_dir / "search", path, source, baseline, seed, jobs)
                for row in jobs:
                    config = row["configuration_id"]
                    raw_dir = model_dir / config
                    if row["audit_status"] != "PASS":
                        ms_prune(raw_dir, row, targets[row["target_speedup"]], path, source, baseline)
                        write_json(folder / "attempts" / f"{config}.json", row)
                        persist()
                    candidate = ms_read(raw_dir / "pruning_manifest.json") if row["audit_status"] == "PASS" else None
                    if candidate:
                        ms_complete_raw(raw_dir, candidate, row, baseline, source, dataset, suite_id)
                    persist()
                    print(config, row["target_status"], row["audit_status"],
                          (candidate or {}).get("validation_status", "NOT_RUN"),
                          (candidate or {}).get("publication", {}).get("status", "NOT_RUN"), flush=True)
                    gc.collect()
                    if TARGET_DEVICE.type == "cuda":
                        torch.cuda.empty_cache()
            (folder / "last_error.json").unlink(missing_ok=True)
            return persist()
        except (Exception, KeyboardInterrupt) as error:
            failure = {"status": "PARTIAL" if persist else "NOT_RUN", "error": safe_error(error) or "Interrupted"}
            write_json(folder / "last_error.json", failure)
            if persist:
                bundle = persist()
                return {**bundle, **failure}
            if (folder / "resume_contract.json").exists():
                raise
            return {**failure, "mode": MODE, "models": []}
