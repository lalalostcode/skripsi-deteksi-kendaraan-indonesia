"""Shared standalone persistence, provenance and W&B helpers for pruning/recovery."""
# ruff: noqa: F821 -- Notebook declarations supply SDK imports and configuration.
import json
import os
import re
import uuid
from contextlib import contextmanager
from datetime import UTC
from pathlib import Path


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + "." + uuid.uuid4().hex + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False), encoding="utf-8")
    os.replace(temporary, path)


def write_csv(path, rows):
    import csv
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + "." + uuid.uuid4().hex + ".tmp")
    keys = list(dict.fromkeys(key for row in rows for key in row))
    with temporary.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=keys)
        writer.writeheader()
        writer.writerows({key: json.dumps(value, ensure_ascii=False) if isinstance(value, (list, dict)) else value
                         for key, value in row.items()} for row in rows)
    os.replace(temporary, path)


def workflow_digest(value):
    import hashlib
    return hashlib.sha256(json.dumps(value, sort_keys=True, default=str, allow_nan=False).encode()).hexdigest()


def workflow_bind(folder, name, contract):
    """Reject semantic changes on resume, allowing absolute paths to relocate."""
    path = Path(folder) / name
    contract = json.loads(json.dumps(contract, sort_keys=True, default=str, allow_nan=False))
    record = {"contract": contract, "sha256": workflow_digest(contract)}
    if path.is_file() and json.loads(path.read_text(encoding="utf-8")) != record:
        raise ValueError(f"Kontrak resume berbeda: {name}. Gunakan suite baru.")
    write_json(path, record)
    return record


def workflow_save_checkpoint(core, path, provenance):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        existing = YOLO(str(path), task="detect").model.cpu().float().eval()
        if model_state_sha256(existing) != model_state_sha256(core):
            raise ValueError(f"Checkpoint immutable sudah berisi state berbeda: {path}")
        return {"path": str(path.resolve()), "sha256": sha256_file(path), "file_bytes": path.stat().st_size}
    temporary = path.with_name(path.stem + "_" + uuid.uuid4().hex + ".pt")
    saved = save_fp32_checkpoint(core, temporary, provenance)
    os.replace(temporary, path)
    return {**saved, "path": str(path.resolve())}


def workflow_suite_dir(output, resume=None):
    if resume:
        path = Path(resume).expanduser().resolve(strict=True)
        if not path.is_dir():
            raise NotADirectoryError(path)
        return path
    from datetime import datetime
    path = Path(output).expanduser().resolve() / (datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ") + "_" + uuid.uuid4().hex[:8])
    path.mkdir(parents=True, exist_ok=False)
    return path


def workflow_status(statuses, *, useful=False):
    statuses = list(statuses)
    if statuses and all(status == "PASS" for status in statuses):
        return "PASS"
    if useful or any(status in {"PASS", "PARTIAL"} for status in statuses):
        return "PARTIAL"
    return "FAIL" if any(status == "FAIL" for status in statuses) else "NOT_RUN"


def workflow_run_name(variant, target, recovery=False):
    tag = f"{target:.2f}".replace(".", "p")
    method = globals().get("RECOVERY_METHOD", "finetune")
    recovery_tag = f"_ft{RECOVERY_EPOCHS}" if method == "finetune" else f"_{method}{RECOVERY_EPOCHS}"
    ft = recovery_tag if recovery else ""
    smoke = "_smoke" if MODE == "smoke" else ""
    return f"prun_lamp_gflops_yolo26{variant}_{PLATFORM}{ft}_su{tag}_seed{SEED}{smoke}"


def workflow_login():
    import getpass

    import wandb
    key = os.environ.get("WANDB_API_KEY")
    if not key and PLATFORM == "kaggle":
        try:
            from kaggle_secrets import UserSecretsClient
            key = UserSecretsClient().get_secret("WANDB_API_KEY")
        except (ImportError, RuntimeError, ValueError):
            pass
    if not key:
        key = getpass.getpass("WANDB_API_KEY: ")
    if not key:
        raise ValueError("W&B memerlukan API key untuk publikasi artefak.")
    os.environ["WANDB_API_KEY"] = key
    wandb.login(key=key, relogin=False)


def workflow_version_ref(reference):
    if not isinstance(reference, str) or not re.fullmatch(r"[^/:]+/[^/:]+/[^/:]+:v\d+", reference):
        raise ValueError("Input W&B memerlukan entity/project/artifact:vN (versi tetap).")
    return reference


@contextmanager
def workflow_tracking(folder, variant, target, *, recovery=False, source_ref=None, group=None, registration=False):
    import wandb
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    run = None
    if not WANDB_TRACKING:
        yield None
        return
    identity_path = folder / "wandb_identity.json"
    existed = identity_path.is_file()
    name = workflow_run_name(variant, target, recovery) + ("_register_raw" if registration else "")
    identity = json.loads(identity_path.read_text(encoding="utf-8")) if existed else {
        "id": uuid.uuid4().hex, "name": name, "group": group or f"lamp-gflops-{PLATFORM}-{folder.parent.name}"}
    if identity["name"] != name:
        raise ValueError("Identitas run resume berbeda dari konfigurasi.")
    write_json(identity_path, identity)
    try:
        workflow_login()
        run = wandb.init(entity=WANDB_ENTITY, project=WANDB_PROJECT, name=name, id=identity["id"],
                         resume="allow" if existed else "never", group=identity["group"],
                         job_type=("register-raw" if registration else
                                   globals().get("RECOVERY_METHOD", "finetune") + "-recovery" if recovery else "pruning-no-recovery"),
                         tags=[PLATFORM, f"yolo26{variant}", "lamp", "depgraph", f"seed{SEED}",
                               "recovery" if recovery else "raw-pruned", MODE],
                         config={"target_speedup": target, "model_variant": variant, "platform": PLATFORM,
                                 "seed": SEED, "mode": MODE})
        if source_ref:
            run.use_artifact(workflow_version_ref(source_ref), type="model")
        yield run
    finally:
        if run is not None:
            run.finish()


class WorkflowUploadPending(RuntimeError):
    """Local computation is committed; only publication needs retrying."""


def workflow_publish(run, folder, label, files, *, metadata_info=None, aliases=None, event=None, kind="model"):
    import wandb
    folder = Path(folder)
    path = folder / "artifact_publications.json"
    publications = json.loads(path.read_text(encoding="utf-8")) if path.is_file() else {}
    files = {name: Path(value).resolve(strict=True) for name, value in files.items()}
    hashes = {name: sha256_file(value) for name, value in files.items()}
    key = label + (":" + event if event else "")
    prior = publications.get(key)
    if prior and prior.get("status") == "CONFIRMED" and prior.get("files_sha256") == hashes:
        return prior
    record = {"status": "NOT_RUN", "files_sha256": hashes, "label": label, "event": event}
    if run is None:
        publications[key] = record
        write_json(path, publications)
        return record
    try:
        artifact = wandb.Artifact(f"{run.name}-{run.id}-{label}", type=kind,
                                 metadata={**(metadata_info or {}), "files_sha256": hashes})
        for name, value in files.items():
            artifact.add_file(str(value), name=name)
        uploaded = run.log_artifact(artifact, aliases=aliases or ["latest"])
        uploaded.wait(timeout=300)
        reference = workflow_version_ref(uploaded.qualified_name)
        record.update(status="CONFIRMED", reference=reference)
    except Exception as error:
        record.update(status="FAIL", error=safe_error(error))
        publications[key] = record
        write_json(path, publications)
        raise WorkflowUploadPending(safe_error(error)) from error
    publications[key] = record
    write_json(path, publications)
    return record


def workflow_report_files(folder):
    folder = Path(folder)
    return {path.relative_to(folder).as_posix(): path for path in folder.rglob("*")
            if path.is_file() and path.suffix in {".json", ".csv", ".yaml", ".md", ".png", ".svg"}
            and "wandb" not in path.relative_to(folder).parts and path.name != "artifact_publications.json"}


def workflow_dataset_manifest(dataset, source_yaml, root, folder):
    """Portable full train/val content identity, never an evaluation train=val placeholder."""
    raw = yaml.safe_load(Path(source_yaml).read_text(encoding="utf-8-sig"))
    names = ordered_names(raw["names"])
    root = Path(root).resolve()
    train = _gflops_split_images(root, raw.get("train"))
    val = _gflops_split_images(root, raw.get("val"))
    if any(image.with_suffix(".npy").is_file() for image in (*train, *val)):
        raise ValueError("Gunakan sumber dataset tanpa cache gambar .npy.")
    train_fp, train_rows = _gflops_fingerprint(root, train, "train", names)
    val_fp, val_rows = _gflops_fingerprint(root, val, "val", names)
    if {row["image_sha256"] for row in train_rows}.intersection(row["image_sha256"] for row in val_rows):
        raise ValueError("Train dan val berbagi konten gambar; recovery tidak boleh memakai train=val.")
    record = {"format_version": 1, "dataset_id": DATASET_ID, "class_names": names,
              "train_fingerprint": train_fp, "val_fingerprint": val_fp,
              "provenance_status": dataset.get("provenance_status", "UNVERIFIED"),
              "train_count": len(train), "val_count": len(val), "source_yaml_sha256": sha256_file(source_yaml)}
    write_json(Path(folder) / "dataset_manifest.json", record)
    return record, train, val
