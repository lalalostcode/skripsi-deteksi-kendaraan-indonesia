"""Run short real recovery tests, verify best/latest packages, then remove test weights."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from scripts.cleanup_smoke_artifacts import save, sha
from scripts.generate_pruned_fine_tuned_notebooks import STUDENT_PINS
from scripts.generate_selected_kd_notebooks import selected_pins
from scripts.verify_selected_finetune_cpu import load_selected as load_ft
from scripts.verify_selected_kd_cpu import load_selected as load_kd

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--online", action="store_true")
    parser.add_argument("--methods", nargs="+", choices=("finetune", "native", "crosskd", "mgd"), default=["finetune", "mgd"])
    parser.add_argument("--output", type=Path, default=ROOT / "outputs/pruning/recovery_storage_smoke_20261108")
    args = parser.parse_args()
    args.output = args.output.resolve()
    args.output.mkdir(parents=True, exist_ok=True)
    os.environ.update(OMP_NUM_THREADS="2", MKL_NUM_THREADS="2", MPLBACKEND="Agg", YOLO_AUTOINSTALL="false",
                      WANDB_MODE="online" if args.online else "disabled")
    if args.online and not os.environ.get("WANDB_API_KEY"):
        from dotenv import dotenv_values
        os.environ["WANDB_API_KEY"] = dotenv_values(ROOT / ".env", interpolate=False)["WANDB_API_KEY"]
    cache = json.loads((ROOT / "outputs/pruning/raw_per_seed_verified/verified_inputs.json").read_text())
    fixture = ROOT / "outputs/pruning/lamp_multiseed_cpu_20261106/fixture"
    teacher = ROOT / "outputs/pruning/teacher_yolo26m_v50_verified/best.pt"
    output = args.output / "verification.json"
    rows = json.loads(output.read_text()) if output.is_file() else {}
    for method in args.methods:
        prior = rows.get(method, {})
        retry_cleanup = args.online and prior.get("status") == "PASS" and prior.get("smoke_artifact_cleanup", {}).get("status") != "PASS"
        if prior.get("status") == "PASS" and not retry_cleanup:
            continue
        pins = STUDENT_PINS if method == "finetune" else selected_pins()
        student = next(name for name, pin in pins.items() if pin["seed"] == 2024)
        # The FT input overlay selects v1; source bytes are inherited unchanged.
        student = student.replace(":v0", ":v1")
        packages = {name: next(value["folder"] for value in cache.values() if value["raw_sha256"] == pin["raw_sha256"])
                    for name, pin in pins.items()}
        packages.update({name.replace(":v0", ":v1"): path for name, path in list(packages.items())})
        epochs = 2 if method == "finetune" else 1
        platform = "kaggle" if method in {"finetune", "crosskd"} else "filkom"
        resume = Path(prior["run_dir"]) if prior.get("run_dir") and prior.get("status") != "PASS" else None
        if method == "finetune":
            scope = load_ft(platform, args.output / method, student, packages, fixture, online=args.online, epochs=epochs, resume=resume)
        else:
            scope = load_kd(method, platform, args.output / method, student, packages, fixture, teacher,
                            online=args.online, epochs=epochs, resume=resume)
        scope["RESUME_UPLOAD_INTERVAL"] = 1  # Diagnostic-only: exercise replacement within two epochs.
        folder = scope["RUN_DIR"]
        if retry_cleanup:
            folder = Path(prior["run_dir"])
            name = "fine_tuned_manifest.json" if method == "finetune" else "kd_manifest.json"
            result = scope["ft_read"](folder / name)
            run_id = scope["ft_read"](folder / "wandb_identity.json")["id"]
            cleanup = scope["storage_cleanup_smoke"](folder, result, run_id)
            result.update(smoke_artifact_cleanup=cleanup, artifact_availability="DELETED_AFTER_SMOKE")
            scope["ceda_write"](folder / name, result)
            scope["ft_registry"](result, folder)
            prior.update(smoke_artifact_cleanup=cleanup, artifact_availability="DELETED_AFTER_SMOKE")
            save(output, rows)
            assert cleanup["status"] == "PASS", cleanup.get("errors")
            print("CLEANUP_RETRY_PASS", method, flush=True)
            continue
        observed = {}
        original = scope["storage_cleanup_smoke"]

        def verify_before_delete(folder, result, run_id):
            api = scope["wandb"].Api(timeout=60)
            journal = scope["ft_read"](folder / "artifact_publications.json")
            latest = journal[f"resume/epoch-{epochs:03d}"]["reference"]
            collection = api.artifact_collection(type_name="model", name=latest.rsplit(":", 1)[0])
            active = [artifact for artifact in collection.artifacts(per_page=100) if artifact.logged_by().id == run_id]
            assert [artifact.qualified_name for artifact in active] == [latest]
            assert result["logging_verification"]["status"] == "PASS"
            retirement = scope["ft_read"](folder / "checkpoint_retention.json")
            assert not any(row.get("errors") for row in retirement.values())
            retired = [row["reference"] for row in journal.values() if row.get("availability_status") == "DELETED_FOR_RETENTION"]
            assert len(retired) == epochs - 1
            index = scope["ft_read"](folder / "latest_state.json")
            for path, digest in ((index["relative_path"], index["sha256"]),
                                 (index["best"]["relative_path"], index["best"]["sha256"])):
                assert sha(folder / path) == digest
            checkpoints = journal["checkpoints/final"]
            scope["storage_verify_package"](checkpoints["reference"], checkpoints["files_sha256"], folder)
            observed.update(status="PASS", run_id=run_id, retired_versions=retired, active_resume_versions=[latest],
                            publications=journal, logging_verification=result["logging_verification"],
                            local_snapshot_count=len(list((folder / "checkpoints").glob("*.pt"))))
            save(folder / "storage_verification_before_cleanup.json", observed)
            return original(folder, result, run_id)

        scope["storage_cleanup_smoke"] = verify_before_delete
        result = scope["ft_execute"](folder)
        row = {key: result.get(key) for key in ("status", "reason", "method", "seed", "epochs", "raw_unchanged", "initial_state_sha256",
            "trained_state_sha256", "artifact_availability", "local_checkpoint_availability", "smoke_artifact_cleanup", "smoke_local_cleanup",
            "logging_verification", "kd_audit", "reload_status", "validation_status", "eda_status", "latency_status")}
        row.update(run_dir=str(folder), platform_variant=platform, actual_device="CPU", storage_verification=observed)
        rows[method] = row
        save(output, rows)
        print("STORAGE_SMOKE", method, row["status"], row["reason"], flush=True)
        assert result["status"] == "PASS", result.get("errors")
        assert result["epochs"] == epochs and result["raw_unchanged"]
        assert result["initial_state_sha256"] != result["trained_state_sha256"]
        assert not list(folder.rglob("*.pt"))
        assert result["smoke_local_cleanup"]["status"] == "PASS"
        if args.online:
            assert observed["status"] == "PASS" and observed["local_snapshot_count"] == 2
            assert result["smoke_artifact_cleanup"]["status"] == "PASS"
            remote = scope["wandb"].Api(timeout=60).run(f"{scope['WANDB_ENTITY']}/{scope['WANDB_PROJECT']}/{observed['run_id']}")
            leftover = [a.qualified_name for a in remote.logged_artifacts() if not a.type.startswith("wandb-") and a.state != "DELETED"]
            assert not leftover, leftover
            row["remote_artifacts_remaining"] = leftover
            save(output, rows)
        if method != "finetune":
            assert result["kd_audit"]["teacher_unchanged"]
        source = next(value for value in cache.values() if value["raw_sha256"] == result["raw_sha256"])
        assert sha(Path(source["folder"]) / "pruned_fp32.pt") == source["raw_sha256"]
        print("VERIFIED_AND_DELETED", method, flush=True)


if __name__ == "__main__":
    main()
