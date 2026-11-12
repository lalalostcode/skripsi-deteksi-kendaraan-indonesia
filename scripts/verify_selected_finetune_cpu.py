"""Exercise selected raw students with real one-epoch CPU training and optional W&B publishing."""
# ruff: noqa: S102 -- Execute trusted standalone notebook declarations for verification.
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path


def load_selected(platform, output, student, raw_packages, fixture, *, online=False, epochs=1, resume=None):
    root = Path(__file__).resolve().parents[1]
    book = root / "notebooks" / platform / "recovery/02_yolo26s_standard_finetuning.ipynb"
    scope = {"__name__": "__main__"}
    for index, cell in enumerate(json.loads(book.read_text(encoding="utf-8"))["cells"]):
        if cell["cell_type"] != "code" or set(cell["metadata"]["tags"]) & {"run", "display", "group-display"}:
            continue
        exec(compile("".join(cell["source"]), f"{book.name}:{index}", "exec"), scope)
        if "config" in cell["metadata"]["tags"]:
            scope.update(STUDENT_INPUTS=[student], MODE="smoke", DEVICE="cpu", IMGSZ=64, BATCH=2,
                         WORKERS=0, INSTALL_DEPENDENCIES=False, WANDB_TRACKING=online,
                         RUN_LATENCY_BENCHMARK=False, EDA_DPI=70, SMOKE_EPOCHS=epochs,
                         SMOKE_DATASET_ROOT=fixture, OUTPUT_DIR=output, LOCAL_RAW_PACKAGES=raw_packages,
                         RESUME_RUN_DIR=resume)
    scope["torch"].set_num_threads(2)
    return scope


def verify_resume(output, student, packages, fixture):
    import torch

    def prepare(name, resume=None):
        scope = load_selected("filkom", output / name, student, packages, fixture, epochs=2, resume=resume)
        original = scope["fine_tuned_train_args"]

        def options(*args, **kwargs):
            return {**original(*args, **kwargs), "nbs": 4, "warmup_epochs": 0.0, "close_mosaic": 0}

        scope["fine_tuned_train_args"] = options
        folder = scope["RUN_DIR"]
        manifest, files = scope["ft_source"](folder)
        raw = scope["fine_tuned_load_raw"](files["pruned_fp32.pt"], manifest)
        data, record, train, yaml = scope["ft_dataset"](manifest, folder / "data")
        contract = scope["ft_contract"](folder, record)
        args = (raw, files["pruned_fp32.pt"], manifest, data, record, train, yaml, folder, None, contract)
        return scope, args

    uninterrupted, args = prepare("uninterrupted")
    expected = uninterrupted["ft_fit"](*args)
    partial, args = prepare("interrupted")
    trainer = partial["SelectedFineTuneTrainer"]
    save_model = trainer.save_model

    def interrupt(self):
        result = save_model(self)
        if self.epoch == 0:
            raise KeyboardInterrupt("Injected interruption after first epoch commit")
        return result

    trainer.save_model = interrupt
    try:
        partial["ft_fit"](*args)
    except KeyboardInterrupt:
        pass
    else:
        raise AssertionError("Interruption did not occur")
    folder = partial["RUN_DIR"]
    first = partial["ft_read"](folder / "latest_state.json")
    state = torch.load(folder / first["relative_path"], map_location="cpu", weights_only=False)
    assert state["epoch"] == 1 and state["gradients"]
    restored, args = prepare("interrupted", resume=folder)
    actual = restored["ft_fit"](*args)
    checks = {}
    for key in ("live_model", "ema"):
        checks[key] = restored["recovery_state_sha256"](expected[key]) == restored["recovery_state_sha256"](actual[key])

    def equal(a, b):
        if isinstance(a, torch.Tensor):
            return torch.equal(a, b)
        if isinstance(a, dict):
            return a.keys() == b.keys() and all(equal(value, b[key]) for key, value in a.items())
        if isinstance(a, (list, tuple)):
            return len(a) == len(b) and all(equal(x, y) for x, y in zip(a, b, strict=True))
        return a == b

    for key in ("optimizer", "scheduler", "scaler", "gradients", "criterion", "ema_updates", "optimizer_steps"):
        checks[key] = equal(expected[key], actual[key])
    result = {"status": "PASS" if all(checks.values()) else "FAIL", "checks": checks,
              "completed_epochs": actual["epoch"], "pending_gradients_after_epoch1": len(state["gradients"]),
              "interrupted_dir": str(folder), "uninterrupted_dir": str(uninterrupted["RUN_DIR"])}
    (output / "resume_verification.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print("RESUME", result, flush=True)
    assert result["status"] == "PASS"


def verify_online(output, packages, fixture, *, refresh_tables=False):
    records = json.loads((output / "verification.json").read_text(encoding="utf-8"))
    audits = {}
    for student, row in records.items():
        assert row["status"] == "PASS", row.get("reason")
        folder = Path(row["run_dir"])
        scope = load_selected("filkom", output, student, packages, fixture, online=True, resume=folder)
        result = scope["ft_read"](folder / "fine_tuned_manifest.json")
        if refresh_tables:
            identity = scope["ft_read"](folder / "wandb_identity.json")
            with scope["wandb"].init(entity=scope["WANDB_ENTITY"], project=scope["WANDB_PROJECT"],
                                      id=identity["id"], resume="must", dir=str(folder)) as run:
                scope["ft_log_eda_tables"](run, result, folder)
        audit = scope["ft_verify_logging"](result, folder)
        remote = scope["wandb"].Api(timeout=30).run(f"{scope['WANDB_ENTITY']}/{scope['WANDB_PROJECT']}/{audit['run_id']}")
        # Early smoke iterations already stored the complete effective config in the results artifact.
        # Also expose those observed settings in the run config for cross-seed filtering.
        training = scope["ft_read"](folder / "training_config.json")["requested"]
        evaluation = dict(result["stages"]["baseline"]["validation"]["validation_arguments_requested"])
        for key in ("data", "project", "name", "seed", "device"):
            training.pop(key, None)
        for key in ("data", "device"):
            evaluation.pop(key, None)
        settings = {"training": training, "evaluation": evaluation,
                    "epoch_loader_policy": "fresh workers and augmentation buffer at each epoch boundary"}
        assert settings == scope["ft_study_settings"]()
        signature = scope["workflow_digest"](settings)
        if remote.config.get("study_signature") != signature:
            remote.config.update(study_settings=settings, study_signature=signature)
            remote.update()
        publications = scope["ft_read"](folder / "artifact_publications.json")
        for publication in publications.values():
            assert publication["status"] == "CONFIRMED"
            artifact = scope["wandb"].Api(timeout=30).artifact(publication["reference"])
            assert artifact.qualified_name == publication["reference"]
            assert artifact.metadata["files_sha256"] == publication["files_sha256"]
        audit.update(student=student, study_signature=signature,
                     publications={key: value["reference"] for key, value in publications.items()})
        audits[student] = audit
        (output / "logging_verification.json").write_text(json.dumps(audits, indent=2), encoding="utf-8")
        print("LOGGING_VERIFIED", student, audit["image_count"], flush=True)
    assert len(audits) == 7
    print("ALL_SEVEN_LOGGING_VERIFIED", flush=True)


def verify_snapshot(output, packages, fixture):
    """Train once, then restore the published complete snapshot into a fresh folder."""
    student = "yolo26s-lamp-su1.20x-seed42-raw:v0"
    first = load_selected("kaggle", output / "initial", student, packages, fixture, online=True)
    initial = first["ft_execute"](first["RUN_DIR"])
    assert initial["status"] == "PASS", initial.get("errors")
    publications = first["ft_read"](first["RUN_DIR"] / "artifact_publications.json")
    snapshot = publications["resume/epoch-001"]
    assert snapshot["status"] == "CONFIRMED" and "training_config.json" in snapshot["files_sha256"]
    identity = first["ft_read"](first["RUN_DIR"] / "wandb_identity.json")
    second = load_selected("kaggle", output / "downloaded_resume", student, packages, fixture, online=True)
    second["RESUME_SNAPSHOT_REF"] = snapshot["reference"]
    resumed = second["ft_execute"](second["RUN_DIR"])
    assert resumed["status"] == "PASS", resumed.get("errors")
    restored = second["ft_read"](second["RUN_DIR"] / "artifact_publications.json")
    checks = {
        "same_run_id": identity["id"] == second["ft_read"](second["RUN_DIR"] / "wandb_identity.json")["id"],
        "same_trained_state": initial["trained_state_sha256"] == resumed["trained_state_sha256"],
        "same_best_sha256": publications["checkpoints/final"]["files_sha256"]["best.pt"]
            == restored["checkpoints/final"]["files_sha256"]["best.pt"],
        "same_snapshot_version": snapshot["reference"] == restored["resume/epoch-001"]["reference"],
        "one_epoch_only": initial["epochs"] == resumed["epochs"] == 1,
        "same_implementation": first["IMPLEMENTATION_SHA256"] == second["IMPLEMENTATION_SHA256"],
    }
    result = {"status": "PASS" if all(checks.values()) else "FAIL", **checks,
              "run_id": identity["id"], "snapshot_ref": snapshot["reference"],
              "implementation_sha256": first["IMPLEMENTATION_SHA256"],
              "initial_run_dir": str(first["RUN_DIR"]), "resumed_run_dir": str(second["RUN_DIR"]),
              "epochs_after_resume": resumed["epochs"], "logging_verification": resumed["logging_verification"]}
    (output / "verification.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print("REMOTE_SNAPSHOT_RESUME", json.dumps(result), flush=True)
    assert result["status"] == "PASS"


def verify_group(output, packages, fixture):
    """Exercise the actual cross-seed reader against finished diagnostic W&B runs."""
    scope = load_selected("filkom", output / "reader", "yolo26s-lamp-su1.20x-seed42-raw:v0",
                          packages, fixture, online=True)
    tables = []
    scope["display"] = tables.append
    rows = scope["ft_group_results"]()
    names = {row["raw_artifact_ref"].rsplit("/", 1)[-1] for row in rows}
    assert names == set(scope["STUDENT_PINS"])
    primary = [row for row in rows if row["platform"] == "filkom" and row["target_speedup"] == 1.2]
    assert {row["seed"] for row in primary} == {42, 0, 2024, 7, 3407}
    assert all(row["gpu"] == "CPU" for row in primary) and tables
    scalar_rows = [{key: value for key, value in row.items()
                    if value is None or isinstance(value, (str, bool, int, float))} for row in rows]
    result = {"status": "PASS", "treatments": len(names), "rows": scalar_rows,
              "primary_seeds_filkom_variant_cpu": sorted(row["seed"] for row in primary),
              "group": scope["ft_group"](), "tables": len(tables)}
    (output / "group_verification.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print("GROUP_READER_VERIFIED", len(names), "treatments", len(rows), "platform/device rows", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--online", action="store_true")
    parser.add_argument("--resume-check", action="store_true")
    parser.add_argument("--verify-only", action="store_true")
    parser.add_argument("--snapshot-check", action="store_true")
    parser.add_argument("--group-check", action="store_true")
    parser.add_argument("--refresh-tables", action="store_true")
    parser.add_argument("--platform", choices=("filkom", "kaggle"), default="filkom")
    parser.add_argument("--students", nargs="+")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--raw-cache", type=Path, default=Path("outputs/pruning/raw_per_seed_verified/verified_inputs.json"))
    parser.add_argument("--fixture", type=Path, default=Path("outputs/pruning/lamp_multiseed_cpu_20261106/fixture"))
    args = parser.parse_args()
    os.environ.update(OMP_NUM_THREADS="2", MKL_NUM_THREADS="2", MPLBACKEND="Agg",
                      WANDB_MODE="online" if args.online else "disabled", YOLO_AUTOINSTALL="false")
    if args.online and not os.environ.get("WANDB_API_KEY"):
        from dotenv import dotenv_values

        key = dotenv_values(Path(__file__).resolve().parents[1] / ".env", interpolate=False).get("WANDB_API_KEY")
        if not key:
            parser.error("Set WANDB_API_KEY before running online verification")
        os.environ["WANDB_API_KEY"] = key
    records = json.loads(args.raw_cache.read_text(encoding="utf-8"))
    packages = {name: row["folder"] for name, row in records.items()}
    args.output.mkdir(parents=True, exist_ok=True)
    if args.snapshot_check or args.group_check:
        if not args.online:
            parser.error("--snapshot-check and --group-check require --online")
        if args.snapshot_check:
            verify_snapshot(args.output.resolve(), packages, args.fixture.resolve())
        if args.group_check:
            verify_group(args.output.resolve(), packages, args.fixture.resolve())
        return
    if args.resume_check:
        verify_resume(args.output.resolve(), "yolo26s-lamp-su1.20x-seed0-raw:v0", packages, args.fixture.resolve())
        return
    if args.verify_only:
        if not args.online:
            parser.error("--verify-only requires --online")
        verify_online(args.output.resolve(), packages, args.fixture.resolve(), refresh_tables=args.refresh_tables)
        return
    verification_path = args.output / "verification.json"
    report = json.loads(verification_path.read_text(encoding="utf-8")) if verification_path.is_file() else {}
    for student in args.students or records:
        if report.get(student, {}).get("status") == "PASS":
            print("ALREADY_VERIFIED", student, flush=True)
            continue
        scope = load_selected(args.platform, args.output.resolve(), student, packages, args.fixture.resolve(), online=args.online)
        result = scope["ft_execute"](scope["RUN_DIR"])
        row = {key: result.get(key) for key in ("status", "reason", "seed", "target_speedup", "epochs", "best_epoch",
               "raw_sha256", "initial_state_sha256", "trained_state_sha256", "raw_unchanged", "reload_status",
               "validation_status", "eda_status", "publication_status", "checkpoint_artifact", "results_artifact")}
        row["run_dir"] = str(scope["RUN_DIR"])
        report[student] = row
        verification_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
        print("SMOKE_RESULT", student, row["status"], row["reason"], flush=True)
        assert row["status"] == "PASS", result.get("errors")
        assert row["epochs"] == 1 and row["raw_unchanged"]
        assert row["initial_state_sha256"] != row["trained_state_sha256"]
        assert len(result["stages"]) == 3
        if args.online:
            assert row["publication_status"] == "CONFIRMED"
    print("COMPLETE", len(report), "one-epoch students", args.output, flush=True)


if __name__ == "__main__":
    main()
