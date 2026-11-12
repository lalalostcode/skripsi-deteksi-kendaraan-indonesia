"""Exercise the selected KD notebooks with real raw students and teacher YOLO26m v50."""
# ruff: noqa: S102 -- Execute trusted repository notebook declarations for verification.
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from scripts.generate_pruned_fine_tuned_notebooks import METHOD_NOTEBOOKS
from scripts.generate_selected_kd_notebooks import selected_pins


def load_selected(method, platform, output, student, packages, fixture, teacher, *, online=False, epochs=1, resume=None):
    root = Path(__file__).resolve().parents[1]
    book = root / "notebooks" / platform / "recovery" / METHOD_NOTEBOOKS[method][0]
    scope = {"__name__": "__main__"}
    for index, cell in enumerate(json.loads(book.read_text(encoding="utf-8"))["cells"]):
        if cell["cell_type"] != "code" or set(cell["metadata"]["tags"]) & {"run", "display", "group-display"}:
            continue
        exec(compile("".join(cell["source"]), f"{book.name}:{index}", "exec"), scope)
        if "config" in cell["metadata"]["tags"]:
            scope.update(STUDENT_INPUTS=[student], MODE="smoke", DEVICE="cpu", IMGSZ=64, BATCH=2,
                         WORKERS=0, INSTALL_DEPENDENCIES=False, WANDB_TRACKING=online,
                         RUN_LATENCY_BENCHMARK=False, EDA_DPI=70, SMOKE_EPOCHS=epochs,
                         SMOKE_DATASET_ROOT=fixture, OUTPUT_DIR=output, LOCAL_RAW_PACKAGES=packages,
                         TEACHER_ARTIFACT_REF=None, TEACHER_LOCAL_PATH=teacher, RESUME_RUN_DIR=resume)
    scope["torch"].set_num_threads(2)
    return scope


def verify_resume(args, method, packages):
    import torch

    student = next(name for name, pin in selected_pins().items() if pin["seed"] == 42)

    def prepare(name, resume=None):
        scope = load_selected(method, "filkom", args.output / method / name, student, packages,
                              args.fixture, args.teacher, epochs=2, resume=resume)
        original = scope["fine_tuned_train_args"]
        scope["fine_tuned_train_args"] = lambda *a, **k: {**original(*a, **k), "nbs": 4, "warmup_epochs": 0., "close_mosaic": 0}
        folder = scope["RUN_DIR"]
        manifest, files = scope["ft_source"](folder)
        raw = scope["fine_tuned_load_raw"](files["pruned_fp32.pt"], manifest)
        scope["ft_verify_raw"](raw, manifest)
        data, record, train, yaml = scope["ft_dataset"](manifest, folder / "data")
        contract = scope["ft_contract"](folder, record)
        return scope, (raw, files["pruned_fp32.pt"], manifest, data, record, train, yaml, folder, None, contract)

    whole, inputs = prepare("uninterrupted")
    expected = whole["ft_fit"](*inputs)
    partial, inputs = prepare("interrupted")
    cls = partial["SelectedKDTrainer"]
    original = cls.save_model

    def interrupt(self):
        result = original(self)
        if self.epoch == 0:
            raise KeyboardInterrupt("Injected epoch commit interruption")
        return result

    cls.save_model = interrupt
    try:
        partial["ft_fit"](*inputs)
    except KeyboardInterrupt:
        pass
    else:
        raise AssertionError("Interruption was not injected")
    index = partial["ft_read"](partial["RUN_DIR"] / "latest_state.json")
    first = torch.load(partial["RUN_DIR"] / index["relative_path"], map_location="cpu", weights_only=False)
    assert first["epoch"] == 1 and first["gradients"] and first["method_state"]["projector_gradients"]
    restored, inputs = prepare("interrupted", resume=partial["RUN_DIR"])
    actual = restored["ft_fit"](*inputs)

    def equal(a, b):
        if isinstance(a, torch.Tensor):
            return torch.equal(a, b)
        if isinstance(a, dict):
            return a.keys() == b.keys() and all(equal(v, b[k]) for k, v in a.items())
        if isinstance(a, (list, tuple)):
            return len(a) == len(b) and all(equal(x, y) for x, y in zip(a, b, strict=True))
        return a == b

    checks = {name: whole["recovery_state_sha256"](expected[name]) == restored["recovery_state_sha256"](actual[name])
              for name in ("live_model", "ema")}
    checks.update({name: equal(expected[name], actual[name]) for name in
                   ("optimizer", "scheduler", "scaler", "gradients", "criterion", "ema_updates", "optimizer_steps", "method_state")})
    result = {"status": "PASS" if all(checks.values()) else "FAIL", "checks": checks, "epochs": actual["epoch"],
              "pending_student_gradients": len(first["gradients"]),
              "pending_projector_gradients": len(first["method_state"]["projector_gradients"]),
              "initial_dir": str(whole["RUN_DIR"]), "resumed_dir": str(restored["RUN_DIR"])}
    path = args.output / method / "resume_verification.json"
    path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print("RESUME", method, json.dumps(result), flush=True)
    assert result["status"] == "PASS"


def verify_snapshot(args, method, packages):
    records = json.loads((args.output / "verification.json").read_text(encoding="utf-8"))
    student = next(name for name, pin in selected_pins().items() if pin["seed"] == 42)
    initial = records[f"{method}/{student}"]
    folder = Path(initial["run_dir"])
    pub = json.loads((folder / "artifact_publications.json").read_text(encoding="utf-8"))
    ref = pub["resume/epoch-001"]["reference"]
    scope = load_selected(method, initial["platform"], args.output / "snapshot_restores" / method, student, packages,
                          args.fixture, args.teacher, online=True)
    # A snapshot's platform is part of the contract; retain the original platform for the restore.
    scope["PLATFORM"] = initial["platform"]
    scope["RESUME_SNAPSHOT_REF"] = ref
    result = scope["ft_execute"](scope["RUN_DIR"])
    assert result["status"] == "PASS", result.get("errors")
    identity = scope["ft_read"](scope["RUN_DIR"] / "wandb_identity.json")
    assert identity["id"] == initial["logging_verification"]["run_id"]
    assert result["epochs"] == 1 and result["trained_state_sha256"] == initial["trained_state_sha256"]
    new = scope["ft_read"](scope["RUN_DIR"] / "artifact_publications.json")
    assert new["resume/epoch-001"]["reference"] == ref
    assert new["checkpoints/final"]["files_sha256"]["best.pt"] == pub["checkpoints/final"]["files_sha256"]["best.pt"]
    proof = {"status": "PASS", "method": method, "snapshot_ref": ref, "same_run_id": True,
             "same_training_state": True, "same_best_sha256": True, "one_epoch_only": True,
             "run_dir": str(scope["RUN_DIR"]), "logging_verification": result["logging_verification"]}
    (args.output / f"snapshot_{method}.json").write_text(json.dumps(proof, indent=2), encoding="utf-8")
    # Use the verified restored folder for subsequent API checks. It represents
    # the same actual epoch, model bytes, and run ID; evaluation/EDA were replayed.
    key = f"{method}/{student}"
    updated = {**initial, **{name: result.get(name) for name in (
        "checkpoint_artifact", "results_artifact", "publication_status", "logging_verification", "kd_audit")},
        "initial_run_dir": initial.get("initial_run_dir", str(folder)),
        "run_dir": str(scope["RUN_DIR"]), "snapshot_restored_run_dir": str(scope["RUN_DIR"])}
    records[key] = updated
    (args.output / "verification.json").write_text(json.dumps(records, indent=2), encoding="utf-8")
    print("SNAPSHOT", method, "PASS", flush=True)


def verify_online(args, packages):
    import math

    records = json.loads((args.output / "verification.json").read_text(encoding="utf-8"))
    assert len(records) == 15
    audits = {}
    report_ids = set()
    for key, record in records.items():
        assert record["status"] == "PASS"
        method, student = key.split("/", 1)
        folder = Path(record["run_dir"])
        scope = load_selected(method, record["platform"], args.output, student, packages,
                              args.fixture, args.teacher, online=True, resume=folder)
        result = scope["ft_read"](folder / "kd_manifest.json")
        audit = scope["ft_verify_logging"](result, folder)
        api = scope["wandb"].Api(timeout=30)
        remote = api.run(f"{scope['WANDB_ENTITY']}/{scope['WANDB_PROJECT']}/{audit['run_id']}")
        identity = scope["ft_identity"]()
        lineage = {artifact.qualified_name for artifact in remote.used_artifacts()}
        expected = {identity[name] for name in ("raw_artifact_ref", "baseline_artifact_ref", "teacher_artifact_ref")}
        assert expected <= lineage and remote.summary["teacher_unchanged"]
        kd_key = "mgd_weighted_loss" if method == "mgd" else "dis_loss"
        history = remote.history(keys=["epoch", f"train/{kd_key}", "train/total_loss"],
                                 samples=max(500, (remote.lastHistoryStep or 0) + 1), pandas=False)
        assert len(history) == 1 and history[0]["epoch"] == 1
        assert all(math.isfinite(history[0][name]) for name in (f"train/{kd_key}", "train/total_loss"))
        publications = scope["ft_read"](folder / "artifact_publications.json")
        for publication in publications.values():
            assert publication["status"] == "CONFIRMED"
            artifact = api.artifact(publication["reference"])
            assert artifact.metadata["files_sha256"] == publication["files_sha256"]
        custom = scope["ceda_read"](folder / "custom_eda_publication.json")
        assert custom["status"] == "CONFIRMED"
        assert api.artifact(custom["reference"]).metadata["files_sha256"] == custom["files_sha256"]
        report_ref = scope["ceda_read"](folder / "custom_eda_report.json")
        wr = scope["ceda_workspace_sdk"]()
        report = wr.Report.from_url(report_ref["url"])
        assert report.id not in report_ids, "Parallel seeded runs must have independent EDA reports"
        report_ids.add(report.id)
        filters = [block.runsets[0].filters for block in report.blocks if isinstance(block, wr.PanelGrid) and block.runsets]
        assert filters and all(audit["run_id"] in value for value in filters)
        audits[key] = {**audit, "lineage": sorted(expected), "kd_epoch_history": history,
                       "artifacts": {k: v["reference"] for k, v in publications.items()},
                       "custom_eda_artifact_ref": custom["reference"], "custom_eda_report": report_ref}
        (args.output / "online_verification.json").write_text(json.dumps(audits, indent=2), encoding="utf-8")
        print("ONLINE_VERIFIED", key, audit["image_count"], flush=True)
    print("ALL_FIFTEEN_ONLINE_VERIFIED", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--methods", nargs="+", default=["native", "crosskd", "mgd"])
    parser.add_argument("--students", nargs="+")
    parser.add_argument("--platform", default="filkom", choices=["kaggle", "filkom"])
    parser.add_argument("--online", action="store_true")
    parser.add_argument("--resume-check", action="store_true")
    parser.add_argument("--snapshot-check", action="store_true")
    parser.add_argument("--group-check", action="store_true")
    parser.add_argument("--verify-only", action="store_true")
    parser.add_argument("--teacher", type=Path, default=Path("outputs/pruning/teacher_yolo26m_v50_verified/best.pt"))
    parser.add_argument("--fixture", type=Path, default=Path("outputs/pruning/lamp_multiseed_cpu_20261106/fixture"))
    args = parser.parse_args()
    args.output = args.output.resolve(); args.teacher = args.teacher.resolve(); args.fixture = args.fixture.resolve()
    args.output.mkdir(parents=True, exist_ok=True)
    os.environ.update(OMP_NUM_THREADS="2", MKL_NUM_THREADS="2", MPLBACKEND="Agg", YOLO_AUTOINSTALL="false",
                      WANDB_MODE="online" if args.online else "disabled")
    if args.online and not os.environ.get("WANDB_API_KEY"):
        from dotenv import dotenv_values
        os.environ["WANDB_API_KEY"] = dotenv_values(Path(__file__).resolve().parents[1] / ".env", interpolate=False)["WANDB_API_KEY"]
    cache = json.loads(Path("outputs/pruning/raw_per_seed_verified/verified_inputs.json").read_text(encoding="utf-8"))
    packages = {name: next(value["folder"] for value in cache.values() if value["raw_sha256"] == pin["raw_sha256"])
                for name, pin in selected_pins().items()}
    if args.verify_only:
        if not args.online:
            parser.error("--verify-only requires --online")
        verify_online(args, packages)
        return
    if args.resume_check or args.snapshot_check:
        for method in args.methods:
            (verify_resume if args.resume_check else verify_snapshot)(args, method, packages)
        return
    if args.group_check:
        scope = load_selected("native", args.platform, args.output / "reader", next(iter(packages)), packages,
                              args.fixture, args.teacher, online=args.online)
        tables = [];scope["display"] = tables.append
        rows = scope["ft_group_results"]()
        assert len({(v["method"], v["seed"]) for v in rows}) == 15
        proof = {"status": "PASS", "rows": rows, "tables": len(tables)}
        (args.output / "group_verification.json").write_text(json.dumps(proof, indent=2), encoding="utf-8")
        print("GROUP_PASS", len(rows), flush=True)
        return
    path = args.output / "verification.json"
    records = json.loads(path.read_text(encoding="utf-8")) if path.is_file() else {}
    for method in args.methods:
        for student in args.students or packages:
            key = f"{method}/{student}"
            if records.get(key, {}).get("status") == "PASS":
                continue
            prior = records.get(key, {})
            resume = Path(prior["run_dir"]) if prior.get("run_dir") else None
            if resume is None:
                candidates = []
                for saved in (args.output / method).rglob("kd_manifest.json"):
                    value = json.loads(saved.read_text(encoding="utf-8"))
                    if (value.get("raw_artifact_ref", "").rsplit("/", 1)[-1] == student
                            and value.get("method") == method and (saved.parent / "resume_contract.json").is_file()):
                        candidates.append(saved)
                if candidates:
                    resume = max(candidates, key=lambda p: p.stat().st_mtime).parent
            if resume is not None and not (resume / "resume_contract.json").is_file():
                resume = None
            scope = load_selected(method, args.platform, args.output / method, student, packages,
                                  args.fixture, args.teacher, online=args.online, resume=resume)
            result = scope["ft_execute"](scope["RUN_DIR"])
            row = {name: result.get(name) for name in ("status", "reason", "seed", "method", "epochs", "raw_sha256",
                   "initial_state_sha256", "trained_state_sha256", "raw_unchanged", "checkpoint_artifact", "results_artifact",
                   "publication_status", "logging_verification", "kd_audit", "reload_status", "validation_status", "eda_status")}
            row.update(run_dir=str(scope["RUN_DIR"]), platform=args.platform)
            records[key] = row;path.write_text(json.dumps(records, indent=2), encoding="utf-8")
            print("KD_SMOKE", key, row["status"], row["reason"], flush=True)
            assert row["status"] == "PASS", result.get("errors")
            assert row["epochs"] == 1 and row["raw_unchanged"] and row["kd_audit"]["teacher_unchanged"]
            assert row["initial_state_sha256"] != row["trained_state_sha256"]
    print("COMPLETE", len(records), flush=True)


if __name__ == "__main__":
    main()
