"""Run one offline epoch with a real notebook-06 raw package and synthetic data."""
# ruff: noqa: C408 -- Keyword settings mirror the notebook config for the fixture.
import argparse
import hashlib
import json
import os
from pathlib import Path
from unittest.mock import patch

from generate_pruned_fine_tuned_notebooks import METHOD_NOTEBOOKS
from notebook_paths import notebook_path
from verify_standard_finetuning_cpu import load_book


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-manifest", required=True, type=Path)
    parser.add_argument("--data", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--teacher", type=Path, help="Local diagnostic YOLO26m fixture or verified teacher.")
    parser.add_argument("--methods", nargs="+", choices=METHOD_NOTEBOOKS, default=tuple(METHOD_NOTEBOOKS))
    args = parser.parse_args()
    if any(method != "finetune" for method in args.methods) and not args.teacher:
        parser.error("--teacher is required for the KD smoke tests")
    os.environ.update(WANDB_MODE="disabled", MPLBACKEND="Agg", YOLO_AUTOINSTALL="false",
                      OMP_NUM_THREADS="2", MKL_NUM_THREADS="2")

    import requests
    import torch

    torch.set_num_threads(2)
    teacher_sha = hashlib.sha256(args.teacher.read_bytes()).hexdigest() if args.teacher else None
    reports = {}
    with patch.object(requests.Session, "request", side_effect=AssertionError("Smoke test must stay offline")):
        for method in args.methods:
            scope = load_book(notebook_path("filkom", METHOD_NOTEBOOKS[method][0]), dict(
                RAW_MANIFEST_PATH=args.raw_manifest, DATASET_ROOT=args.data.resolve().parent, DATA_YAML=args.data.resolve(),
                DATASET_ID="synthetic-fixture", OUTPUT_DIR=args.output / method, MODE="smoke", DEVICE="cpu", SEED=7,
                IMGSZ=64, BATCH=2, WORKERS=0, INSTALL_DEPENDENCIES=False, WANDB_TRACKING=False, EDA_DPI=65,
                TEACHER_ARTIFACT_REF=None, TEACHER_LOCAL_PATH=args.teacher, TEACHER_SHA256=teacher_sha,
            ))
            folder = scope["RUN_DIR"]
            report = scope["fine_tuned_execute"](folder)
            print(method, report["status"], folder, report.get("errors"), flush=True)
            checks = {key: report.get(key) for key in ("status", "input_status", "training_status", "reload_status",
                      "validation_status", "eda_status", "raw_unchanged", "teacher_frozen", "raw_sha256", "checkpoint",
                      "trained_parameters_sha256", "initial_parameters_sha256", "teacher", "epochs", "resume_status")}
            checks["manifest"] = str(folder / "fine_tuned_manifest.json")
            if report["status"] == "PASS":
                assert report["raw_unchanged"] and report["epochs"] == 1
                assert report["raw_pruning_seed"] != report["seed"]
                assert report["initial_parameters_sha256"] != report["trained_parameters_sha256"]
                assert report["stages"]["raw_pruned"]["parameters"] == report["stages"]["recovered"]["parameters"]
                assert report["teacher_frozen"] is (True if method != "finetune" else None)
                epoch = scope["recovery_read_json"](folder / "epoch_metrics.json")[0]
                if method != "finetune":
                    key = "mgd_weighted_loss" if method == "mgd" else "dis_loss"
                    assert epoch["losses"][key] > 0
                    checks["kd_loss"] = epoch["losses"][key]
                assert epoch["optimizer_steps"] > 0
                checks["optimizer_steps"] = epoch["optimizer_steps"]
            reports[method] = checks
    (args.output / "verification.json").write_text(json.dumps(reports, indent=2), encoding="utf-8")
    assert all(report["status"] == "PASS" for report in reports.values())


if __name__ == "__main__":
    main()
