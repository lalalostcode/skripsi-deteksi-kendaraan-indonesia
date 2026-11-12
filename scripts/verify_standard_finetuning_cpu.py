"""Exercise standalone recovery with local native pruned checkpoints and a fixture."""
# ruff: noqa: S102, C408 -- Execute the generated notebook's trusted authoring cells.
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]


def load_book(path, overrides):
    scope = {"display": lambda value: None}
    for cell in json.loads(path.read_text(encoding="utf-8"))["cells"]:
        tags = cell["metadata"].get("tags", [])
        if cell["cell_type"] != "code" or set(tags) & {"run", "display"}:
            continue
        exec(compile("".join(cell["source"]), str(path), "exec"), scope)
        if "config" in tags:
            scope.update(overrides)
    return scope


def check_resume(scope, output):
    import torch

    dataset, data_record, yaml, _ = scope["recovery_dataset"](output / "data")
    inputs, errors = scope["recovery_discover_inputs"](output)
    assert not errors and len(inputs) == 1
    source = inputs[0][1]
    raw, raw_path, _, _ = scope["recovery_input"](source, dataset, data_record)
    sha = source["manifest"]["raw_checkpoint"]["sha256"]
    settings = {**scope["recovery_settings"](), "epochs": 3, "warmup_epochs": 0.0, "nbs": 8, "close_mosaic": 1}
    common = (settings, "local-resume-fixture-v2")
    fit = scope["recovery_fit"]
    names = dataset["class_names"]
    uninterrupted, _ = fit(raw, raw_path, sha, yaml, names, output / "uninterrupted", *common, val_batch=2)
    resumed_path = output / "interrupted"
    calls = []

    def interrupt(index, row):
        calls.append(index["epoch"])
        raise scope["WorkflowUploadPending"]("injected interruption after local epoch commit")

    try:
        fit(raw, raw_path, sha, yaml, names, resumed_path, *common, val_batch=2, committed_callback=interrupt)
    except scope["WorkflowUploadPending"]:
        assert calls == [1]
    else:
        raise AssertionError("Interruption was not observed.")
    checkpoint, _ = scope["recovery_committed_state"](resumed_path, common[1], sha)
    assert checkpoint["epoch"] == 1 and checkpoint["gradients"] and checkpoint["optimizer_steps"] == 0
    retried = []
    resumed, _ = fit(raw, raw_path, sha, yaml, names, resumed_path, *common, val_batch=2,
                     committed_callback=lambda index, _: retried.append(index["epoch"]))
    assert retried == [1, 2, 3]
    digest = scope["recovery_state_sha256"]
    for key in ("live_model", "ema"):
        if digest(resumed[key]) != digest(uninterrupted[key]):
            diffs = [float((v - uninterrupted[key].state_dict()[k]).abs().max())
                     for k, v in resumed[key].state_dict().items()]
            raise AssertionError(f"Resume {key} mismatch; max abs delta {max(diffs)}")
    for key in ("optimizer_steps", "last_opt_step", "scheduler", "criterion", "epoch_rows"):
        assert resumed[key] == uninterrupted[key], key
    for key in resumed["gradients"]:
        assert torch.equal(resumed["gradients"][key], uninterrupted["gradients"][key]), key
    assert scope["sha256_file"](raw_path) == sha
    return {"status": "PASS", "epochs": resumed["epoch"], "optimizer_steps": resumed["optimizer_steps"],
            "pending_gradient_tensors_after_epoch_1": len(checkpoint["gradients"]),
            "live_sha256": digest(resumed["live_model"]), "ema_sha256": digest(resumed["ema"]),
            "raw_unchanged": True, "manifest": str(output / "interrupted/latest_state.json")}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pruning-output", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--models", nargs="+", choices=["n", "s"], default=["n", "s"])
    parser.add_argument("--resume-check", action="store_true")
    args = parser.parse_args()
    os.environ.update(WANDB_MODE="disabled", MPLBACKEND="Agg", YOLO_AUTOINSTALL="false", OMP_NUM_THREADS="2", MKL_NUM_THREADS="2")
    import requests
    import torch

    torch.set_num_threads(2)
    dataset = args.pruning_output.resolve() / "synthetic_data"
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    evidence = {}
    for variant in args.models:
        suite = next((args.pruning_output / f"yolo26{variant}").glob("*/student_candidates.json")).resolve().parent
        platform = "filkom" if variant == "n" else "kaggle"
        prefix = "01" if variant == "n" else "02"
        path = ROOT / "notebooks" / platform / "recovery" / f"{prefix}_yolo26{variant}_standard_finetuning.ipynb"
        options = dict(MODE="smoke", DATASET_ROOT=dataset, DATA_YAML=dataset / "data.yaml", DATASET_ID="synthetic-fixture",
                       IMGSZ=64, DEVICE="cpu", BATCH=2, WORKERS=0, RECOVERY_BATCH=2, RECOVERY_NBS=2, RECOVERY_WORKERS=0,
                       RECOVERY_EPOCHS=1, WANDB_TRACKING=False, INSTALL_DEPENDENCIES=False, RECOVERY_TARGETS=(1.02,),
                       OUTPUT_DIR=output / f"yolo26{variant}", PRUNING_SUITE_DIR=suite, EDA_DPI=75)
        with patch.object(requests.Session, "request", side_effect=AssertionError("CPU fixture may not access HTTP")):
            scope = load_book(path, options)
            folder = scope["SUITE_DIR"]
            report = scope["recovery_execute"](folder)
            job = report["jobs"][0] if report["jobs"] else {}
            checks = ("input", "training", "structural", "reload", "validation")
            print("CPU pipeline", variant, {key: job.get(key + "_status") for key in checks}, job.get("errors"), flush=True)
            if not all(job.get(key + "_status") == "PASS" for key in checks) or not job.get("raw_unchanged"):
                raise RuntimeError(f"CPU recovery failed. See {folder / 'suite_manifest.json'}")
            eda = folder / "jobs/target-1.02x/eda"
            for panel in ("accuracy", "complexity", "latency", "changes", "per_class"):
                assert (eda / f"comparison_{panel}.png").is_file() and (eda / f"comparison_{panel}.svg").is_file()
            retry = scope["recovery_execute"](folder)["jobs"][0]
            assert retry["resume_checkpoint"] == job["resume_checkpoint"] and retry["recovered_checkpoint"] == job["recovered_checkpoint"]
            evidence[variant] = {"suite": str(folder / "suite_manifest.json"), "stages": {key: job[key + "_status"] for key in (*checks, "eda")},
                                 "status": job["status"], "latency": job["latency_status"], "upload": job["upload_status"],
                                 "raw_sha256": job["raw_sha256"], "recovered_sha256": job["recovered_checkpoint"]["sha256"],
                                 "optimizer_steps": job["optimizer_steps"], "same_epoch_on_retry": True}
            if args.resume_check:
                evidence[variant]["resume"] = check_resume(scope, folder / "resume_verification")
        scope["write_json"](output / "verification.json", evidence)


if __name__ == "__main__":
    main()
