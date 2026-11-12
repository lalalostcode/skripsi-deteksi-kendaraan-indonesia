"""Run all standalone V2 notebook cells against an explicitly supplied CPU checkpoint.

No login, network, package installation or learning is permitted. This validates the
implementation only; unavailable V3/GPU metrics remain NOT_RUN, never substituted.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, help="Backward-compatible alias for --checkpoint-n")
    parser.add_argument("--checkpoint-n", type=Path)
    parser.add_argument("--checkpoint-s", type=Path)
    parser.add_argument("--models", nargs="+", choices=["n", "s"], help="Default: models with an explicit checkpoint")
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--policy", choices=["conservative", "extended", "both"], default="both")
    parser.add_argument("--ratios", nargs="+", type=float, default=[0.1, 0.2, 0.3, 0.4, 0.5])
    parser.add_argument("--data", type=Path)
    parser.add_argument("--output", type=Path, default=ROOT / "outputs/pruning/lamp_v2_ns_cpu_verification")
    parser.add_argument("--report", type=Path, default=ROOT / "reports/pruning/comparisons/lamp_v2_ns_ratio050_cpu_verification_20261103.json")
    args = parser.parse_args()
    sys.stdout.reconfigure(encoding="utf-8")
    supplied = {"n": args.checkpoint_n or args.checkpoint, "s": args.checkpoint_s}
    selected = args.models or [variant for variant, path in supplied.items() if path is not None]
    if not selected or any(supplied[variant] is None for variant in selected):
        parser.error("Provide an explicit checkpoint for every selected model; no download fallback is allowed")
    checkpoints = {variant: supplied[variant].resolve(strict=True) for variant in selected}
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    tempfile.tempdir = str(output)
    os.environ.update({"TEMP": str(output), "TMP": str(output), "MPLBACKEND": "Agg",
                       "WANDB_MODE": "disabled", "YOLO_AUTOINSTALL": "false"})
    import requests
    import torch
    from ultralytics import YOLO

    torch.set_num_threads(2)

    def forbidden(*_args, **_kwargs):
        raise AssertionError("CPU verification attempted learning or a network request")

    YOLO.train = forbidden
    torch.Tensor.backward = forbidden
    torch.autograd.backward = forbidden
    requests.sessions.Session.request = forbidden
    for optimizer in (torch.optim.Optimizer, torch.optim.SGD, torch.optim.Adam, torch.optim.AdamW):
        optimizer.step = forbidden
    path = ROOT / "notebooks/kaggle/pruning/03_lamp_yolo26n_extended_no_recovery.ipynb"
    notebook = json.loads(path.read_text(encoding="utf-8"))
    digests = {variant: hashlib.sha256(path.read_bytes()).hexdigest() for variant, path in checkpoints.items()}
    scope = {"display": lambda _value: None, "__name__": "__main__"}
    for index, cell in enumerate(notebook["cells"]):
        if cell["cell_type"] != "code":
            continue
        print(f"Execute cell {index + 1}", flush=True)
        exec(compile("".join(cell["source"]), f"notebook03-cell-{index + 1}", "exec"), scope)  # noqa: S102 -- Execute trusted local notebook with guards.
        if "config" in cell.get("metadata", {}).get("tags", []):
            scope.update({"OUTPUT_DIR": output, "MODEL_VARIANTS": selected,
                          "LOCAL_CHECKPOINT_PATHS": {variant: str(path) for variant, path in checkpoints.items()},
                          "LOCAL_EXPECTED_SHA256S": digests, "DEVICE": "cpu", "IMGSZ": args.imgsz,
                          "PRUNING_POLICY": args.policy, "PRUNING_RATIOS": args.ratios,
                          "WANDB_TRACKING": False, "UPLOAD_WANDB_ARTIFACTS": False,
                          "INSTALL_DEPENDENCIES": False, "WORKERS": 0, "BATCH": 1,
                          "DATA_YAML": args.data.resolve() if args.data else output / "V3_unavailable.yaml",
                          "DATASET_ROOT": args.data.resolve().parent if args.data else output})
    bundle = scope["BUNDLE"]
    verified_models = []
    for suite in bundle["models"]:
        verified_models.append({"model_variant": suite["model_variant"], "source": suite["source"],
                               "baseline": suite["baseline"], "summary": scope["summary_rows"](suite),
                               "baseline_validation_status": suite["baseline_validation_status"],
                               "results": [{key: result[key] for key in
                                            ("model_variant", "policy", "ratio", "structural_smoke_status", "validation_status", "checks",
                                             "actual_eligible_channel_ratio", "eligible_channels_before", "eligible_channels_after",
                                             "physical_conv_output_channels_removed", "pruned_group_count", "reload", "comparison", "tracking", "eda_status")}
                                           for result in suite["results"]]})
    report = {"kind": "actual N/S baseline artifact CPU implementation verification; no Kaggle/GPU claim",
              "notebook": path.relative_to(ROOT).as_posix(),
              "notebook_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
              "source_checkpoint_sha256": digests, "source_artifacts": {variant: scope["MODEL_ARTIFACTS"][variant] for variant in selected},
              "device": "cpu", "imgsz": args.imgsz, "suite_dir": str(scope["SUITE_DIR"]),
              "models": verified_models,
              "environment": bundle["models"][0]["environment"], "status": bundle["status"],
              "guards": ["YOLO.train", "Tensor.backward", "autograd.backward", "Optimizer/SGD/Adam/AdamW.step", "requests.Session.request"],
              "limitations": ["Single seed", "No recovery", "W&B upload tested only by mock",
                              "Target V3 mAP NOT_RUN unless explicitly supplied", "CUDA latency NOT_RUN"]}
    args.report.parent.mkdir(parents=True, exist_ok=True)
    scope["write_json"](args.report, report)
    print("COMPLETE", bundle["status"], "models:", selected, "configs:",
          sum(len(suite["results"]) for suite in bundle["models"]), "report:", args.report, flush=True)


if __name__ == "__main__":
    main()
