"""Offline notebook04 integration on explicit actual checkpoints and a tiny synthetic dataset.

One epoch and 64px verify the pipeline, never Poltrada accuracy or GPU performance.
Training/backward/optimizer are allowed only inside the ordinary recovery helper.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint-n", type=Path, required=True)
    parser.add_argument("--checkpoint-s", type=Path, required=True)
    parser.add_argument("--ratio", type=float, default=0.5)
    args = parser.parse_args()
    output = ROOT / "outputs/pruning/lamp_recovery_cpu_verification"
    output.mkdir(parents=True, exist_ok=True)
    tempfile.tempdir = str(output)
    os.environ.update({"TEMP": str(output), "TMP": str(output), "MPLBACKEND": "Agg",
                       "WANDB_MODE": "disabled", "YOLO_AUTOINSTALL": "false"})
    import numpy as np
    import requests
    import torch
    import yaml
    from PIL import Image
    from ultralytics import YOLO

    torch.set_num_threads(2)
    checkpoints = {v: p.resolve(strict=True) for v, p in (("n", args.checkpoint_n), ("s", args.checkpoint_s))}
    payload = torch.load(checkpoints["n"], map_location="cpu", weights_only=False)
    model = payload.get("ema") or payload["model"]
    names = [model.names[i] for i in range(len(model.names))]
    del payload, model
    dataset = output / "synthetic_fixture"
    for split in ("train", "val"):
        (dataset / split / "images").mkdir(parents=True, exist_ok=True)
        (dataset / split / "labels").mkdir(parents=True, exist_ok=True)
        for i in range(2):
            rng = np.random.default_rng(42 + i + (0 if split == "train" else 10))
            Image.fromarray(rng.integers(0, 256, (64, 64, 3), dtype=np.uint8)).save(dataset / split / "images" / f"{i}.jpg")
            (dataset / split / "labels" / f"{i}.txt").write_text(f"{i} 0.5 0.5 0.4 0.4\n", encoding="utf-8")
    data_yaml = dataset / "data.yaml"
    data_yaml.write_text(yaml.safe_dump({"path": str(dataset.resolve()), "train": "train/images",
                                        "val": "val/images", "names": names}), encoding="utf-8")
    gate = {"recovery": False, "backward_calls": 0}

    def forbidden(*_args, **_kwargs):
        raise AssertionError("Offline recovery verification attempted network or high-level YOLO.train")

    requests.sessions.Session.request = forbidden
    YOLO.train = forbidden
    backward = torch.Tensor.backward

    def guarded_backward(self, *a, **kw):
        if not gate["recovery"]:
            raise AssertionError("Backward outside the ordinary fine-tuning stage")
        gate["backward_calls"] += 1
        return backward(self, *a, **kw)

    torch.Tensor.backward = guarded_backward
    for cls in (torch.optim.Optimizer, torch.optim.AdamW, torch.optim.Adam, torch.optim.SGD):
        original = cls.step

        def guarded_step(self, *a, _original=original, **kw):
            if not gate["recovery"]:
                raise AssertionError("Optimizer step outside ordinary fine-tuning")
            return _original(self, *a, **kw)

        cls.step = guarded_step
    notebook_path = ROOT / "notebooks/kaggle/pruning/04_lamp_yolo26ns_finetune_recovery.ipynb"
    notebook = json.loads(notebook_path.read_text(encoding="utf-8"))
    scope = {"__name__": "__main__", "display": lambda _value: None}
    wrapped = False
    for index, cell in enumerate(notebook["cells"]):
        if cell["cell_type"] != "code":
            continue
        print("Execute notebook04 cell", index + 1, flush=True)
        if "BUNDLE = execute_models(" in "".join(cell["source"]):
            original_finetune = scope["fine_tune_pruned_checkpoint"]

            def recovery_only(*a, _original=original_finetune, **kw):
                gate["recovery"] = True
                try:
                    return _original(*a, **kw)
                finally:
                    gate["recovery"] = False

            scope["fine_tune_pruned_checkpoint"] = recovery_only
            wrapped = True
        exec(compile("".join(cell["source"]), f"notebook04-cell-{index + 1}", "exec"), scope)  # noqa: S102 -- Trusted local notebook integration.
        if "config" in cell.get("metadata", {}).get("tags", []):
            scope.update({"OUTPUT_DIR": output, "MODEL_VARIANTS": ["n", "s"],
                "LOCAL_CHECKPOINT_PATHS": checkpoints,
                "LOCAL_EXPECTED_SHA256S": {v: hashlib.sha256(p.read_bytes()).hexdigest() for v, p in checkpoints.items()},
                "DEVICE": "cpu", "IMGSZ": 64, "BATCH": 2, "WORKERS": 0,
                "PRUNING_POLICY": "extended", "PRUNING_RATIOS": [args.ratio],
                "RECOVERY_EPOCHS": 1, "RECOVERY_CLOSE_MOSAIC": 0, "RECOVERY_WARMUP_EPOCHS": 0,
                "RECOVERY_BATCH": 2, "RECOVERY_NBS": 2, "RECOVERY_WORKERS": 0,
                "INSTALL_DEPENDENCIES": False, "WANDB_TRACKING": False, "UPLOAD_WANDB_ARTIFACTS": False,
                "RUN_LATENCY_BENCHMARK": False, "DATA_YAML": data_yaml, "DATASET_ROOT": dataset})
    assert wrapped and gate["backward_calls"] > 0
    bundle = scope["BUNDLE"]
    assert bundle["status"] == "COMPLETE"
    results = []
    for suite in bundle["models"]:
        for result in suite["results"]:
            assert result["structural_smoke_status"] == result["recovery_status"] == "PASS"
            assert result["validation_status"] == result["recovery_validation_status"] == "PASS"
            assert result["recovery_training"]["optimizer_steps"] > 0
            assert result["pruned"]["parameters"] == result["recovered"]["parameters"]
            results.append({key: result[key] for key in (
                "model_variant", "ratio", "structural_smoke_status", "validation_status", "recovery_status",
                "recovery_validation_status", "recovery_training", "recovery_reload", "recovered_vs_pruned")})
    report = {"status": "PASS", "kind": "actual N/S source checkpoints; synthetic dataset implementation test only",
              "notebook_sha256": hashlib.sha256(notebook_path.read_bytes()).hexdigest(),
              "suite_dir": str(scope["SUITE_DIR"]), "results": results, "guards": gate,
              "settings": {"epochs": 1, "imgsz": 64, "device": "cpu", "batch": 2, "nbs": 2},
              "limitations": ["Not Poltrada accuracy", "Not full recovery budget", "No GPU latency", "W&B mocked only"]}
    report_path = ROOT / "reports/pruning/recovery/lamp_recovery_cpu_verification_20261103.json"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    print("PASS actual N/S sources, synthetic one-epoch pipeline:", report_path, flush=True)


if __name__ == "__main__":
    main()
