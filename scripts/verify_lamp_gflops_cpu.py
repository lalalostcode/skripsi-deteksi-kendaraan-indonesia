"""Execute separate N/S notebooks offline on cached checkpoints with a synthetic split."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
NOTEBOOKS = {"n": ROOT / "notebooks/kaggle/pruning/05_lamp_yolo26n_gflops_no_recovery.ipynb",
             "s": ROOT / "notebooks/kaggle/pruning/06_lamp_yolo26s_gflops_no_recovery.ipynb"}
CHECKPOINTS = {"n": ROOT / "outputs/lamp_target_artifact/artifact/best.pt",
               "s": ROOT / "outputs/lamp_yolo26s_target_artifact/artifact/best.pt"}


def fixture_dataset(folder):
    import cv2
    import numpy as np
    import yaml
    from ultralytics import YOLO

    classes = YOLO(str(CHECKPOINTS["n"]), task="detect").names
    for split in ("train", "valid"):
        for name in ("images", "labels"):
            (folder / split / name).mkdir(parents=True, exist_ok=True)
    generator = np.random.default_rng(42)
    for split in ("valid", "train"):
        for index in range(6):
            image = generator.integers(0, 256, (64, 64, 3), dtype=np.uint8)
            if not cv2.imwrite(str(folder / f"{split}/images/{index}.jpg"), image):
                raise OSError("Gagal menulis gambar fixture.")
            (folder / f"{split}/labels/{index}.txt").write_text(f"{index} 0.5 0.5 0.4 0.4\n", encoding="utf-8")
    path = folder / "data.yaml"
    path.write_text(yaml.safe_dump({"train": "train/images", "val": "valid/images",
                                    "names": list(classes.values())}, allow_unicode=True), encoding="utf-8")
    return path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--model", nargs="+", choices=["n", "s"], default=["n", "s"])
    parser.add_argument("--fixture", action="store_true")
    parser.add_argument("--representative", action="store_true",
                        help="Exercise all configured targets without rendering per-checkpoint EDA.")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "outputs/pruning/lamp_gflops_cpu")
    args = parser.parse_args()
    destination = args.output_dir.resolve()
    destination.mkdir(parents=True, exist_ok=True)
    os.environ.update({"WANDB_MODE": "disabled", "MPLBACKEND": "Agg", "YOLO_AUTOINSTALL": "false",
                       "OMP_NUM_THREADS": "2", "MKL_NUM_THREADS": "2"})
    import requests
    import torch
    from ultralytics import YOLO

    torch.set_num_threads(2)
    dataset = destination / "synthetic_data"
    data_yaml = fixture_dataset(dataset) if args.fixture else destination / "missing/data.yaml"

    def forbidden(*args, **kwargs):
        raise AssertionError("Smoke pruning offline melarang learning dan HTTP.")

    with (patch.object(YOLO, "train", forbidden), patch.object(torch.Tensor, "backward", forbidden),
          patch.object(torch.autograd, "backward", forbidden), patch.object(torch.optim.AdamW, "step", forbidden),
          patch.object(requests.Session, "request", forbidden)):
        for variant in args.model:
            path = NOTEBOOKS[variant]
            notebook = json.loads(path.read_text(encoding="utf-8"))
            scope = {"display": lambda value: None}
            for index, cell in enumerate(notebook["cells"]):
                if cell["cell_type"] != "code":
                    continue
                tags = cell["metadata"].get("tags", [])
                if args.representative and "run" in tags:
                    scope.update(build_run_eda=lambda *_: {
                        "status": "NOT_RUN", "reason": "Render EDA dilewati pada verifikasi representatif CPU."})
                exec(compile("".join(cell["source"]), f"{path.name}#cell-{index}", "exec"), scope)  # noqa: S102
                if "config" in tags:
                    scope.update(MODE="smoke", MODEL_VARIANTS=[variant], IMGSZ=args.imgsz, DEVICE="cpu", WORKERS=0,
                                 LOCAL_CHECKPOINT_PATHS=CHECKPOINTS, DATASET_ROOT=dataset, DATA_YAML=data_yaml,
                                 DATASET_ID="synthetic-fixture" if args.fixture else None,
                                 OUTPUT_DIR=destination / f"yolo26{variant}",
                                 INSTALL_DEPENDENCIES=False, ALLOW_ARTIFACT_DOWNLOAD=False, WANDB_TRACKING=False,
                                 EDA_DPI=80)
            report = scope["BUNDLE"]
            experiments = [row for model in report["models"] for row in model["rows"] if row.get("target_speedup")]
            if ([row["target_speedup"] for row in experiments] != list(scope["GFLOPS_TARGETS"])
                    or any(row["structural_status"] != "PASS" or row["reload_status"] != "PASS"
                           or (args.fixture and row["validation_status"] != "PASS") for row in experiments)):
                raise RuntimeError(f"Pipeline CPU {report['status']}. Buka {scope['SUITE_DIR'] / 'suite_manifest.json'}")


if __name__ == "__main__":
    main()
