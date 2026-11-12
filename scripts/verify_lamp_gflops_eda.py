"""Replay each N/S notebook's analysis on cached evidence without rerunning experiments."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import uuid
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
NOTEBOOKS = {"n": ROOT / "notebooks/kaggle/pruning/05_lamp_yolo26n_gflops_no_recovery.ipynb",
             "s": ROOT / "notebooks/kaggle/pruning/06_lamp_yolo26s_gflops_no_recovery.ipynb"}


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def replay(args, variant):
    notebook_path = NOTEBOOKS[variant]
    source = args.suite.resolve()
    bundle = json.loads((source / "suite_manifest.json").read_text(encoding="utf-8"))
    bundle["models"] = [model for model in bundle["models"] if model["model_variant"] == variant]
    if not bundle["models"]:
        raise ValueError(f"Suite sumber tidak berisi varian {variant}.")
    destination = args.output_dir.resolve() / uuid.uuid4().hex[:12]
    destination.mkdir(parents=True, exist_ok=False)
    for model in bundle["models"]:
        model_dir = model["model_name"].lower()
        for run in model["results"]:
            eda = Path(model_dir) / run["configuration_id"] / "eda"
            if (source / eda).is_dir():
                shutil.copytree(source / eda, destination / eda)
        for filename in ("p5_contrasts.csv", "p5_per_class.csv"):
            relative = Path(model_dir) / filename
            if (source / relative).is_file():
                (destination / model_dir).mkdir(exist_ok=True)
                shutil.copyfile(source / relative, destination / relative)
    os.environ.update({"MPLBACKEND": "Agg", "WANDB_MODE": "disabled", "YOLO_AUTOINSTALL": "false"})
    import requests
    import torch
    from IPython.display import Image
    from ultralytics import YOLO

    images, tables = [], []

    def capture(value):
        if isinstance(value, Image):
            if not value.data:
                raise AssertionError("Tampilan gambar EDA kosong.")
            images.append(str(value.filename))
        elif hasattr(value, "columns"):
            tables.append({"rows": len(value), "columns": list(value.columns)})

    def forbidden(*args, **kwargs):
        raise AssertionError("Replay EDA tidak boleh menjalankan learning, network, atau evaluasi ulang.")

    scope = {"display": capture}
    notebook = json.loads(notebook_path.read_text(encoding="utf-8"))
    old = {str(path): sha256(path) for path in notebook_path.parent.glob("0[1234]_*.ipynb")}
    comparison_plotted = False
    with (patch.object(YOLO, "train", forbidden), patch.object(YOLO, "val", forbidden),
          patch.object(torch.Tensor, "backward", forbidden), patch.object(torch.optim.AdamW, "step", forbidden),
          patch.object(requests.Session, "request", forbidden)):
        for index, cell in enumerate(notebook["cells"]):
            tags = cell.get("metadata", {}).get("tags", [])
            if cell["cell_type"] != "code" or "run" in tags or "bootstrap" in tags:
                continue
            if "display" in tags:
                scope.update(BUNDLE=bundle, SUITE_DIR=destination)
                if not comparison_plotted:
                    scope["plot_model_comparison"](bundle["models"], destination / "comparison")
                    comparison_plotted = True
            exec(compile("".join(cell["source"]), f"{notebook_path.name}#cell-{index}", "exec"), scope)  # noqa: S102
            if "config" in tags:
                scope.update(EDA_DPI=80, MODE=bundle["mode"])
    analysis = scope["ANALYSIS"]
    if old != {path: sha256(Path(path)) for path in old}:
        raise AssertionError("Notebook 01–04 berubah saat replay.")
    report = {"status": analysis["status"], "scope": "Replay EDA on cached evidence; no experiment rerun",
              "source_suite": str(source), "source_manifest_sha256": sha256(source / "suite_manifest.json"),
              "notebook": str(notebook_path), "notebook_sha256": sha256(notebook_path),
              "models": [model["model_name"] for model in bundle["models"]],
              "configurations": sum(len(model["results"]) for model in bundle["models"]),
              "table_rows": analysis["table_rows"], "displayed_images": len(images),
              "displayed_tables": len(tables), "analysis_plots": analysis["plots"],
              "old_notebooks_sha256": old, "experiment_rerun": "NOT_RUN", "poltrada_v3": "NOT_RUN",
              "kaggle": "NOT_RUN", "cuda_latency": "NOT_RUN", "output_dir": str(destination)}
    (destination / "verification.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({key: report[key] for key in ("status", "models", "configurations", "displayed_images", "output_dir")}))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--suite", type=Path, required=True)
    parser.add_argument("--model", nargs="+", choices=["n", "s"], default=["n", "s"])
    parser.add_argument("--output-dir", type=Path, default=ROOT / "outputs/pruning/lamp_gflops_eda_review")
    args = parser.parse_args()
    for variant in args.model:
        replay(args, variant)


if __name__ == "__main__":
    main()
