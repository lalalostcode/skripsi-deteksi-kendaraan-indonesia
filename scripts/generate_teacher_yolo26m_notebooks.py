"""Generate the five-seed supervised teacher workflow for Kaggle and Filkom."""
from __future__ import annotations

import ast
import json

try:
    from scripts.notebook_paths import MULTISEED_TEACHER, notebook_path
except ModuleNotFoundError:
    from notebook_paths import MULTISEED_TEACHER, notebook_path

SEEDS = (42, 0, 2024, 7, 3407)  # Seed choices of the current pretrained N/S baselines.
KAGGLE_DATA = "/kaggle/input/datasets/ilhamrafiqin/poltrada-v3-yolo26-clean/data.yaml"
CONFIG = '''
EXPERIMENT_ID = "02_teacher_yolo26m_multiseed"
METHOD = "none"
MODEL_WEIGHTS = "yolo26m.pt"  # Bisa diganti path file pretrained YOLO26m lokal.
SEEDS_TO_RUN = {seeds}
EPOCHS = 100
IMGSZ = 640
BATCH = 16
DEVICE = 0                   # GPU yang terlihat oleh kernel; "cpu" untuk uji lokal.
WORKERS = 2
OPTIMIZER = "AdamW"
LR = 0.001
PATIENCE = 100
AMP = True
WEIGHT_DECAY = 0.0005
WARMUP_EPOCHS = 3.0
CLOSE_MOSAIC = 10
EVAL_SPLIT = "val"
INSTALL_DEPENDENCIES = True
WANDB_ENABLED = True
WANDB_PROJECT = "yolom26-kd-indotraffic"
WANDB_ENTITY = "ilhamrafiqin-brawijaya-university"
WANDB_GROUP = "supervised"
BASE_WANDB_TAGS = ["teacher", "model-m", "poltrada-v3", EXPERIMENT_ID, {platform!r}]
TEACHER_CHECKPOINT = None
'''

ENVIRONMENT = '''
import gc
import hashlib
import importlib.metadata as metadata
import json
import os
import subprocess
import sys
import tempfile
import time
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean, stdev

pins = {"ultralytics": "8.4.155", "wandb": "0.28.1"}


def installed_version(name):
    try:
        return metadata.version(name)
    except metadata.PackageNotFoundError:
        return None


preserved = {name: installed_version(name) for name in ("torch", "torchvision")}
if not all(preserved.values()):
    raise RuntimeError("Gunakan kernel yang sudah memiliki Torch dan torchvision.")
needed = [f"{name}=={version}" for name, version in pins.items() if installed_version(name) != version]
if needed:
    if not INSTALL_DEPENDENCIES:
        raise RuntimeError(f"Dependensi belum sesuai: {needed}")
    with tempfile.TemporaryDirectory(prefix="teacher-yolo26m-") as temp:
        constraints = Path(temp) / "torch-constraints.txt"
        constraints.write_text("".join(f"{name}=={version}\\n" for name, version in preserved.items()), encoding="utf-8")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-c", str(constraints), *needed])
    if {name: installed_version(name) for name in preserved} != preserved:
        raise RuntimeError("Torch/torchvision berubah; gunakan kembali environment GPU awal.")

import torch
import ultralytics
import wandb
from ultralytics import YOLO, settings
from ultralytics.utils.torch_utils import init_seeds

if ultralytics.__version__ != pins["ultralytics"] or wandb.__version__ != pins["wandb"]:
    raise RuntimeError("Restart kernel setelah pemasangan dependensi.")
if DEVICE != "cpu" and not torch.cuda.is_available():
    raise RuntimeError("GPU CUDA tidak tersedia pada kernel ini.")
settings.update({"wandb": False})
ENVIRONMENT_INFO = {"python": sys.version, "kernel": sys.executable, "torch": torch.__version__,
                    "torchvision": preserved["torchvision"], "cuda": torch.version.cuda,
                    "gpu": torch.cuda.get_device_name(DEVICE) if DEVICE != "cpu" else None, **pins}
print(json.dumps(ENVIRONMENT_INFO, indent=2))
print("Dataset:", DATA, "| Output:", PROJECT)
'''

AUTH = {
    "kaggle": '''
if WANDB_ENABLED:
    key = os.environ.get("WANDB_API_KEY")
    if not key:
        from kaggle_secrets import UserSecretsClient
        key = UserSecretsClient().get_secret("WANDB_API_KEY")
    if not key:
        raise RuntimeError("Tambahkan WANDB_API_KEY melalui Kaggle Secrets.")
    wandb.login(key=key)
    del key
''',
    "filkom": '''
if WANDB_ENABLED:
    from getpass import getpass
    key = os.environ.get("WANDB_API_KEY") or getpass("WANDB_API_KEY: ")
    if not key:
        raise RuntimeError("Sediakan WANDB_API_KEY untuk logging online.")
    wandb.login(key=key)
    del key
''',
}

DATA_PREFLIGHT = '''
from ultralytics.data.utils import check_det_dataset


def sha256_file(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


if DATA != "coco8.yaml" and not Path(DATA).is_file():
    raise FileNotFoundError(f"Atur DATA ke data.yaml Poltrada V3 yang tersedia: {DATA}")
if EVAL_SPLIT not in {"val", "test"}:
    raise ValueError("Pilih EVAL_SPLIT val atau test.")
data_info = check_det_dataset(DATA, autodownload=DATA == "coco8.yaml", split=EVAL_SPLIT)
for split in {"train", "val", EVAL_SPLIT}:
    paths = data_info.get(split)
    if not paths or not all(Path(path).exists() for path in ([paths] if isinstance(paths, str) else paths)):
        raise FileNotFoundError(f"Split {split} tidak tersedia. Periksa path di YAML sumber {DATA}.")
ENVIRONMENT_INFO["dataset_yaml_sha256"] = sha256_file(data_info["yaml_file"])
ENVIRONMENT_INFO["class_names"] = [data_info["names"][index] for index in range(data_info["nc"])]
print(f"Split evaluasi: {EVAL_SPLIT} | Kelas: {ENVIRONMENT_INFO['class_names']}")
'''

CHECKPOINT_HELPERS = '''
def load_initial_model():
    global INITIAL_SHA256
    model = YOLO(MODEL_WEIGHTS)
    digest = sha256_file(model.ckpt_path)
    if INITIAL_SHA256 is not None and digest != INITIAL_SHA256:
        raise RuntimeError("Bobot pretrained berubah di tengah antrean seed.")
    INITIAL_SHA256 = digest
    return model


def summarize_seed_metrics(results, seeds):
    summary = {"seeds_requested": seeds, "seeds_completed": [item["seed"] for item in results],
               "initial_checkpoint_sha256": INITIAL_SHA256, "eval_split": EVAL_SPLIT, "metrics": {}}
    for name in ("map50_95", "map50", "precision", "recall"):
        values = [item[name] for item in results]
        summary["metrics"][name] = {"mean": mean(values), "sample_std": stdev(values) if len(values) > 1 else None}
    return summary
'''


def replace_once(source, old, new):
    if source.count(old) != 1:
        raise ValueError(f"Template baseline berubah pada {old!r}")
    return source.replace(old, new, 1)


def replace_function(source, name, replacement):
    node = next(node for node in ast.parse(source).body if isinstance(node, ast.FunctionDef) and node.name == name)
    return replace_once(source, ast.get_source_segment(source, node), replacement)


def teacher_training(source):
    source = replace_function(source, "seed_run_name", 'def seed_run_name(seed):\n    return f"base_teacher_yolo26m_seed{seed}"')
    source = replace_function(source, "prepare_seed_teacher", "")
    edits = [
        ('".yolo26kd_teacher_s_gpu.lock"', '".yolo26m_teacher_gpu.lock"'),
        ('"run": wb_run.id', '"run": getattr(wb_run, "id", None)'),
        ('    fitness = None if trainer.fitness', '''    if trainer.batch_size != BATCH or trainer.args.imgsz != IMGSZ or bool(trainer.amp) != AMP:
        raise RuntimeError("Batch, resolusi, atau AMP efektif berubah dari konfigurasi.")
    wb_state["saved_epochs"].add(epoch)
    fitness = None if trainer.fitness'''),
        ('    artifact = wandb.Artifact(', '''    if wb_run is None:
        return
    best_digest = sha256_file(best_path) if is_best and best_path.is_file() else None
    artifact = wandb.Artifact('''),
        ('"is_best_epoch": bool(is_best), "best_epoch_so_far": wb_state["best_epoch"],',
         '"is_best_epoch": bool(is_best), "best_epoch_so_far": wb_state["best_epoch"],\n            "best_pt_sha256": best_digest,'),
        ('    wb_run.log_artifact(artifact, aliases=aliases).wait()', '''    uploaded = wb_run.log_artifact(artifact, aliases=aliases)
    uploaded.wait()
    if best_digest:
        reference = uploaded.qualified_name
        version = reference.rpartition(":")[2]
        if not version.startswith("v") or not version[1:].isdigit():
            raise RuntimeError("Upload best.pt tidak mengembalikan referensi W&B berversi.")
        wb_state["best_artifact"] = {"reference": reference, "best_pt_sha256": best_digest}'''),
        ('            model = YOLO(MODEL_WEIGHTS)', '            model = load_initial_model()'),
        ('            wb_run.config.update({"training_args": train_args, "kd": KD_CONFIG})',
         '            if wb_run is not None:\n                wb_run.config.update({"training_args": train_args, "initial_checkpoint_sha256": INITIAL_SHA256})'),
        ('            return {"run_dir": str(directory), "best_pt": str(best),',
         '''            if len(wb_state["saved_epochs"]) != EPOCHS:
                raise RuntimeError("Training berakhir sebelum anggaran epoch lengkap.")
            return {"run_dir": str(directory), "best_pt": str(best),'''),
        ('"completed_epochs": len(wb_state["uploaded_epochs"]), "best_epoch": wb_state["best_epoch"]}',
         ('"completed_epochs": len(wb_state["saved_epochs"]), "best_epoch": wb_state["best_epoch"],\n'
          '                    "initial_checkpoint_sha256": INITIAL_SHA256, "best_artifact": wb_state.get("best_artifact")}')),
        ('    diagnostic = DATA == "coco8.yaml"', '    diagnostic = DATA == "coco8.yaml" or DEVICE == "cpu"'),
        ('"teacher_artifact": TEACHER_ARTIFACT if METHOD != "none" else None,\n                ', ''),
        ('    global TEACHER_CHECKPOINT, TEACHER_PROVENANCE, teacher_path\n', ''),
        (('    if METHOD != "none":\n        TEACHER_CHECKPOINT, TEACHER_PROVENANCE, teacher_path = None, {}, None\n'
          '        KD_TRAIN_ARGS["distill_model"] = None\n'), ''),
        ('("_smoke" if DATA == "coco8.yaml" else "")',
         '("_smoke" if DATA == "coco8.yaml" or DEVICE == "cpu" else "")'),
        ('        start_seed_run()\n        prepare_seed_teacher()', '        if WANDB_ENABLED:\n            start_seed_run()'),
        ('student baru dari {MODEL_WEIGHTS}', 'teacher baru dari {MODEL_WEIGHTS}'),
        (('result.update(seed=seed, wandb_run_id=wb_run.id, wandb_run_name=WANDB_RUN_NAME,\n'
          '                      wandb_url=wb_run.url, seed_directory=str(SESSION_DIR))'),
         ('result.update(seed=seed, wandb_run_id=getattr(wb_run, "id", None), wandb_run_name=WANDB_RUN_NAME,\n'
          '                      wandb_url=getattr(wb_run, "url", None), seed_directory=str(SESSION_DIR))')),
        ('        wb_run.finish()\n        log_status("SEED", f"Seed {seed} selesai; run W&B sudah ditutup.")',
         '        if wb_run is not None:\n            wb_run.finish()\n        log_status("SEED", f"Seed {seed} selesai.")'),
        ('    if not WANDB_ENABLED:\n        raise ValueError("Antrean penelitian ini memerlukan W&B online.")',
         '    global INITIAL_SHA256\n    INITIAL_SHA256 = None'),
        ('            progress["completed"].append(', '''            (queue_dir / "multiseed_summary.json").write_text(
                json.dumps(summarize_seed_metrics(results, seeds), indent=2, allow_nan=False), encoding="utf-8")
            progress["completed"].append('''),
    ]
    for old, new in edits:
        source = replace_once(source, old, new)
    return source.replace("if torch.cuda.is_available():", 'if DEVICE != "cpu":')


def teacher_evaluation(source):
    source = replace_once(source, "device=DEVICE, plots=True, verbose=False,",
                          'device=DEVICE, plots=True, verbose=False,\n'
                          '            project=str(SESSION_DIR), name="evaluation", exist_ok=False,')
    source = replace_once(source, 'results["best_epoch"] = wb_state["best_epoch"] if WANDB_ENABLED else None',
                          '''results.update(best_epoch=training["best_epoch"], completed_epochs=training["completed_epochs"],
                       initial_checkpoint_sha256=training["initial_checkpoint_sha256"],
                       best_pt_sha256=sha256_file(best_pt), best_artifact=training["best_artifact"])''')
    source = replace_once(source, '"wandb_enabled": WANDB_ENABLED, "wandb_project": WANDB_PROJECT,',
                          '"wandb_enabled": WANDB_ENABLED, "wandb_project": WANDB_PROJECT, "environment": ENVIRONMENT_INFO,')
    source = replace_once(source, '            wb_run.summary["best_epoch"] = wb_state["best_epoch"]',
                          '            wb_run.summary["best_epoch"] = wb_state["best_epoch"]\n'
                          '            wb_run.summary["best_artifact"] = results["best_artifact"]')
    return source.replace("if torch.cuda.is_available():", 'if DEVICE != "cpu":')


def build_notebook(platform):
    if platform not in AUTH:
        raise ValueError(f"Platform teacher tidak didukung: {platform}")
    baseline = json.loads(notebook_path("kaggle", "03_baseline_yolo26n.ipynb").read_text(encoding="utf-8"))
    shared = ["".join(cell["source"]) for cell in baseline["cells"]]
    paths = (f'DATA = {KAGGLE_DATA!r}\nPROJECT = "/kaggle/working/results"'
             if platform == "kaggle" else '''import os
from pathlib import Path

WORKSPACE_ROOT = Path(os.environ.get("SKRIPSI_ROOT", Path.cwd())).expanduser().resolve()
DATA = str(WORKSPACE_ROOT / "datasets/poltrada-v3/data.yaml")
PROJECT = str(WORKSPACE_ROOT / "outputs/filkom/teacher")''')
    setup = ("Tambahkan dataset Poltrada V3 YOLO26 Clean dan secret WANDB_API_KEY di Kaggle."
             if platform == "kaggle" else "Gunakan kernel GPU Filkom melalui SSH/Jupyter. Periksa DATA dan PROJECT pada sel konfigurasi.")
    cells = [
        ("markdown", f"""# Teacher YOLO26m · 5 seed · {platform.title()}

Latih teacher deteksi supervised di Poltrada V3 untuk mengukur variasi hasil antar-seed.
Seed 42, 0, 2024, 7, dan 3407 mengikuti baseline pretrained N/S. Run All menjalankan
100 epoch per seed. Setiap seed memuat ulang bobot awal `yolo26m.pt` dan mengevaluasi
`best.pt` pada validation. {setup}

Setiap seed mendapat folder dan ID W&B baru. Simpan referensi versi tetap dan SHA-256
dari `best_artifact` untuk menggunakan teacher baru pada KD. Artefak teacher `v50`
adalah rujukan untuk eksperimen KD yang sudah dikonfigurasi.""", None),
        ("markdown", "## 1. Konfigurasi eksperimen", None),
        ("code", paths + "\n" + CONFIG.format(seeds=list(SEEDS), platform=platform), "config"),
        ("markdown", "## 2. Preflight kernel dan W&B", None),
        ("code", ENVIRONMENT + AUTH[platform], "bootstrap"),
        ("markdown", "## 3. Periksa dataset train dan validation", None),
        ("code", DATA_PREFLIGHT, "dataset"),
        ("markdown", "## 4. Training supervised dan antrean seed", None),
        ("code", shared[8].split("\nprint(", 1)[0] + "\n\n" + shared[11] + "\n" + CHECKPOINT_HELPERS, "helpers"),
        ("code", teacher_training(shared[13]), "helpers"),
        ("markdown", "## 5. Evaluasi best checkpoint dan ekspor per seed", None),
        ("code", teacher_evaluation(shared[15]), "helpers"),
        ("markdown", """## 6. Jalankan antrean

Satu seed menuntaskan training, evaluasi, upload, dan penutupan W&B sebelum seed berikutnya.
Ringkasan `multiseed_summary.json` memuat mean dan simpangan baku sampel per metrik.
Bila antrean berhenti, periksa `multiseed_progress.json` dan isi `SEEDS_TO_RUN` dengan seed
yang masih perlu dijalankan. Pilih satu platform/GPU untuk seluruh lima seed.""", None),
        ("code", shared[17], "run"),
    ]
    notebook = {"nbformat": 4, "nbformat_minor": 5, "metadata": {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python"}, "accelerator": "GPU",
        "research_workflow": {"platform": platform, "model_variant": "m", "stage": "supervised_teacher"},
    }, "cells": []}
    for index, (kind, source, tag) in enumerate(cells):
        cell = {"cell_type": kind, "id": f"teacher-multiseed-{index:02d}",
                "metadata": {}, "source": source.strip().splitlines(keepends=True)}
        if kind == "code":
            ast.parse(source.strip())
            cell.update(execution_count=None, outputs=[], metadata={"tags": [tag]})
        notebook["cells"].append(cell)
    return notebook


def main():
    for platform in AUTH:
        path = notebook_path(platform, MULTISEED_TEACHER)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(build_notebook(platform), ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
        print(path)


if __name__ == "__main__":
    main()
