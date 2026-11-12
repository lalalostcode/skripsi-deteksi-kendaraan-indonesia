"""Generate standalone standard fine-tuning and KD recovery for Kaggle and Filkom."""
from __future__ import annotations

import ast
import hashlib
import json
import runpy
from pathlib import Path

if __name__ == "__main__" and not __package__:
    from generate_pruned_fine_tuned_notebooks import (
        build_notebook as build_source_matched,
    )
else:
    from scripts.generate_pruned_fine_tuned_notebooks import (
        build_notebook as build_source_matched,
    )

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
DESTINATIONS = {
    ("finetune", "n"): "01_yolo26n_standard_finetuning.ipynb",
    ("finetune", "s"): "02_yolo26s_standard_finetuning.ipynb",
    ("native", "ns"): "03_yolo26ns_native_kd.ipynb",
    ("crosskd", "ns"): "04_yolo26ns_crosskd.ipynb",
    ("mgd", "ns"): "05_yolo26ns_mgd_kd.ipynb",
}
LABELS = {"finetune": "Fine-tuning standar", "native": "Native KD", "crosskd": "CrossKD", "mgd": "MGD"}

CONFIG = '''
MODE = "full_val"  # "smoke" untuk satu epoch dan subset tetap maksimal 128 gambar per split.
PRUNING_SUITE_DIR = None  # Folder yang berisi student_candidates.json.
RAW_MANIFEST_PATHS = ()   # Alternatif untuk paket raw lokal tepercaya.
RAW_ARTIFACT_REFS = ()    # Contoh format: ("entity/project/raw-artifact:v0",)
EXPECTED_RAW_SHA256 = {}  # Isi {target: "sha256"} dari student_lock.json untuk perbandingan KD.
RECOVERY_TARGETS = ()     # () = semua target yang tersedia. Contoh: (1.02,)
GFLOPS_TARGETS = (1.02, 1.05, 1.10, 1.20, 1.70)
DATASET_MANIFEST_PATH = None  # Manifest sumber Poltrada V3 untuk salinan Filkom.
RESUME_SUITE_DIR = None
WANDB_TRACKING = True   # False hanya untuk smoke lokal.
'''

TEACHER_CONFIG = '''
TEACHER_ARTIFACT_REF = "ilhamrafiqin-brawijaya-university/yolom26-kd-indotraffic/01_teacher_yolo26m-fux5wdcb-checkpoints:v50"
TEACHER_LOCAL_PATH = None
TEACHER_SHA256 = "d749ef0baa0d6c3ae94e83ded1d9999d83c804b76ad5a841fc2ebec953af1428"
TEACHER_CHECKPOINT_FILENAME = "best.pt"
'''

METHOD_CONFIG = {"native": "NATIVE_DIS_WEIGHT = 1.0", "crosskd": "KD_CLS_WEIGHT = 1.0\nKD_BOX_WEIGHT = 1.0\nKD_CLS_GAMMA = 1.0",
                 "mgd": "MGD_ALPHA = 0.00002\nMGD_MASK_RATIO = 0.65"}

SETTINGS = '''
RECOVERY_EPOCHS = 1 if MODE == "smoke" else 100
RECOVERY_BATCH = 16
RECOVERY_NBS = 64
RECOVERY_WORKERS = 2


def recovery_settings():
    return {"epochs": RECOVERY_EPOCHS, "optimizer": "AdamW", "lr0": 0.0001, "lrf": 0.01,
            "weight_decay": 0.0005, "momentum": 0.937, "warmup_epochs": 3, "warmup_bias_lr": 0.0,
            "close_mosaic": 10 if MODE == "full_val" else 0, "patience": 0,
            "batch": RECOVERY_BATCH, "nbs": RECOVERY_NBS, "workers": RECOVERY_WORKERS, "imgsz": IMGSZ,
            "device": str(TARGET_DEVICE), "seed": SEED, "deterministic": True, "amp": False,
            "resume": False, "freeze": None, "cache": False, "save": True, "val": True,
            "plots": True, "cos_lr": False, "mosaic": 1.0, "mixup": 0.0, "copy_paste": 0.0, "cutmix": 0.0,
            "hsv_h": 0.015, "hsv_s": 0.7, "hsv_v": 0.4, "degrees": 0.0, "translate": 0.1,
            "scale": 0.5, "shear": 0.0, "perspective": 0.0, "flipud": 0.0, "fliplr": 0.5, "multi_scale": False}
'''


def build_notebook(platform="filkom", method="finetune", variant="n"):
    if (method, variant) not in DESTINATIONS or platform not in {"filkom", "kaggle"}:
        raise ValueError("Platform, metode, atau varian notebook tidak terdaftar.")
    if variant != "n":
        return build_source_matched(platform, method)
    pruning_builder = runpy.run_path(str(SCRIPTS / "generate_lamp_gflops_notebook.py"))["build_notebook"]
    default_variant = "n" if variant == "n" else "s"
    pruning = pruning_builder(default_variant, platform)
    shared = ["".join(cell["source"]) for cell in pruning["cells"]]
    config = shared[2].split("# full_val:")[0]
    # Keep common pins, paths, latency and validator settings. Drop pruning-only controls.
    pruning_only = ("MODEL_VARIANTS", "LOCAL_CHECKPOINT_PATHS", "LOCAL_EXPECTED_SHA256S", "CHECKPOINT_SELECTOR",
                    "PRUNING_POLICY", "PRUNING_RATIOS", "GLOBAL_PRUNING", "ITERATIVE_STEPS", "MAX_PRUNING_RATIO",
                    "ROUND_TO", "IMPORTANCE_P", "RUN_FULL_VALIDATION", "WANDB_TRACKING", "UPLOAD_WANDB_ARTIFACTS",
                    "RECOVERY_ENABLED", "FINETUNING_ENABLED", "KD_ENABLED")
    config = "\n".join(line for line in config.splitlines() if not any(line.startswith(key + " =") for key in pruning_only))
    config = "\n".join(line for line in config.splitlines() if not line.startswith("OUTPUT_DIR ="))
    config += f"\n\nPLATFORM = {platform!r}\nRECOVERY_METHOD = {method!r}\nNOTEBOOK_VARIANT = {default_variant!r}\n"
    config += "# Eksperimen saat ini memakai 's'. Pilih varian sesuai artefak raw.\n" if variant == "ns" else ""
    config += "SOURCE_CHECKPOINT_SHA256 = " + shared[2].split("SOURCE_CHECKPOINT_SHA256 = ", 1)[1].split("\nLOCAL_EXPECTED_SHA256S")[0]
    config += '\nDATASET_ID = "ilhamrafiqin/poltrada-v3-yolo26-clean"\n'
    config += ("OUTPUT_DIR = WORKSPACE_ROOT / f'outputs/filkom/recovery/{RECOVERY_METHOD}/yolo26{NOTEBOOK_VARIANT}'\n"
               if platform == "filkom" else
               "OUTPUT_DIR = Path(f'/kaggle/working/recovery/{RECOVERY_METHOD}/yolo26{NOTEBOOK_VARIANT}')\n")
    config += CONFIG
    if method != "finetune":
        config += TEACHER_CONFIG + "\n" + METHOD_CONFIG[method]
    config += "\n" + SETTINGS
    # Pull the actual evaluator used by pruning, not an alternative copy of its protocol.
    runtime = (SCRIPTS / "lamp_gflops_runtime_source.py").read_text(encoding="utf-8")
    common_measurements = "\n\n".join(ast.get_source_segment(runtime, node) for node in ast.parse(runtime).body
                                     if isinstance(node, ast.FunctionDef) and node.name in {"gflops_latency", "gflops_validation"})
    method_sources = []
    if method != "finetune":
        dependencies = runpy.run_path(str(SCRIPTS / "recovery_kd_method_dependencies.py"))["method_dependencies"]
        method_sources = dependencies(ROOT, method) + [(SCRIPTS / "recovery_kd_methods_source.py").read_text(encoding="utf-8")]
    training = [(SCRIPTS / f"{name}_source.py").read_text(encoding="utf-8") for name in (
        "lamp_recovery_training", "standard_finetuning_training", "standard_finetuning_eda", "standard_finetuning_runtime")]
    declarations = [shared[i] for i in (4, 5, 6, 7, 9, 10, 17, 18, 26, 28, 29, 30, 21)] + [common_measurements]
    implementation = hashlib.sha256("\n".join(declarations + training + method_sources + [CONFIG, SETTINGS]).encode()).hexdigest()
    cells = []

    def add(kind, source, tag="helpers"):
        source = source.strip()
        item = {"cell_type": kind, "source": source.splitlines(keepends=True), "metadata": {},
                "id": hashlib.sha256(f"{len(cells)}:{source}".encode()).hexdigest()[:8]}
        if kind == "code":
            ast.parse(source)
            item.update(execution_count=None, outputs=[], metadata={"tags": [tag]})
        cells.append(item)

    model = f"YOLO26{default_variant}"
    notes = ("Loss deteksi native. Kontrol fine-tuning ini memakai raw dan anggaran training yang sama dengan eksperimen KD."
             if method == "finetune" else "Teacher YOLO26m v50 dikunci ke SHA-256 penelitian. Gunakan versi dan SHA-256 raw yang sama dengan fine-tuning dan metode KD lain.")
    add("markdown", f"""# 01 — Recovery {model} · {LABELS[method]} · {platform.title()}

Notebook standalone untuk recovery structured pruning LAMP + DepGraph pada Poltrada V3. {notes}
Isi `RAW_ARTIFACT_REFS` dengan versi tetap W&B yang kamu terima dari tahap pruning. `PRUNING_SUITE_DIR` atau `RAW_MANIFEST_PATHS` dapat dipakai untuk hasil lokal.
Ketika input belum diisi, notebook melaporkan `NOT_RUN`. Default eksperimen 100 epoch dengan AdamW dan FP32. Untuk pemeriksaan pipeline, pilih `MODE="smoke"`.

EDA membandingkan baseline, raw pruned, dan recovered pada validation yang sama. Pilihan struktur student tetap memakai full validation tahap raw pruning.
""")
    add("markdown", "## 02 — Konfigurasi data, artefak, device, dan protokol")
    add("code", config + f"\nIMPLEMENTATION_SHA256 = {implementation!r}", "config")
    add("markdown", "## 03 — Bootstrap kernel dan preflight")
    for source in declarations[:4]:
        add("code", source)
    add("code", shared[8], "bootstrap")
    add("code", declarations[4], "imports")
    add("code", declarations[5])
    add("code", shared[11], "bootstrap")
    add("markdown", """## 04 — Audit input dan evaluator identik
Fingerprint dibentuk dari isi train dan val. Setiap checkpoint diperiksa dari manifest dan hash, lalu diukur dengan evaluator pruning yang sama.
Latensi memakai raw forward dual-head FP32 pada device, ukuran input, dan batch yang sama. Setiap hasil mencatat scope pengukuran.""")
    for source in declarations[6:]:
        add("code", source)
    add("markdown", """## 05 — Training, checkpoint per epoch, dan resume
Trainer mempertahankan objek DetectionModel dengan channel pruned. Checkpoint epoch menyimpan model aktif, EMA FP32, state training dan gradien pending.
W&B mengonfirmasi raw sebelum training. Epoch yang tersimpan diunggah dengan alias epoch, latest, dan best bila cocok. Untuk resume, gunakan `RESUME_SUITE_DIR` lama dengan konfigurasi yang sama.""")
    add("code", training[0])
    add("code", training[1])
    for source in method_sources:
        add("code", source)
    add("code", training[2])
    add("code", training[3])
    add("code", '''BUNDLE = recovery_execute(SUITE_DIR)
print("Output", SUITE_DIR, "| Status", BUNDLE["status"], "|", BUNDLE.get("reason", ""))
display(pd.DataFrame(BUNDLE.get("errors", [])))''', "run")
    add("markdown", """## 06 — EDA dan evaluasi per target
Panel terpisah membandingkan akurasi, AP per kelas, ukuran model, channel, GFLOPs, latensi, speed-up, dan recovery gain. Nilai yang belum diukur tetap kosong.
Folder EDA juga memuat kurva training/validation, PR/F1, confusion matrix, serta audit channel, heatmap dan LAMP dari paket raw jika tersedia.""")
    add("code", '''from IPython.display import Image as NotebookImage, Markdown

for job in BUNDLE["jobs"]:
    configuration = job["configuration_id"]
    display(Markdown(f"### {configuration} · {job['target_status']} · {job['status']}"))
    display(pd.DataFrame([{"stage": key, **{name: stage.get(name) for name in ("validation_status", "parameters", "gflops")}}
                          for key, stage in job.get("stages", {}).items()]))
    eda = SUITE_DIR / "jobs" / configuration / "eda"
    if (eda / "stage_metrics.csv").is_file():
        display(pd.read_csv(eda / "stage_metrics.csv"))
    if job.get("alias_of"):
        display(Markdown(f"Checkpoint dan audit struktur dibagi dengan {job['alias_of']}."))
    for figure in sorted(eda.rglob("*.png")):
        display(NotebookImage(filename=str(figure)))''', "display")
    add("markdown", """## 07 — Ekspor hasil dan lineage
Gunakan manifest raw serta SHA-256 yang sama untuk membandingkan seluruh metode recovery. `recovered_fp32.pt` adalah model inference. State resume disimpan terpisah dalam `checkpoints/`.""")
    add("code", '''for name in ("suite_manifest.json", "recovery_summary.csv", "stage_metrics.csv", "per_class_metrics.csv"):
    path = SUITE_DIR / name
    if path.is_file():
        print(path)
        if path.suffix == ".csv" and path.with_suffix(".json").is_file():
            display(pd.DataFrame(recovery_read_json(path.with_suffix(".json"))))
for job in BUNDLE["jobs"]:
    print(job["configuration_id"], job.get("raw_artifact_ref"), job.get("raw_sha256"), job.get("recovered_publication"))''', "display")
    book = {**pruning, "cells": cells, "metadata": {**pruning["metadata"], "research_workflow": {
        "platform": platform, "model_variant": variant, "stage": "recovery", "method": method}}}
    if __name__ == "__main__" and not __package__:
        from recovery_custom_eda import attach_custom_eda
    else:
        from scripts.recovery_custom_eda import attach_custom_eda
    return attach_custom_eda(book, ROOT, selected=False)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--method", choices=LABELS)
    method_filter = parser.parse_args().method
    for platform in ("kaggle", "filkom"):
        for (method, variant), filename in DESTINATIONS.items():
            if method_filter and method_filter != method:
                continue
            path = ROOT / "notebooks" / platform / "recovery" / filename
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps(build_notebook(platform, method, variant), ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
            print(path)
