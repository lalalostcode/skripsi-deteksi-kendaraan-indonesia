"""Build standalone notebook04 from the established pruning declarations.

Notebook03 and its generator are read-only inputs. Repository imports occur only
during authoring; all runtime helpers are embedded in the resulting notebook.
"""
from __future__ import annotations

import ast
import hashlib
import json
import re
import runpy
from copy import deepcopy
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DESTINATION = ROOT / "notebooks/kaggle/pruning/04_lamp_yolo26ns_finetune_recovery.ipynb"
RUNTIME = ROOT / "scripts/lamp_recovery_runtime_source.py"
TRAINING = ROOT / "scripts/lamp_recovery_training_source.py"

MARKDOWN = {
    1: """# 01 — LAMP Structured Pruning dan Recovery Fine-tuning Biasa
Notebook standalone **YOLO26n dan YOLO26s**: baseline terlatih → pruning struktural LAMP →
evaluasi sebelum recovery → fine-tuning biasa → evaluasi setelah recovery. Tidak memakai
teacher, KD, sparse training, atau perubahan loss deteksi. Notebook 01/02/03 dipertahankan.

Default memproses extended pruning 10%, 20%, 30%, 40%, dan 50% channel eligible untuk N/S.
Setiap rasio memulai dari baseline asli, dengan satu budget recovery yang sama.
Pruning/reload yang valid tidak menjamin akurasi pulih; hasil negatif tetap disimpan.

Rujukan: [LAMP asli](https://arxiv.org/abs/2010.07611),
[Torch-Pruning 1.6.1](https://github.com/VainF/Torch-Pruning/tree/v1.6.1),
[Ultralytics Advanced Customization](https://docs.ultralytics.com/usage/engine/),
[Ultralytics Train](https://docs.ultralytics.com/modes/train/), dan
[MP-YOLO, Zhou dkk.](https://doi.org/10.1016/j.jvcir.2025.104560).
MP-YOLO memakai pruning dan fine-tuning; notebook ini merupakan adaptasi pada YOLO26,
bukan replikasi penuh arsitektur, stopping budget, maupun jadwal paper.

Kaggle: aktifkan GPU + Internet, tambahkan **Poltrada V3 YOLO26 Clean** baseline dan secret
`WANDB_API_KEY`, lalu Run All pada kernel baru. Sepuluh konfigurasi dapat memerlukan waktu panjang;
pilih satu model/rasio untuk uji pipeline.""",
    2: """## 02 — Konfigurasi Pruning dan Recovery
Semua parameter pengguna berada pada cell ini. Default N/S, extended, rasio 10–50% menghasilkan
sepuluh eksperimen independen. `MODEL_VARIANTS=["n"]` atau `["s"]` memilih satu model;
`PRUNING_RATIOS=[0.10]` membatasi uji. Baseline N memakai artifact `v84`, S memakai `v62`;
`best.pt` harus unik. Hash checkpoint lokal bersifat eksplisit per model.

Rasio adalah proporsi channel eligible, bukan seluruh parameter. Cap group tetap 50%, satu
langkah, tanpa rounding. Fine-tuning memakai optimizer baru (`resume=False`), AdamW dan
learning rate eksplisit, FP32 tanpa AMP, loss deteksi native Ultralytics, dan validation
untuk memilih best EMA. Anggaran `RECOVERY_EPOCHS` sama pada seluruh konfigurasi.
Tidak ada recovery kumulatif antar-rasio atau penggunaan hasil N sebagai baseline S.""",
    3: """## 03 — Environment dan Dependencies
Pin Ultralytics 8.4.155, Torch-Pruning 1.6.1, dan W&B 0.28.1. Bootstrap mempertahankan
Torch/torchvision Kaggle serta berhenti saat dependency tidak kompatibel. Source trainer
versi terpasang menjadi kontrak implementasi; dokumentasi terbaru dapat memiliki API berbeda.
Pengukuran dilakukan pada salinan unfused FP32; prediction/validation memakai wrapper terpisah.""",
    8: """## 08 — Eksperimen Independen dan Fine-tuning Recovery
Setiap konfigurasi memuat baseline asli dan memverifikasi state hash sebelum pruning.
Baseline diukur/divalidasi sekali per model; checkpoint pruned asli tetap disimpan sebagai
pembanding sebelum recovery. Trainer mempertahankan objek DetectionModel struktural dan
memeriksa state/shape awal agar YAML standar tidak membangun ulang channel yang dihapus.

Recovery menjalankan training biasa pada split train dengan loss deteksi native, optimizer
baru dan seed yang dicatat. Validation memilih best EMA. Semua bobot yang dapat dilatih,
termasuk head, dapat diperbarui selama recovery; perlindungan head berarti dimensi tidak
dipangkas. Dimensi hasil pruning wajib tetap sama sebelum/sesudah fine-tuning.
Dataset tidak tersedia menghasilkan NOT_RUN tanpa download pengganti; dataset invalid
atau kegagalan training disimpan ke laporan lalu exception diteruskan.""",
    9: """## 09 — Checkpoint Pruned, Recovered, dan Fresh Reload
Checkpoint sebelum recovery dipertahankan sebagai `pruned_fp32.pt`; checkpoint hasil
best EMA disimpan sebagai `recovery/recovered_fp32.pt`. Keduanya memuat objek DetectionModel
native struktural FP32 dengan `ema=None`, `optimizer=None`, tanpa kelas trainer khusus.
Checkpoint training best/last dan laporan epoch disimpan terpisah.

Fresh Python tanpa import repository memuat YOLO(checkpoint), menjalankan kedua cabang,
prediction, dan parity FP32 raw/decoded outputs (`rtol=1e-4`, `atol=1e-5`). Bentuk tensor,
jumlah parameter dan class order wajib cocok dengan model pruned; akurasi dinilai terpisah.""",
    10: """## 10 — Validasi Baseline, Pruned, dan Recovered
Ketiga tahap memakai protokol identik: val, 640, confidence 0.001, IoU 0.7, max_det 300,
nms=None, FP32 dan tanpa TTA. Class order wajib sama. Dataset tersedia tetapi invalid
menghasilkan FAIL; dataset tidak tersedia menghasilkan NOT_RUN. Tidak memakai fallback V2.
Latency CUDA berupa raw unfused model forward, batch satu, 50 warm-up, 200 pengukuran
dan sinkronisasi; preprocessing/NMS/disk IO tidak termasuk.

Bandingkan penurunan akurasi akibat pruning, perubahan akibat fine-tuning, dan selisih akhir
terhadap baseline. Status pipeline dan besar pemulihan akurasi dilaporkan secara terpisah.""",
    12: """## 12 — EDA Efisiensi dan Metrik pada Tiga Tahap
Tabel membedakan `stage=baseline`, `pruned`, dan `recovered`. Parameter/GFLOPs/channel
setelah recovery harus sama dengan struktur pruned. Ukuran checkpoint memakai FP32 yang
sama; latency disertai interval p05–p95. Diagram channel dan ledger menjelaskan keputusan
pruning, sedangkan training curve menjelaskan jalannya fine-tuning.

Delta mAP dalam poin persentase dan perubahan relatif dicatat terpisah, termasuk gain
recovered versus pruned serta selisih recovered versus baseline. AP/P/R/F1/support per kelas,
kurva PR/F1/confidence, dan confusion matrix memakai hasil validator aktual. IoU pencocokan
confusion matrix dibedakan dari IoU konfigurasi validasi. Metrik yang belum diukur tetap null.""",
    13: """## 13 — Trade-off Kompresi dan Pemulihan Akurasi
Grafik memakai hasil aktual dengan identitas model, rasio, policy, dan tahap sebelum/sesudah
recovery. Metrik yang tidak dijalankan tetap NOT_RUN. Fine-tuning dapat memperbaiki akurasi,
tetapi pemulihan penuh tidak dijanjikan, terutama pada rasio agresif.
Seed tunggal tidak membuktikan generalisasi statistik; perubahan channel/akurasi tidak
otomatis membuktikan layer tertentu sebagai penyebab. Hasil notebook tanpa recovery
yang lama tetap bukti historis dan tidak dicampur sebagai konfigurasi suite ini.""",
    14: """## 14 — Artifact, Status, dan Reproducibility
Project W&B tetap `yolom26-kd-indotraffic`; setiap model/policy/rasio menggunakan UUID/run
baru dengan label fine-tuning recovery dan `resume="never"`. Baseline, pruned, serta recovered
memiliki metrik terpisah. Tracking/upload yang gagal tidak menghapus hasil lokal dan tidak
dianggap berhasil. Kredensial tidak disimpan dalam config, artifact, atau laporan.

Output suite baru berada di `pruning/lamp_v2_finetune/{suite_id}/`, berisi direktori `yolo26n/`,
`yolo26s/`, dan `comparison/`. Simpan checkpoint pruned, `recovery/recovered_fp32.pt`,
best/last training, audit channel/skor, EDA, config, provenance, environment, log epoch,
manifest recovery, reload/prediction report, serta metrik ketiga tahap.
Notebook lama dan laporan historis tidak ditulis ulang. Structural validity, training
completion, evaluation validity, tracking status, dan scientific trade-off tetap terpisah.""",
}


def _sections(path):
    groups = {}
    key = None
    for line in path.read_text(encoding="utf-8").splitlines(keepends=True):
        if line.startswith("# %% "):
            key = line.strip().removeprefix("# %% ")
            groups[key] = ""
        elif key is not None:
            groups[key] += line
    return {key: value.strip() for key, value in groups.items()}


def _cell(text, kind="code", tag="helpers"):
    text = text.strip()
    cell = {"cell_type": kind, "metadata": {}, "source": text.splitlines(keepends=True)}
    if kind == "code":
        ast.parse(text)
        cell.update({"execution_count": None, "outputs": []})
        cell["metadata"]["tags"] = [tag]
    return cell


def build_notebook():
    source = runpy.run_path(str(ROOT / "scripts/generate_lamp_v2_notebook.py"))["build_notebook"]()
    runtime = _sections(RUNTIME)
    training = TRAINING.read_text(encoding="utf-8")
    cells = []
    config_insertions = recovery_insertions = 0
    for original in source["cells"]:
        cell = deepcopy(original)
        text = "".join(cell["source"])
        if cell["cell_type"] == "markdown":
            match = re.match(r"^#{1,2} (\d{2}) —", text)
            if match and int(match[1]) in MARKDOWN:
                text = MARKDOWN[int(match[1])]
        elif "config" in cell.get("metadata", {}).get("tags", []):
            text = text.replace("pruning/lamp_v2", "pruning/lamp_v2_finetune")
            text = text.replace("RECOVERY_ENABLED = False", "RECOVERY_ENABLED = True")
            text = text.replace("FINETUNING_ENABLED = False", "FINETUNING_ENABLED = True")
            text += '''

RECOVERY_EPOCHS = 5
RECOVERY_LR0 = 0.0001
RECOVERY_OPTIMIZER = "AdamW"
RECOVERY_PATIENCE = 0
RECOVERY_CLOSE_MOSAIC = 1
RECOVERY_WARMUP_EPOCHS = 3
RECOVERY_WEIGHT_DECAY = 0.0005
RECOVERY_BATCH = 16
RECOVERY_NBS = 64
RECOVERY_WORKERS = 2
RECOVERY_AMP = False'''
        elif text.startswith("BUNDLE = execute_models("):
            cells.append(_cell(training))
            cells.append(_cell(runtime["recovery_helpers"]))
            recovery_insertions += 1
        elif "SUMMARY = pd.DataFrame(" in text:
            text = text.replace('["model", "policy", "ratio",', '["model", "policy", "stage", "ratio",')
        elif "PER_CLASS = []" in text:
            text = '''display(SUMMARY[["model", "policy", "stage", "ratio", "parameters", "parameter_reduction_pct", "gflops", "gflops_reduction_pct", "checkpoint_bytes", "latency_median_ms"]])
display(SUMMARY[["model", "policy", "stage", "ratio", "map50_95", "map50", "precision", "recall", "delta_map50_95_pp", "relative_map50_95_pct", "validation_status"]])
PER_CLASS = []
for suite in SUITES:
    PER_CLASS.extend({"model": suite["model_name"], "policy": "baseline", "stage": "baseline", "ratio": 0, **row}
                     for row in (suite["baseline"].get("validation") or {}).get("per_class", []))
    for result in suite["results"]:
        for stage in ("pruned", "recovered"):
            PER_CLASS.extend({"model": suite["model_name"], "policy": result["policy"], "stage": stage,
                              "ratio": result["ratio"], **row}
                             for row in (result.get(stage, {}).get("validation") or {}).get("per_class", []))
write_json(SUITE_DIR / "comparison" / "per_class_metrics.json", PER_CLASS)
write_csv(SUITE_DIR / "comparison" / "per_class_metrics.csv", PER_CLASS)'''
        elif '"reproducibility.json"' in text:
            text = text.replace('"No recovery", ', '')
            text = text.replace('    "tracking_statuses":', '''    "recovery_statuses": [{"model": s["model_name"], "policy": r["policy"], "ratio": r["ratio"], "status": r.get("recovery_status", "NOT_RUN")} for s in SUITES for r in s["results"]],
    "recovery_validation_statuses": [{"model": s["model_name"], "policy": r["policy"], "ratio": r["ratio"], "status": r.get("recovery_validation_status", "NOT_RUN")} for s in SUITES for r in s["results"]],
    "tracking_statuses":''')
        cell["source"] = text.strip().splitlines(keepends=True)
        cells.append(cell)
        if cell["cell_type"] == "code" and "def validate_configuration(" in text:
            cells.append(_cell(runtime["configuration_helpers"]))
            config_insertions += 1
    assert config_insertions == recovery_insertions == 1
    for index, cell in enumerate(cells):
        text = "".join(cell["source"])
        if cell["cell_type"] == "code":
            ast.parse(text)
        cell["id"] = hashlib.sha256(f"{index}:{text}".encode()).hexdigest()[:8]
    return {**source, "cells": cells}


if __name__ == "__main__":
    DESTINATION.write_text(json.dumps(build_notebook(), ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    print(DESTINATION)
