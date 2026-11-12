"""Build one standalone MGD notebook; never train or rewrite previous notebooks."""
from __future__ import annotations

import ast
import hashlib
import json
from pathlib import Path

try:
    from scripts.mgd_method_source import MGD_FEATURES_SOURCE, MGD_LOSS_SOURCE
    from scripts.mgd_runtime_source import MGD_CALLBACKS_SOURCE, MGD_RUNTIME_SOURCE
    from scripts.notebook_paths import notebook_path
except ModuleNotFoundError:
    from mgd_method_source import MGD_FEATURES_SOURCE, MGD_LOSS_SOURCE
    from mgd_runtime_source import MGD_CALLBACKS_SOURCE, MGD_RUNTIME_SOURCE
    from notebook_paths import notebook_path

ROOT = Path(__file__).resolve().parents[1]
TARGET = notebook_path("kaggle", "15_mgd_yolo26n_teacher_yolo26s.ipynb")
CONFIG = '''EXPERIMENT_ID = "15_mgd_yolo26n_teacher_yolo26s"
METHOD = "mgd"
DATA = "/kaggle/input/datasets/ilhamrafiqin/poltrada-v3-yolo26-clean/data.yaml"
MODEL_WEIGHTS = "yolo26n.pt"
EPOCHS = 100
IMGSZ = 640
BATCH = 16
SEEDS_TO_RUN = [
    42,
    # 123,
    # 2024,
    # 7,
    # 3407,
]
SEED = SEEDS_TO_RUN[0] if SEEDS_TO_RUN else None
DEVICE = 0
WORKERS = 2
OPTIMIZER = "AdamW"
LR = 0.001
PATIENCE = 100
AMP = True
WEIGHT_DECAY = 0.0005
WARMUP_EPOCHS = 3.0
CLOSE_MOSAIC = 10
EVAL_SPLIT = "val"
PROJECT = "/kaggle/working/results"
RUN_FULL_TRAINING = True
RUN_SMOKE_TEST = False  # True hanya untuk smoke 1 epoch tambahan, bukan hasil penelitian.
KEEP_LAST_LOCAL_EPOCHS = 1  # Versi setiap epoch tetap di W&B; snapshot gagal upload tidak dihapus.
MGD_ALPHA = 2e-5  # Default deteksi one-stage resmi; belum dituning untuk YOLO26.
MGD_MASK_RATIO = 0.65  # Proporsi posisi spasial yang dimatikan, bukan keep ratio.
EXPECTED_NC = 6
WANDB_ENABLED = True
WANDB_ENTITY = "ilhamrafiqin-brawijaya-university"
WANDB_PROJECT = "yolom26-kd-indotraffic"
WANDB_GROUP = "T26s_S26n"
TEACHER_RUN_ID = "q5kp9k9d"
TEACHER_ARTIFACT = "ilhamrafiqin-brawijaya-university/yolom26-kd-indotraffic/02_baseline_yolo26s-q5kp9k9d-checkpoints:v62"
TEACHER_SHA256 = "5bc3ae61f8da1c9f03916922de10146805c27c64a11e46edd766a00d0f6f18fd"
MGD_REFERENCE_COMMIT = "2c9da0b28625eb948db57afc02c824452c3910fe"
MGD_REFERENCE_SHA256 = "a91595b98b906363abff7374662638ccaa7cc1557b44ab61640e5d610b9649de"
RESUME_STATUS = "RESUME_NOT_VERIFIED"
'''

NUMERICAL = '''def numerical_verification():
    """Small raw-SSE/batch oracle; official-source loss/gradient parity is tested offline."""
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(0)
        block = MGDFeatureLoss(2, 2, "oracle", alpha_mgd=0.1, lambda_mgd=0.0)
        with torch.no_grad():
            for layer in block.generation:
                if isinstance(layer, nn.Conv2d):
                    layer.weight.zero_()
                    layer.bias.zero_()
                    for channel in range(2):
                        layer.weight[channel, channel, 1, 1] = 1
        student = torch.ones(2, 2, 3, 3, requires_grad=True)
        teacher = torch.zeros_like(student, requires_grad=True)
        loss = block(student, teacher)
        torch.testing.assert_close(loss, torch.tensor(1.8))  # 0.1 * (2*2*3*3) / 2.
        loss.backward()
        torch.testing.assert_close(student.grad, torch.full_like(student, 0.1))
        if teacher.grad is not None:
            raise RuntimeError("Teacher target tidak boleh menerima gradien.")
        block.lambda_mgd = 1.0
        torch.testing.assert_close(block(student.detach(), teacher.detach()), torch.tensor(0.0))
    return {"status": "PASS CPU ORACLE", "sse_divided_by_batch": True,
            "reference_sha256": MGD_REFERENCE_SHA256, "teacher_gradient_isolated": True}

print(numerical_verification())
'''


def node_source(text, name):
    node = next(node for node in ast.parse(text).body if isinstance(node, (ast.FunctionDef, ast.ClassDef)) and node.name == name)
    return ast.get_source_segment(text, node) + "\n"


def make_cell(kind, name, text):
    result = {"cell_type": kind, "id": "mgd-" + name, "metadata": {}, "source": text.strip().splitlines(keepends=True)}
    if kind == "code":
        compile(text, "mgd-" + name, "exec")
        result.update(execution_count=None, outputs=[])
    return result


def build():
    previous = {p: hashlib.sha256(p.read_bytes()).hexdigest() for p in (ROOT / "notebooks").rglob("*.ipynb") if p != TARGET}
    base = json.loads(notebook_path("kaggle", "12_pkd_yolo26n.ipynb").read_bytes())
    cells = {c["id"]: "".join(c["source"]) for c in base["cells"]}
    environment = cells["pkd-environment"].replace("GPU diagnostic tidak boleh diganti CPU", "Training penelitian memerlukan GPU")
    teacher = cells["pkd-teacher"].replace('    del payload', '    if digest != TEACHER_SHA256:\n        raise ValueError("SHA-256 teacher berbeda dari Native/CrossKD.")\n    del payload')
    protocol = cells["pkd-protocol"].replace("PKD", "MGD").replace("pkd", "mgd")
    protocol = protocol.replace("epochs=EPOCHS", "epochs=1 if diagnostic else EPOCHS")
    protocol = protocol.replace("batch=DIAGNOSTIC_BATCH if diagnostic else BATCH", "batch=BATCH")
    checkpoints = cells["pkd-checkpoints"].replace("PKD", "MGD").replace("pkd", "mgd")
    checkpoints = checkpoints.replace('"mgd_weight": MGD_WEIGHT', '"mgd_alpha": MGD_ALPHA, "mgd_mask_ratio": MGD_MASK_RATIO')
    evaluation = node_source(cells["pkd-diagnostic"], "evaluate_student")
    parts = [
        ("markdown", "intro", """# MGD YOLO26s → YOLO26n — Kaggle

Satu notebook standalone. Tambahkan Poltrada V3, aktifkan GPU/Internet, dan Secret `WANDB_API_KEY`. Run All menjalankan audit/numerical verification lalu training 100 epoch untuk setiap seed aktif. Seed 42 aktif; empat seed lainnya tersedia sebagai komentar. Tidak membutuhkan checkout repository atau MMDetection.

Metode mengikuti **Masked Generative Distillation**, Yang dkk., ECCV 2022: [paper](https://arxiv.org/abs/2205.01529), [kode deteksi penulis](https://github.com/yzd-v/MGD/blob/2c9da0b28625eb948db57afc02c824452c3910fe/det/mmdet/distillation/losses/mgd.py). Referensi lokal: `references/mgd`, commit dan SHA dikunci pada konfigurasi. Kode referensi berlisensi Apache-2.0; adaptasi PyTorch/YOLO26 ini mempertahankan atribusi penulis.
"""),
        ("markdown", "config-md", """## 1. Konfigurasi penelitian

Mengikuti Native KD 10/CrossKD 11: teacher s v62 yang sama, student pretrained nano, 100 epoch, 640/batch16, AdamW lr0.001, AMP, seed yang dipasangkan. Default engine termasuk nbs64, scheduler dan augmentasi tetap berasal dari Ultralytics 8.4.155. Setiap seed dimulai dari student dan MGD baru. Bobot alpha dan mask ratio adalah default resmi detektor one-stage, bukan hasil tuning YOLO.
"""),
        ("code", "config", CONFIG),
        ("markdown", "environment-md", "## 2. Lingkungan dan autentikasi\nPyTorch/CUDA bawaan Kaggle dipertahankan; tidak ada secret di kode notebook."),
        ("code", "environment", environment),
        ("code", "auth", cells["pkd-auth"]),
        ("markdown", "dataset-md", "## 3. Dataset dan teacher\nSplit, label, urutan kelas dan checkpoint teacher tidak diganti. Audit versi, producer, manifest, SHA-256, schema dan label dilakukan untuk setiap seed."),
        ("code", "dataset", cells["pkd-dataset"]),
        ("code", "teacher", teacher),
        ("markdown", "method-md", """## 4. Masked Generative Distillation

Paper hal.5–6 Eq.2–6: alignment 1×1 → spatial mask → Conv3×3/ReLU/Conv3×3 → rekontruksi seluruh fitur teacher. Hal.8 §4.2: semua feature map neck, alpha2e-5/lambda0.65 untuk one-stage. Hal.13 mendukung dua conv penuh; generator tidak diperkecil.

Kode deteksi membentuk mask B×1×H×W dengan `U > 1-lambda` sebagai posisi nol, tanpa rescaling. Paper memakai `R < lambda`: distribusinya setara, bukan mask identik untuk angka acak yang sama. Implementasi mengikuti threshold kode. Loss per level = alpha × SSE / actual_batch; ketiga level **dijumlahkan**, tidak dirata-rata atau dinormalisasi terhadap C/H/W. Pembagian batch mengikuti kode resmi, tidak dituliskan eksplisit pada Eq.5.

Adaptasi YOLO memakai P3/P4/P5 (stride8/16/32), alignment bias dan generator penuh setiap level. Generator/alignment/reduksi berjalan FP32; AMP detector tetap aktif. Tambahan sekitar 6.37M parameter hanya untuk training. Paper menggunakan SGD/24 epoch dan pewarisan bobot teacher pada struktur yang cocok; eksperimen ini mempertahankan AdamW/100 epoch dan `yolo26n.pt` agar sebanding dengan Native/CrossKD. Tidak ada pewarisan neck/head teacher ke student atau KD metode lain.
"""),
        ("code", "features", MGD_FEATURES_SOURCE),
        ("code", "loss", MGD_LOSS_SOURCE),
        ("code", "numerical", NUMERICAL),
        ("markdown", "protocol-md", """## 5. Trainer dan checkpoint

Detection criterion O2M/O2O, optimizer, accumulation, clipping, EMA dan GradScaler native tetap. Teacher frozen/eval; teacher Detect tidak dijalankan. MGD ditambahkan dengan faktor actual batch tepat sekali. Tidak ada resize, batch-halving, AMP fallback atau recovery NaN otomatis. Native KD tidak dipanggil.

Validasi/inferensi hanya student; tidak mengonsumsi RNG mask. Snapshot `best.pt`/`last.pt` berupa student-only; state archive mencatat student/MGD/EMA/optimizer/scheduler/scaler/RNG. **RESUME_NOT_VERIFIED**: archive tidak menjamin pemulihan persis engine/prefetch; resume=True ditolak.
"""),
        ("code", "protocol", protocol),
        ("code", "checkpoints", checkpoints),
        ("markdown", "tracking-md", "## 6. W&B dan antrean seed\nGroup T26s_S26n; run ID baru, resume=never. Upload checkpoint setiap epoch ditunggu. Default hanya snapshot epoch terbaru yang sukses diunggah dipertahankan lokal (termasuk best.pt dan state best). Versi setiap epoch tetap di W&B. Snapshot gagal/belum terkonfirmasi upload tidak dihapus; kegagalan menghentikan antrean. Hasil seed lama tidak diimpor otomatis."),
        ("code", "callbacks", MGD_CALLBACKS_SOURCE),
        ("code", "runtime", evaluation + "\n\n" + MGD_RUNTIME_SOURCE),
        ("markdown", "execute-md", "## 7. Jalankan training\nDefault hanya seed42, 100 epoch. Membuka komentar seed berikutnya menambahkan run serial yang independen, bukan melanjutkan model seed sebelumnya. Smoke opsional memakai satu epoch penuh dan run terpisah."),
        ("code", "execute", 'MULTISEED_RESULTS = run_seed_queue()\nprint(json.dumps(MULTISEED_RESULTS, indent=2))'),
        ("markdown", "limitations", """## 8. Batas pengujian

Default MGD belum dianggap optimal untuk YOLO26. Laporan numerik/CPU sintetis menguji software, bukan mAP Poltrada, kapasitas GPU atau stabilitas 100 epoch. **GPU Kaggle: NOT TESTED** sampai ada run aktual. Checkpoint dan log setiap epoch memberi bukti progres, bukan jaminan resume sesi.
"""),
    ]
    notebook = {"nbformat": 4, "nbformat_minor": 5,
                "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                             "language_info": {"name": "python"}, "accelerator": "GPU"},
                "cells": [make_cell(*part) for part in parts]}
    TARGET.parent.mkdir(parents=True, exist_ok=True)
    TARGET.write_text(json.dumps(notebook, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    assert previous == {p: hashlib.sha256(p.read_bytes()).hexdigest() for p in previous}, "Previous notebook changed"
    print(f"Generated {TARGET.name}: {len(notebook['cells'])} cells; previous notebooks unchanged.")


if __name__ == "__main__":
    build()
