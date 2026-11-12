"""Build notebook03; every required runtime definition is embedded, never imported.

The established V1 serialization, evaluation and dependency bootstrap are copied as
source. Existing notebooks are read-only inputs and are never regenerated here.
"""
from __future__ import annotations

import ast
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "notebooks/kaggle/pruning/01_lamp_yolo26n_no_recovery.ipynb"
DESTINATION = ROOT / "notebooks/kaggle/pruning/03_lamp_yolo26n_extended_no_recovery.ipynb"


def sections(path):
    result = {}
    key = None
    for line in path.read_text(encoding="utf-8").splitlines(keepends=True):
        if line.startswith("# %% "):
            key = line.strip().removeprefix("# %% ")
            result[key] = ""
        elif key is not None:
            result[key] += line
    return {key: value.strip() for key, value in result.items()}


def pick_definitions(book, cell, names):
    text = "".join(book["cells"][cell]["source"])
    nodes = [node for node in ast.parse(text).body if isinstance(node, ast.FunctionDef) and node.name in names]
    assert {node.name for node in nodes} == set(names)
    return "\n\n\n".join(ast.get_source_segment(text, node) for node in nodes)


def build_notebook():
    book = json.loads(SOURCE.read_text(encoding="utf-8"))
    runtime = sections(ROOT / "scripts/lamp_v2_runtime_source.py")
    policy = (ROOT / "scripts/lamp_v2_policy_source.py").read_text(encoding="utf-8")
    eda = (ROOT / "scripts/lamp_v2_eda_source.py").read_text(encoding="utf-8")
    validation = (ROOT / "scripts/lamp_v2_validation_source.py").read_text(encoding="utf-8")
    cells = []

    def add(kind, text, tag=None):
        text = text.strip()
        identity = hashlib.sha256(f"{len(cells)}:{text}".encode()).hexdigest()[:8]
        cell = {"cell_type": kind, "id": identity, "metadata": {}, "source": text.splitlines(keepends=True)}
        if kind == "code":
            ast.parse(text)
            cell.update({"execution_count": None, "outputs": []})
            cell["metadata"]["tags"] = [tag or "helpers"]
        cells.append(cell)

    def md(text):
        add("markdown", text)

    def code(text, tag="helpers"):
        add("code", text, tag)

    md("""# 01 — Tujuan dan Hipotesis Eksperimen
LAMP structured pruning V2 **YOLO26n dan YOLO26s**, tanpa recovery: bandingkan conservative dan extended
pada 10%, 20%, 30%, 40%, dan 50% channel eligible. Himpunan eligible berbeda sehingga rasio target yang
sama tidak menyatakan kompresi parameter/GFLOPs yang sama. Perluasan cakupan adalah hipotesis,
bukan jaminan perbaikan akurasi. Tidak ada training, sparse training, backward, optimizer,
fine-tuning, teacher, atau KD. Notebook ini standalone dan tidak memerlukan clone repository.

Run All Kaggle: aktifkan GPU + Internet, tambahkan **Poltrada V3 YOLO26 Clean** baseline,
aktifkan secret `WANDB_API_KEY`, lalu jalankan kernel baru. Tidak memakai dataset pengganti.

Dasar: [Lee et al., ICLR 2021](https://arxiv.org/abs/2010.07611),
[Torch-Pruning 1.6.1](https://github.com/VainF/Torch-Pruning/tree/v1.6.1),
dan source Ultralytics 8.4.155. LAMP asli berbasis magnitude bobot; `LAMPImportance` pada
dependency group adalah adaptasi structured channel pruning.
Studi Gabriel Winasta Sinisuka, Rizal Setya Perdana, dan Irawati Nurmala Sari,
*Kompresi Model YOLOv8 melalui Pruning, Sparsity, dan Knowledge Distillation ...*,
PDF lokal `references/kompresi-model/paper/Kompresi Model YOLOv8 melalui Pruning, Sparsity, dan Knowledge Distillation untuk Deteksi Kendaraan dan Plat Nomor secara Real-Time pada Perangkat Edge.pdf`, hlm. 3–5, memakai BNScale, sparse training,
dan KD recovery. Hasil akhir studi tersebut bukan hasil LAMP tanpa recovery; perubahan
C2f_v2 YOLOv8 dan pruning hidden head tidak diterapkan pada YOLO26. Tidak menyimpulkan
akurasi akan membaik dari hasil kompresi struktural saja.""")
    md("""## 02 — Konfigurasi
Semua parameter pengguna berada pada cell ini. Default memproses N dan S, masing-masing
dengan `extended` pada 10%, 20%, 30%, 40%, dan 50%: sepuluh eksperimen independen, evaluasi penuh,
dan upload W&B aktif. Pilih `MODEL_VARIANTS=["n"]` atau `["s"]` untuk satu model.
Pilih policy `both` untuk menambahkan conservative sebagai pembanding pada setiap model.
Untuk uji cepat pilih satu policy dan `PRUNING_RATIOS=[0.10]`. `v84` adalah versi artifact,
bukan epoch. `best.pt` harus unik; gunakan selector relatif lengkap bila ambigu.
Rasio dihitung dari channel eligible, bukan seluruh parameter. Cap per group tetap 50%.
Rasio 40%/50% menguji kompresi tanpa recovery; structural PASS tidak berarti akurasi terjaga.""")
    code('''from pathlib import Path

MODEL_VARIANTS = ["n", "s"]
WANDB_ENTITY = "ilhamrafiqin-brawijaya-university"
WANDB_PROJECT = "yolom26-kd-indotraffic"
MODEL_ARTIFACTS = {
    "n": "ilhamrafiqin-brawijaya-university/yolom26-kd-indotraffic/03_baseline_yolo26n-9uu3meve-checkpoints:v84",
    "s": "ilhamrafiqin-brawijaya-university/yolom26-kd-indotraffic/02_baseline_yolo26s-q5kp9k9d-checkpoints:v62",
}
LOCAL_CHECKPOINT_PATHS = {"n": None, "s": None}
LOCAL_EXPECTED_SHA256S = {"n": None, "s": None}
CHECKPOINT_SELECTOR = "best.pt"
DATASET_ROOT = Path("/kaggle/input/datasets/ilhamrafiqin/poltrada-v3-yolo26-clean")
DATA_YAML = DATASET_ROOT / "data.yaml"
OUTPUT_DIR = Path("/kaggle/working/pruning/lamp_v2") if Path("/kaggle").exists() else Path("outputs/pruning/lamp_v2")
PRUNING_POLICY = "extended"
PRUNING_RATIOS = [0.10, 0.20, 0.30, 0.40, 0.50]
GLOBAL_PRUNING = True
ITERATIVE_STEPS = 1
MAX_PRUNING_RATIO = 0.50
ROUND_TO = None
IMPORTANCE_P = 2
SEED = 42
IMGSZ = 640
DEVICE = "auto"
RUN_FULL_VALIDATION = True
RUN_LATENCY_BENCHMARK = True
BATCH = 16
WORKERS = 2
LATENCY_WARMUP = 50
LATENCY_REPEATS = 200
RELOAD_TIMEOUT_SECONDS = 300
INSTALL_DEPENDENCIES = True
WANDB_TRACKING = True
UPLOAD_WANDB_ARTIFACTS = True
EDA_DPI = 300
EDA_MAX_LAYERS_PER_PAGE = 32
EDA_DISPLAY_RATIO = 0.50
RECOVERY_ENABLED = False
FINETUNING_ENABLED = False
KD_ENABLED = False''', "config")
    md("""## 03 — Environment dan Dependencies
Pertahankan Torch/torchvision bawaan Kaggle; konflik dependency menghentikan instalasi.
Model pengukuran tetap unfused FP32. Prediction/validation memakai salinan terpisah karena
backend dapat melakukan fusion. Pin dan hash source importance dicatat tanpa kredensial.""")
    common = "".join(book["cells"][3]["source"]).split("\ndef new_state():")[0]
    code(common)
    code(runtime["configuration_helpers"])
    code("".join(book["cells"][6]["source"]))
    code('''validate_configuration()
SUITE_DIR = Path(OUTPUT_DIR).expanduser().resolve() / (
    datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "_" + uuid.uuid4().hex[:8]
)
SUITE_DIR.mkdir(parents=True, exist_ok=False)
(SUITE_DIR / "baseline").mkdir()
write_json(SUITE_DIR / "config.json", configuration())
try:
    ensure_dependencies(SUITE_DIR, install=INSTALL_DEPENDENCIES)
except Exception as error:
    write_json(SUITE_DIR / "environment_failure.json", {"status": "FAIL", "error": safe_error(error)})
    raise''', "run")
    # Import block has no side-effectful stage wrapper; errors remain visible.
    imports = "".join(book["cells"][8]["source"]).splitlines()[1:]
    code("\n".join(line.removeprefix("    ") for line in imports), "imports")
    code("".join(book["cells"][9]["source"]))
    code('''TARGET_DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu") if DEVICE == "auto" else torch.device(DEVICE)
seed_environment(SEED)
try:
    ENVIRONMENT = environment_report()
    write_json(SUITE_DIR / "environment.json", ENVIRONMENT)
except Exception as error:
    write_json(SUITE_DIR / "environment_failure.json", {"status": "FAIL", "error": safe_error(error)})
    raise
print("Runtime:", TARGET_DEVICE, "FP32", ENVIRONMENT["packages"])''', "run")
    md("""## 04 — Pengambilan Baseline dari W&B
Unduh artifact satu kali per model. UUID/nama run eksperimen berbeda dari baseline/KD; project tetap sama.
Salinan lokal memerlukan pilihan eksplisit dan tingkat verifikasi provenance yang jujur.
SHA-256 dan identitas artifact disimpan; jumlah kelas tidak diasumsikan dari nama project.""")
    code(pick_definitions(book, 12, ["select_checkpoint", "checkpoint_provenance"]))
    code(runtime["acquisition_helpers"])
    code('''display(pd.DataFrame([model_spec(variant) for variant in MODEL_VARIANTS]))
print("Pengambilan sumber dan verifikasi SHA-256 dilakukan terpisah pada setiap model saat eksekusi suite.")''', "run")
    md("""## 05 — Audit Arsitektur YOLO26n/S
Validasi actual checkpoint: skala nano/stem 16 atau small/stem 32, class names/order, head one-to-many dan one-to-one,
residual, concat, CSP, attention, dan sumber fitur. Pengukuran GFLOPs memakai THOP Conv/BN
pada resolusi penuh ditambah counter matmul attention, dengan metode identik sebelum/sesudah.
Tabel layer memakai hitungan operasi forward Conv/BN; operasi attention terpisah pada total.""")
    architecture = "".join(book["cells"][15]["source"])
    architecture = architecture.replace('def architecture_audit(core):', 'def architecture_audit(core, model_variant="n"):')
    architecture = architecture.replace('first_conv.out_channels != 16', 'first_conv.out_channels != {"n": 16, "s": 32}[model_variant]')
    architecture = architecture.replace('cfg.get("scale") not in (None, "", "n")', 'cfg.get("scale") not in (None, "", model_variant)')
    architecture = architecture.replace('YOLO26n standar', 'YOLO26 N/S standar').replace('Checkpoint bukan skala nano.', 'Checkpoint bukan skala model yang dipilih.')
    code(architecture)
    code("".join(book["cells"][24]["source"]))
    md("""## 06 — Audit Dependency Graph dan Eligible Layer
Conservative mempertahankan hidden Bottleneck V1. Extended memeriksa seluruh group graph
dan hanya menerima operasi yang didukung, tanpa indeks layer hardcoded. Head dan envelope
attention utuh; split/reshape dan mapping tidak valid ditolak. Group residual yang mengikat
beberapa output dihitung sekali pada denominator channel independen, bukan setiap Conv.
Daftar lengkap accepted/rejected groups dan alasan ditulis sebelum channel dihapus.""")
    code(policy)
    code(eda)
    md("""## 07 — Implementasi LAMP Structured Pruning
`LAMPImportance(p=2, group_reduction="mean", bias=False)` dipakai oleh BasePruner.
Graph dibangun pada eval dengan autograd aktif, memakai raw outputs kedua cabang; tanpa backward.
Cap 50% dibatasi sebelum ranking menggunakan sentinel +inf bagi kandidat di luar kapasitas;
skor LAMP kandidat tetap asli. Setiap proposal diaudit ulang sebelum penghapusan fisik.
Pruning bukan masking. Operasi tidak dikenal menghentikan proposal terpilih.""")
    md("""## 08 — Eksperimen Rasio Pruning
Baseline asli dimuat ulang dan state hash diverifikasi pada setiap policy/rasio.
Tidak ada cumulative pruning. Baseline disimpan, diukur dan divalidasi sekali per model.
Jika konfigurasi gagal, laporan parsial disimpan dan exception diteruskan; job berikutnya
tetap NOT_RUN. Log error tidak diubah menjadi PASS.""")
    md("""## 09 — Save dan Fresh Reload
Simpan objek model struktural FP32 dengan `ema=None`, `optimizer=None`; tidak memakai YOLO.save().
Proses Python baru tanpa PYTHONPATH repository wajib memuat YOLO(checkpoint), menjalankan
kedua cabang dan prediction, lalu membandingkan FP32 raw/decoded outputs pada rtol=1e-4,
atol=1e-5. Prediksi kosong pada input sintetis tidak dianggap bukti akurasi deteksi nyata.""")
    code("".join(book["cells"][27]["source"]))
    md("""## 10 — Validasi Baseline vs Pruned
Protokol identik: val, 640, confidence 0.001, IoU 0.7, max_det 300, nms=None, FP32,
tanpa TTA. Class order wajib cocok. Dataset tidak tersedia: NOT_RUN; dataset ada tetapi
invalid: FAIL. Tidak ada fallback Poltrada versi lain. Latency CUDA memakai batch satu,
50 warm-up, 200 pengukuran dan sinkronisasi; tidak termasuk preprocessing/NMS/disk IO.""")
    code("".join(book["cells"][30]["source"]))
    code(validation)
    code(runtime["tracking_helpers"])
    code(runtime["experiment_helpers"])
    code(runtime["plot_helpers"])
    code('''BUNDLE = execute_models(SUITE_DIR)
SUITES = BUNDLE["models"]
print("Suite:", BUNDLE["status"], "| model:", len(SUITES), "| eksperimen:", sum(len(s["results"]) for s in SUITES))''', "run")
    md("""## 11 — EDA Identitas, Cakupan, dan Channel
Inventory Conv/BN, eligible/protected groups, indeks asli channel, skor LAMP finite,
dan perubahan per stage/region disimpan untuk setiap model/rasio. Diagram tiga panel:
**Before Pruning** biru, **After Pruning** merah, **Channel Comparison** overlay;
seluruh Conv-output memiliki urutan dan skala sumbu yang sama pada ketiga panel.
Overview serta detail maksimum 32 layer disimpan PNG 300 DPI, SVG, dan CSV sumber.
Heatmap menunjukkan channel bobot bertahan/terhapus, bukan aktivasi feature map.
Contoh detail default rasio 0.50; seluruh rasio tetap dianalisis dan disimpan.""")
    code('''SUMMARY = pd.DataFrame([row for suite in SUITES for row in summary_rows(suite)])
display(SUMMARY[["model", "policy", "ratio", "actual_eligible_ratio", "pruned_group_count", "structural_status", "validation_status"]])
from IPython.display import Image as NotebookImage
for suite in SUITES:
    for result in suite["results"]:
        if abs(result["ratio"] - EDA_DISPLAY_RATIO) < 1e-12:
            print(result["model_name"], result["policy"], f"pr{result['ratio']:g}")
            overview = SUITE_DIR / result["model_name"].lower() / result["policy"] / f"pr{result['ratio']:g}" / "eda" / "channels_overview.png"
            if overview.is_file():
                display(NotebookImage(filename=str(overview)))
            display(pd.DataFrame(result["audit"]["eligible_groups"])[["name", "initial_channels", "group_id"]])''', "run")
    md("""## 12 — EDA Efisiensi dan Metrik Deteksi
Rasio channel independen dan penghapusan fisik Conv-output berbeda pada residual yang terikat.
Bandingkan parameter/GFLOPs per layer/stage, ukuran checkpoint FP32, serta latency raw forward
dengan p05–p95. Semua perbandingan memakai baseline model masing-masing, bukan N sebagai baseline S.
`delta_map50_95_pp = 100 × (mAP_pruned − mAP_baseline)` menyatakan poin persentase.
Perubahan relatif memakai baseline sebagai denominator dan kosong bila baseline nol.
AP/P/R/F1 dan support per kelas, kurva PR/F1/confidence, confusion matrix numerik dan threshold
aktual disimpan dari validator. IoU pencocokan confusion matrix dibedakan dari IoU konfigurasi
validasi. AP kelas yang tidak tersedia tetap null; tanpa validasi tidak ada mAP buatan.""")
    code('''display(SUMMARY[["model", "policy", "ratio", "parameters", "parameter_reduction_pct", "gflops", "gflops_reduction_pct", "checkpoint_bytes", "latency_median_ms"]])
display(SUMMARY[["model", "policy", "ratio", "map50_95", "map50", "precision", "recall", "delta_map50_95_pp", "relative_map50_95_pct", "validation_status"]])
PER_CLASS = []
for suite in SUITES:
    PER_CLASS.extend({"model": suite["model_name"], "policy": "baseline", "ratio": 0, **row}
                     for row in (suite["baseline"].get("validation") or {}).get("per_class", []))
    for result in suite["results"]:
        PER_CLASS.extend({"model": suite["model_name"], "policy": result["policy"], "ratio": result["ratio"], **row}
                         for row in (result["pruned"].get("validation") or {}).get("per_class", []))
write_json(SUITE_DIR / "comparison" / "per_class_metrics.json", PER_CLASS)
write_csv(SUITE_DIR / "comparison" / "per_class_metrics.csv", PER_CLASS)''', "run")
    md("""## 13 — Analisis Trade-off Akurasi–Efisiensi
Grafik menggunakan hasil suite aktual, terpisah per model dan policy. Perbandingan N/S
memuat identitas model/rasio pada setiap titik. Panel akurasi tanpa data
ditandai NOT_RUN. Kompresi lebih besar dengan akurasi buruk adalah temuan negatif yang valid;
tidak ada PASS otomatis untuk trade-off atau bukti kausal layer tertentu menyebabkan penurunan mAP.
Seed tunggal tidak membuktikan generalisasi statistik.
Angka Kaggle V1 pada brief memiliki ratio UNKNOWN sampai manifest asli tersedia, sehingga
tidak dimasukkan sebagai titik eksperimen terverifikasi pada grafik perbandingan utama.""")
    code('''print("Perbandingan model:")
display(BUNDLE["comparison_plots"])
for suite in SUITES:
    print(suite["model_name"], "| validasi:", suite["baseline_validation_status"])
    display(pd.DataFrame([{"ratio": r["ratio"], "policy": r["policy"], "eda_status": r["eda_status"]} for r in suite["results"]]))''', "run")
    md("""## 14 — Ringkasan, Artifact, dan Reproducibility
Project W&B tetap `yolom26-kd-indotraffic`, satu UUID/run baru tiap model/policy/rasio,
nama `prun_lamp2_<model>[_c]_pruned-pr<rasio>` dan resume=never.
Source/hash, dataset YAML hash, versi, seed dan protokol dicatat. Model/report artifact
dikonfirmasi lewat wait sebelum diklaim uploaded; kegagalan upload tercatat tanpa menghapus
hasil lokal. Checkpoint siap dipakai pada studi recovery berikutnya, tetapi recovery belum dijalankan.

Output: `yolo26n/`, `yolo26s/` berisi baseline, policy/ratio, EDA, dan summary; `comparison/` lintas model;
config/provenance/environment dan
`suite_manifest.json`. Setiap konfigurasi memuat audit, perubahan layer, checkpoint FP32,
RESTORE.md, fresh reload, metrik, status dan error. Simpan seluruh direktori suite dari Kaggle.
Laporan penelitian tetap membedakan structural PASS, evaluation PASS, dan analisis trade-off.""")
    code('''write_json(SUITE_DIR / "comparison" / "reproducibility.json", {
    "suite_id": BUNDLE["suite_id"], "configuration": configuration(),
    "sources": [{"model": s["model_name"], "source": s["source"]} for s in SUITES],
    "baseline_validation_statuses": {s["model_name"]: s["baseline_validation_status"] for s in SUITES},
    "structural_statuses": [{"model": s["model_name"], "policy": r["policy"], "ratio": r["ratio"], "status": r["structural_smoke_status"]} for s in SUITES for r in s["results"]],
    "tracking_statuses": [{"model": s["model_name"], "policy": r["policy"], "ratio": r["ratio"], **r["tracking"]} for s in SUITES for r in s["results"]],
    "plots": BUNDLE["comparison_plots"], "limitations": ["No recovery", "Single seed", "GPU/mAP only when actually measured", "Descriptive associations, no causal layer attribution"]
})
print("Output:", SUITE_DIR)
display(SUMMARY)''', "run")
    return {"cells": cells, "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                                           "language_info": {"name": "python", "version": "3.11"}},
            "nbformat": 4, "nbformat_minor": 5}


if __name__ == "__main__":
    DESTINATION.write_text(json.dumps(build_notebook(), ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    print(DESTINATION)
