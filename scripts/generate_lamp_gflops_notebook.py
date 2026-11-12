"""Build standalone Kaggle/Filkom LAMP GFLOPs notebooks for YOLO26n and YOLO26s."""
from __future__ import annotations

import ast
import hashlib
import json
import re
import runpy
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DESTINATIONS = {
    "n": ROOT / "notebooks/kaggle/pruning/05_lamp_yolo26n_gflops_no_recovery.ipynb",
    "s": ROOT / "notebooks/kaggle/pruning/06_lamp_yolo26s_gflops_no_recovery.ipynb",
}
NOTEBOOK_TARGETS = (1.02, 1.05, 1.10, 1.20, 1.30, 1.40, 1.50, 1.60, 1.70)

CONFIG = '''
# full_val: eksperimen model ini. Gabungkan ekspor N/S untuk memilih satu student akhir.
MODE = "full_val"
# MODE = "smoke"
# MODE = "test"  # Memerlukan STUDENT_LOCK_PATH dari full_val.
GFLOPS_TARGETS = (1.02, 1.05, 1.10, 1.20, 1.70)
RUN_DIAGNOSTIC_CONTROLS = False
RESUME_SUITE_DIR = None
DATASET_MANIFEST_PATH = None
EXPORT_DATASET_MANIFEST = True
DATASET_MANIFEST_ONLY = False
SOURCE_CHECKPOINT_SHA256 = {
    "n": "380a087fd621e8a43fe2b9b4d6b1bb6fb00a4d0443cc5afe81ef3efdf0775f0f",
    "s": "5bc3ae61f8da1c9f03916922de10146805c27c64a11e46edd766a00d0f6f18fd",
}
LOCAL_EXPECTED_SHA256S = SOURCE_CHECKPOINT_SHA256.copy()
DATASET_ID = "ilhamrafiqin/poltrada-v3-yolo26-clean"
EXPECTED_DATASET_FINGERPRINT = None
STUDENT_LOCK_PATH = None
TEST_STUDENT_CHECKPOINT_PATH = None
TEST_BASELINE_CHECKPOINT_PATH = None
ALLOW_ARTIFACT_DOWNLOAD = True
# Setelah kedua full_val selesai, lampirkan seluruh folder output varian lain.
# Isi dengan path student_candidates.json varian lain; sel terakhir dapat dijalankan ulang.
COMPARISON_CANDIDATE_PATHS = ()
# () menampilkan overview semua konfigurasi. Detail dapat diperluas tanpa mengulang pruning.
EDA_CONFIGURATIONS = ()
EDA_DETAIL_CONFIGURATIONS = ()
'''


def shared_checkpoint_runtime(source):
    """Preserve the 1.70× diagnostic artifact for any higher targets with the same plan."""
    edits = (
        ('        for target in search["targets"]:',
         '''        # Feasibility and the requested grid are already recorded in target_search.json.
        # Run the original 1.70× diagnostic before other infeasible targets so
        # identical maximum plans reuse its checkpoint and recovery manifest.
        ordered = sorted(search["targets"], key=lambda row: (
            row["status"] == "TARGET_NOT_REACHED" and row["target"] != 1.70))
        for target in ordered:'''),
        ('        "actual_eligible_ratio": result.get("actual_eligible_channel_ratio"),',
         '''        "artifact_configuration_id": result.get("artifact_configuration_id", result["configuration_id"]),
        "actual_eligible_ratio": result.get("actual_eligible_channel_ratio"),'''),
        ('                if target["status"] != "REACHED":',
         '                if target["status"] != "REACHED" and alias["status"] == "PASS":'),
        ('        suite["status"] = workflow_status([result.get("status", "FAIL") for result in results], useful=bool(baseline))',
         '''        # Preserve ascending target order in the tables and manifests.
        rows.sort(key=lambda row: row.get("target_speedup") or 0)
        results.sort(key=lambda row: row.get("target_speedup") or 0)
        suite["status"] = workflow_status([result.get("status", "FAIL") for result in results], useful=bool(baseline))'''),
    )
    for old, new in edits:
        if source.count(old) != 1:
            raise ValueError(f"Anchor ekspor notebook GFLOPs berubah: {old}")
        source = source.replace(old, new, 1)
    return source


def build_notebook(variant="n", platform="kaggle"):
    if variant not in DESTINATIONS:
        raise ValueError("Variant harus n atau s.")
    if platform not in {"kaggle", "filkom"}:
        raise ValueError("Platform harus kaggle atau filkom.")
    model_name = f"YOLO26{variant}"
    artifact_version = {"n": "v84", "s": "v62"}[variant]
    v2 = runpy.run_path(str(ROOT / "scripts/generate_lamp_v2_notebook.py"))["build_notebook"]()
    shared = ["".join(cell["source"]) for cell in v2["cells"]]
    declarations = [(ROOT / f"scripts/lamp_gflops_{subsystem}_source.py").read_text(encoding="utf-8")
                    for subsystem in ("policy", "evaluation", "eda", "runtime")]
    historical = (ROOT / "reports/pruning/lamp_gflops_historical_unverified.csv").read_text(encoding="utf-8")
    workflow = (ROOT / "scripts/lamp_workflow_source.py").read_text(encoding="utf-8")
    # Preserve the method/evaluator contract. The resume contract independently
    # records the actual target grid, so a five-target suite is not overwritten.
    implementation_sha256 = hashlib.sha256("\n".join([*shared, *declarations, workflow, CONFIG, historical]).encode()).hexdigest()
    formatted_targets = [f"{target:.2f}" for target in NOTEBOOK_TARGETS]
    target_labels = [target.replace(".", ",") + "×" for target in formatted_targets]
    config_defaults = CONFIG.replace("GFLOPS_TARGETS = (1.02, 1.05, 1.10, 1.20, 1.70)",
                                     "GFLOPS_TARGETS = (" + ", ".join(formatted_targets) + ")")
    declarations[-1] = shared_checkpoint_runtime(declarations[-1])
    cells = []

    def cell(kind, source, tag="helpers"):
        source = source.strip()
        if kind == "markdown":
            source = source.replace("1,1×, 1,2×, 1,3×", ", ".join(target_labels))
            source = source.replace("1,1×/1,2×/1,3×", "/".join(target_labels))
            source = source.replace("1,1× / 1,2× / 1,3×", " / ".join(target_labels))
            source = source.replace("UNREACHABLE_UNDER_POLICY", "TARGET_NOT_REACHED")
            source = source.replace('default `("target-1.1x",)`', 'default `()` untuk semua konfigurasi')
            source = source.replace("Kontrol extended 10–50% dan kedua probe P5 ikut dievaluasi.",
                                    "Kontrol extended/P5 opsional melalui RUN_DIAGNOSTIC_CONTROLS; default nonaktif.")
            source = source.replace("Logging W&B opsional.", "Logging W&B dan unggah artefak aktif; raw dipublikasikan sebelum recovery.")
            source = source.replace("kedua full_val VERIFIED yang selesai PASS", "kedua full_val VERIFIED dengan baris kandidat lengkap, termasuk dari suite PARTIAL")
            if platform == "filkom":
                source = source.replace("Pada Kaggle aktifkan GPU, Internet untuk instalasi/unduh artifact, dan tambahkan **Poltrada V3 YOLO26 Clean**.",
                    "Gunakan kernel Jupyter Server GPU Filkom. Isi path Poltrada V3, checkpoint, output, dan manifest dataset sumber.")
                source = source.replace("secret `WANDB_API_KEY`", "environment `WANDB_API_KEY` atau login tersembunyi")
                source = source.replace("pemeriksaan awal Kaggle", "pemeriksaan awal Filkom")
        item = {"cell_type": kind, "source": source.splitlines(keepends=True), "metadata": {},
                "id": hashlib.sha256(f"{len(cells)}:{source}".encode()).hexdigest()[:8]}
        if kind == "code":
            ast.parse(source)
            item.update({"execution_count": None, "outputs": [], "metadata": {"tags": [tag]}})
        cells.append(item)

    cell("markdown", f"""# 01 — Pruning {model_name} dengan Target GFLOPs
Notebook standalone khusus **{model_name}** untuk pruning Poltrada V3. Pruning terstruktur memakai LAMP + DepGraph.
Target 1,1×, 1,2×, 1,3× adalah GFLOPs baseline dibagi GFLOPs pruned.
Latency median baseline dibagi latency median pruned dilaporkan terpisah.
Contoh MP-YOLO memakai 1,7×. Adaptasi YOLO26 ini mempertahankan dual head dan dimensi antarmuka,
serta memakai ulang baseline yang sama untuk setiap kandidat. Training, recovery, dan KD tidak dijalankan.
[MP-YOLO](https://github.com/Wang-jj-zs/MP-YOLO/blob/master/MP-YOLO/compress.py) ·
[LAMP](https://arxiv.org/abs/2010.07611) · [DepGraph](https://arxiv.org/abs/2301.12900)

Pada Kaggle aktifkan GPU, Internet untuk instalasi/unduh artifact, dan tambahkan **Poltrada V3 YOLO26 Clean**.
Sediakan secret `WANDB_API_KEY` untuk mengunduh checkpoint atau isi path checkpoint lokal model ini.
SHA-256 checkpoint {model_name} {artifact_version} dikunci. Logging W&B opsional.
Default `MODEL_VARIANTS=["{variant}"]` dan `MODE="full_val"` menjalankan model ini
pada seluruh validation Poltrada V3. Varian lainnya mempunyai notebook sendiri.
Untuk pemeriksaan awal Kaggle, pilih `MODE="smoke"` sebelum Run All;
smoke mengevaluasi subset tetap hingga 128 citra val. `test` memerlukan lock dari full_val.
Hasil Kaggle dan mAP Poltrada V3 harus berasal dari eksekusi nyata.""")
    cell("markdown", f"""## 02 — Konfigurasi {model_name}
Default eksperimen memakai {model_name} {artifact_version}, seed 42, FP32 640 px, batch validasi 16,
serta target GFLOPs 1,1×/1,2×/1,3×. Kontrol extended 10–50% dan kedua probe P5 ikut dievaluasi.
Setiap konfigurasi memuat baseline asli variannya. Pruning menjalankan nol epoch training.
Pilih mode, varian, sumber checkpoint, dan dataset pada sel berikut.

| Mode | Data dan tujuan |
| --- | --- |
| `smoke` | Maksimal 128 citra val dengan subset tetap untuk memeriksa pipeline |
| `full_val` | Seluruh val model ini; ekspor kandidat untuk perbandingan N/S |
| `test` | Split independen untuk baseline dan student yang sudah dikunci |""")
    config = shared[2].replace("pruning/lamp_v2", f"pruning/lamp_gflops_yolo26{variant}")
    config = config.replace('MODEL_VARIANTS = ["n", "s"]', f'MODEL_VARIANTS = ["{variant}"]')
    if platform == "filkom":
        config = re.sub(r'^DATASET_ROOT = .*$', 'WORKSPACE_ROOT = Path(os.environ.get("SKRIPSI_ROOT", Path.cwd()))\nDATASET_ROOT = WORKSPACE_ROOT / "datasets/poltrada-v3"', config, flags=re.MULTILINE)
        config = re.sub(r'^OUTPUT_DIR = .*$', f'OUTPUT_DIR = WORKSPACE_ROOT / "outputs/filkom/pruning/yolo26{variant}"', config, flags=re.MULTILINE)
    cell("code", "import os\n" + config + "\n" + config_defaults + f"\nPLATFORM = {platform!r}\nNOTEBOOK_VARIANT = {variant!r}\nIMPLEMENTATION_SHA256 = {implementation_sha256!r}", "config")
    cell("markdown", """## 03 — Environment dan Dependensi
Notebook memakai versi Ultralytics dan Torch-Pruning yang dikunci serta mencatat lingkungan tiap suite.""")
    for index in (4, 5, 6):
        cell("code", shared[index])
    cell("code", workflow)
    cell("code", """SUITE_DIR = workflow_suite_dir(OUTPUT_DIR, RESUME_SUITE_DIR)
try:
    ensure_dependencies(SUITE_DIR, install=INSTALL_DEPENDENCIES)
except Exception as error:
    failure = {"status": "FAIL", "stage": "environment", "error": safe_error(error)}
    write_json(SUITE_DIR / "environment_failure.json", failure)
    if not (SUITE_DIR / "suite_manifest.json").is_file():
        write_json(SUITE_DIR / "suite_manifest.json", failure)
    raise""", "bootstrap")
    cell("code", shared[8], "imports")
    cell("code", shared[9])
    cell("code", """TARGET_DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu") if DEVICE == "auto" else torch.device(DEVICE)
seed_environment(SEED)
ENVIRONMENT = environment_report()
write_json(SUITE_DIR / "environment.json", ENVIRONMENT)
print("Runtime", TARGET_DEVICE, "| FP32 | Seed", SEED)
print("Kernel", sys.executable, "| cwd", Path.cwd(), "| platform", PLATFORM)
print("GPU", ENVIRONMENT.get("gpu"), "| CUDA", torch.version.cuda, "| Output", SUITE_DIR)
display(pd.DataFrame(ENVIRONMENT["packages"].items(), columns=["package", "version"]))""", "bootstrap")
    cell("markdown", """## 04 — Pengambilan Baseline
Periksa sumber dan SHA-256 checkpoint per varian. Sumber yang dipilih dimuat saat eksperimen berjalan.""")
    cell("code", shared[12])
    acquisition = "def wandb_session_login():\n    workflow_login()\n\n\n" + shared[13][shared[13].index("def obtain_baseline("):]
    acquisition = acquisition.replace("if not WANDB_TRACKING:", "if not ALLOW_ARTIFACT_DOWNLOAD:")
    acquisition = acquisition.replace("Isi LOCAL_CHECKPOINT_PATH atau aktifkan WANDB_TRACKING",
                                      "Isi LOCAL_CHECKPOINT_PATHS atau aktifkan ALLOW_ARTIFACT_DOWNLOAD")
    cell("code", acquisition)
    cell("code", """if MODE != "test":
    display(pd.DataFrame([
        {**model_spec(variant), "source_sha256": SOURCE_CHECKPOINT_SHA256[variant]}
        for variant in MODEL_VARIANTS
    ]).reindex(columns=["model_name", "artifact", "local_checkpoint", "source_sha256"]))""", "bootstrap")
    cell("markdown", f"""## 05 — Audit Arsitektur {model_name}
Periksa struktur backbone, neck, dan kedua cabang Detect. Ukur operasi pada model unfused.""")
    for index in (16, 17):
        cell("code", shared[index])
    cell("markdown", """## 06 — Audit Dependency Graph dan Eligible Layer
Gunakan DepGraph untuk memeriksa dimensi yang terikat, indeks channel asli, dan group eligible.""")
    cell("code", shared[19])
    eda = shared[20].replace('latency.get("mean_ms")', 'latency.get("median_ms")')
    cell("code", eda.replace("Raw model forward latency (ms)", "Median raw model forward latency (ms)"))
    capacity_notes = """

Feasibility dihitung per model dari GFLOPs maksimum pada ketiga tier sebelum kandidat target dievaluasi.
Target di atas batas terukur diberi `TARGET_NOT_REACHED` dengan GFLOPs dan speed-up aktual.
Baris dengan plan identik menunjuk ke `artifact_configuration_id` yang sama.
Kandidat maksimum memakai ID `target-1.70x` bila berbagi plan dengan target lain yang tidak tercapai.
Gunakan checkpoint kanonis satu kali jika menjalankan recovery terpisah.
Untuk grid sembilan target, gunakan folder suite baru."""
    cell("markdown", """## 07 — Implementasi LAMP dan Pencarian Target GFLOPs
Kandidat target mempertahankan proteksi head, attention, split CSP dan antarmuka fitur pada seluruh tier cap.
Pencarian mencatat GFLOPs aktual dan target yang melampaui kapasitas.""" + capacity_notes)
    cell("code", declarations[0])
    cell("markdown", """## 08 — Eksperimen Independen dan Kontrol P5
Setiap konfigurasi memulai dari checkpoint baseline variannya.

| Konfigurasi | Pembanding |
| --- | --- |
| Extended 10–50% | Baseline dari varian yang sama |
| P5-only | Baseline, dengan indeks P5 diambil dari extended 10% |
| P5-protected | Extended 10% dengan jumlah channel independen yang dibuang sama |
| Target 1,1× / 1,2× / 1,3× | Baseline dan GFLOPs aktual kandidat |""")
    cell("markdown", """## 09 — Simpan Checkpoint dan Fresh Reload
Uji muat ulang objek DetectionModel FP32 pada proses baru dan bandingkan output kedua cabang Detect.""")
    cell("code", shared[24])
    cell("markdown", """## 10 — Validasi Baseline dan Pruned
Ukur baseline dan pruned dengan split, citra, serta konfigurasi evaluator yang sama.
Structural PASS, reload PASS, evaluasi, dan latency memiliki status terpisah.""")
    cell("code", shared[26])
    validation = shared[27]
    call = 'arguments = validation_arguments(dataset["yaml"], output_dir, label)'
    capture = 'validator = captured.get("validator", getattr(wrapper, "validator", None))'
    val_call = "        metric = wrapper.val(**arguments)"
    assert validation.count(call) == validation.count(capture) == validation.count(val_call) == 1
    validation = validation.replace(call, "arguments = gflops_validation_arguments(dataset, output_dir, label)")
    validation = validation.replace(capture, capture + "\n    result.update(gflops_validation_metadata(validator, arguments, dataset))")
    validation = validation.replace(
        val_call, "        with gflops_require_complete_nms(), gflops_disable_label_cache():\n    " + val_call
    )
    cell("code", validation)
    for source in declarations[1:]:
        cell("code", source)
    cell("code", """BUNDLE = gflops_execute(SUITE_DIR)
print("Output", SUITE_DIR, "| Status", BUNDLE["status"])""", "run")
    cell("markdown", """## 11 — EDA Identitas, Cakupan, dan Channel
Analisis mengikuti urutan notebook 03 dan 04. Rekap mencakup **semua konfigurasi model pada notebook ini**.
Overview channel ditampilkan untuk setiap konfigurasi yang EDA-nya berhasil.
`EDA_CONFIGURATIONS=()` berarti semua hasil; isi tuple ID untuk membatasi tampilan.
Detail layer, ledger, mask, distribusi LAMP, dan kurva individual mengikuti
`EDA_DETAIL_CONFIGURATIONS`, default `("target-1.1x",)`. Gunakan `()` pada keduanya untuk
menampilkan seluruh detail. Pengaturan ini hanya mengatur tampilan; ekspor analisis tetap lengkap.
Sel 11–14 dapat dijalankan ulang dari `BUNDLE` tanpa pruning atau validasi ulang.

### 11.1 Identitas eksperimen dan cakupan arsitektur
Periksa checkpoint, hash, pasangan baseline, jumlah citra, fingerprint, dan mode evaluasi.
Tabel cakupan membedakan group eligible dari graph, kapasitas policy target, channel independen,
dan perubahan fisik Conv-output. Group yang ditolak graph beserta alasannya tetap tersedia.
Layer pada group graph yang diterima dapat mempunyai kapasitas nol karena proteksi target tambahan.""")
    cell("code", """from IPython.display import Image as NotebookImage
from IPython.display import Markdown

try:
    ANALYSIS = gflops_build_analysis(BUNDLE, SUITE_DIR)
except Exception as error:
    print("EDA suite PARTIAL:", safe_error(error))
    ANALYSIS = {"status": "PARTIAL", "plots": [], "tables": gflops_eda_tables(BUNDLE, SUITE_DIR)}
TABLES = ANALYSIS["tables"]
SUMMARY = pd.DataFrame(TABLES["metrics"])
EDA_RUNS = []


def show_eda_table(rows, columns=None):
    frame = pd.DataFrame(rows)
    if frame.empty:
        print("UNAVAILABLE — lihat status pengukuran pada tabel ringkasan.")
        return
    if columns:
        frame = frame.reindex(columns=columns)
    with pd.option_context("display.max_rows", None, "display.max_columns", None, "display.max_colwidth", 100):
        display(frame)


def show_eda_images(folder, *patterns):
    for pattern in patterns:
        for path in sorted(folder.glob(pattern)):
            display(NotebookImage(filename=str(path)))


show_eda_table(TABLES["sources"], ["model", "requested_artifact", "sha256", "source_verification",
                                  "baseline_validation_parity_status"])
show_eda_table(TABLES["metrics"], ["model", "configuration_id", "policy", "stage", "ratio", "actual_eligible_ratio",
    "pruned_group_count", "evaluation_mode", "validation_image_count", "dataset_provenance_status",
    "structural_status", "reload_status", "validation_status", "eda_status"])
show_eda_table(TABLES["architecture"], ["model", "index", "type", "from", "region", "parameters"])
show_eda_table(TABLES["coverage"])
""", "display")
    cell("markdown", """### 11.2 Perbandingan channel sebelum dan sesudah pruning
Diagram **Before Pruning** biru, **After Pruning** merah, dan **Channel Comparison** overlay
memakai urutan seluruh Conv dan batas sumbu yang sama. Layer yang tetap utuh ikut ditampilkan.
Bandingkan pola extended, P5-only, P5-protected, dan kandidat GFLOPs pada model yang sama.
Jumlah Conv-output yang terhapus dapat melebihi channel independen karena residual mengikat
beberapa output. Konsentrasi penghapusan merupakan observasi distribusi; kontras P5 di bagian 13
memerlukan evaluasi mAP untuk menguji hipotesis sensitivitas.""")
    cell("code", """for model in BUNDLE["models"]:
    model_name = model["model_name"]
    available = {result["configuration_id"]: result for result in model["results"]
                 if result["eda_status"] == "PASS"}
    if not available:
        print(f"{model_name} — EDA belum tersedia. Lihat status tiap konfigurasi pada tabel.")
    for key, result in available.items():
        if EDA_CONFIGURATIONS and key not in EDA_CONFIGURATIONS:
            continue
        folder = SUITE_DIR / model_name.lower() / result.get("artifact_configuration_id", key) / "eda"
        EDA_RUNS.append((model_name, key, result, folder))
        display(Markdown(f"#### {model_name} · {key}"))
        show_eda_images(folder, "channels_overview.png", "channel_reduction_percent.png")
EDA_DETAILS = [entry for entry in EDA_RUNS if not EDA_DETAIL_CONFIGURATIONS or entry[1] in EDA_DETAIL_CONFIGURATIONS]
for model_name, key, result, folder in EDA_DETAILS:
    display(Markdown(f"#### Detail layer · {model_name} · {key}"))
    selected = [row for row in TABLES["layers"] if row["model"] == model_name and row["configuration_id"] == key]
    show_eda_table(selected, ["layer_name", "module_type", "region", "in_channels_before", "in_channels_after",
        "out_channels_before", "out_channels_after", "out_channels_removed", "out_channels_reduction_pct",
        "parameters_before", "parameters_after", "estimated_flops_before", "estimated_flops_after",
        "dependency_group", "protection_reason"])
    show_eda_images(folder, "channels_detail_*.png")""", "display")
    cell("markdown", """### 11.3 Keputusan LAMP, kapasitas group, dan ledger indeks asli
Lihat group yang diterima/ditolak, rentang skor finite, kapasitas, dan jumlah channel yang dipilih.
Skor `+inf` dari pembatas kapasitas tidak diperlakukan sebagai importance.
Heatmap memakai indeks asli sebelum pruning. Ledger juga mencatat kolom input konsumen dan BN,
termasuk offset concat dan seluruh konsumen fan-out. Tabel lengkap seluruh konfigurasi diekspor
ke `comparison/analysis/groups.csv` dan `ledger.csv`. Heatmap menggambarkan channel bobot.""")
    cell("code", """for model_name, key, result, folder in EDA_DETAILS:
    display(Markdown(f"#### LAMP dan dependency · {model_name} · {key}"))
    selected = [row for row in TABLES["groups"] if row["model"] == model_name and row["configuration_id"] == key]
    show_eda_table(selected, ["root", "group_id", "decision", "initial_channels", "capacity_eligible",
        "removed", "score_min", "score_median", "score_max", "coupled_output_layers", "reason"])
    changed = [row for row in TABLES["ledger"] if row["model"] == model_name and row["configuration_id"] == key
               and row.get("removed_original_indices")]
    show_eda_table(changed, ["layer", "module_type", "dimension", "channels_before", "channels_after",
                            "kept_original_indices", "removed_original_indices", "dependency_group_ids"])
    show_eda_images(folder, "channel_mask_*.png", "lamp_score_distribution.png", "lamp_group_capacity.png")""", "display")
    cell("markdown", """## 12 — EDA Efisiensi dan Metrik Deteksi
### 12.1 Efisiensi per model, layer, dan stage
Ringkasan memuat seluruh konfigurasi, masing-masing dibandingkan dengan baseline variannya.
GFLOPs speed-up dan latency speed-up memakai baseline yang sama untuk varian tersebut.
Latency adalah raw dual-head forward FP32 batch satu, dengan p50 dan p95, di luar preprocessing,
NMS, dan I/O. Pengukuran CUDA yang belum dijalankan tetap `NOT_RUN`.
Estimasi operasi per layer/stage mencakup Conv/BN; total GFLOPs model juga memasukkan attention.
Ukuran checkpoint pascapruning dibandingkan dengan baseline FP32 yang disimpan ulang.
Ukuran file baseline sumber ditampilkan terpisah karena format penyimpanannya dapat berbeda.""")
    cell("code", """show_eda_table(TABLES["metrics"], [
    "model", "configuration_id", "stage", "parameters", "parameter_reduction_pct",
    "gflops", "gflops_reduction_pct", "gflops_speedup", "checkpoint_bytes", "checkpoint_reduction_pct",
    "latency_median_ms", "latency_p95_ms", "latency_speedup", "latency_status",
])
for model_name, key, result, folder in EDA_DETAILS:
    display(Markdown(f"#### Efisiensi stage · {model_name} · {key}"))
    selected = [row for row in TABLES["stages"] if row["model"] == model_name and row["configuration_id"] == key]
    show_eda_table(selected)
    show_eda_images(folder, "stage_physical_conv_outputs.png", "stage_parameters.png", "stage_estimated_flops.png")""", "display")
    cell("markdown", """### 12.2 Akurasi keseluruhan dan AP/P/R/F1 per kelas untuk seluruh konfigurasi
`delta_map50_95_pp = 100 × (mAP_pruned − mAP_baseline)` menyatakan poin persentase.
Perubahan relatif menggunakan mAP baseline sebagai denominator dan tidak terdefinisi jika baseline nol.
Rekap per kelas mengikuti notebook 03, dilengkapi selisih AP terhadap baseline dengan ID kelas yang sama.
Support menunjukkan jumlah ground-truth; kelas tanpa metrik tetap kosong.
Pada `smoke`, semua angka hanya memeriksa pipeline di subset tetap. Interpretasi akurasi penelitian
serta pemilihan student membutuhkan seluruh validation Poltrada V3.""")
    cell("code", """show_eda_table(TABLES["metrics"], [
    "model", "configuration_id", "stage", "map50_95", "map50", "precision", "recall",
    "delta_map50_95_pp", "relative_map50_95_pct", "validation_status", "evaluation_mode", "validation_image_count",
])
PER_CLASS = pd.DataFrame(TABLES["per_class_metrics"])
for model in BUNDLE["models"]:
    model_name = model["model_name"]
    display(Markdown(f"#### Semua konfigurasi · {model_name}"))
    selected = [row for row in TABLES["per_class_metrics"] if row["model"] == model_name]
    show_eda_table(selected, ["configuration_id", "stage", "class_id", "name", "support", "ap50_95", "ap50",
                            "precision", "recall", "f1", "delta_ap50_95_pp", "delta_ap50_pp"])
    show_eda_images(SUITE_DIR / "comparison/analysis" / model_name.lower(),
                    "configuration_metrics.png", "per_class_ap_all.png")""", "display")
    cell("markdown", """### 12.3 Kurva PR/F1/confidence, confusion matrix, dan protokol efektif
Tampilkan kurva baseline dan pruned dari validator aktual untuk konfigurasi detail.
Precision/recall/F1 per kelas berasal dari titik confidence terbaik menurut smoothed mean F1 SDK
pada IoU 0,50. Confusion matrix memakai **baris predicted, kolom true**, termasuk background.
Threshold confidence dan IoU pencocokan matrix berbeda dari IoU NMS validator; baca metadata aktual.
Seluruh konfigurasi menyimpan kurva dan matrix di folder EDA masing-masing saat evaluasi tersedia.
Tabel protokol mengungkap split, jumlah citra, FP32, one-to-many/NMS, dan fingerprint setiap evaluasi.""")
    cell("code", """show_eda_table(TABLES["validation_protocol"], [
    "model", "configuration_id", "validation_status", "evaluation_mode", "split", "validation_image_count",
    "dataset_fingerprint", "dataset_provenance_status", "imgsz", "batch", "half", "conf", "iou", "max_det",
    "nms", "end2end", "augment", "rect", "device",
])
for model_name, key, result, folder in EDA_DETAILS:
    display(Markdown(f"#### Kurva dan matrix · {model_name} · {key}"))
    for stage in ("baseline", "pruned"):
        metrics = result[stage].get("validation") or {}
        display(Markdown(f"##### {stage.capitalize()}"))
        if not metrics:
            print("NOT_RUN — metrik deteksi belum tersedia.")
            continue
        display(metrics.get("metric_semantics", {}))
        matrix = metrics.get("confusion_matrix", {})
        display({field: matrix.get(field) for field in (
            "status", "labels", "confidence_threshold", "matching_iou_threshold", "validation_iou")})
        show_eda_images(folder / stage, "class_metrics.png", "evaluation_curve_*.png", "confusion_matrix.png")""", "display")
    cell("markdown", """## 13 — Analisis Trade-off Akurasi dan Efisiensi
### 13.1 Bukti awal extended dari tabel historis pengguna
Tabel 12 baris ini dipertahankan dengan provenance **UNVERIFIED**. Pada extended 10%, mAP50–95
yang dilaporkan turun dari 0,5261 ke 0,1692 pada N dan 0,5612 ke 0,2546 pada S; pada 20%,
keduanya di bawah 0,003. Status PASS pada tabel adalah status asli yang dilaporkan.
Artefak run, fingerprint, konfigurasi evaluator, dan counter perlu diverifikasi.
Tabel ini diarsipkan tersendiri dan tidak dimasukkan ke grafik hasil suite aktual.""")
    cell("code", "from io import StringIO\nHISTORICAL_EXTENDED_CSV = " + repr(historical) + "\n" + """
HISTORICAL_EXTENDED = pd.read_csv(StringIO(HISTORICAL_EXTENDED_CSV))
show_eda_table(HISTORICAL_EXTENDED.to_dict("records"))
(SUITE_DIR / "comparison/analysis/historical_unverified.csv").write_text(HISTORICAL_EXTENDED_CSV, encoding="utf-8")""", "display")
    cell("markdown", """### 13.2 Kontrol extended dan pencarian target komputasi
Seperti notebook 03, lihat hubungan rasio kontrol dengan mAP dan reduksi parameter/GFLOPs.
Kandidat utama memakai target komputasi 1,1×/1,2×/1,3×. Tabel tier menunjukkan batas kapasitas
dan tabel target menunjukkan capaian atau `UNREACHABLE_UNDER_POLICY`.
Profil pencarian hanya mengukur GFLOPs; mAP dinilai pada kandidat eksperimen yang dievaluasi.
Bandingkan GFLOPs speed-up dan latency speed-up yang terukur untuk menilai manfaat di perangkat.""")
    cell("code", """show_eda_table(TABLES["target_tiers"])
show_eda_table(TABLES["target_search"], ["model", "target", "status", "tier", "channels", "gflops", "speed_up"])
for model in BUNDLE["models"]:
    model_name = model["model_name"]
    display(Markdown(f"#### Sweep dan kapasitas · {model_name}"))
    show_eda_images(SUITE_DIR / "comparison/analysis" / model_name.lower(),
                    "extended_sweep.png", "target_search_capacity.png")""", "display")
    cell("markdown", """### 13.3 Trade-off seluruh konfigurasi model ini
Setiap titik berlabel ID konfigurasi. Grafik per varian memakai baseline variannya sendiri;
Hasil N/S berasal dari notebook terpisah. Penurunan mAP besar tetap merupakan temuan negatif
walaupun struktur dan reload lulus. Angka historis pengguna berprovenance `UNVERIFIED` disimpan
terpisah dari titik suite aktual. Satu seed tidak mengukur variasi antar-seed.""")
    cell("code", """show_eda_table(ANALYSIS["plots"], ["model", "plot", "status", "measured_values", "reason"])
for model in BUNDLE["models"]:
    display(Markdown(f"#### Trade-off · {model['model_name']}"))
    show_eda_images(SUITE_DIR / "comparison/analysis" / model["model_name"].lower(), "tradeoff_*.png")
show_eda_images(SUITE_DIR / "comparison", "map_vs_*.png")""", "display")
    cell("markdown", """### 13.4 Kontras sensitivitas P5 dan kelayakan student
**P5-only vs baseline** mengukur perubahan saat hanya channel P5 pilihan extended 10% yang dibuang.
**P5-protected vs extended 10%** membandingkan alokasi ulang dengan jumlah penghapusan independen yang sama.
Bandingkan delta mAP dan AP per kelas sambil membaca perbedaan GFLOPs kedua alokasi.
Distribusi channel sendiri belum membuktikan penyebab runtuhnya akurasi.

Kelayakan student mengikuti aturan full_val terverifikasi, target tercapai, kehilangan mAP50–95
maksimal dua poin, serta parameter/GFLOPs/latency yang lengkap. Pemilihan berasal dari pipeline full_val
dan mencakup gabungan N/S. Tabel EDA tidak mengganti atau mengunci ulang pilihan tersebut.""")
    cell("code", """show_eda_table(TABLES["p5_contrasts"])
show_eda_table(TABLES["p5_per_class"], ["model", "contrast", "reference", "class_id", "name", "support",
    "reference_ap50_95", "ap50_95", "delta_ap50_95_pp"])
show_eda_table(TABLES["student_eligibility"])
display(BUNDLE.get("selection", BUNDLE.get("test")))""", "display")
    cell("markdown", """## 14 — Ringkasan, Artefak, dan Reproducibility
Ringkasan mempertahankan status struktur, oracle, replay, reload, evaluasi, latency, dan EDA secara terpisah.
`PASS` pada struktur tidak menjadi penilaian akurasi. Metrik yang belum diukur tetap kosong atau `NOT_RUN`.
Seluruh tabel analisis tersedia sebagai CSV/JSON dan grafik sebagai PNG/SVG pada `comparison/analysis/`.
Folder EDA tiap konfigurasi berisi inventory, ledger, skor, diagram channel/stage, dan hasil validator.
Simpan juga manifest dataset, `target_search.json`, kontras P5, environment, checkpoint, dan hash.
Setiap notebook mengekspor `student_candidates.json`. Untuk pemilihan satu student N/S, simpan
folder output notebook pertama lalu lampirkan sebagai dataset pada notebook kedua. Isi
`COMPARISON_CANDIDATE_PATHS` dengan path file kandidat varian lain dan jalankan ulang sel terakhir.
Pemilihan memerlukan kedua full_val VERIFIED yang selesai PASS, fingerprint dataset yang sama,
serta protokol, versi implementasi, dan perangkat latency yang sama. Checkpoint terpilih beserta
baseline-nya wajib tersedia dan cocok dengan SHA-256 sebelum `student_lock.json` dibuat.
Gunakan objek DetectionModel FP32 beserta manifest untuk loader KD yang mempertahankan lebar pruned.""")
    cell("code", """print("Folder suite", SUITE_DIR)
for filename in ("suite_manifest.json", "config.json", "environment.json", "metrics.csv", "student_candidates.json", "student_lock.json"):
    path = SUITE_DIR / filename
    if path.is_file():
        print(path.name, path)
print("Analisis", SUITE_DIR / "comparison/analysis", "| Status ekspor", ANALYSIS["status"])
show_eda_table(TABLES["verification"])
show_eda_table(TABLES["metrics"])
if MODE == "full_val" and COMPARISON_CANDIDATE_PATHS:
    BUNDLE["selection"] = gflops_finalize_student(SUITE_DIR, COMPARISON_CANDIDATE_PATHS)
    write_json(SUITE_DIR / "suite_manifest.json", BUNDLE)
display(BUNDLE.get("selection", BUNDLE.get("test")))""", "display")
    return {**v2, "cells": cells, "metadata": {**v2["metadata"], "research_workflow": {
        "platform": platform, "model_variant": variant, "stage": "pruning"}}}


if __name__ == "__main__":
    for platform in ("kaggle", "filkom"):
        for variant, source in DESTINATIONS.items():
            destination = ROOT / "notebooks" / platform / "pruning" / source.name
            destination.parent.mkdir(parents=True, exist_ok=True)
            destination.write_text(json.dumps(build_notebook(variant, platform), ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
            print(destination)
