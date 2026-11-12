"""Build the standalone Kaggle/Filkom YOLO26s multi-seed pruning notebook."""
from __future__ import annotations

import ast
import hashlib
import json
import runpy
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
DESTINATION = ROOT / "notebooks/kaggle/pruning/06_lamp_yolo26s_multiseed_wandb.ipynb"

# SHA-256 dihitung dari best.pt pada versi tetap, dengan seed dicek pada metadata W&B.
BASELINE_PINS = {
    42: ("02_baseline_yolo26s-q5kp9k9d-checkpoints:v62", "5bc3ae61f8da1c9f03916922de10146805c27c64a11e46edd766a00d0f6f18fd"),
    0: ("02_baseline_yolo26s-7f5jmtwb-checkpoints:v71", "d02d8d90d5000fc8379379acb0c0cb3cde0c01143b197d237b25be1bae99bf4f"),
    2024: ("02_baseline_yolo26s-1o9ggnyw-checkpoints:v41", "47cfab586889d2b464c882239c86224b645363a5a4fa3badc31c46991ad27d70"),
    7: ("02_baseline_yolo26s-9cq437l3-checkpoints:v49", "3e59d5f5302fb524da729e139ef7e9b8456c08d0952c19b08af19c7f39bce6ab"),
    3407: ("02_baseline_yolo26s-bmshdzv9-checkpoints:v43", "de9f06892953a343629a4b3a59df880d771973ae95160dee02cd96e37de59e46"),
}

CONFIG = '''
import os
from pathlib import Path

PLATFORM = "kaggle" if Path("/kaggle").is_dir() else "filkom"
WORKSPACE_ROOT = Path(os.environ.get("SKRIPSI_ROOT", Path.cwd()))
DATASET_ROOT = (Path("/kaggle/input/datasets/ilhamrafiqin/poltrada-v3-yolo26-clean") if PLATFORM == "kaggle"
                else WORKSPACE_ROOT / "datasets/poltrada-v3")
DATA_YAML = DATASET_ROOT / "data.yaml"
DATASET_MANIFEST_PATH = None  # Filkom: Path ke dataset_manifest.json dari mount Kaggle resmi.
DATASET_MANIFEST_ONLY = False
OUTPUT_DIR = (Path("/kaggle/working") if PLATFORM == "kaggle" else WORKSPACE_ROOT / "outputs") / "pruning/lamp_yolo26s_per_seed"
RESUME_SUITE_DIR = None
LOCAL_BASELINE_PATHS = {}  # Opsional, misalnya {42: Path(".../best.pt")}. Hash harus cocok dengan pin.

MODE = "full_val"  # "smoke" memakai hingga 128 citra val dan menyimpan hasil secara lokal.
DEVICE = "auto"
IMGSZ, BATCH, WORKERS = 640, 16, 2
INSTALL_DEPENDENCIES = True
RUN_LATENCY_BENCHMARK = True
LATENCY_WARMUP, LATENCY_REPEATS = 50, 200
RELOAD_TIMEOUT_SECONDS = 300
EDA_DPI, EDA_MAX_LAYERS_PER_PAGE = 300, 32
EDA_DISPLAY_RATIO = 0.50

# Proteksi dan tier cap dikunci di helper audit GFLOPs.
GLOBAL_PRUNING = True
RECOVERY_ENABLED = FINETUNING_ENABLED = KD_ENABLED = False
'''

RECOVERY_CELL = '''
def recovery_config_from_registry(path, seed, target=1.20, *, allow_diagnostic=False, include_baseline=False):
    import json
    from pathlib import Path
    import re

    registry = json.loads(Path(path).expanduser().read_text(encoding="utf-8"))
    version = registry.get("format_version")
    if (version not in {1, 2} or registry.get("workflow") != "lamp-yolo26s-multiseed"
            or registry.get("mode") != "full_val"):
        raise ValueError("Gunakan checkpoint_registry.json dari full validation LAMP YOLO26s.")
    matches = [row for row in registry["rows"] if row.get("target_speedup") == float(target)
               and row.get("recovery_seed") == seed]
    if len(matches) != 1:
        raise ValueError("Target dan seed recovery harus memilih tepat satu baris.")
    row = matches[0]
    if row.get("target_status") != "REACHED" and not (
            allow_diagnostic and row.get("target_status") == "TARGET_NOT_REACHED"):
        raise ValueError("Target belum tercapai. Pilih allow_diagnostic=True untuk kandidat maksimum yang terverifikasi.")
    reference, digest = row.get("artifact_ref"), row.get("raw_sha256")
    version_ref = r"[^/:]+/[^/:]+/[^/:]+:v[0-9]+"
    sha256 = r"[0-9a-f]{64}"
    if (row.get("model_variant") != "s" or row.get("evaluation_mode") != "full_val"
            or row.get("dataset_provenance_status") != "VERIFIED" or row.get("publication_status") != "CONFIRMED"
            or row.get("recovery_ready") is not True
            or not isinstance(reference, str) or not re.fullmatch(version_ref, reference)
            or not isinstance(digest, str) or not re.fullmatch(sha256, digest)):
        raise ValueError("Pilih baris terverifikasi dengan versi W&B tetap dan SHA-256 raw.")
    config = (reference, digest, row["recovery_seed"])
    if version == 2:
        baseline_ref = row.get("source_baseline_artifact_ref")
        baseline_sha = row.get("source_baseline_sha256")
        collection = f"yolo26s-lamp-su{float(target):.2f}x-seed{seed}-raw"
        if (row.get("baseline_seed") != seed or row.get("pruning_seed") != seed
                or not registry.get("suite_id") or row.get("suite_id") != registry["suite_id"]
                or reference.split("/")[-1].rsplit(":", 1)[0] != collection
                or not isinstance(baseline_ref, str) or not re.fullmatch(version_ref, baseline_ref)
                or not isinstance(baseline_sha, str) or not re.fullmatch(sha256, baseline_sha)):
            raise ValueError("Identitas suite, perlakuan, atau baseline dari baris registry tidak valid.")
        if include_baseline:
            return (*config, baseline_ref, baseline_sha)
    elif include_baseline:
        raise ValueError("Identitas baseline per seed memerlukan registry format 2.")
    return config


# Salin sel ini sesudah konfigurasi recovery 02–05, lalu aktifkan pengaturan berikut.
# RAW_ARTIFACT_REF, EXPECTED_RAW_SHA256, SEED, REGISTRY_BASELINE_REF, REGISTRY_BASELINE_SHA256 = (
#     recovery_config_from_registry("checkpoint_registry.json", seed=42, include_baseline=True)
# )
# RAW_MANIFEST_PATH = None
# Di sel terpisah sesudah bootstrap recovery dan sebelum run, pasang lock dari baris yang sama.
# BASELINE_REF, BASELINE_SHA256 = REGISTRY_BASELINE_REF, REGISTRY_BASELINE_SHA256
'''


def definitions(source, *names):
    found = {node.name: ast.get_source_segment(source, node) for node in ast.parse(source).body
             if isinstance(node, ast.FunctionDef) and node.name in names}
    return "\n\n\n".join(found[name] for name in names)


def build_notebook():
    original = runpy.run_path(str(SCRIPTS / "generate_lamp_gflops_notebook.py"))["build_notebook"]("s", "kaggle")
    code = ["".join(cell["source"]) for cell in original["cells"] if cell["cell_type"] == "code"]
    template_markdown, template_display = {}, {}
    for cell in original["cells"]:
        source = "".join(cell["source"])
        if cell["cell_type"] == "markdown":
            number = source.splitlines()[0].split()[1]
            template_markdown[number] = source
        elif "display" in cell["metadata"].get("tags", []):
            template_display[number] = source

    def shared(prefix):
        source, = (source for source in code if source.startswith(prefix))
        return source

    config_values = {node.targets[0].id: ast.literal_eval(node.value) for node in ast.parse(shared("import os\n")).body
                     if isinstance(node, ast.Assign) and isinstance(node.targets[0], ast.Name)
                     and node.targets[0].id in {"WANDB_ENTITY", "WANDB_PROJECT", "DATASET_ID"}}
    config = CONFIG + "\n" + "\n".join(f"{key} = {value!r}" for key, value in config_values.items())
    pins = [
        f'    {seed}: {{\n        "ref": f"{{WANDB_ENTITY}}/{{WANDB_PROJECT}}/{ref}",\n        "sha256": "{sha}",\n    }},'
        for seed, (ref, sha) in BASELINE_PINS.items()
    ]
    config += "\n\nBASELINE_SOURCES = {\n" + "\n".join(pins) + "\n}"
    workflow = definitions(shared('"""Shared standalone persistence'), "write_json", "write_csv", "workflow_digest",
                           "workflow_bind", "workflow_save_checkpoint", "workflow_suite_dir", "workflow_status",
                           "workflow_version_ref", "workflow_dataset_manifest")
    early = [shared("import csv\n"), "from datetime import UTC\nimport re\n\n" + workflow,
             definitions(shared("# Backward-compatible N helper aliases"), "safe_error", "model_state_sha256", "metric_comparison"),
             shared("def ensure_dependencies(")]
    reload = shared("RELOAD_SCRIPT =").replace('child_env.pop("PYTHONPATH", None)',
                                               'child_env.pop("PYTHONPATH", None)\n    child_env.pop("WANDB_API_KEY", None)')
    runtime = definitions(shared('"""Notebook 05 runtime'), "gflops_latency", "gflops_validation",
                          "gflops_baseline", "gflops_reload", "gflops_make_graph", "gflops_plan_from_history")
    # Keep the evaluator and the eligibility predicate used by the shared EDA tables.
    evaluation = shared('"""Dataset and student-selection').split("\ndef gflops_select_student(")[0]
    shared_sources = [shared("import numpy as np"), shared("def seed_environment("), shared("def select_checkpoint("),
                      shared("def ordered_names("), shared("def assert_layer_consistency("),
                      shared('"""Standalone LAMP V2 policy'), shared('"""Standalone analysis helpers'),
                      shared('"""GFLOPs-target planner'), reload,
                      definitions(shared("def prepare_validation_dataset("), "cuda_latency"),
                      shared('"""Validation declarations'), evaluation, shared('"""Cross-configuration analysis'), runtime]
    multi = (SCRIPTS / "lamp_multiseed_runtime_source.py").read_text(encoding="utf-8")
    # Resume binds the per-seed source locks and executable declarations.
    implementation = hashlib.sha256("\n".join([config, *early, *shared_sources, multi]).encode()).hexdigest()
    acquisition = [definitions(multi, name) for name in ("ms_wandb_environment", "ms_source")]
    multi_runtime = multi
    for source in acquisition:
        multi_runtime = multi_runtime.replace(source, "", 1)
    cells = []

    def add(source, tag=None):
        source = source.strip()
        item = {"cell_type": "code" if tag else "markdown", "source": source.splitlines(keepends=True), "metadata": {},
                "id": hashlib.sha256(f"{len(cells)}:{source}".encode()).hexdigest()[:8]}
        if tag:
            ast.parse(source)
            item.update(execution_count=None, outputs=[], metadata={"tags": [tag]})
        cells.append(item)

    def section(number, *descriptions):
        source = template_markdown[number]
        if descriptions:
            headings = [line for line in source.splitlines() if line.startswith("#")]
            source = "\n\n".join(f"{heading}\n\n{body.strip()}".strip()
                                 for heading, body in zip(headings, descriptions, strict=True))
        add(source)

    section("01", """

**NO RECOVERY · NO TRAINING · NO KD**

Notebook standalone khusus **YOLO26s** untuk pruning terstruktur Poltrada V3 dengan LAMP +
DepGraph. Ukur variasi perubahan akurasi dan biaya komputasi pada target 1,20× di lima student
baseline yang dilatih dengan seed berbeda. Pembanding 1,10× dan 1,70× memakai baseline seed 42.
Setiap perlakuan dibandingkan dengan baseline seed yang sama memakai mAP50–95, AP per kelas,
parameter, GFLOPs, dan latency p50/p95 pada protokol yang sama.

Speed-up GFLOPs adalah GFLOPs baseline dibagi GFLOPs pruned. Speed-up latency memakai rasio
median raw forward. Adaptasi YOLO26 mempertahankan kedua cabang Detect dan dimensi antarmuka.
Contoh konfigurasi MP-YOLO memakai 1,70× sebagai target komputasi.
[MP-YOLO](https://github.com/Wang-jj-zs/MP-YOLO/blob/master/MP-YOLO/compress.py) ·
[LAMP](https://arxiv.org/abs/2010.07611) · [Kode penulis LAMP](https://github.com/jaeho-lee/layer-adaptive-sparsity) ·
[DepGraph](https://arxiv.org/abs/2301.12900)

Di Kaggle pilih Tesla T4, aktifkan internet, pasang input **Poltrada V3 YOLO26 Clean**, dan
hubungkan Secret `WANDB_API_KEY`. `Run All` menjalankan antrean serial pada `cuda:0`, termasuk
saat runtime menyediakan dua GPU. Di Filkom pilih kernel CUDA, isi path dataset dan manifest
sumber terverifikasi, serta set `WANDB_API_KEY` pada environment Jupyter.
""")
    section("02", """
Default `MODE="full_val"` dan `RESUME_SUITE_DIR=None` membuat suite baru dari lima `best.pt` W&B
yang dikunci. Gunakan FP32 640 px dan batch validasi 16. Path dataset, perangkat, output, dan
salinan baseline lokal opsional tersedia di sel konfigurasi. Set `DATASET_MANIFEST_ONLY=True`
pada mount resmi Kaggle untuk mengekspor manifest yang bisa disalin ke Filkom.

| Mode | Data dan tujuan |
| --- | --- |
| `smoke` | Subset tetap hingga 128 citra validation untuk memeriksa alur dan menyimpan hasil lokal |
| `full_val` | Seluruh validation Poltrada V3 terverifikasi, benchmark CUDA, dan publikasi checkpoint W&B |
""")
    add(config + f"\nIMPLEMENTATION_SHA256 = {implementation!r}\n" + '''
# Filter tampilan EDA berdasarkan ID perlakuan, misalnya "target-1.20x-seed0". () menampilkan semua.
EDA_CONFIGURATIONS = ()
EDA_DETAIL_CONFIGURATIONS = ()
''', "config")
    section("03", """
Gunakan Python 3.11 atau lebih baru. Bootstrap mengunci Ultralytics 8.4.155, Torch-Pruning 1.6.1,
dan W&B 0.28.1 sambil mempertahankan Torch serta torchvision dari kernel. Suite mencatat versi
dependensi, perangkat, build CUDA, presisi, dan hash implementasi importance.
""")
    for index in (0, 2, 3, 1):
        add(early[index], "helpers")
    add('''
SUITE_DIR = workflow_suite_dir(OUTPUT_DIR, RESUME_SUITE_DIR)
os.environ["YOLO_AUTOINSTALL"] = "false"
os.environ.setdefault("YOLO_CONFIG_DIR", str(SUITE_DIR / "ultralytics_config"))
ensure_dependencies(SUITE_DIR, install=INSTALL_DEPENDENCIES)
''', "bootstrap")
    add(shared_sources[0], "imports")
    add(shared_sources[1], "helpers")
    add('''
TARGET_DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu") if DEVICE == "auto" else torch.device(DEVICE)
ENVIRONMENT = environment_report()
print("Runtime", PLATFORM, "|", TARGET_DEVICE, "|", ENVIRONMENT["gpu"], "|", SUITE_DIR)
display(pd.DataFrame(ENVIRONMENT["packages"].items(), columns=["package", "version"]))
''', "bootstrap")
    section("04", """
Periksa seed, versi W&B, dan SHA-256 `best.pt` sumber. Suite memeriksa metadata seed dan SHA-256
unduhan sebelum memuat model. Salinan dalam `LOCAL_BASELINE_PATHS` harus cocok dengan SHA-256 seed
yang bersangkutan. Baseline dimuat kembali dari `best.pt` untuk setiap perlakuan.
""")
    add(shared_sources[2], "helpers")
    add("\n\n\n".join(acquisition), "runtime")
    add('''
display(pd.DataFrame([{"baseline_seed": seed, "artifact": source["ref"], "source_sha256": source["sha256"],
                       "local_checkpoint": LOCAL_BASELINE_PATHS.get(seed)} for seed, source in BASELINE_SOURCES.items()]))
''', "bootstrap")
    section("05", """
Audit backbone, neck, dan Detect memeriksa identitas skala YOLO26s, dimensi keluaran, urutan kelas,
serta keberadaan cabang one-to-many dan one-to-one. Inventaris menyimpan tipe, asal fitur, dan
parameter tiap stage. Counter operasi memakai model FP32 unfused.
""")
    for source in shared_sources[3:5]:
        add(source, "helpers")
    section("06", """
DepGraph menelusuri dimensi channel yang terikat antara Conv, BatchNorm, dan konsumennya.
Audit mencatat group eligible, alasan penolakan atau proteksi, serta indeks channel asli.
Pemetaan mengikuti residual, offset concat, dan seluruh cabang fan-out.
""")
    for source in shared_sources[5:7]:
        add(source, "helpers")
    section("07", """
Ranking global LAMP memakai skor berbasis bobot dan DepGraph menentukan channel yang dipangkas
bersama. Proteksi head, attention, split CSP, downsampling, dan antarmuka fitur mengikuti audit 06.
Feasibility diukur pada tiga tier batas penghapusan 25%/10%, 35%/20%, dan 50%/50% untuk group
hidden Bottleneck dan group eligible lain. Pencarian mencari anggaran channel terkecil yang
mencapai target pada tier pertama yang layak. Jika target melebihi kapasitas, evaluasi checkpoint
maksimum dan catat `TARGET_NOT_REACHED` beserta speed-up aktual.

Profil pencarian memakai bobot baseline masing-masing seed. Tiap perlakuan memuat ulang baseline
dan mengulang plan terpilih untuk memeriksa ledger, struktur, serta hash tensor dan buffer.
""")
    add(shared_sources[7], "helpers")
    section("08", """
Antrean serial memakai baseline sesuai seed. Ketiga target seed 42 memulai pruning dari
`best.pt` yang sama dan menghasilkan paket raw terpisah.

| Urutan | Target GFLOPs | Seed | Baseline | Peran |
| ---: | --- | ---: | --- | --- |
| 1 | 1,10× | 42 | `v62` | Pembanding |
| 2 | 1,20× | 42 | `v62` | Utama |
| 3 | 1,70× | 42 | `v62` | Pembanding kapasitas |
| 4 | 1,20× | 0 | `v71` | Utama |
| 5 | 1,20× | 2024 | `v41` | Utama |
| 6 | 1,20× | 7 | `v49` | Utama |
| 7 | 1,20× | 3407 | `v43` | Utama |

Kontrol extended, P5-only, dan P5-protected berstatus `NOT_RUN` pada suite ini. Bagian 13
menyediakan konteks historis dan auditnya.
""")
    section("09", """
Simpan checkpoint setiap perlakuan sebagai objek `DetectionModel` FP32 dengan struktur hasil pruning.
Oracle membandingkan output pruned dengan baseline yang diberi mask dari ledger. Fresh reload
di proses Python baru memeriksa struktur, state, dan kesetaraan output kedua cabang Detect dengan
model pruned sebelum disimpan. Manifest raw menyimpan hash checkpoint serta referensi baseline
dan bukti audit.
""")
    add(shared_sources[8], "helpers")
    section("10", """
Evaluasi lima baseline dan tujuh checkpoint raw pada split, daftar citra, dan konfigurasi validator
yang sama. `full_val` memakai seluruh validation Poltrada V3 dengan FP32 dan one-to-many/NMS.
Konfigurasi validator memakai confidence 0,001, IoU NMS 0,70, dan maksimum 300 deteksi per citra.
Status audit struktur, reload, validasi, latency CUDA, EDA, dan publikasi tercatat terpisah.
Hasil baseline seed 42 dipakai sebagai pembanding untuk ketiga target seed 42.
""")
    for source in shared_sources[9:]:
        add(source, "helpers")
    add(multi_runtime, "runtime")
    add('''
BUNDLE = ms_execute(SUITE_DIR)
print(BUNDLE["status"], BUNDLE.get("error") or BUNDLE.get("reason", ""))
''', "run")
    section("11", """
Ringkasan memuat ketujuh percobaan dengan pasangan baseline dan checkpoint raw masing-masing.
Detail dan grafik tersedia untuk setiap perlakuan. `EDA_CONFIGURATIONS` mengatur tampilan overview
dan `EDA_DETAIL_CONFIGURATIONS` mengatur detail. Isi `()` untuk melihat seluruh hasil.
Sel 11–14 bisa dijalankan ulang dari `BUNDLE` untuk menampilkan hasil tersimpan.
""", """
Periksa checkpoint sumber dan hash, jumlah citra, fingerprint, serta mode evaluasi. Tabel cakupan
membedakan group eligible dari graph, kapasitas policy, channel independen, dan perubahan fisik
output Conv. Registry mengikat seed, versi baseline, dan status dari tiap perlakuan.
""")
    display_setup = '''
from IPython.display import Image as NotebookImage, Markdown, display

REGISTRY = BUNDLE.get("registry", {})
REGISTRY_ROWS = REGISTRY.get("rows", [])
if REGISTRY_ROWS:
    try:
        with FileLock(str(SUITE_DIR / ".suite.lock"), timeout=0):
            ANALYSIS = gflops_build_analysis(BUNDLE, SUITE_DIR)
    except Exception as error:
        print("EDA agregat PARTIAL", safe_error(error))
        ANALYSIS = {"status": "PARTIAL", "plots": [], "tables": gflops_eda_tables(BUNDLE, SUITE_DIR)}
else:
    print("Status suite", BUNDLE["status"], BUNDLE.get("error") or BUNDLE.get("reason", ""))
    ANALYSIS = {"status": "NOT_RUN", "plots": [], "tables": gflops_eda_tables(BUNDLE, SUITE_DIR)}
TABLES = ANALYSIS["tables"]
'''
    table_helpers = definitions(template_display["11"], "show_eda_table", "show_eda_images")
    add(display_setup + "\n\n" + table_helpers + '''

show_eda_table(TABLES["sources"], ["model", "baseline_seed", "requested_artifact", "sha256", "source_verification",
                                  "baseline_validation_parity_status"])
show_eda_table(REGISTRY_ROWS, ["configuration_id", "baseline_seed", "source_baseline_artifact_ref", "source_status",
                             "target_speedup", "target_status", "actual_gflops_speedup", "structural_status",
                             "reload_status", "validation_status", "baseline_map50_95", "map50_95",
                             "delta_map50_95_pp", "eda_status", "evaluation_mode",
                             "validation_image_count", "dataset_provenance_status", "dataset_fingerprint"])
show_eda_table(TABLES["architecture"], ["model", "index", "type", "from", "region", "parameters"])
show_eda_table(TABLES["coverage"])
''', "display")
    section("11.2", """
Diagram **Before Pruning** biru, **After Pruning** merah, dan **Channel Comparison** overlay memakai
urutan Conv dan batas sumbu yang sama, termasuk layer dengan channel tetap. Detail membandingkan
channel input dan output, parameter, serta estimasi operasi setiap layer sebelum dan sesudah
pruning. Satu penghapusan channel independen dapat memengaruhi beberapa output Conv yang terikat
oleh residual.
""")
    add("EDA_RUNS = []\n" + template_display["11.2"], "display")
    section("11.3", """
Lihat group yang diterima atau ditolak, rentang skor finite, kapasitas, dan jumlah channel
yang dipilih. Skor `+inf` dari pembatas kapasitas dikeluarkan dari distribusi importance.
Heatmap memakai indeks asli sebelum pruning. Ledger mencatat kolom input Conv konsumen dan
BatchNorm, termasuk offset concat dan seluruh konsumen fan-out. Ekspor per perlakuan
tersedia di `comparison/analysis/groups.csv` dan `comparison/analysis/ledger.csv`.
""")
    add(template_display["11.3"], "display")
    section("12", "", """
Bandingkan parameter, GFLOPs, dan ukuran checkpoint setiap kandidat dengan baseline seed yang
sama yang disimpan ulang sebagai FP32. Ukuran file sumber ditampilkan terpisah karena formatnya dapat berbeda.
Estimasi per layer dan stage mencakup Conv/BN, sedangkan total GFLOPs juga memasukkan attention.
Latency p50 dan p95 memakai raw dual-head forward FP32 batch 1 di luar preprocessing, NMS,
dan I/O. Speed-up latency membagi median baseline dengan median pruned. Jika CUDA belum diukur,
status latency tetap `NOT_RUN`.
""")
    add(template_display["12"], "display")
    section("12.2", """
Delta `mAP50–95` dihitung terhadap baseline dan dilaporkan dalam poin persentase. Perubahan
relatif memakai mAP baseline sebagai denominator dan kosong jika baseline nol. Delta AP per kelas
memakai ID kelas yang sama. Support menghitung jumlah objek ground-truth, dan kelas tanpa metrik
tetap kosong. Tabel dan grafik memakai hasil setiap perlakuan. Ringkasan lintas seed pada 13.3
merangkum delta berpasangan untuk target utama. Angka `smoke` berasal dari subset untuk pemeriksaan alur.
""")
    add(template_display["12.2"], "display")
    section("12.3", """
Kurva dan confusion matrix berasal dari validator aktual. Tampilkan panel baseline sekali per
seed, lalu panel tiap checkpoint raw. Precision, recall, dan F1 per kelas memakai
titik confidence terbaik menurut smoothed mean F1 SDK pada IoU 0,50. Confusion matrix memakai
**baris predicted dan kolom true**, termasuk background. Confidence dan IoU pencocokan confusion
matrix dilaporkan pada metadata panel. Tabel protokol mencatat split, jumlah citra, fingerprint,
presisi, serta pengaturan one-to-many/NMS yang digunakan.
""")
    add('''
show_eda_table(TABLES["validation_protocol"], [
    "model", "configuration_id", "validation_status", "evaluation_mode", "split", "validation_image_count",
    "dataset_fingerprint", "dataset_provenance_status", "imgsz", "batch", "quantize", "conf", "iou", "max_det",
    "nms", "end2end", "augment", "rect", "device",
])
baseline_curves_shown = set()
for model_name, key, result, folder in EDA_DETAILS:
    for stage in ("baseline", "pruned"):
        if stage == "baseline" and model_name in baseline_curves_shown:
            continue
        label = "Baseline" if stage == "baseline" else key
        display(Markdown(f"#### Kurva dan confusion matrix · {model_name} · {label}"))
        metrics = result[stage].get("validation") or {}
        validation_status = result["validation_status"] if stage == "pruned" else result[stage].get("validation_status", "NOT_RUN")
        if stage == "baseline":
            baseline_curves_shown.add(model_name)
        if validation_status != "PASS" or not metrics:
            print(validation_status, "— metrik deteksi belum tersedia.")
            continue
        display(metrics.get("metric_semantics", {}))
        matrix = metrics.get("confusion_matrix", {})
        display({field: matrix.get(field) for field in (
            "status", "labels", "confidence_threshold", "matching_iou_threshold", "validation_iou")})
        show_eda_images(folder / stage, "class_metrics.png", "evaluation_curve_*.png", "confusion_matrix.png")
''', "display")
    section("13", "", """
Enam baris YOLO26s dari tabel historis pengguna disajikan dengan provenance `UNVERIFIED`.
Pada extended 10%, mAP50–95 yang dilaporkan turun dari 0,5612 ke 0,2546. Pada extended 20%,
nilainya di bawah 0,003. Status di tabel berasal dari catatan awal. Artefak run, fingerprint,
protokol evaluator, dan counter pada catatan tersebut masih perlu diverifikasi. Data historis
disimpan terpisah dari grafik hasil suite saat ini.
""")
    historical_csv = next(ast.literal_eval(node.value) for node in ast.parse(template_display["13"]).body
                          if isinstance(node, ast.Assign) and isinstance(node.targets[0], ast.Name)
                          and node.targets[0].id == "HISTORICAL_EXTENDED_CSV")
    add(f"from io import StringIO\nHISTORICAL_EXTENDED_CSV = {historical_csv!r}\n" + '''
historical = pd.read_csv(StringIO(HISTORICAL_EXTENDED_CSV))
historical_rows = historical.loc[historical["model_variant"].eq("s")].to_dict("records")
show_eda_table(historical_rows)
if ANALYSIS["status"] == "PASS":
    with FileLock(str(SUITE_DIR / ".suite.lock"), timeout=0):
        write_csv(SUITE_DIR / "comparison/analysis/historical_unverified.csv", historical_rows)
''', "display")
    section("13.2", """
Kontrol extended berstatus `NOT_RUN`. Tabel tier dan grafik kapasitas menampilkan profil pencarian
setiap seed, termasuk tiga target untuk seed 42. Profil tersimpan di `yolo26s-seed<seed>/search/`.
Pencarian mengukur GFLOPs. Nilai mAP berasal dari validasi kandidat sesudah audit dan reload.
""")
    add(template_display["13.2"].replace('"extended_sweep.png", "target_search_capacity.png"',
                                         '"target_search_capacity.png"'), "display")
    section("13.3", """
Setiap grafik membandingkan kandidat dengan baseline seed yang sama. Bandingkan mAP50–95 terhadap
parameter, GFLOPs, dan median latency CUDA sesuai status pengukuran masing-masing. Ringkasan target
1,20× mencatat capaian tiap seed dan mean serta simpangan baku sampel dari delta berpasangan
yang tersedia. Hitung dan baca hasil parsial sesuai jumlah pasangan dan status pencapaian target.
""")
    add('''
display(Markdown("#### Ringkasan lintas seed · target 1,20×"))
primary_rows = [row for row in REGISTRY_ROWS if row["target_speedup"] == PRIMARY_TARGET]
show_eda_table(primary_rows, ["baseline_seed", "target_status", "actual_gflops_speedup",
                              "baseline_map50_95", "map50_95", "delta_map50_95_pp",
                              "baseline_latency_p50_ms", "latency_p50_ms", "validation_status"])
deltas = pd.Series([row["delta_map50_95_pp"] for row in primary_rows if row["delta_map50_95_pp"] is not None
                    and row["validation_status"] == row["baseline_validation_parity_status"] == "PASS"], dtype=float)
if not deltas.empty:
    display({"mode": BUNDLE["mode"], "pasangan_terukur": int(deltas.count()), "pasangan_direncanakan": len(primary_rows),
             "target_tercapai": sum(row["target_status"] == "REACHED" for row in primary_rows),
             "rerata_delta_mAP50_95_pp": deltas.mean(), "simpangan_baku_sampel_pp": deltas.std() if len(deltas) > 1 else None})
show_eda_table([plot for plot in ANALYSIS["plots"] if plot["plot"] != "extended_sweep"],
               ["model", "plot", "status", "measured_values", "reason"])
for model in BUNDLE["models"]:
    display(Markdown(f"#### Trade-off · {model['model_name']}"))
    show_eda_images(SUITE_DIR / "comparison/analysis" / model["model_name"].lower(), "tradeoff_*.png")
''', "display")
    section("13.4", """
Dalam audit 06, **P5-only vs baseline** memeriksa penghapusan channel P5 pilihan extended 10%.
**P5-protected vs extended 10%** memeriksa alokasi ulang pada jumlah channel independen yang sama.
Kontras itu membaca delta mAP dan AP per kelas bersama perubahan GFLOPs. Distribusi channel
sendiri memberi konteks tentang lokasi pruning. Keduanya berstatus `NOT_RUN` pada suite ini.

Garis baseline −2 poin persentase pada grafik metrik merujuk pada kriteria akurasi di notebook
06 final. Kolom `recovery_ready` pada suite ini menandai kelengkapan audit, full validation,
dan publikasi paket. Gunakan tabel akurasi untuk menilai hasil deteksi. Baris `TARGET_NOT_REACHED`
menunjukkan kandidat maksimum dan membutuhkan pilihan diagnostik eksplisit saat checkpoint terkonfirmasi.
""")
    add('''
print("P5-only", "NOT_RUN", "| P5-protected", "NOT_RUN")
show_eda_table(REGISTRY_ROWS, ["target_speedup", "actual_gflops_speedup", "target_status", "recovery_seed",
                             "source_baseline_artifact_ref", "structural_status", "reload_status",
                             "validation_status", "publication_status", "recovery_ready"])
''', "display")
    section("14", """
Setiap perlakuan punya run dan Model Artifact sendiri dalam satu grup W&B. Nama mengikuti pola
`yolo26s-lamp-su1.20x-seed0-raw`. Metadata menyimpan versi dan SHA-256 baseline, target, capaian
aktual, seed, GFLOPs, parameter, mAP, latency, SHA-256 raw, dan fingerprint dataset. Ringkasan
mencatat reproducibility replay tiap perlakuan. Perbandingan lintas seed memakai lima baseline
berbeda.

`CONFIRMED` berarti semua SHA-256 paket cocok setelah diunduh dari versi tetap `:vN`. Suite baru
selesai saat tujuh baris registry memiliki hasil evaluasi dan referensi W&B terverifikasi.
Pakai versi `:vN` dari registry suite yang dijalankan. Versi `:v0` seed 42 dari desain sebelumnya
tetap tercatat sebagai riwayat W&B.

Untuk melanjutkan, isi `RESUME_SUITE_DIR` dengan folder suite tersimpan. Gunakan `OUTPUT_DIR`
persisten di Filkom. Di Kaggle, salin suite dari `/kaggle/working` ke lokasi yang dapat ditulis
sebelum menjalankan ulang notebook di sesi baru. Kontrak resume mengunci kelima versi dan SHA-256
baseline, fingerprint dataset, implementasi, dan runtime. Pencarian, validasi, EDA, dan progres
publikasi tersimpan per tahap. Tabel CSV/JSON dan grafik PNG/SVG tersedia di `comparison/analysis/`
dan folder `eda/` tiap perlakuan.

Salin sel pemuat registry di bawah ke notebook recovery sesudah konfigurasi dan sebelum bootstrap.
Isi path JSON dan seed recovery. Target default `1.20` memilih eksperimen utama. Ambil versi raw
dan identitas baseline dari baris yang sama. Pasang lock baseline sesudah bootstrap recovery.
Untuk kandidat `TARGET_NOT_REACHED` yang sudah terkonfirmasi, aktifkan `allow_diagnostic=True`.
""")
    add('''
from IPython.display import FileLink

print("Folder suite", SUITE_DIR)
print("Status suite", BUNDLE["status"], "| Status EDA", ANALYSIS["status"])
print("Suite", REGISTRY.get("suite_id", "NOT_RUN"), "| Checkpoint raw", REGISTRY.get("pruned_checkpoint_count", 0),
      "| Artefak terverifikasi", REGISTRY.get("confirmed_artifact_count", 0), "/", len(PRUNING_QUEUE))
for filename in ("suite_manifest.json", "resume_contract.json", "dataset_manifest.json", "environment.json",
                 "checkpoint_registry.csv", "checkpoint_registry.json", "comparison/analysis/analysis_status.json",
                 "comparison/analysis/metrics.csv", "comparison/analysis/per_class_metrics.csv"):
    path = SUITE_DIR / filename
    if path.is_file():
        display(FileLink(Path(os.path.relpath(path)).as_posix()))
show_eda_table(TABLES["verification"])
show_eda_table(REGISTRY_ROWS, ["configuration_id", "pruning_seed", "recovery_seed", "reproducibility_status",
                             "target_status", "checkpoint_relative_path", "raw_sha256", "integrity_status",
                             "source_baseline_artifact_ref", "source_baseline_sha256", "baseline_fp32_sha256",
                             "publication_status", "artifact_ref", "wandb_run_id", "publication_error"])
''', "display")
    add(RECOVERY_CELL, "recovery-input")
    return {"cells": cells, "nbformat": 4, "nbformat_minor": 5, "metadata": {
        "kernelspec": original["metadata"]["kernelspec"], "language_info": original["metadata"]["language_info"],
        "research_workflow": {"platforms": ["kaggle", "filkom"], "model_variant": "s", "stage": "pruning"}}}


if __name__ == "__main__":
    DESTINATION.write_text(json.dumps(build_notebook(), ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
