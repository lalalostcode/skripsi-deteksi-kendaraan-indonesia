"""Generate only recovery KD 03–05, reusing the verified selected-student workflow."""
from __future__ import annotations

import ast
import hashlib
import json
import re
import runpy
from pprint import pformat

if __package__:
    from . import generate_pruned_fine_tuned_notebooks as base
else:
    import generate_pruned_fine_tuned_notebooks as base

METHODS = ("native", "crosskd", "mgd")

# This deployment override applies only to FILKOM MGD. Keep shared templates and
# the other generated notebooks byte-identical to their existing contracts.
FILKOM_MGD_PATHS = '''WORKSPACE_ROOT = Path(os.environ.get("SKRIPSI_ROOT", Path.home() / "skripsi-kd")).expanduser()
REPO_ROOT = Path(os.environ.get("SKRIPSI_REPO_ROOT", WORKSPACE_ROOT / "repo")).expanduser()
DATASET_ROOT = Path(os.environ.get("SKRIPSI_DATASET_ROOT", WORKSPACE_ROOT / "datasets/roboflow-poltrada-v3")).expanduser()
DATA_YAML = Path(os.environ.get("SKRIPSI_DATA_YAML", DATASET_ROOT / "data.yaml")).expanduser()
DATASET_ID = "filkom/roboflow-poltrada-v3"
# Base output only: RECOVERY_METHOD is appended once below.
OUTPUT_DIR = Path(os.environ.get("SKRIPSI_OUTPUT_DIR", WORKSPACE_ROOT / "outputs/filkom/recovery/selected/yolo26s")).expanduser()'''

FILKOM_MGD_PREFLIGHT = '''import sys

if sys.version_info < (3, 11):
    raise RuntimeError(
        "MGD YOLO26s memerlukan Python >=3.11 sesuai pyproject.toml; "
        f"kernel aktif memakai Python {sys.version.split()[0]} ({sys.executable}). "
        "Aktifkan environment HPC dengan Python >=3.11 dan Torch/torchvision CUDA yang sesuai, "
        "daftarkan environment tersebut sebagai kernel Jupyter, lalu pilih kernel itu dan Run All ulang. "
        "Mengganti import datetime saja tidak membuat proyek mendukung Python 3.8."
    )
'''


def selected_pins():
    original = {key: value for key, value in base.STUDENT_PINS.items() if value["target"] == 1.2}
    receipt = base.ROOT / "reports/pruning/lamp_yolo26s_raw_restored_20261107.json"
    if not receipt.exists():
        return original
    restored = json.loads(receipt.read_text(encoding="utf-8"))
    if restored["status"] != "PASS" or len(restored["rows"]) != 5:
        raise ValueError("Pemulihan raw belum terverifikasi untuk kelima seed.")
    pins = {}
    for row in restored["rows"]:
        pin = original[row["historical_input"]]
        name = row["artifact_ref"].rsplit("/", 1)[-1]
        expected_name = row["historical_input"].rsplit(":", 1)[0]
        namespace = base.TEACHER_REF.rsplit("/", 1)[0]
        if (row["status"] != "CONFIRMED" or row["raw_sha256"] != pin["raw_sha256"]
                or row["manifest_sha256"] != pin["manifest_sha256"] or row["seed"] != pin["seed"]
                or row["artifact_ref"].rsplit("/", 1)[0] != namespace
                or row["historical_artifact_ref"] != namespace + "/" + row["historical_input"]
                or not re.fullmatch(re.escape(expected_name) + r":v\d+", name)):
            raise ValueError("Paket raw pulihan berbeda dari pin suite pruning.")
        pins[name] = pin
    return pins


def build_notebook(platform, method):
    if method not in METHODS:
        raise ValueError(method)
    book = base.build_selected_notebook(platform, restore_inputs=False)
    config, training, evaluation = base.source_contract(method)
    config.pop("SEED")
    pins = selected_pins()
    by_tag = {cell["metadata"]["tags"][0]: cell for cell in book["cells"] if cell["cell_type"] == "code"
              and cell["metadata"]["tags"][0] != "helpers"}

    def put(cell, source):
        ast.parse(source) if cell["cell_type"] == "code" else None
        cell["source"] = source.strip().splitlines(keepends=True)
        cell["id"] = hashlib.sha256(source.encode()).hexdigest()[:8]

    source = "".join(by_tag["config"]["source"])
    source = source.replace('RECOVERY_METHOD = "finetune"', f"RECOVERY_METHOD = {method!r}")
    source = re.sub(r"SOURCE_NOTEBOOK = .*", f"SOURCE_NOTEBOOK = {base.METHOD_NOTEBOOKS[method][1].relative_to(base.ROOT).as_posix()!r}", source)
    if (platform, method) == ("filkom", "mgd"):
        source = re.sub(r"DATASET_ROOT = .*\nDATA_YAML = .*\nOUTPUT_DIR = .*", FILKOM_MGD_PATHS, source)
    # The checked-in FILKOM MGD notebook selects seed 0; regeneration must retain it.
    selected_seed = 0 if (platform, method) == ("filkom", "mgd") else 42
    choices = "STUDENT_INPUTS = [\n" + "\n".join(
        f'    {"" if pin["seed"] == selected_seed else "# "}"{name}",' for name, pin in pins.items()) + "\n]"
    source = re.sub(r"STUDENT_INPUTS = \[.*?\n\]", choices, source, flags=re.DOTALL)
    for key, value in config.items():
        if re.search(rf"^{key} = ", source, re.MULTILINE):
            source = re.sub(rf"^{key} = .*", f"{key} = {value}", source, flags=re.MULTILINE)
        else:
            source += f"\n{key} = {value}"
    source += "\n" + base.TEACHER_CONFIG
    source += f"\nLOCKED_TEACHER_REF = {base.TEACHER_REF!r}\nLOCKED_TEACHER_SHA256 = {base.TEACHER_SHA!r}\n"
    source += "NATIVE_DIS_WEIGHT = KD_WEIGHT if RECOVERY_METHOD == 'native' else 1.0\n"
    source += "OUTPUT_DIR = OUTPUT_DIR / RECOVERY_METHOD\n"
    put(by_tag["config"], source)
    selection = "".join(by_tag["selection"]["source"])
    begin, end = selection.index("STUDENT_PINS ="), selection.index("if len(STUDENT_INPUTS)")
    put(by_tag["selection"], selection[:begin] + "STUDENT_PINS = " + pformat(pins, sort_dicts=False, width=110) + "\n" + selection[end:])
    put(by_tag["protocol"], f'''def fine_tuned_train_args(DATA, PROJECT, RUN_NAME, teacher_path=None):
    args = {base.kwargs_expression(training)}
    if MODE == "smoke":
        args.update(epochs=SMOKE_EPOCHS, amp=False, nbs=BATCH)
    return args


def fine_tuned_val_args(DATA):
    return {base.kwargs_expression(evaluation)}
''')
    bootstrap = "".join(by_tag["bootstrap"]["source"])
    bootstrap = bootstrap.replace('STUDENT_INPUTS[0].split(":")[0] + f"_', 'STUDENT_INPUTS[0].split(":")[0] + "-" + RECOVERY_METHOD + f"_')
    if (platform, method) == ("filkom", "mgd"):
        # Fail before output creation, dependency inspection/install, or model imports.
        bootstrap = FILKOM_MGD_PREFLIGHT + "\n" + bootstrap
    put(by_tag["bootstrap"], bootstrap)
    source = "".join(by_tag["runtime"]["source"]).split("\nIMPLEMENTATION_SHA256 =")[0]
    source = source.replace("selected-raw-finetune-v2", "selected-raw-kd-v1")
    source = source.replace("fine_tuned_manifest.json", "kd_manifest.json").replace("finetuning_registry", "kd_registry")
    source = source.replace(', method="finetune"', "").replace('job_type="finetune"', 'job_type="kd"')
    source = source.replace('["yolo26s", "lamp", "finetune",', '["yolo26s", "lamp", "kd",')
    dependencies = runpy.run_path(str(base.SCRIPTS / "recovery_kd_method_dependencies.py"))["method_dependencies"]
    extra = "\n\n".join((*dependencies(base.ROOT, method, source_matched=True),
        (base.SCRIPTS / "recovery_kd_methods_source.py").read_text(encoding="utf-8"),
        base.definitions("source_matched_kd_source.py", "source_matched_teacher"),
        (base.SCRIPTS / "pruned_kd_selected_source.py").read_text(encoding="utf-8")))
    put(by_tag["runtime"], source + "\n\n" + extra)
    if (platform, method) == ("filkom", "mgd"):
        dataset_policy = (base.SCRIPTS / "filkom_mgd_dataset_source.py").read_text(encoding="utf-8").strip()
        put(by_tag["runtime"], "".join(by_tag["runtime"]["source"]) + "\n\n" + dataset_policy)
    for tag in ("display", "run"):
        put(by_tag[tag], "".join(by_tag[tag]["source"]).replace("fine_tuned_manifest.json", "kd_manifest.json")
            .replace("finetuning_registry", "kd_registry"))
    if (platform, method) == ("filkom", "mgd"):
        put(by_tag["run"], '''print(
    f"MGD | mode={MODE} | seed={SEED} | target_epochs={SMOKE_EPOCHS if MODE == 'smoke' else EPOCHS} | run={RUN_DIR}",
    flush=True,
)
RESULT = ft_execute(RUN_DIR)
print(RESULT["status"], RESULT.get("reason", ""), "|", RUN_DIR, flush=True)
if RESULT.get("status") != "PASS":
    errors = RESULT.get("errors") or []
    failed_stage = errors[-1].get("stage", "unknown") if errors else "unknown"
    reason = RESULT.get("reason") or "Baca errors pada kd_manifest.json untuk detail."
    raise RuntimeError(
        f"MGD {RESULT.get('status', 'UNKNOWN')} pada tahap {failed_stage}: {reason} "
        f"Manifest: {RUN_DIR / 'kd_manifest.json'}"
    )''')
    put(by_tag["group-display"], '''if LOAD_GROUP_RESULTS:
    group_frame = pd.DataFrame(ft_group_results())
    columns = ["method", "seed", "target_speedup", "actual_gflops_speedup", "platform", "gpu",
               "epochs_completed", "best_epoch", "baseline_map50_95", "raw_pruned_map50_95",
               "recovered_map50_95", "recovered_map50", "recovered_precision", "recovered_recall",
               "checkpoint_artifact_ref", "wandb_run_url"]
    display(group_frame[[name for name in columns if name in group_frame]])
else:
    print("Aktifkan LOAD_GROUP_RESULTS untuk ringkasan 15 perlakuan KD.")''')
    markdown = [cell for cell in book["cells"] if cell["cell_type"] == "markdown"]
    descriptions = [
        f'''# Recovery YOLO26s · {base.METHOD_LABELS[method]} · {platform.title()}

Pilih satu raw LAMP su1.20x dari lima seed melalui comment/uncomment. Bobot, struktur, dan seed
student diwarisi dari raw. Teacher bersama adalah YOLO26m v50, dengan versi serta SHA-256 terkunci.
Satu sesi menjalankan satu perlakuan; gunakan kernel dan DEVICE masing-masing untuk paralel.
Default 100 epoch mengikuti konfigurasi metode asal. Evaluasi membandingkan baseline asal,
raw pruned, dan student setelah KD pada validation Poltrada V3 yang sama.''',
        '''## 01 — Pilih student, teacher, perangkat, dan output

Aktifkan tepat satu STUDENT_INPUTS. Target hanya 1.20x, seed 42/0/2024/7/3407. Model student
tetap memakai objek native pruned. Seed augmentasi/dataloader mengikuti identitas manifest.
Teacher YOLO26m v50 dipakai untuk seluruh seed. WANDB_API_KEY memakai environment atau Kaggle Secret.''',
        '''## 02 — Kernel dan verifikasi input

Periksa versi dependency, hash raw, baseline, manifest, audit, serta teacher sebelum training.
Audit teacher memeriksa arsitektur M, kelas, kedua head, mode eval, dan optimizer tanpa teacher.
Kontrak resume mengikat metode, parameter KD, teacher, student, dataset, dan runtime.''',
        '''## 03 — KD dan logging epoch

Loss mengikuti metode asal dengan adapter untuk lebar fitur pruned. Teacher tetap frozen.
Snapshot lengkap lokal setiap epoch mencakup student/EMA, adapter dan gradien tertunda,
optimizer, scheduler, scaler, criterion, RNG, serta dataloader. Unggah tiap lima epoch dan akhir.
Best/last inference memuat student native FP32. Resume memakai folder lokal atau artefak versi tetap.''',
        '''## 04 — EDA custom sebelum pruning, raw pruned, dan setelah KD

Tampilkan mAP/P/R, AP/P/R/F1/support per kelas, parameter, GFLOPs, channel, ukuran checkpoint,
latency p50/p95, delta dan gain. Pertahankan diagram channel, ledger, heatmap indeks, LAMP,
PR/F1, confusion matrix, kurva deteksi/KD/LR, dan trade-off. Seluruh CSV menjadi W&B Tables,
grafik menjadi Images, dengan identitas seed, target, capaian, metode, dan teacher.
Angka yang belum terukur tetap mempunyai status eksplisit.''',
        '''## 05 — Ringkasan 15 perlakuan lintas metode dan seed

LOAD_GROUP_RESULTS membaca hasil terkonfirmasi dari tiga metode × lima seed. Tampilkan
ketersediaan parsial, tabel per seed, dan mean/std tiap metode. Platform, perangkat latency,
teacher, serta konfigurasi training harus sebanding. Hasil smoke berada pada grup terpisah.''',
    ]
    if (platform, method) == ("filkom", "mgd"):
        descriptions[2] += '''

Dataset memakai path lokal FILKOM. Fingerprint train/validation dicatat tanpa kewajiban sama
dengan dataset asal raw pruning. Urutan kelas dan pemisahan train/validation tetap diperiksa.
Baseline, raw, dan hasil MGD dievaluasi ulang pada validation lokal yang sama. Perbedaan dataset
asal ditandai pada manifest; ringkasan lintas run hanya memilih fingerprint lokal yang cocok.'''
    for cell, text in zip(markdown, descriptions, strict=True):
        put(cell, text)
    if (platform, method) == ("filkom", "mgd"):
        # The shared bootstrap and recovery training helper both import datetime.UTC.
        # Normalize the embedded copies only, preserving the other platform/methods.
        for cell in book["cells"]:
            if cell["cell_type"] == "code":
                source = "".join(cell["source"])
                portable = source.replace("from datetime import UTC, datetime",
                                          "from datetime import datetime, timezone\nUTC = timezone.utc")
                portable = portable.replace("from datetime import UTC as _RECOVERY_UTC",
                                            "from datetime import timezone as _RecoveryTimezone\n_RECOVERY_UTC = _RecoveryTimezone.utc")
                if portable != source:
                    put(cell, portable)
    implementation = hashlib.sha256("\n".join("".join(c["source"]) for c in book["cells"]
        if c["cell_type"] == "code" and c["metadata"]["tags"][0] not in {"run", "display", "group-display"}).encode()).hexdigest()
    put(by_tag["runtime"], "".join(by_tag["runtime"]["source"]) + f"\nIMPLEMENTATION_SHA256 = {implementation!r}")
    book["metadata"]["research_workflow"].update(method=method, protocol="selected-raw-kd-v1")
    if __package__:
        from .recovery_custom_eda import attach_custom_eda
        from .recovery_postprocess import attach_postprocessing
    else:
        from recovery_custom_eda import attach_custom_eda
        from recovery_postprocess import attach_postprocessing
    return attach_custom_eda(attach_postprocessing(book, base.ROOT), base.ROOT)


if __name__ == "__main__":
    for platform in ("kaggle", "filkom"):
        for method in METHODS:
            path = base.notebook_path(platform, base.METHOD_NOTEBOOKS[method][0])
            path.write_text(json.dumps(build_notebook(platform, method), ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
            print(path)
