"""Generate the two standalone YOLO26s fine-tuning notebooks from the baseline contract."""
from __future__ import annotations

import ast
import hashlib
import json
import runpy
from pprint import pformat
from textwrap import indent

if __package__:
    from .notebook_paths import ROOT, notebook_path
else:
    from notebook_paths import ROOT, notebook_path

SCRIPTS = ROOT / "scripts"
BASELINE = notebook_path("kaggle", "02_baseline_yolo26s.ipynb")
PRUNING = notebook_path("kaggle", "06_lamp_yolo26s_gflops_no_recovery.ipynb")
CONFIG_KEYS = ("EPOCHS", "IMGSZ", "BATCH", "SEED", "DEVICE", "WORKERS", "OPTIMIZER", "LR", "PATIENCE", "AMP",
               "WEIGHT_DECAY", "WARMUP_EPOCHS", "CLOSE_MOSAIC", "EVAL_SPLIT", "WANDB_ENTITY", "WANDB_PROJECT")
METHOD_NOTEBOOKS = {
    "finetune": ("02_yolo26s_standard_finetuning.ipynb", BASELINE),
    "native": ("03_yolo26ns_native_kd.ipynb", notebook_path("kaggle", "04_native_yolo26s.ipynb")),
    "crosskd": ("04_yolo26ns_crosskd.ipynb", notebook_path("kaggle", "06_crosskd_yolo26s.ipynb")),
    "mgd": ("05_yolo26ns_mgd_kd.ipynb", notebook_path("kaggle", "15_mgd_yolo26n_teacher_yolo26s.ipynb")),
}
METHOD_LABELS = {"finetune": "Fine-tuning standar", "native": "Native KD", "crosskd": "CrossKD", "mgd": "MGD"}
TEACHER_REF = "ilhamrafiqin-brawijaya-university/yolom26-kd-indotraffic/01_teacher_yolo26m-fux5wdcb-checkpoints:v50"
TEACHER_SHA = "d749ef0baa0d6c3ae94e83ded1d9999d83c804b76ad5a841fc2ebec953af1428"
TEACHER_CONFIG = f'''TEACHER_ARTIFACT_REF = {TEACHER_REF!r}
TEACHER_LOCAL_PATH = None  # Kosongkan TEACHER_ARTIFACT_REF jika memakai salinan lokal.
TEACHER_SHA256 = {TEACHER_SHA!r}
'''


def definitions(file, *names):
    source = (SCRIPTS / file).read_text(encoding="utf-8")
    return "\n\n".join(ast.unparse(node) for node in ast.parse(source).body
                        if isinstance(node, (ast.FunctionDef, ast.ClassDef)) and node.name in names)


def notebook_code(path):
    return [ast.parse("".join(cell["source"])) for cell in json.loads(path.read_text(encoding="utf-8"))["cells"]
            if cell["cell_type"] == "code"]


def source_contract(method):
    path = METHOD_NOTEBOOKS[method][1]
    source = notebook_code(path)
    nodes = [node for cell in source for node in cell.body if isinstance(node, ast.Assign)]
    values = {target.id: node.value for node in nodes for target in node.targets if isinstance(target, ast.Name)}
    config = {key: ast.unparse(values[key]) for key in CONFIG_KEYS}
    method_config = {"finetune": (), "native": ("KD_WEIGHT",), "crosskd": ("KD_CLS_WEIGHT", "KD_BOX_WEIGHT", "KD_CLS_GAMMA"),
                     "mgd": ("MGD_ALPHA", "MGD_MASK_RATIO")}[method]
    config.update({key: ast.unparse(values[key]) for key in method_config})
    if method == "mgd":
        config["SEED"] = ast.unparse(values["SEEDS_TO_RUN"].elts[0])
        training = next(node.value for cell in source for node in ast.walk(cell)
                        if isinstance(node, ast.Return) and isinstance(node.value, ast.Call)
                        and any(kw.arg == "distill_model" for kw in node.value.keywords))
        validation = next(node.value for cell in source for node in ast.walk(cell)
                          if isinstance(node, ast.Assign) and isinstance(node.value, ast.Call)
                          and isinstance(node.value.func, ast.Attribute) and node.value.func.attr == "val")
        # Paths and seed are supplied by the shared runner.
        args = {kw.arg: ast.unparse(kw.value) for kw in training.keywords}
        args.update(project="PROJECT", name="RUN_NAME", seed="SEED", epochs="EPOCHS", plots="True")
    else:
        args = {kw.arg: ast.unparse(kw.value) for kw in values["train_args"].keywords}
        if method != "finetune":
            args.update({key.value: ast.unparse(value) for key, value in
                         zip(values["KD_TRAIN_ARGS"].keys, values["KD_TRAIN_ARGS"].values, strict=True)})
        validation = values["metrics"]
    if method != "finetune":
        args["distill_model"] = "str(teacher_path) if teacher_path else None"
    val_args = {kw.arg: ast.unparse(kw.value) for kw in validation.keywords if kw.arg not in {"project", "name"}}
    return config, args, val_args


def kwargs_expression(args):
    return "dict(" + ", ".join(f"{key}={value}" for key, value in args.items()) + ")"


def build_legacy_notebook(platform, method="finetune"):
    if platform not in {"kaggle", "filkom"} or method not in METHOD_NOTEBOOKS:
        raise ValueError((platform, method))
    config_values, training, evaluation = source_contract(method)
    method_config = "\n".join(f"{key} = {value}" for key, value in config_values.items())
    if platform == "kaggle":
        paths = '''DATASET_ROOT = Path("/kaggle/input/datasets/ilhamrafiqin/poltrada-v3-yolo26-clean")
OUTPUT_DIR = Path("/kaggle/working/recovery/source-matched/yolo26s") / RECOVERY_METHOD'''
    else:
        paths = '''WORKSPACE_ROOT = Path(os.environ.get("SKRIPSI_ROOT", Path.cwd())).expanduser()
DATASET_ROOT = WORKSPACE_ROOT / "datasets/poltrada-v3"
OUTPUT_DIR = WORKSPACE_ROOT / "outputs/filkom/recovery/source-matched/yolo26s" / RECOVERY_METHOD'''
    config = f'''import os
from pathlib import Path

PLATFORM = {platform!r}
RECOVERY_METHOD = {method!r}
NOTEBOOK_VARIANT = 's'
SOURCE_NOTEBOOK = {METHOD_NOTEBOOKS[method][1].relative_to(ROOT).as_posix()!r}
{paths}
DATA_YAML = DATASET_ROOT / "data.yaml"
RAW_ARTIFACT_REF = None   # "entity/project/raw-artifact:vN"
RAW_MANIFEST_PATH = None  # Path(".../raw_pruned_manifest.json"), alternatif input lokal
EXPECTED_RAW_SHA256 = None  # Opsional, SHA-256 raw dari student lock untuk pairing KD
RESUME_SUITE_DIR = None    # Protokol ini mulai baru, resume belum terverifikasi.
INSTALL_DEPENDENCIES = True
WANDB_TRACKING = True
RUN_LATENCY_BENCHMARK = True
MODE = "full_val"  # "smoke" untuk pemeriksaan satu epoch, AMP nonaktif dan subset val

# Pilih satu SEED per run. Gunakan seed yang sama pada semua metode pembanding.
{method_config}
'''
    if method != "finetune":
        config += "\n" + TEACHER_CONFIG
    protocol = f'''def fine_tuned_train_args(DATA, PROJECT, RUN_NAME, teacher_path=None):
    args = {kwargs_expression(training)}
    if MODE == "smoke":
        args.update(epochs=1, amp=False, nbs=BATCH)
    return args


def fine_tuned_val_args(DATA):
    return {kwargs_expression(evaluation)}
'''
    cells = []

    def add(source, *, tag=None):
        source = source.strip()
        item = {"cell_type": "code" if tag else "markdown", "source": source.splitlines(keepends=True),
                "metadata": {}, "id": hashlib.sha256(f"{len(cells)}:{source}".encode()).hexdigest()[:8]}
        if tag:
            ast.parse(source)
            item.update(execution_count=None, outputs=[], metadata={"tags": [tag]})
        cells.append(item)

    def guarded(source, tag="helpers"):
        add("if (RAW_ARTIFACT_REF or RAW_MANIFEST_PATH) and not RESUME_SUITE_DIR:\n" + indent(source.strip(), "    "), tag=tag)

    adaptation = " MGD mempertahankan loss dan konfigurasi asal dengan pasangan YOLO26m → YOLO26s pruned." if method == "mgd" else ""
    add(f"""# Recovery YOLO26s · {METHOD_LABELS[method]} · {platform.title()}

Mulai dari satu student raw LAMP + DepGraph dari notebook pruning 06. Konfigurasi berasal dari `{METHOD_NOTEBOOKS[method][1].relative_to(ROOT).as_posix()}`.{adaptation}
Evaluasi baseline, raw, dan recovered pada validation yang sama. Gunakan versi dan SHA-256 raw yang sama untuk pairing. Dengan input kosong notebook berstatus `NOT_RUN`. Resume berstatus `RESUME_NOT_VERIFIED`.""")
    add("## 01 — Input student dan konfigurasi asal")
    add(config, tag="config")
    add(protocol, tag="protocol")
    add("## 02 — Kernel, verifikasi checkpoint, dan fingerprint train/val")
    guarded((SCRIPTS / "pruned_fine_tuned_bootstrap_source.py").read_text(encoding="utf-8"), "bootstrap")
    guarded("\n\n".join((
        definitions("lamp_workflow_source.py", "write_json", "write_csv", "workflow_digest", "workflow_bind",
                    "workflow_login", "workflow_version_ref", "workflow_run_name", "workflow_tracking",
                    "WorkflowUploadPending", "workflow_publish", "workflow_dataset_manifest"),
        definitions("lamp_gflops_evaluation_source.py", "_gflops_split_images", "_gflops_fingerprint",
                    "gflops_prepare_dataset", "gflops_disable_label_cache", "gflops_require_complete_nms"),
        definitions("lamp_v2_runtime_source.py", "model_state_sha256", "safe_error"),
        definitions("lamp_recovery_training_source.py", "recovery_ordered_names", "recovery_structure_signature",
                    "recovery_state_sha256", "_recovery_finite_state"),
        "ordered_names = recovery_ordered_names",
        definitions("standard_finetuning_runtime_source.py", "recovery_read_json", "recovery_raw_reference"),
    )))
    add("## 03 — Trainer native, full validation dan EDA")
    eda = (SCRIPTS / "standard_finetuning_eda_source.py").read_text(encoding="utf-8").split("def recovery_plot_stage_bars", 1)[0]
    measurements = [ast.unparse(node) for cell in notebook_code(PRUNING) for node in cell.body
                    if isinstance(node, ast.FunctionDef) and node.name in {"model_measurements", "cuda_latency"}]
    guarded("\n\n".join((
        definitions("lamp_v2_validation_source.py", "_validation_number", "_validation_array", "_validation_json_array",
                    "_validation_per_class", "_validation_curves", "_validation_confusion_matrix"),
        definitions("lamp_v2_eda_source.py", "_eda_finite", "_eda_save_figure", "_eda_write_csv", "_eda_write_json", "layer_eda_rows",
                    "stage_eda_rows", "make_channel_figure", "_eda_channel_plots", "_eda_mapping_plots", "_eda_lamp_plots",
                    "_eda_curve_plots"), eda,
        definitions("standard_finetuning_eda_source.py", "_recovery_epoch_plots", "recovery_log_comparison",
                    "_recovery_pruning_diagnostics", "_recovery_stage_validation_plots"),
        *measurements,
        (SCRIPTS / "pruned_fine_tuned_source.py").read_text(encoding="utf-8"),
    )))
    if method != "finetune":
        dependencies = runpy.run_path(str(SCRIPTS / "recovery_kd_method_dependencies.py"))["method_dependencies"]
        guarded("\n\n".join((*dependencies(ROOT, method, source_matched=True),
            f"LOCKED_TEACHER_REF = {TEACHER_REF!r}\nLOCKED_TEACHER_SHA256 = {TEACHER_SHA!r}",
            definitions("recovery_kd_methods_source.py", "recovery_kd_head", "recovery_kd_teacher_head"),
            (SCRIPTS / "source_matched_kd_source.py").read_text(encoding="utf-8"))))
    add("## 04 — Jalankan fine-tuning dan tampilkan hasil")
    add('''from IPython.display import display, Image

if RESUME_SUITE_DIR:
    RESULT = {"status": "NOT_RUN", "resume_status": "RESUME_NOT_VERIFIED", "reason": "Mulai run baru. Resume protokol ini belum terverifikasi."}
elif RAW_ARTIFACT_REF or RAW_MANIFEST_PATH:
    RESULT = fine_tuned_execute(RUN_DIR)
else:
    RESULT = {"status": "NOT_RUN", "reason": "Isi RAW_ARTIFACT_REF atau RAW_MANIFEST_PATH setelah student tersedia."}
print(RESULT["status"], "|", RESULT.get("reason", RESULT.get("output", "")))
if RESULT.get("stages"):
    display(pd.DataFrame(fine_tuned_comparison(RESULT)[0]))
    for figure in sorted((RUN_DIR / "eda").rglob("*.png")):
        display(Image(filename=str(figure)))''', tag="run")
    return {"cells": cells, "nbformat": 4, "nbformat_minor": 5, "metadata": {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "research_workflow": {"platform": platform, "variant": "s", "protocol": "source-matched-v1",
                              "stage": "recovery", "method": method}}}


STUDENT_PINS = {
    "yolo26s-lamp-su1.10x-seed42-raw:v0": {
        "seed": 42,
        "target": 1.1,
        "raw_sha256": "913672ebac25afdebc09957659006506562b11cc56b7771a81b473a0041fc427",
        "baseline_ref": "ilhamrafiqin-brawijaya-university/yolom26-kd-indotraffic/02_baseline_yolo26s-q5kp9k9d-checkpoints:v62",
        "baseline_source_sha256": "5bc3ae61f8da1c9f03916922de10146805c27c64a11e46edd766a00d0f6f18fd",
        "baseline_fp32_sha256": "48733cbdb927ea6dde0985ec7a006cbd8f92d1b84def45d927e3c6c86fab5d50",
        "manifest_sha256": "76ac3d0dfc46d6144a9dc37dcdd5ef91a2d7ce98816ace39eba26bc170f42806"
    },
    "yolo26s-lamp-su1.20x-seed42-raw:v0": {
        "seed": 42,
        "target": 1.2,
        "raw_sha256": "1c20b10f3a33de89377e1cf80e1ff244489fb9fc737b121735cd6d24618d7159",
        "baseline_ref": "ilhamrafiqin-brawijaya-university/yolom26-kd-indotraffic/02_baseline_yolo26s-q5kp9k9d-checkpoints:v62",
        "baseline_source_sha256": "5bc3ae61f8da1c9f03916922de10146805c27c64a11e46edd766a00d0f6f18fd",
        "baseline_fp32_sha256": "48733cbdb927ea6dde0985ec7a006cbd8f92d1b84def45d927e3c6c86fab5d50",
        "manifest_sha256": "2ed9f4eec2143262ee0a75749edaf58fae077ffad965eeb3e00fdfee0a2724df"
    },
    "yolo26s-lamp-su1.70x-seed42-raw:v0": {
        "seed": 42,
        "target": 1.7,
        "raw_sha256": "e181881b8e5bbd4dc021f6dfad54c579361bcdc4a908724247087204a155f026",
        "baseline_ref": "ilhamrafiqin-brawijaya-university/yolom26-kd-indotraffic/02_baseline_yolo26s-q5kp9k9d-checkpoints:v62",
        "baseline_source_sha256": "5bc3ae61f8da1c9f03916922de10146805c27c64a11e46edd766a00d0f6f18fd",
        "baseline_fp32_sha256": "48733cbdb927ea6dde0985ec7a006cbd8f92d1b84def45d927e3c6c86fab5d50",
        "manifest_sha256": "602bc7e4d3cded6c4abf1f49a1e105176f5bb925817ae8e9c413353ffd6f73af"
    },
    "yolo26s-lamp-su1.20x-seed0-raw:v0": {
        "seed": 0,
        "target": 1.2,
        "raw_sha256": "99df514946a6a283e91d90ba22c8c159e6fd107baba739f8860a20c79af2e8aa",
        "baseline_ref": "ilhamrafiqin-brawijaya-university/yolom26-kd-indotraffic/02_baseline_yolo26s-7f5jmtwb-checkpoints:v71",
        "baseline_source_sha256": "d02d8d90d5000fc8379379acb0c0cb3cde0c01143b197d237b25be1bae99bf4f",
        "baseline_fp32_sha256": "fb811a61b070580b13c5a7908b93898d4e88c96bd3e82df192227699e275bac1",
        "manifest_sha256": "7a9597a815e75b5e5909de3f281c14f69ec5bad6ee9e5eee6aa1fdc479963a05"
    },
    "yolo26s-lamp-su1.20x-seed2024-raw:v0": {
        "seed": 2024,
        "target": 1.2,
        "raw_sha256": "3187dde6e33139a1e4c1afd5a6fbc2235d05cd0cc676be1b494bc2d08e45fcf6",
        "baseline_ref": "ilhamrafiqin-brawijaya-university/yolom26-kd-indotraffic/02_baseline_yolo26s-1o9ggnyw-checkpoints:v41",
        "baseline_source_sha256": "47cfab586889d2b464c882239c86224b645363a5a4fa3badc31c46991ad27d70",
        "baseline_fp32_sha256": "e3b464a21029166408bd2006eb388b2fc1399f0d25fc9172307df0aeef604742",
        "manifest_sha256": "356bbd6a24b1580beeb17c969f1f810f368fd2fb75e23b270ef16948d6a6356a"
    },
    "yolo26s-lamp-su1.20x-seed7-raw:v0": {
        "seed": 7,
        "target": 1.2,
        "raw_sha256": "69ed0b860c79626ab435fa06afa1fe4fcf57e4c1478aec498713b4970430cf4a",
        "baseline_ref": "ilhamrafiqin-brawijaya-university/yolom26-kd-indotraffic/02_baseline_yolo26s-9cq437l3-checkpoints:v49",
        "baseline_source_sha256": "3e59d5f5302fb524da729e139ef7e9b8456c08d0952c19b08af19c7f39bce6ab",
        "baseline_fp32_sha256": "7c3224a2ecdf113c0a0279d59241082565029a1af76d23d78c155dc1e8c7c0cc",
        "manifest_sha256": "4a9b1f98e5fd200d4a6a3953ca0658f02630be22204ca36b692062f064937581"
    },
    "yolo26s-lamp-su1.20x-seed3407-raw:v0": {
        "seed": 3407,
        "target": 1.2,
        "raw_sha256": "9a6e64cdf17fb64dfb38082e7f5231a59cf4d6b32f01f05a14aaaf937667d455",
        "baseline_ref": "ilhamrafiqin-brawijaya-university/yolom26-kd-indotraffic/02_baseline_yolo26s-bmshdzv9-checkpoints:v43",
        "baseline_source_sha256": "de9f06892953a343629a4b3a59df880d771973ae95160dee02cd96e37de59e46",
        "baseline_fp32_sha256": "f1dd45dcaec079d67c90ca1c60da50be272bda959184928851578076f2ebb967",
        "manifest_sha256": "8fa0b9acf87681f48c4130fbff662555717b5c29175522b183f88510cedd1e11"
    }
}


def build_selected_notebook(platform, *, restore_inputs=True):
    if platform not in {"kaggle", "filkom"}:
        raise ValueError(platform)
    config, train_args, val_args = source_contract("finetune")
    config.pop("SEED")
    root = 'Path("/kaggle/working")' if platform == "kaggle" else 'Path(os.environ.get("SKRIPSI_ROOT", Path.cwd())) / "outputs/filkom"'
    dataset = ('Path("/kaggle/input/datasets/ilhamrafiqin/poltrada-v3-yolo26-clean")' if platform == "kaggle"
               else 'Path(os.environ.get("SKRIPSI_ROOT", Path.cwd())) / "datasets/poltrada-v3"')
    configuration = f'''import os
from pathlib import Path

PLATFORM = {platform!r}
RECOVERY_METHOD = "finetune"
NOTEBOOK_VARIANT = "s"
SOURCE_NOTEBOOK = {BASELINE.relative_to(ROOT).as_posix()!r}
DATASET_ROOT = {dataset}
DATA_YAML = DATASET_ROOT / "data.yaml"
OUTPUT_DIR = {root} / "recovery/selected/yolo26s"

# Aktifkan tepat satu baris. Identitas seed dan bobot diwarisi dari raw.
STUDENT_INPUTS = [
    # "yolo26s-lamp-su1.10x-seed42-raw:v0",
    "yolo26s-lamp-su1.20x-seed42-raw:v0",
    # "yolo26s-lamp-su1.70x-seed42-raw:v0",
    # "yolo26s-lamp-su1.20x-seed0-raw:v0",
    # "yolo26s-lamp-su1.20x-seed2024-raw:v0",
    # "yolo26s-lamp-su1.20x-seed7-raw:v0",
    # "yolo26s-lamp-su1.20x-seed3407-raw:v0",
]
RESUME_RUN_DIR = None  # Folder run tersimpan yang dapat ditulis.
RESUME_SNAPSHOT_REF = None  # Alternatif: entity/project/<nama-run>-resume:vN.
LOCAL_RAW_PACKAGES = {{}}  # Opsional, pemetaan nama raw ke folder paket terverifikasi.
MODE = "full_val"
SMOKE_EPOCHS = 1
SMOKE_DATASET_ROOT = None  # Opsional khusus smoke sintetis dengan urutan kelas yang sama.
RESUME_UPLOAD_INTERVAL = 5
WANDB_TRACKING = True
RUN_LATENCY_BENCHMARK = True
INSTALL_DEPENDENCIES = True
EDA_DPI = 150
EDA_MAX_LAYERS_PER_PAGE = 32
LOAD_GROUP_RESULTS = False  # Aktifkan sel terakhir untuk membaca hasil grup W&B.
{chr(10).join(f"{key} = {value}" for key, value in config.items())}
'''
    pins = "PRUNING_SUITE_ID = '20261107T020459Z_45575a7f'\nSTUDENT_PINS = " + pformat(STUDENT_PINS, sort_dicts=False, width=110) + '''
if len(STUDENT_INPUTS) != 1 or STUDENT_INPUTS[0] not in STUDENT_PINS:
    raise ValueError("Aktifkan tepat satu student raw pada STUDENT_INPUTS.")
SEED = STUDENT_PINS[STUDENT_INPUTS[0]]["seed"]
if RESUME_RUN_DIR and RESUME_SNAPSHOT_REF:
    raise ValueError("Pilih satu sumber resume.")
if RESUME_UPLOAD_INTERVAL < 1 or SMOKE_EPOCHS < 1:
    raise ValueError("Interval upload dan budget smoke harus positif.")
'''
    protocol = f'''def fine_tuned_train_args(DATA, PROJECT, RUN_NAME, teacher_path=None):
    args = {kwargs_expression(train_args)}
    if MODE == "smoke":
        args.update(epochs=SMOKE_EPOCHS, amp=False, nbs=BATCH)
    return args


def fine_tuned_val_args(DATA):
    return {kwargs_expression(val_args)}
'''
    bootstrap = (SCRIPTS / "pruned_fine_tuned_bootstrap_source.py").read_text(encoding="utf-8")
    begin, end = bootstrap.index("if bool(RAW_ARTIFACT_REF)"), bootstrap.index("PINS =")
    bootstrap = bootstrap[:begin] + '''if RESUME_RUN_DIR:
    RUN_DIR = Path(RESUME_RUN_DIR).expanduser().resolve(strict=True)
    if not (RUN_DIR / "resume_contract.json").is_file():
        raise ValueError("Folder resume belum memiliki kontrak run.")
else:
    RUN_DIR = OUTPUT_DIR.expanduser().resolve() / (
        STUDENT_INPUTS[0].split(":")[0] + f"_{datetime.now(UTC):%Y%m%dT%H%M%SZ}_{uuid.uuid4().hex[:8]}")
    RUN_DIR.mkdir(parents=True, exist_ok=False)

''' + bootstrap[end:]
    bootstrap = "\n".join(line for line in bootstrap.splitlines()
                          if not line.startswith(("BASELINE_REF =", "BASELINE_SHA256 =")))
    bootstrap += "\nimport csv\nimport os\nimport random\nimport re\n"
    helpers = "\n\n".join((
        definitions("lamp_workflow_source.py", "write_json", "write_csv", "workflow_digest", "workflow_bind", "workflow_login",
                    "workflow_version_ref", "workflow_dataset_manifest"),
        definitions("lamp_gflops_evaluation_source.py", "_gflops_split_images", "_gflops_fingerprint", "gflops_prepare_dataset",
                    "gflops_disable_label_cache", "gflops_require_complete_nms"),
        definitions("lamp_v2_runtime_source.py", "model_state_sha256", "safe_error"),
        (SCRIPTS / "lamp_recovery_training_source.py").read_text(encoding="utf-8"),
        "ordered_names = recovery_ordered_names",
        (SCRIPTS / "standard_finetuning_training_source.py").read_text(encoding="utf-8").split("\ndef recovery_fit(")[0],
    ))
    measurements = [ast.unparse(node) for cell in notebook_code(PRUNING) for node in cell.body
                    if isinstance(node, ast.FunctionDef) and node.name in {
                        "model_measurements", "cuda_latency", "finite_tensors", "raw_branches", "forward_snapshot"}]
    pruning_book = json.loads(PRUNING.read_text(encoding="utf-8"))
    reload_source = next("".join(cell["source"]) for cell in pruning_book["cells"]
                         if "".join(cell["source"]).startswith("RELOAD_SCRIPT ="))
    reload_source = reload_source.replace('child_env.pop("PYTHONPATH", None)',
                                          'child_env.pop("PYTHONPATH", None)\n    child_env.pop("WANDB_API_KEY", None)')
    eda = (SCRIPTS / "standard_finetuning_eda_source.py").read_text(encoding="utf-8").split("\ndef recovery_suite_tables")[0]
    eda = eda.replace('"Recovered", "#217854"', '"Fine-tuned", "#217854"')
    metrics = "\n\n".join((
        definitions("lamp_v2_validation_source.py", "_validation_number", "_validation_array", "_validation_json_array",
                    "_validation_per_class", "_validation_curves", "_validation_confusion_matrix"),
        definitions("lamp_v2_eda_source.py", "_eda_finite", "_eda_save_figure", "_eda_write_csv", "_eda_write_json", "layer_eda_rows",
                    "stage_eda_rows", "make_channel_figure", "_eda_channel_plots", "_eda_mapping_plots", "_eda_lamp_plots",
                    "_eda_curve_plots"),
        eda, *measurements, "import subprocess", reload_source,
        definitions("pruned_fine_tuned_source.py", "fine_tuned_check_heads", "fine_tuned_load_raw", "fine_tuned_check_images", "fine_tuned_measure",
                    "fine_tuned_comparison"),
    ))
    selected = (SCRIPTS / "pruned_finetune_selected_source.py").read_text(encoding="utf-8")
    implementation = hashlib.sha256(f"{protocol}\n{bootstrap}\n{helpers}\n{metrics}\n{selected}\n{pins}".encode()).hexdigest()
    cells = []

    def add(source, tag=None):
        source = source.strip()
        item = {"cell_type": "code" if tag else "markdown", "source": source.splitlines(keepends=True),
                "metadata": {}, "id": hashlib.sha256(f"{len(cells)}:{source}".encode()).hexdigest()[:8]}
        if tag:
            ast.parse(source)
            item.update(execution_count=None, outputs=[], metadata={"tags": [tag]})
        cells.append(item)

    add(f"""# Recovery YOLO26s · Fine-tuning satu student · {platform.title()}

Pilih satu dari tujuh raw LAMP melalui comment/uncomment. Seed, bobot, dan struktur berasal dari
raw terpilih. Setiap sesi memiliki folder output dan run W&B sendiri. Gunakan kernel dan DEVICE
masing-masing untuk eksekusi paralel. Protokol training mengikuti baseline student dengan default 100 epoch.

Evaluasi tiga kondisi memakai baseline asal, raw pruned, dan hasil fine-tuning pada validation
Poltrada V3 yang sama. LAMP memakai [kode penulis](https://github.com/jaeho-lee/layer-adaptive-sparsity)
dan [DepGraph](https://arxiv.org/abs/2301.12900). Model pruned diteruskan sebagai objek native.
""")
    add("""## 01 — Pilih student, perangkat, dan output

Aktifkan tepat satu STUDENT_INPUTS. Namespace W&B dilengkapi otomatis. SEED dibaca dari pin raw,
kemudian diperiksa terhadap manifest. Gunakan WANDB_API_KEY pada environment atau Kaggle Secret.
MODE full_val menggunakan dataset Poltrada V3 asli. MODE smoke menjalankan satu epoch diagnostik.
""")
    add(configuration, "config")
    add(pins, "selection")
    add(protocol, "protocol")
    add("""## 02 — Kernel dan verifikasi input

Versi dependency mengikuti baseline. Periksa SHA-256 manifest, raw, baseline FP32, dan audit sebelum
memuat model. Kontrak resume mengikat versi input, seed, dataset, protokol, dan lingkungan.
""")
    add(bootstrap, "bootstrap")
    add(helpers, "helpers")
    add(metrics, "helpers")
    add(selected + f"\nIMPLEMENTATION_SHA256 = {implementation!r}", "runtime")
    add("""## 03 — Fine-tuning dan logging epoch

Training memulai bobot persis dari raw. Snapshot lengkap disimpan tiap epoch dan diunggah tiap
lima epoch serta akhir training. Resume menggunakan folder lokal atau versi tetap artefak resume.
Checkpoint best dan last dipublikasikan beserta audit dan hash. Nama run memakai akhiran -ft.
""")
    add('''RESULT = ft_execute(RUN_DIR)
print(RESULT["status"], RESULT.get("reason", ""), "|", RUN_DIR)''', "run")
    add("""## 04 — EDA custom sebelum pruning, raw pruned, dan fine-tuned

Perbandingan khusus seed dan target ini mencakup metrik deteksi, AP/P/R/F1 per kelas, support,
parameter, GFLOPs, channel, checkpoint, dan latency p50/p95. Delta terhadap baseline dan gain
terhadap raw dilaporkan dalam poin persentase. Panel memakai warna baseline biru, raw merah,
dan fine-tuned hijau. Heatmap, ledger, LAMP, PR/F1, confusion matrix, training, dan trade-off
tersimpan sebagai tabel serta media W&B. Status belum terukur ditampilkan pada tabel.
""")
    add('''from IPython.display import Image, FileLink, display

if RESULT.get("stages"):
    display(pd.DataFrame(fine_tuned_comparison(RESULT)[0]))
    display(pd.DataFrame(fine_tuned_comparison(RESULT)[1]))
    for figure in sorted((RUN_DIR / "eda").rglob("*.png")):
        display(Image(filename=str(figure)))
for name in ("fine_tuned_manifest.json", "finetuning_registry.json", "finetuning_registry.csv", "epoch_metrics.json"):
    if (RUN_DIR / name).is_file():
        display(FileLink(os.path.relpath(RUN_DIR / name)))
''', "display")
    add("""## 05 — Ringkasan grup lintas seed dan pembanding

Aktifkan LOAD_GROUP_RESULTS untuk membaca run terkonfirmasi dari grup yang sama. Tabel memilih
hasil sukses terbaru per perlakuan, platform, dan perangkat latency. Lima seed target 1,20×
diringkas bersama. Target 1,10× dan kandidat maksimum 1,70× seed 42 tampil sebagai pembanding.
Hasil parsial memperlihatkan jumlah perlakuan yang sudah tersedia.
""")
    add('''if LOAD_GROUP_RESULTS:
    group_frame = pd.DataFrame(ft_group_results())
    identity_columns = ["seed", "target_speedup", "actual_gflops_speedup", "target_status", "platform", "gpu",
                        "epochs_completed", "best_epoch", "checkpoint_artifact_ref", "wandb_run_url"]
    metric_columns = [f"{stage}_{metric}" for stage in ("baseline", "raw_pruned", "recovered")
                      for metric in ("map50_95", "map50", "precision", "recall")]
    group_columns = [name for name in identity_columns + metric_columns if name in group_frame]
    display(group_frame[group_columns])
    if not group_frame.empty:
        print("Pembanding seed 42 · tiga target · baseline asal, raw, dan fine-tuned")
        display(group_frame.loc[group_frame["seed"].eq(42), group_columns].sort_values("target_speedup"))
else:
    print("Aktifkan LOAD_GROUP_RESULTS untuk membaca hasil grup W&B.")''', "group-display")
    book = {"cells": cells, "nbformat": 4, "nbformat_minor": 5, "metadata": {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "research_workflow": {"platform": platform, "variant": "s", "protocol": "selected-raw-finetune-v2",
                              "stage": "recovery", "method": "finetune"}}}
    if restore_inputs:
        if __package__:
            from .finetune_artifact_binding import bind_restored_inputs
            from .recovery_custom_eda import attach_custom_eda
            from .recovery_postprocess import attach_postprocessing
        else:
            from finetune_artifact_binding import bind_restored_inputs
            from recovery_custom_eda import attach_custom_eda
            from recovery_postprocess import attach_postprocessing
        return attach_custom_eda(attach_postprocessing(bind_restored_inputs(book, ROOT, STUDENT_PINS), ROOT), ROOT)
    return book


def build_notebook(platform, method="finetune"):
    if method == "finetune":
        return build_selected_notebook(platform)
    if __package__:
        from .generate_selected_kd_notebooks import build_notebook as selected_kd
    else:
        from generate_selected_kd_notebooks import build_notebook as selected_kd
    return selected_kd(platform, method)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--methods", nargs="+", choices=METHOD_NOTEBOOKS, default=["finetune"])
    arguments = parser.parse_args()
    for platform in ("kaggle", "filkom"):
        for method in arguments.methods:
            filename = METHOD_NOTEBOOKS[method][0]
            path = notebook_path(platform, filename)
            path.write_text(json.dumps(build_notebook(platform, method), ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
            print(path)
