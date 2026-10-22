"""Generate standalone Colab copies without modifying their Kaggle sources."""
from __future__ import annotations

import ast
import copy
import hashlib
import json
import os
from pathlib import Path

try:
    from scripts.notebook_paths import notebook_path, notebook_paths
except ModuleNotFoundError:
    from notebook_paths import notebook_path, notebook_paths

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "notebooks/kaggle"
DESTINATION = ROOT / "notebooks/colab"
SOURCE_DATA = "/kaggle/input/datasets/ilhamrafiqin/poltrada-v3-yolo26-clean/data.yaml"
WORKFLOWS = [
    ("00_eda_poltrada_v3", "00_eda_poltrada_v3", "EDA", "—", "—"),
    ("01_teacher_yolo26m", "01_teacher_yolo26m", "Supervised teacher", "—", "26m"),
    ("02_baseline_yolo26s", "02_baseline_yolo26s", "Supervised", "—", "26s"),
    ("03_baseline_yolo26n_pretrained", "03_baseline_yolo26n", "Pretrained baseline", "—", "26n"),
    ("04_baseline_yolo26n_scratch", "03_baseline_yolo26n_scratch", "Scratch baseline", "—", "26n"),
    ("05_native_yolo26m_to_yolo26s", "04_native_yolo26s", "Native KD", "26m v50", "26s"),
    ("06_native_yolo26m_to_yolo26n", "05_native_yolo26n", "Native KD", "26m v50", "26n"),
    ("07_crosskd_yolo26m_to_yolo26s", "06_crosskd_yolo26s", "CrossKD", "26m v50", "26s"),
    ("08_crosskd_yolo26m_to_yolo26n", "07_crosskd_yolo26n", "CrossKD", "26m v50", "26n"),
    ("09_csakd_yolo26m_to_yolo26s", "08_csakd_yolo26s", "CSAKD", "26m v50", "26s"),
    ("10_csakd_yolo26m_to_yolo26n", "09_csakd_yolo26n", "CSAKD", "26m v50", "26n"),
    ("11_native_yolo26s_to_yolo26n", "10_native_yolo26n_teacher_yolo26s", "Native KD", "26s v62", "26n"),
    ("12_crosskd_yolo26s_to_yolo26n", "11_crosskd_yolo26n_teacher_yolo26s", "CrossKD", "26s v62", "26n"),
    ("13_csakd_yolo26s_to_yolo26n", "12_csakd_yolo26n_teacher_yolo26s", "CSAKD", "26s v62", "26n"),
    ("14_pearson_pkd_yolo26s_to_yolo26n", "12_pkd_yolo26n", "Pearson/PKD", "26s v62", "26n"),
    ("15_head_yolo26s_to_yolo26n", "13_head_yolo26n_teacher_yolo26s", "HEAD", "26s v62", "26n"),
    ("16_rank_mimicking_yolo26s_to_yolo26n", "14_rm_yolo26n_teacher_yolo26s", "Rank Mimicking", "26s v62", "26n"),
]


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def cell(kind, identifier, source, **metadata):
    result = {"cell_type": kind, "id": identifier, "metadata": metadata, "source": source.splitlines(keepends=True)}
    if kind == "code":
        result.update(execution_count=None, outputs=[])
    return result


def assignment_values(source):
    return {target.id: node.value for node in ast.parse(source).body if isinstance(node, ast.Assign)
            for target in node.targets if isinstance(target, ast.Name)}


def adapt_source(source):
    """Bounded platform edits, keeping method implementations and training kwargs intact."""
    # Teacher metadata describes the original source, not its runtime filesystem location.
    source = source.replace('!= DATA.replace("\\\\", "/")', '!= SOURCE_DATA_DECLARATION.replace("\\\\", "/")')
    source = source.replace('!= str(DATA).replace("\\\\", "/")', '!= SOURCE_DATA_DECLARATION.replace("\\\\", "/")')
    tree = ast.parse(source)
    replacements = []
    for node in ast.walk(tree):
        # Select the innermost containing if (the WANDB_API_KEY environment check).
        if (isinstance(node, ast.If) and "kaggle_secrets" in (ast.get_source_segment(source, node) or "")
                and not any(isinstance(child, ast.If) and "kaggle_secrets" in (ast.get_source_segment(source, child) or "")
                            for child in ast.iter_child_nodes(node))):
            replacements.append((node, 'os.environ["WANDB_API_KEY"] = colab_api_key()'))
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
            if node.func.attr == "download":
                for keyword in node.keywords:
                    if keyword.arg == "root":
                        replacements.append((keyword.value, 'str(COLAB_LOCAL_DIR / "teacher")'))
            if isinstance(node.func.value, ast.Name) and node.func.value.id == "wandb" and node.func.attr == "init":
                for keyword in node.keywords:
                    old = ast.get_source_segment(source, keyword.value)
                    if keyword.arg == "config":
                        replacements.append((keyword.value, '{**(' + old + '), "colab_dataset": COLAB_DATA_PROVENANCE, "runtime": "google-colab"}'))
                    if keyword.arg == "tags":
                        replacements.append((keyword.value, '[*(' + old + '), "colab"]'))
    # Apply right-to-left by source location; segments are unchanged until their edit.
    for node, new in sorted(replacements, key=lambda item: (item[0].lineno, item[0].col_offset), reverse=True):
        old = ast.get_source_segment(source, node)
        if source.count(old) != 1:
            raise ValueError(f"Ambiguous platform edit: {old[:80]}")
        source = source.replace(old, new, 1)
    assignments = assignment_values(source)
    for name, new in {"DATA": "COLAB_DATA", "DATA_YAML": "Path(COLAB_DATA)",
                      "PROJECT": "str(COLAB_OUTPUT_DIR)", "OUTPUT_DIR": "COLAB_OUTPUT_DIR"}.items():
        if name in assignments:
            # Replace the complete assignment rather than other identical literals.
            pattern = next(node for node in ast.parse(source).body if isinstance(node, ast.Assign)
                           and any(isinstance(target, ast.Name) and target.id == name for target in node.targets))
            source = source.replace(ast.get_source_segment(source, pattern), f"{name} = {new}", 1)
    # No operational Kaggle paths remain. The original declaration lives in the setup cell.
    source = source.replace('Path("/kaggle/input").resolve()', '(COLAB_LOCAL_DIR / "source").resolve()')
    source = source.replace("/kaggle/working", "/content")
    source = source.replace("Kaggle: Settings > Accelerator", "Colab: Runtime > Change runtime type > GPU")
    source = source.replace("Secret Kaggle", "Colab Secret").replace("GPU CUDA Kaggle", "GPU CUDA Colab")
    source = source.replace("GPU Kaggle/Colab", "GPU Colab").replace("GPU Kaggle", "GPU Colab")
    source = source.replace("Dataset Kaggle", "Dataset Colab").replace("dataset bersih di Kaggle", "dataset dari ZIP Drive")
    source = source.replace("Dataset bersih di Kaggle.", "Runtime YAML prepared from the Drive ZIP.")
    source = source.replace("Set sebagai Kaggle Secret atau environment variable", "Set sebagai Colab Secret atau environment variable")
    source = source.replace("Tambahkan dataset Poltrada V3 di Kaggle:", "Periksa DATASET_ZIP dan dataset preparation:")
    source = source.replace("Tambahkan dataset final melalui ", "Periksa DATASET_ZIP; ")
    source = source.replace("Kaggle Input", "dataset ZIP source")
    source = source.replace("  # Colab: ganti ke /content/results", "  # Persistent Google Drive output for this run.")
    source = source.replace("YAML asli (dibaca tanpa penulisan ulang)", "YAML runtime Colab (YAML sumber ZIP disimpan terpisah)")
    # Remove obsolete authentication instructions left beside the replaced block.
    source = "\n".join(line for line in source.splitlines() if "# Kaggle: tambahkan secret" not in line) + "\n"
    compile(source, "<adapted-colab-cell>", "exec")
    return source


def bootstrap_cells(stem, training):
    config = f'''from pathlib import Path
from datetime import datetime, timezone
import uuid

# Edit these two paths to match your Google Drive.
DATASET_ZIP = "/content/drive/MyDrive/Skripsi/datasets/poltrada-v3-yolo26-clean.zip"
DRIVE_OUTPUT_ROOT = "/content/drive/MyDrive/Skripsi/yolo26-kd-indotraffic/results"
DATASET_YAML_RELATIVE = ""  # Set only if the ZIP contains more than one YAML.
SOURCE_DATA_DECLARATION = {SOURCE_DATA!r}  # Original teacher provenance; not an active Colab path.
COLAB_RUN_ID = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S_%fZ") + "_" + uuid.uuid4().hex[:6]
COLAB_LOCAL_DIR = Path("/content/yolo26_colab") / COLAB_RUN_ID
COLAB_OUTPUT_DIR = Path(DRIVE_OUTPUT_ROOT) / "{stem}" / COLAB_RUN_ID
COLAB_DATA = str(COLAB_LOCAL_DIR / "runtime/data.yaml")
'''
    mount = '''from google.colab import drive
import os

drive.mount("/content/drive")
COLAB_LOCAL_DIR.mkdir(parents=True, exist_ok=False)
COLAB_OUTPUT_DIR.mkdir(parents=True, exist_ok=False)
os.chdir(COLAB_LOCAL_DIR)  # Downloads and temporary files stay on the Colab VM.
os.environ["WANDB_CACHE_DIR"] = str(COLAB_LOCAL_DIR / "wandb-cache")
os.environ["WANDB_DIR"] = str(COLAB_LOCAL_DIR)
os.environ["PIP_CACHE_DIR"] = str(COLAB_LOCAL_DIR / "pip-cache")
os.environ["TORCH_HOME"] = str(COLAB_LOCAL_DIR / "torch-cache")
print("Persistent output:", COLAB_OUTPUT_DIR)
'''
    packages = [('ultralytics', 'ultralytics', '8.4.155'), ('wandb', 'wandb', '0.28.1')] if training else [
        ('numpy', 'numpy', None), ('pandas', 'pandas', None), ('matplotlib', 'matplotlib', None),
        ('Pillow', 'Pillow', None), ('ipython', 'ipython', None)]
    packages.append(('PyYAML', 'PyYAML', None))
    dependencies = f'''from importlib import metadata
import subprocess
import sys

# Preserve Colab's installed PyTorch/torchvision build during dependency resolution.
constraints = []
for package in ("torch", "torchvision"):
    try:
        constraints.append(package + "==" + metadata.version(package))
    except metadata.PackageNotFoundError:
        if {training!r}:
            raise RuntimeError("Use a Colab GPU runtime with PyTorch installed.") from None
constraint_file = COLAB_LOCAL_DIR / "colab-torch-constraints.txt"
constraint_file.write_text("\\n".join(constraints), encoding="utf-8")
for distribution, package, version in {packages!r}:
    try:
        installed = metadata.version(distribution)
    except metadata.PackageNotFoundError:
        installed = None
    if installed is None or (version is not None and installed != version):
        requirement = package if version is None else package + "==" + version
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "-c", str(constraint_file), requirement])
'''
    prepare = f'''import json
{('import torch' + chr(10) + 'colab_require_gpu(torch)') if training else '# EDA does not require a GPU or a W&B secret.'}
COLAB_DATA, COLAB_DATA_PROVENANCE = prepare_colab_dataset(
    DATASET_ZIP, COLAB_LOCAL_DIR, DATASET_YAML_RELATIVE, SOURCE_DATA_DECLARATION)
{'DATA = COLAB_DATA' if training else 'DATA_YAML = Path(COLAB_DATA)'}
(COLAB_OUTPUT_DIR / "dataset_provenance.json").write_text(
    json.dumps(COLAB_DATA_PROVENANCE, indent=2, ensure_ascii=False), encoding="utf-8")
print("Dataset:", COLAB_DATA)
print("Split image counts:", COLAB_DATA_PROVENANCE["split_image_counts"])
'''
    runtime = (ROOT / "scripts/colab_runtime.py").read_text(encoding="utf-8")
    return config, [
        cell("markdown", "colab-setup-md", "## Colab: Drive, dependencies, and local dataset\n\nTrain from the local extracted ZIP; outputs go to Drive and existing W&B callbacks. Authorize Drive mounting when prompted."),
        cell("code", "colab-drive", mount), cell("code", "colab-dependencies", dependencies),
        cell("code", "colab-runtime", runtime, collapsed=True, cellView="form"),
        cell("code", "colab-dataset", prepare),
    ]


def build_notebook(row):
    stem, original, method, teacher, student = row
    path = notebook_path("kaggle", original + ".ipynb")
    destination = notebook_path("colab", stem + ".ipynb")
    notebook = copy.deepcopy(json.loads(path.read_text(encoding="utf-8")))
    config_index = next(i for i, c in enumerate(notebook["cells"]) if c["cell_type"] == "code"
                        and ("EXPERIMENT_ID =" in "".join(c["source"]) or "DATASET_NAME =" in "".join(c["source"])))
    original_config = "".join(notebook["cells"][config_index]["source"])
    values = assignment_values(original_config)
    experiment = ast.literal_eval(values["EXPERIMENT_ID"]) if "EXPERIMENT_ID" in values else "eda_poltrada_v3"
    seeds = ast.literal_eval(values["SEEDS_TO_RUN"]) if "SEEDS_TO_RUN" in values else [ast.literal_eval(values["SEED"])]
    config, bootstrap = bootstrap_cells(stem, method != "EDA")
    intro = notebook["cells"][0]
    method_notes = "\n".join("".join(intro["source"]).splitlines()[1:]).strip().replace("Kaggle", "Colab")
    intro["source"] = (f"# {stem.replace('_', ' ')} — Google Colab\n\n"
                       "1. Set **Runtime > Change runtime type > GPU** for training.\n"
                       "2. Edit `DATASET_ZIP` and `DRIVE_OUTPUT_ROOT` below.\n"
                       "3. Add `WANDB_API_KEY` in **Colab Secrets** and enable notebook access (training only).\n"
                       "4. Run All. Research hyperparameters and seed schedules are in the next configuration cell.\n\n"
                       f"Source: `{original}.ipynb`. Experiment IDs/artifact names are preserved. "
                       "This copy is standalone; no repository checkout is required.\n\n"
                       + ("### Method notes\n\n" + method_notes + "\n" if method_notes else "")).splitlines(keepends=True)
    remaining = []
    for i, c in enumerate(notebook["cells"]):
        if i in (0, config_index - 1, config_index):
            continue
        if c["cell_type"] == "code":
            c["source"] = adapt_source("".join(c["source"])).splitlines(keepends=True)
        else:
            c["source"] = "".join(c["source"]).replace("Kaggle", "Colab").replace("/kaggle/working", "/content").splitlines(keepends=True)
        remaining.append(c)
    source_config = notebook["cells"][config_index]
    source_config["source"] = adapt_source(original_config).splitlines(keepends=True)
    notebook["cells"] = [intro, cell("markdown", "colab-config-md", "## Colab paths\n\nKeep the ZIP unchanged in Drive. Extraction and the derived YAML are local; source images, labels and YAML are not edited. For multiple YAML files, set `DATASET_YAML_RELATIVE` to the desired archive-relative path."),
                         cell("code", "colab-config", config),
                         cell("markdown", "colab-research-config-md", "## Research configuration\n\nPreserved from the original notebook. The trio defaults to 100 epochs with smoke disabled. Existing multi-seed queues remain unchanged; 100 epochs is per seed."),
                         source_config, *bootstrap, *remaining]
    for c in notebook["cells"]:
        c["metadata"] = {key: value for key, value in c.get("metadata", {}).items() if key in {"collapsed", "cellView"}}
        if c["cell_type"] == "code":
            c.update(execution_count=None, outputs=[])
            compile("".join(c["source"]), stem + ":" + c["id"], "exec")
    notebook["metadata"] = {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                            "language_info": {"name": "python"}, "accelerator": "GPU",
                            "colab": {"name": stem + ".ipynb", "provenance": []},
                            "colab_source": {"notebook": Path(os.path.relpath(path, destination.parent)).as_posix(), "sha256": sha256(path)}}
    return notebook, {"colab": destination.relative_to(DESTINATION).as_posix(),
                      "source": path.relative_to(SOURCE).as_posix(), "source_sha256": sha256(path),
                      "experiment_id": experiment, "method": method, "teacher": teacher, "student": student,
                      "seeds": seeds, "output": stem + "/<run-id>/", "colab_gpu_execution": "NOT TESTED"}


def readme(rows):
    lines = ["# YOLO26 research notebooks — Google Colab", "",
             "17 standalone workflows, grouped into eda, teacher, baseline, kd/teacher_m, and kd/teacher_s. Original Kaggle notebooks are in [../kaggle/](../kaggle/README.md). EDA saved-results and AMD/DirectML–Roboflow copies are intentionally excluded.", "",
             "## Run in Colab", "",
             "1. Upload the chosen notebook to Colab; select a GPU runtime for training.",
             "2. Put the unchanged Poltrada V3 ZIP in Drive. Edit `DATASET_ZIP` (default `/content/drive/MyDrive/Skripsi/datasets/poltrada-v3-yolo26-clean.zip`).",
             "3. Edit `DRIVE_OUTPUT_ROOT` if needed (default `/content/drive/MyDrive/Skripsi/yolo26-kd-indotraffic/results`).",
             "4. Training notebooks: add `WANDB_API_KEY` in Colab Secrets, enable notebook access, and retain access to the existing W&B project/teacher artifacts. An existing environment variable takes precedence. EDA needs no secret.",
             "5. Run All and authorize Drive. The ZIP is copied/extracted under `/content`, never read image-by-image from Drive. If multiple YAML files exist, set `DATASET_YAML_RELATIVE`.", "",
             "The original YAML is retained. A separate runtime YAML (and derived TXT split lists when needed) relocates paths without changing images, labels, class order, or split membership. ZIP/original/runtime YAML hashes and resolved counts are recorded in `dataset_provenance.json` under the run output and in training W&B config. This records your ZIP; it does not independently prove byte equality with the previous Kaggle dataset.", "",
             "Training/EDA outputs are stored below Drive's output root using the Colab filename and a unique UTC/UUID run directory. Existing experiment/seed subdirectories remain inside it. Teacher downloads, package caches and temporary working files use `/content`. Existing W&B per-epoch uploads are retained.", "",
             "## Notebook index", "",
             "Filename numbers are navigation only; original experiment IDs and teacher artifact names are unchanged.", "",
             "| Colab notebook | Original Kaggle notebook | Method | Teacher | Student | Seeds | Drive output below root |",
             "|---|---|---|---|---|---|---|"]
    for row in rows:
        lines.append(f"| [{row['colab']}]({row['colab']}) | [{row['source']}](../kaggle/{row['source']}) | {row['method']} | {row['teacher']} | {row['student']} | {', '.join(map(str, row['seeds']))} | `{row['output']}` |")
    lines += ["", "## Training defaults and verification", "",
              "HEAD, Rank Mimicking and Pearson/PKD: `EPOCHS=100`, `RUN_FULL_TRAINING=True`, `RUN_SMOKE_TEST=False`. No additional smoke run is dispatched by default. Teacher audits and numerical checks remain active. Other workflows retain their original hyperparameters, seed queues and diagnostic behavior.", "",
              "Dataset declaration checks use the original Kaggle source identity separately from the active Colab YAML. Artifact versions, existing checksums/schema/class audits, objectives and seed schedules are preserved. A Drive checkpoint does not establish exact resume support; existing `RESUME_NOT_VERIFIED` restrictions remain.", "",
              "Generation and local tests do not constitute Colab GPU execution or proof of 100-epoch stability. Historical Kaggle evidence is in [experiment reports](../../reports/hasil_percobaan/README.md).", "",
              "Local adaptation checks and their limits are documented in the [Colab verification report](../../reports/hasil_percobaan/colab_collection_validation_20261022.md).", "",
              "Regenerate from the repository root: `python scripts/generate_colab_notebooks.py`. The notebooks themselves need no generator or repository files in Colab.", "",
              "References: [Google Colab Drive/archive guidance](https://research.google.com/colaboratory/faq.html), [Colab Secrets API](https://github.com/googlecolab/colabtools/blob/main/google/colab/userdata.py).", ""]
    return "\n".join(lines)


def main():
    before = {p: sha256(p) for p in notebook_paths("kaggle")}
    DESTINATION.mkdir(parents=True, exist_ok=True)
    rows = []
    for row in WORKFLOWS:
        notebook, record = build_notebook(row)
        destination = DESTINATION / record["colab"]
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(json.dumps(notebook, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
        rows.append(record)
    (DESTINATION / "manifest.json").write_text(json.dumps(rows, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    (DESTINATION / "README.md").write_text(readme(rows), encoding="utf-8")
    assert before == {p: sha256(p) for p in notebook_paths("kaggle")}, "Kaggle source changed during generation"
    print(f"Generated {len(rows)} standalone Colab notebooks; all Kaggle sources unchanged.")


if __name__ == "__main__":
    main()
