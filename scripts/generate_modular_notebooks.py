"""Regenerate the four small notebook frontends without running any experiment."""

from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DESTINATION = ROOT / "notebooks" / "modular"
REPOSITORY_URL = (
    "https://github.com/lalalostcode/skripsi-deteksi-kendaraan-indonesia.git"
)


def cell(kind: str, source: str) -> dict:
    result = {
        "cell_type": kind,
        "metadata": {},
        "source": source.splitlines(keepends=True),
    }
    if kind == "code":
        result.update({"execution_count": None, "outputs": []})
    return result


def make_notebook(architecture: str) -> dict:
    title = "FPN-PAN baseline" if architecture == "baseline" else architecture.upper()
    bootstrap = f'''from pathlib import Path
import subprocess
import sys

ON_KAGGLE = Path("/kaggle/working").is_dir()
REPO_URL = "{REPOSITORY_URL}"
REPO_REF = "main"  # record the resolved commit below for reproducibility

if ON_KAGGLE:
    repo_root = Path("/kaggle/working/skripsi-code")
    if not (repo_root / ".git").exists():
        subprocess.run(["git", "clone", "--depth", "1", "--branch", REPO_REF, REPO_URL, str(repo_root)], check=True)
    subprocess.run(["git", "-C", str(repo_root), "fetch", "--depth", "1", "origin", REPO_REF], check=True)
    subprocess.run(["git", "-C", str(repo_root), "checkout", "--detach", "FETCH_HEAD"], check=True)
    subprocess.run([sys.executable, "-m", "pip", "install", "-q",
                    "ultralytics==8.4.155", "wandb==0.28.1", "roboflow==1.4.0",
                    "pycocotools>=2.0.10", "thop", "pyyaml", "pandas", "pillow", "matplotlib"], check=True)
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", "-e", str(repo_root), "--no-deps"], check=True)
else:
    repo_root = Path.cwd().resolve()
    for candidate in (repo_root, *repo_root.parents):
        if (candidate / "pyproject.toml").exists() and (candidate / "src" / "skripsi").is_dir():
            repo_root = candidate
            break
    else:
        raise FileNotFoundError("Buka notebook dari checkout repository skripsi")
    # `uv sync` environments may not ship pip; imports work directly from src.
    sys.path.insert(0, str(repo_root / "src"))

git_commit = subprocess.check_output(["git", "-C", str(repo_root), "rev-parse", "HEAD"], text=True).strip()
print("Repository:", repo_root)
print("Commit:", git_commit)
'''
    config = f'''from skripsi import ExperimentConfig, preflight, run_training, evaluate_checkpoint

architecture = "{architecture}"
active_seed = 42  # ganti menjadi 123 atau 2026 untuk seed berikutnya
local_v2_candidates = (
    repo_root / "data" / "raw" / "poltrada-original-raw-2-v2",
    repo_root / "data" / "audited_v2",
)
dataset_dir = next((path for path in local_v2_candidates if path.is_dir()), None) if not ON_KAGGLE else None
output_root = Path("/kaggle/working/yolo26n_research") if ON_KAGGLE else repo_root / "outputs" / "research"

config = ExperimentConfig(
    architecture=architecture,
    seed=active_seed,
    dataset_dir=dataset_dir,
    output_root=output_root,
    git_commit=git_commit,
)
print(config)
'''
    return {
        "cells": [
            cell(
                "markdown",
                f"# YOLO26n {title} — pipeline modular\n\nNotebook ini memakai paket `skripsi`. Dataset v2, audit, evaluasi, dan protokol training sama untuk keempat arsitektur. **Preflight tidak melakukan training.** Training dan evaluasi hanya berjalan ketika flag sel masing-masing diaktifkan. Di Kaggle, aktifkan GPU dan Internet serta simpan `ROBOFLOW_API_KEY` dan (opsional) `WANDB_API_KEY` pada Secrets.\n",
            ),
            cell(
                "markdown",
                "## 1. Bootstrap kode dan dependensi\n\nDi Kaggle, kode di-clone dari repository publik. Setelah perubahan lokal dipush, buat versi notebook Kaggle baru agar checkout mengambil commit baru.\n",
            ),
            cell("code", bootstrap),
            cell(
                "markdown",
                "## 2. Konfigurasi eksperimen\n\nGunakan satu seed per run Kaggle. Ubah hanya `active_seed` untuk run berikutnya.\n",
            ),
            cell("code", config),
            cell(
                "markdown",
                "## 3. Preflight tanpa training\n\nSalin atau unduh dataset v2 ke area kerja, konversi polygon pada salinan, audit fingerprint, dan periksa graph dengan satu forward sintetis. Pada lokal tanpa dataset v2, tambahkan `dataset_dir` atau `ROBOFLOW_API_KEY`.\n",
            ),
            cell(
                "code",
                "dataset = preflight(config)\nprint('Fingerprint:', dataset.fingerprint)\nprint('Runtime YAML:', dataset.runtime_yaml)\n",
            ),
            cell(
                "markdown",
                "## 4. Training — jalankan hanya bila Anda menghendaki\n\nPemanggilan ini disiapkan untuk pengguna; default `False` sehingga Run All tidak memulai training.\n",
            ),
            cell(
                "code",
                "RUN_TRAINING = False\nif RUN_TRAINING:\n    training_result = run_training(config, dataset)\n    print(training_result)\n",
            ),
            cell(
                "markdown",
                "## 5. Evaluasi checkpoint — tanpa training\n\nSetelah `best.pt` tersedia, aktifkan sel ini. Validation memilih ambang confidence; test menghasilkan tabel 3.5–3.8.\n",
            ),
            cell(
                "code",
                "RUN_EVALUATION = False\ncheckpoint = config.run_dir / 'weights' / 'best.pt'\nif RUN_EVALUATION:\n    evaluation_result = evaluate_checkpoint(config, checkpoint, dataset)\n    print(evaluation_result['overall'])\n    print('Hasil:', evaluation_result['evaluation_dir'])\n",
            ),
        ],
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3",
            },
            "language_info": {"name": "python"},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }


def main() -> None:
    DESTINATION.mkdir(parents=True, exist_ok=True)
    for index, architecture in enumerate(
        ("baseline", "bifpn", "asff", "afpn"), start=2
    ):
        output = DESTINATION / f"{index:02d}_yolo26n_{architecture}.ipynb"
        output.write_text(
            json.dumps(make_notebook(architecture), ensure_ascii=False, indent=1)
            + "\n",
            encoding="utf-8",
        )
        print(output)


if __name__ == "__main__":
    main()
