# YOLO26 research notebooks — Google Colab

17 standalone workflows, grouped into eda, teacher, baseline, kd/teacher_m, and kd/teacher_s. Original Kaggle notebooks are in [../kaggle/](../kaggle/README.md). EDA saved-results and AMD/DirectML–Roboflow copies are intentionally excluded.

## Run in Colab

1. Upload the chosen notebook to Colab; select a GPU runtime for training.
2. Put the unchanged Poltrada V3 ZIP in Drive. Edit `DATASET_ZIP` (default `/content/drive/MyDrive/Skripsi/datasets/poltrada-v3-yolo26-clean.zip`).
3. Edit `DRIVE_OUTPUT_ROOT` if needed (default `/content/drive/MyDrive/Skripsi/yolo26-kd-indotraffic/results`).
4. Training notebooks: add `WANDB_API_KEY` in Colab Secrets, enable notebook access, and retain access to the existing W&B project/teacher artifacts. An existing environment variable takes precedence. EDA needs no secret.
5. Run All and authorize Drive. The ZIP is copied/extracted under `/content`, never read image-by-image from Drive. If multiple YAML files exist, set `DATASET_YAML_RELATIVE`.

The original YAML is retained. A separate runtime YAML (and derived TXT split lists when needed) relocates paths without changing images, labels, class order, or split membership. ZIP/original/runtime YAML hashes and resolved counts are recorded in `dataset_provenance.json` under the run output and in training W&B config. This records your ZIP; it does not independently prove byte equality with the previous Kaggle dataset.

Training/EDA outputs are stored below Drive's output root using the Colab filename and a unique UTC/UUID run directory. Existing experiment/seed subdirectories remain inside it. Teacher downloads, package caches and temporary working files use `/content`. Existing W&B per-epoch uploads are retained.

## Notebook index

Filename numbers are navigation only; original experiment IDs and teacher artifact names are unchanged.

| Colab notebook | Original Kaggle notebook | Method | Teacher | Student | Seeds | Drive output below root |
|---|---|---|---|---|---|---|
| [eda/00_eda_poltrada_v3.ipynb](eda/00_eda_poltrada_v3.ipynb) | [eda/00_eda_poltrada_v3.ipynb](../kaggle/eda/00_eda_poltrada_v3.ipynb) | EDA | — | — | 42 | `00_eda_poltrada_v3/<run-id>/` |
| [teacher/01_teacher_yolo26m.ipynb](teacher/01_teacher_yolo26m.ipynb) | [teacher/01_teacher_yolo26m.ipynb](../kaggle/teacher/01_teacher_yolo26m.ipynb) | Supervised teacher | — | 26m | 42 | `01_teacher_yolo26m/<run-id>/` |
| [baseline/02_baseline_yolo26s.ipynb](baseline/02_baseline_yolo26s.ipynb) | [baseline/02_baseline_yolo26s.ipynb](../kaggle/baseline/02_baseline_yolo26s.ipynb) | Supervised | — | 26s | 42 | `02_baseline_yolo26s/<run-id>/` |
| [baseline/03_baseline_yolo26n_pretrained.ipynb](baseline/03_baseline_yolo26n_pretrained.ipynb) | [baseline/03_baseline_yolo26n.ipynb](../kaggle/baseline/03_baseline_yolo26n.ipynb) | Pretrained baseline | — | 26n | 7, 3407 | `03_baseline_yolo26n_pretrained/<run-id>/` |
| [baseline/04_baseline_yolo26n_scratch.ipynb](baseline/04_baseline_yolo26n_scratch.ipynb) | [baseline/03_baseline_yolo26n_scratch.ipynb](../kaggle/baseline/03_baseline_yolo26n_scratch.ipynb) | Scratch baseline | — | 26n | 42 | `04_baseline_yolo26n_scratch/<run-id>/` |
| [kd/teacher_m/05_native_yolo26m_to_yolo26s.ipynb](kd/teacher_m/05_native_yolo26m_to_yolo26s.ipynb) | [kd/teacher_m/04_native_yolo26s.ipynb](../kaggle/kd/teacher_m/04_native_yolo26s.ipynb) | Native KD | 26m v50 | 26s | 42 | `05_native_yolo26m_to_yolo26s/<run-id>/` |
| [kd/teacher_m/06_native_yolo26m_to_yolo26n.ipynb](kd/teacher_m/06_native_yolo26m_to_yolo26n.ipynb) | [kd/teacher_m/05_native_yolo26n.ipynb](../kaggle/kd/teacher_m/05_native_yolo26n.ipynb) | Native KD | 26m v50 | 26n | 42 | `06_native_yolo26m_to_yolo26n/<run-id>/` |
| [kd/teacher_m/07_crosskd_yolo26m_to_yolo26s.ipynb](kd/teacher_m/07_crosskd_yolo26m_to_yolo26s.ipynb) | [kd/teacher_m/06_crosskd_yolo26s.ipynb](../kaggle/kd/teacher_m/06_crosskd_yolo26s.ipynb) | CrossKD | 26m v50 | 26s | 42 | `07_crosskd_yolo26m_to_yolo26s/<run-id>/` |
| [kd/teacher_m/08_crosskd_yolo26m_to_yolo26n.ipynb](kd/teacher_m/08_crosskd_yolo26m_to_yolo26n.ipynb) | [kd/teacher_m/07_crosskd_yolo26n.ipynb](../kaggle/kd/teacher_m/07_crosskd_yolo26n.ipynb) | CrossKD | 26m v50 | 26n | 42 | `08_crosskd_yolo26m_to_yolo26n/<run-id>/` |
| [kd/teacher_m/09_csakd_yolo26m_to_yolo26s.ipynb](kd/teacher_m/09_csakd_yolo26m_to_yolo26s.ipynb) | [kd/teacher_m/08_csakd_yolo26s.ipynb](../kaggle/kd/teacher_m/08_csakd_yolo26s.ipynb) | CSAKD | 26m v50 | 26s | 42 | `09_csakd_yolo26m_to_yolo26s/<run-id>/` |
| [kd/teacher_m/10_csakd_yolo26m_to_yolo26n.ipynb](kd/teacher_m/10_csakd_yolo26m_to_yolo26n.ipynb) | [kd/teacher_m/09_csakd_yolo26n.ipynb](../kaggle/kd/teacher_m/09_csakd_yolo26n.ipynb) | CSAKD | 26m v50 | 26n | 42 | `10_csakd_yolo26m_to_yolo26n/<run-id>/` |
| [kd/teacher_s/11_native_yolo26s_to_yolo26n.ipynb](kd/teacher_s/11_native_yolo26s_to_yolo26n.ipynb) | [kd/teacher_s/10_native_yolo26n_teacher_yolo26s.ipynb](../kaggle/kd/teacher_s/10_native_yolo26n_teacher_yolo26s.ipynb) | Native KD | 26s v62 | 26n | 123, 2024, 7, 3407 | `11_native_yolo26s_to_yolo26n/<run-id>/` |
| [kd/teacher_s/12_crosskd_yolo26s_to_yolo26n.ipynb](kd/teacher_s/12_crosskd_yolo26s_to_yolo26n.ipynb) | [kd/teacher_s/11_crosskd_yolo26n_teacher_yolo26s.ipynb](../kaggle/kd/teacher_s/11_crosskd_yolo26n_teacher_yolo26s.ipynb) | CrossKD | 26s v62 | 26n | 123, 2024, 7, 3407 | `12_crosskd_yolo26s_to_yolo26n/<run-id>/` |
| [kd/teacher_s/13_csakd_yolo26s_to_yolo26n.ipynb](kd/teacher_s/13_csakd_yolo26s_to_yolo26n.ipynb) | [kd/teacher_s/12_csakd_yolo26n_teacher_yolo26s.ipynb](../kaggle/kd/teacher_s/12_csakd_yolo26n_teacher_yolo26s.ipynb) | CSAKD | 26s v62 | 26n | 42 | `13_csakd_yolo26s_to_yolo26n/<run-id>/` |
| [kd/teacher_s/14_pearson_pkd_yolo26s_to_yolo26n.ipynb](kd/teacher_s/14_pearson_pkd_yolo26s_to_yolo26n.ipynb) | [kd/teacher_s/12_pkd_yolo26n.ipynb](../kaggle/kd/teacher_s/12_pkd_yolo26n.ipynb) | Pearson/PKD | 26s v62 | 26n | 42 | `14_pearson_pkd_yolo26s_to_yolo26n/<run-id>/` |
| [kd/teacher_s/15_head_yolo26s_to_yolo26n.ipynb](kd/teacher_s/15_head_yolo26s_to_yolo26n.ipynb) | [kd/teacher_s/13_head_yolo26n_teacher_yolo26s.ipynb](../kaggle/kd/teacher_s/13_head_yolo26n_teacher_yolo26s.ipynb) | HEAD | 26s v62 | 26n | 42 | `15_head_yolo26s_to_yolo26n/<run-id>/` |
| [kd/teacher_s/16_rank_mimicking_yolo26s_to_yolo26n.ipynb](kd/teacher_s/16_rank_mimicking_yolo26s_to_yolo26n.ipynb) | [kd/teacher_s/14_rm_yolo26n_teacher_yolo26s.ipynb](../kaggle/kd/teacher_s/14_rm_yolo26n_teacher_yolo26s.ipynb) | Rank Mimicking | 26s v62 | 26n | 42 | `16_rank_mimicking_yolo26s_to_yolo26n/<run-id>/` |

## Training defaults and verification

HEAD, Rank Mimicking and Pearson/PKD: `EPOCHS=100`, `RUN_FULL_TRAINING=True`, `RUN_SMOKE_TEST=False`. No additional smoke run is dispatched by default. Teacher audits and numerical checks remain active. Other workflows retain their original hyperparameters, seed queues and diagnostic behavior.

Dataset declaration checks use the original Kaggle source identity separately from the active Colab YAML. Artifact versions, existing checksums/schema/class audits, objectives and seed schedules are preserved. A Drive checkpoint does not establish exact resume support; existing `RESUME_NOT_VERIFIED` restrictions remain.

Generation and local tests do not constitute Colab GPU execution or proof of 100-epoch stability. Historical Kaggle evidence is in [experiment reports](../../reports/hasil_percobaan/README.md).

Local adaptation checks and their limits are documented in the [Colab verification report](../../reports/hasil_percobaan/colab_collection_validation_20261022.md).

Regenerate from the repository root: `python scripts/generate_colab_notebooks.py`. The notebooks themselves need no generator or repository files in Colab.

References: [Google Colab Drive/archive guidance](https://research.google.com/colaboratory/faq.html), [Colab Secrets API](https://github.com/googlecolab/colabtools/blob/main/google/colab/userdata.py).
