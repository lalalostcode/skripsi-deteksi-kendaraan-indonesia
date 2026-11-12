# Indeks Notebook Penelitian YOLO26

Notebook dikelompokkan berdasarkan **platform**, lalu jenis eksperimen. Nama file, experiment ID, seed, artifact W&B, dan lokasi output eksperimen dipertahankan.

## Struktur

```text
notebooks/
├── kaggle/{eda,teacher,baseline,baseline/fine-tunned,kd/teacher_m,kd/teacher_s,pruning,recovery}/
├── filkom/{teacher,pruning,recovery,fine-tunned}/
├── colab/{eda,teacher,baseline,kd/teacher_m,kd/teacher_s}/
├── amd/baseline/
├── local/{eda,analysis}/
└── archieve/
```

Notebook KD dari baseline dikelompokkan menurut teacher. Notebook KD dari raw pruned ada di `recovery/`, dipisah menurut metode. Fine-tuning recovery memakai satu notebook per varian.

Notebook Kaggle, Colab, dan Filkom memuat helper runtime sendiri. Registry path hanya digunakan generator dan tes.

62 notebook terdaftar dan 10 notebook arsip. Gunakan `recovery/02`–`05` untuk YOLO26s dengan konfigurasi asal. Salinan `fine-tunned/` adalah rujukan historis. [Matriks notebook dan protokol bersama](../docs/lamp_gflops_recovery_workflow.md)

Petunjuk platform tersedia untuk [Filkom](filkom/README.md), [Kaggle](kaggle/README.md), dan [Colab](colab/README.md). Lihat juga panduan LAMP historis [YOLO26n](../docs/lamp_yolo26n_no_recovery.md) dan [YOLO26s](../docs/lamp_yolo26s_no_recovery.md).

[Panduan LAMP V2 N/S dan EDA](../docs/lamp_yolo26n_extended_no_recovery.md) memakai sepuluh eksperimen extended pada rasio 10-50% secara default, dengan diagram channel sebelum/sesudah serta metrik efisiensi dan akurasi; policy conservative tetap tersedia. [Indeks laporan](../reports/README.md) memisahkan bukti baseline, KD, pruning, dan pengujian platform.

[Recovery fine-tuning biasa](../docs/lamp_yolo26ns_finetune_recovery.md) ada pada notebook 04 tersendiri. Notebook 03 tetap menjadi pembanding tanpa recovery; hasil ketiga tahap dibandingkan tanpa mengubah bukti historis.

## Kaggle

| Notebook | Metode / peran | Teacher | Model / student |
|---|---|---|---|
| [kaggle/eda/00_eda_poltrada_v3.ipynb](kaggle/eda/00_eda_poltrada_v3.ipynb) | EDA | — | — |
| [kaggle/eda/00_results_eda-poltrada.ipynb](kaggle/eda/00_results_eda-poltrada.ipynb) | EDA / hasil tersimpan | — | — |
| [kaggle/teacher/01_teacher_yolo26m.ipynb](kaggle/teacher/01_teacher_yolo26m.ipynb) | Pelatihan teacher | — | YOLO26m |
| [kaggle/teacher/02_teacher_yolo26m_multiseed.ipynb](kaggle/teacher/02_teacher_yolo26m_multiseed.ipynb) | Pelatihan teacher lima seed | — | YOLO26m |
| [kaggle/baseline/02_baseline_yolo26s.ipynb](kaggle/baseline/02_baseline_yolo26s.ipynb) | Baseline supervised | — | YOLO26s |
| [kaggle/baseline/03_baseline_yolo26n.ipynb](kaggle/baseline/03_baseline_yolo26n.ipynb) | Baseline supervised | — | YOLO26n |
| [kaggle/baseline/03_baseline_yolo26n_scratch.ipynb](kaggle/baseline/03_baseline_yolo26n_scratch.ipynb) | Baseline scratch | — | YOLO26n |
| [kaggle/kd/teacher_m/04_native_yolo26s.ipynb](kaggle/kd/teacher_m/04_native_yolo26s.ipynb) | Native KD | YOLO26m | YOLO26s |
| [kaggle/kd/teacher_m/05_native_yolo26n.ipynb](kaggle/kd/teacher_m/05_native_yolo26n.ipynb) | Native KD | YOLO26m | YOLO26n |
| [kaggle/kd/teacher_m/06_crosskd_yolo26s.ipynb](kaggle/kd/teacher_m/06_crosskd_yolo26s.ipynb) | CrossKD | YOLO26m | YOLO26s |
| [kaggle/kd/teacher_m/07_crosskd_yolo26n.ipynb](kaggle/kd/teacher_m/07_crosskd_yolo26n.ipynb) | CrossKD | YOLO26m | YOLO26n |
| [kaggle/kd/teacher_m/08_csakd_yolo26s.ipynb](kaggle/kd/teacher_m/08_csakd_yolo26s.ipynb) | CSAKD | YOLO26m | YOLO26s |
| [kaggle/kd/teacher_m/09_csakd_yolo26n.ipynb](kaggle/kd/teacher_m/09_csakd_yolo26n.ipynb) | CSAKD | YOLO26m | YOLO26n |
| [kaggle/kd/teacher_s/10_native_yolo26n_teacher_yolo26s.ipynb](kaggle/kd/teacher_s/10_native_yolo26n_teacher_yolo26s.ipynb) | Native KD | YOLO26s | YOLO26n |
| [kaggle/kd/teacher_s/11_crosskd_yolo26n_teacher_yolo26s.ipynb](kaggle/kd/teacher_s/11_crosskd_yolo26n_teacher_yolo26s.ipynb) | CrossKD | YOLO26s | YOLO26n |
| [kaggle/kd/teacher_s/12_csakd_yolo26n_teacher_yolo26s.ipynb](kaggle/kd/teacher_s/12_csakd_yolo26n_teacher_yolo26s.ipynb) | CSAKD | YOLO26s | YOLO26n |
| [kaggle/kd/teacher_s/12_pkd_yolo26n.ipynb](kaggle/kd/teacher_s/12_pkd_yolo26n.ipynb) | Pearson / PKD | YOLO26s | YOLO26n |
| [kaggle/kd/teacher_s/13_head_yolo26n_teacher_yolo26s.ipynb](kaggle/kd/teacher_s/13_head_yolo26n_teacher_yolo26s.ipynb) | HEAD | YOLO26s | YOLO26n |
| [kaggle/kd/teacher_s/14_rm_yolo26n_teacher_yolo26s.ipynb](kaggle/kd/teacher_s/14_rm_yolo26n_teacher_yolo26s.ipynb) | Rank Mimicking | YOLO26s | YOLO26n |
| [kaggle/kd/teacher_s/15_mgd_yolo26n_teacher_yolo26s.ipynb](kaggle/kd/teacher_s/15_mgd_yolo26n_teacher_yolo26s.ipynb) | Masked Generative Distillation / MGD | YOLO26s | YOLO26n |
| [kaggle/pruning/01_lamp_yolo26n_no_recovery.ipynb](kaggle/pruning/01_lamp_yolo26n_no_recovery.ipynb) | LAMP structured, tanpa recovery | — | YOLO26n |
| [kaggle/pruning/02_lamp_yolo26s_no_recovery.ipynb](kaggle/pruning/02_lamp_yolo26s_no_recovery.ipynb) | LAMP structured, tanpa recovery | — | YOLO26s |
| [kaggle/pruning/03_lamp_yolo26n_extended_no_recovery.ipynb](kaggle/pruning/03_lamp_yolo26n_extended_no_recovery.ipynb) | LAMP V2 conservative/extended dan EDA channel, tanpa recovery | — | YOLO26n / YOLO26s |
| [kaggle/pruning/04_lamp_yolo26ns_finetune_recovery.ipynb](kaggle/pruning/04_lamp_yolo26ns_finetune_recovery.ipynb) | LAMP structured dan recovery fine-tuning supervised biasa | — | YOLO26n / YOLO26s |
| [kaggle/pruning/05_lamp_yolo26n_gflops_no_recovery.ipynb](kaggle/pruning/05_lamp_yolo26n_gflops_no_recovery.ipynb) | LAMP target GFLOPs, kontrol P5, EDA, dan ekspor full val | — | YOLO26n |
| [kaggle/pruning/06_lamp_yolo26s_gflops_no_recovery.ipynb](kaggle/pruning/06_lamp_yolo26s_gflops_no_recovery.ipynb) | LAMP target GFLOPs, kontrol P5, EDA, dan ekspor full val | — | YOLO26s |
| [kaggle/pruning/06_lamp_yolo26s_multiseed_wandb.ipynb](kaggle/pruning/06_lamp_yolo26s_multiseed_wandb.ipynb) | LAMP tujuh perlakuan dengan baseline sesuai seed dan registry W&B untuk Kaggle atau Filkom | — | YOLO26s |
| [kaggle/recovery/01_yolo26n_standard_finetuning.ipynb](kaggle/recovery/01_yolo26n_standard_finetuning.ipynb) | Fine-tuning raw pruned | — | YOLO26n |
| [kaggle/recovery/02_yolo26s_standard_finetuning.ipynb](kaggle/recovery/02_yolo26s_standard_finetuning.ipynb) | Fine-tuning raw pruned | — | YOLO26s |
| [kaggle/recovery/03_yolo26ns_native_kd.ipynb](kaggle/recovery/03_yolo26ns_native_kd.ipynb) | Native KD dari raw pruned | YOLO26m v50 | YOLO26s |
| [kaggle/recovery/04_yolo26ns_crosskd.ipynb](kaggle/recovery/04_yolo26ns_crosskd.ipynb) | CrossKD dari raw pruned | YOLO26m v50 | YOLO26s |
| [kaggle/recovery/05_yolo26ns_mgd_kd.ipynb](kaggle/recovery/05_yolo26ns_mgd_kd.ipynb) | MGD dari raw pruned | YOLO26m v50 | YOLO26s |
| [kaggle/baseline/fine-tunned/01_yolo26s_pruned_fine_tuned.ipynb](kaggle/baseline/fine-tunned/01_yolo26s_pruned_fine_tuned.ipynb) | Rujukan historis fine-tuning baseline | — | YOLO26s |

## Filkom

| Notebook | Metode / peran | Teacher | Model / student |
|---|---|---|---|
| [filkom/teacher/02_teacher_yolo26m_multiseed.ipynb](filkom/teacher/02_teacher_yolo26m_multiseed.ipynb) | Pelatihan teacher lima seed | — | YOLO26m |
| [filkom/pruning/05_lamp_yolo26n_gflops_no_recovery.ipynb](filkom/pruning/05_lamp_yolo26n_gflops_no_recovery.ipynb) | LAMP target GFLOPs | — | YOLO26n |
| [filkom/pruning/06_lamp_yolo26s_gflops_no_recovery.ipynb](filkom/pruning/06_lamp_yolo26s_gflops_no_recovery.ipynb) | LAMP target GFLOPs | — | YOLO26s |
| [filkom/recovery/01_yolo26n_standard_finetuning.ipynb](filkom/recovery/01_yolo26n_standard_finetuning.ipynb) | Fine-tuning raw pruned | — | YOLO26n |
| [filkom/recovery/02_yolo26s_standard_finetuning.ipynb](filkom/recovery/02_yolo26s_standard_finetuning.ipynb) | Fine-tuning raw pruned | — | YOLO26s |
| [filkom/recovery/03_yolo26ns_native_kd.ipynb](filkom/recovery/03_yolo26ns_native_kd.ipynb) | Native KD dari raw pruned | YOLO26m v50 | YOLO26s |
| [filkom/recovery/04_yolo26ns_crosskd.ipynb](filkom/recovery/04_yolo26ns_crosskd.ipynb) | CrossKD dari raw pruned | YOLO26m v50 | YOLO26s |
| [filkom/recovery/05_yolo26ns_mgd_kd.ipynb](filkom/recovery/05_yolo26ns_mgd_kd.ipynb) | MGD dari raw pruned | YOLO26m v50 | YOLO26s |
| [filkom/fine-tunned/01_yolo26s_pruned_fine_tuned.ipynb](filkom/fine-tunned/01_yolo26s_pruned_fine_tuned.ipynb) | Rujukan historis fine-tuning baseline | — | YOLO26s |

## Google Colab

| Notebook | Metode / peran | Teacher | Model / student |
|---|---|---|---|
| [colab/eda/00_eda_poltrada_v3.ipynb](colab/eda/00_eda_poltrada_v3.ipynb) | EDA | — | — |
| [colab/teacher/01_teacher_yolo26m.ipynb](colab/teacher/01_teacher_yolo26m.ipynb) | Supervised teacher | — | 26m |
| [colab/baseline/02_baseline_yolo26s.ipynb](colab/baseline/02_baseline_yolo26s.ipynb) | Supervised | — | 26s |
| [colab/baseline/03_baseline_yolo26n_pretrained.ipynb](colab/baseline/03_baseline_yolo26n_pretrained.ipynb) | Pretrained baseline | — | 26n |
| [colab/baseline/04_baseline_yolo26n_scratch.ipynb](colab/baseline/04_baseline_yolo26n_scratch.ipynb) | Scratch baseline | — | 26n |
| [colab/kd/teacher_m/05_native_yolo26m_to_yolo26s.ipynb](colab/kd/teacher_m/05_native_yolo26m_to_yolo26s.ipynb) | Native KD | 26m v50 | 26s |
| [colab/kd/teacher_m/06_native_yolo26m_to_yolo26n.ipynb](colab/kd/teacher_m/06_native_yolo26m_to_yolo26n.ipynb) | Native KD | 26m v50 | 26n |
| [colab/kd/teacher_m/07_crosskd_yolo26m_to_yolo26s.ipynb](colab/kd/teacher_m/07_crosskd_yolo26m_to_yolo26s.ipynb) | CrossKD | 26m v50 | 26s |
| [colab/kd/teacher_m/08_crosskd_yolo26m_to_yolo26n.ipynb](colab/kd/teacher_m/08_crosskd_yolo26m_to_yolo26n.ipynb) | CrossKD | 26m v50 | 26n |
| [colab/kd/teacher_m/09_csakd_yolo26m_to_yolo26s.ipynb](colab/kd/teacher_m/09_csakd_yolo26m_to_yolo26s.ipynb) | CSAKD | 26m v50 | 26s |
| [colab/kd/teacher_m/10_csakd_yolo26m_to_yolo26n.ipynb](colab/kd/teacher_m/10_csakd_yolo26m_to_yolo26n.ipynb) | CSAKD | 26m v50 | 26n |
| [colab/kd/teacher_s/11_native_yolo26s_to_yolo26n.ipynb](colab/kd/teacher_s/11_native_yolo26s_to_yolo26n.ipynb) | Native KD | 26s v62 | 26n |
| [colab/kd/teacher_s/12_crosskd_yolo26s_to_yolo26n.ipynb](colab/kd/teacher_s/12_crosskd_yolo26s_to_yolo26n.ipynb) | CrossKD | 26s v62 | 26n |
| [colab/kd/teacher_s/13_csakd_yolo26s_to_yolo26n.ipynb](colab/kd/teacher_s/13_csakd_yolo26s_to_yolo26n.ipynb) | CSAKD | 26s v62 | 26n |
| [colab/kd/teacher_s/14_pearson_pkd_yolo26s_to_yolo26n.ipynb](colab/kd/teacher_s/14_pearson_pkd_yolo26s_to_yolo26n.ipynb) | Pearson/PKD | 26s v62 | 26n |
| [colab/kd/teacher_s/15_head_yolo26s_to_yolo26n.ipynb](colab/kd/teacher_s/15_head_yolo26s_to_yolo26n.ipynb) | HEAD | 26s v62 | 26n |
| [colab/kd/teacher_s/16_rank_mimicking_yolo26s_to_yolo26n.ipynb](colab/kd/teacher_s/16_rank_mimicking_yolo26s_to_yolo26n.ipynb) | Rank Mimicking | 26s v62 | 26n |

## AMD ROCm

| Notebook | Metode / peran | Teacher | Model / student |
|---|---|---|---|
| [amd/baseline/03_baseline_yolo26n_amd_roboflow.ipynb](amd/baseline/03_baseline_yolo26n_amd_roboflow.ipynb) | Baseline AMD ROCm / Roboflow | — | YOLO26n |

## Lokal

| Notebook | Metode / peran | Teacher | Model / student |
|---|---|---|---|
| [local/eda/01_eda_dataset.ipynb](local/eda/01_eda_dataset.ipynb) | EDA | — | — |
| [local/analysis/03_baseline_seed42_failure_analysis.ipynb](local/analysis/03_baseline_seed42_failure_analysis.ipynb) | Failure analysis baseline | — | YOLO26n |

Catatan legacy lokal. Notebook `local/eda/01_eda_dataset.ipynb` memiliki satu code cell dengan sintaks Python tidak valid sejak sebelum penataan (indeks cell 7, baris 3). Pemeriksaan sintaks mencakup Kaggle, Filkom, Colab, dan AMD. Tes schema juga mencakup notebook lokal.

## Arsip

[Notebook modular](archieve/modular/), [multi-scale feature fusion](archieve/multi_scale_feature_fusion/), dan [Feature Based Knowledge Distilation](archieve/Feature%20Based%20Knowledge%20Distilation/) tetap di tempat semula. Isi dan hash sepuluh notebook arsip tidak diubah.

## Pemetaan lokasi lama ke baru

Laporan historis mempertahankan path eksekusi dan hash pada saat pengujian. Gunakan tabel ini untuk menemukan lokasi aktifnya. Folder lama tidak berisi salinan notebook atau symlink.

| Lokasi lama (relatif root repository) | Lokasi sekarang |
|---|---|
| `notebooks/yolo26-kd-indotraffic/notebooks/00_eda_poltrada_v3.ipynb` | [notebooks/kaggle/eda/00_eda_poltrada_v3.ipynb](kaggle/eda/00_eda_poltrada_v3.ipynb) |
| `notebooks/yolo26-kd-indotraffic/notebooks/00_results_eda-poltrada.ipynb` | [notebooks/kaggle/eda/00_results_eda-poltrada.ipynb](kaggle/eda/00_results_eda-poltrada.ipynb) |
| `notebooks/yolo26-kd-indotraffic/notebooks/01_teacher_yolo26m.ipynb` | [notebooks/kaggle/teacher/01_teacher_yolo26m.ipynb](kaggle/teacher/01_teacher_yolo26m.ipynb) |
| `notebooks/yolo26-kd-indotraffic/notebooks/02_baseline_yolo26s.ipynb` | [notebooks/kaggle/baseline/02_baseline_yolo26s.ipynb](kaggle/baseline/02_baseline_yolo26s.ipynb) |
| `notebooks/yolo26-kd-indotraffic/notebooks/03_baseline_yolo26n.ipynb` | [notebooks/kaggle/baseline/03_baseline_yolo26n.ipynb](kaggle/baseline/03_baseline_yolo26n.ipynb) |
| `notebooks/yolo26-kd-indotraffic/notebooks/03_baseline_yolo26n_scratch.ipynb` | [notebooks/kaggle/baseline/03_baseline_yolo26n_scratch.ipynb](kaggle/baseline/03_baseline_yolo26n_scratch.ipynb) |
| `notebooks/yolo26-kd-indotraffic/notebooks/04_native_yolo26s.ipynb` | [notebooks/kaggle/kd/teacher_m/04_native_yolo26s.ipynb](kaggle/kd/teacher_m/04_native_yolo26s.ipynb) |
| `notebooks/yolo26-kd-indotraffic/notebooks/05_native_yolo26n.ipynb` | [notebooks/kaggle/kd/teacher_m/05_native_yolo26n.ipynb](kaggle/kd/teacher_m/05_native_yolo26n.ipynb) |
| `notebooks/yolo26-kd-indotraffic/notebooks/06_crosskd_yolo26s.ipynb` | [notebooks/kaggle/kd/teacher_m/06_crosskd_yolo26s.ipynb](kaggle/kd/teacher_m/06_crosskd_yolo26s.ipynb) |
| `notebooks/yolo26-kd-indotraffic/notebooks/07_crosskd_yolo26n.ipynb` | [notebooks/kaggle/kd/teacher_m/07_crosskd_yolo26n.ipynb](kaggle/kd/teacher_m/07_crosskd_yolo26n.ipynb) |
| `notebooks/yolo26-kd-indotraffic/notebooks/08_csakd_yolo26s.ipynb` | [notebooks/kaggle/kd/teacher_m/08_csakd_yolo26s.ipynb](kaggle/kd/teacher_m/08_csakd_yolo26s.ipynb) |
| `notebooks/yolo26-kd-indotraffic/notebooks/09_csakd_yolo26n.ipynb` | [notebooks/kaggle/kd/teacher_m/09_csakd_yolo26n.ipynb](kaggle/kd/teacher_m/09_csakd_yolo26n.ipynb) |
| `notebooks/yolo26-kd-indotraffic/notebooks/10_native_yolo26n_teacher_yolo26s.ipynb` | [notebooks/kaggle/kd/teacher_s/10_native_yolo26n_teacher_yolo26s.ipynb](kaggle/kd/teacher_s/10_native_yolo26n_teacher_yolo26s.ipynb) |
| `notebooks/yolo26-kd-indotraffic/notebooks/11_crosskd_yolo26n_teacher_yolo26s.ipynb` | [notebooks/kaggle/kd/teacher_s/11_crosskd_yolo26n_teacher_yolo26s.ipynb](kaggle/kd/teacher_s/11_crosskd_yolo26n_teacher_yolo26s.ipynb) |
| `notebooks/yolo26-kd-indotraffic/notebooks/12_csakd_yolo26n_teacher_yolo26s.ipynb` | [notebooks/kaggle/kd/teacher_s/12_csakd_yolo26n_teacher_yolo26s.ipynb](kaggle/kd/teacher_s/12_csakd_yolo26n_teacher_yolo26s.ipynb) |
| `notebooks/yolo26-kd-indotraffic/notebooks/12_pkd_yolo26n.ipynb` | [notebooks/kaggle/kd/teacher_s/12_pkd_yolo26n.ipynb](kaggle/kd/teacher_s/12_pkd_yolo26n.ipynb) |
| `notebooks/yolo26-kd-indotraffic/notebooks/13_head_yolo26n_teacher_yolo26s.ipynb` | [notebooks/kaggle/kd/teacher_s/13_head_yolo26n_teacher_yolo26s.ipynb](kaggle/kd/teacher_s/13_head_yolo26n_teacher_yolo26s.ipynb) |
| `notebooks/yolo26-kd-indotraffic/notebooks/14_rm_yolo26n_teacher_yolo26s.ipynb` | [notebooks/kaggle/kd/teacher_s/14_rm_yolo26n_teacher_yolo26s.ipynb](kaggle/kd/teacher_s/14_rm_yolo26n_teacher_yolo26s.ipynb) |
| `notebooks/pruning/01_lamp_yolo26n_no_recovery.ipynb` | [notebooks/kaggle/pruning/01_lamp_yolo26n_no_recovery.ipynb](kaggle/pruning/01_lamp_yolo26n_no_recovery.ipynb) |
| `notebooks/yolo26-kd-indotraffic/colab/00_eda_poltrada_v3.ipynb` | [notebooks/colab/eda/00_eda_poltrada_v3.ipynb](colab/eda/00_eda_poltrada_v3.ipynb) |
| `notebooks/yolo26-kd-indotraffic/colab/01_teacher_yolo26m.ipynb` | [notebooks/colab/teacher/01_teacher_yolo26m.ipynb](colab/teacher/01_teacher_yolo26m.ipynb) |
| `notebooks/yolo26-kd-indotraffic/colab/02_baseline_yolo26s.ipynb` | [notebooks/colab/baseline/02_baseline_yolo26s.ipynb](colab/baseline/02_baseline_yolo26s.ipynb) |
| `notebooks/yolo26-kd-indotraffic/colab/03_baseline_yolo26n_pretrained.ipynb` | [notebooks/colab/baseline/03_baseline_yolo26n_pretrained.ipynb](colab/baseline/03_baseline_yolo26n_pretrained.ipynb) |
| `notebooks/yolo26-kd-indotraffic/colab/04_baseline_yolo26n_scratch.ipynb` | [notebooks/colab/baseline/04_baseline_yolo26n_scratch.ipynb](colab/baseline/04_baseline_yolo26n_scratch.ipynb) |
| `notebooks/yolo26-kd-indotraffic/colab/05_native_yolo26m_to_yolo26s.ipynb` | [notebooks/colab/kd/teacher_m/05_native_yolo26m_to_yolo26s.ipynb](colab/kd/teacher_m/05_native_yolo26m_to_yolo26s.ipynb) |
| `notebooks/yolo26-kd-indotraffic/colab/06_native_yolo26m_to_yolo26n.ipynb` | [notebooks/colab/kd/teacher_m/06_native_yolo26m_to_yolo26n.ipynb](colab/kd/teacher_m/06_native_yolo26m_to_yolo26n.ipynb) |
| `notebooks/yolo26-kd-indotraffic/colab/07_crosskd_yolo26m_to_yolo26s.ipynb` | [notebooks/colab/kd/teacher_m/07_crosskd_yolo26m_to_yolo26s.ipynb](colab/kd/teacher_m/07_crosskd_yolo26m_to_yolo26s.ipynb) |
| `notebooks/yolo26-kd-indotraffic/colab/08_crosskd_yolo26m_to_yolo26n.ipynb` | [notebooks/colab/kd/teacher_m/08_crosskd_yolo26m_to_yolo26n.ipynb](colab/kd/teacher_m/08_crosskd_yolo26m_to_yolo26n.ipynb) |
| `notebooks/yolo26-kd-indotraffic/colab/09_csakd_yolo26m_to_yolo26s.ipynb` | [notebooks/colab/kd/teacher_m/09_csakd_yolo26m_to_yolo26s.ipynb](colab/kd/teacher_m/09_csakd_yolo26m_to_yolo26s.ipynb) |
| `notebooks/yolo26-kd-indotraffic/colab/10_csakd_yolo26m_to_yolo26n.ipynb` | [notebooks/colab/kd/teacher_m/10_csakd_yolo26m_to_yolo26n.ipynb](colab/kd/teacher_m/10_csakd_yolo26m_to_yolo26n.ipynb) |
| `notebooks/yolo26-kd-indotraffic/colab/11_native_yolo26s_to_yolo26n.ipynb` | [notebooks/colab/kd/teacher_s/11_native_yolo26s_to_yolo26n.ipynb](colab/kd/teacher_s/11_native_yolo26s_to_yolo26n.ipynb) |
| `notebooks/yolo26-kd-indotraffic/colab/12_crosskd_yolo26s_to_yolo26n.ipynb` | [notebooks/colab/kd/teacher_s/12_crosskd_yolo26s_to_yolo26n.ipynb](colab/kd/teacher_s/12_crosskd_yolo26s_to_yolo26n.ipynb) |
| `notebooks/yolo26-kd-indotraffic/colab/13_csakd_yolo26s_to_yolo26n.ipynb` | [notebooks/colab/kd/teacher_s/13_csakd_yolo26s_to_yolo26n.ipynb](colab/kd/teacher_s/13_csakd_yolo26s_to_yolo26n.ipynb) |
| `notebooks/yolo26-kd-indotraffic/colab/14_pearson_pkd_yolo26s_to_yolo26n.ipynb` | [notebooks/colab/kd/teacher_s/14_pearson_pkd_yolo26s_to_yolo26n.ipynb](colab/kd/teacher_s/14_pearson_pkd_yolo26s_to_yolo26n.ipynb) |
| `notebooks/yolo26-kd-indotraffic/colab/15_head_yolo26s_to_yolo26n.ipynb` | [notebooks/colab/kd/teacher_s/15_head_yolo26s_to_yolo26n.ipynb](colab/kd/teacher_s/15_head_yolo26s_to_yolo26n.ipynb) |
| `notebooks/yolo26-kd-indotraffic/colab/16_rank_mimicking_yolo26s_to_yolo26n.ipynb` | [notebooks/colab/kd/teacher_s/16_rank_mimicking_yolo26s_to_yolo26n.ipynb](colab/kd/teacher_s/16_rank_mimicking_yolo26s_to_yolo26n.ipynb) |
| `notebooks/yolo26-kd-indotraffic/notebooks/03_baseline_yolo26n_amd_roboflow.ipynb` | [notebooks/amd/baseline/03_baseline_yolo26n_amd_roboflow.ipynb](amd/baseline/03_baseline_yolo26n_amd_roboflow.ipynb) |
| `notebooks/01_eda_dataset.ipynb` | [notebooks/local/eda/01_eda_dataset.ipynb](local/eda/01_eda_dataset.ipynb) |
| `notebooks/03_baseline_seed42_failure_analysis.ipynb` | [notebooks/local/analysis/03_baseline_seed42_failure_analysis.ipynb](local/analysis/03_baseline_seed42_failure_analysis.ipynb) |

## Tooling dan verifikasi

`scripts/notebook_paths.py` berisi pemetaan eksplisit dan `notebook_path(platform, filename)`. Penambahan notebook baru harus didaftarkan agar generator dan tes tidak menebak peran teacher.

Generator Colab: `python scripts/generate_colab_notebooks.py`. Generator membaca source Kaggle pada kategori baru dan menulis output Colab pada kategori yang sama; nama folder output Drive tetap mengikuti filename semula.

Generator modular tetap menghasilkan notebook dalam arsip modular. Penataan ini tidak menjalankan training, download dataset/artifact, atau eksperimen GPU.
