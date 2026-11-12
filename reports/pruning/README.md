# Laporan pruning LAMP

[Notebook 04 dan bukti recovery fine-tuning biasa](recovery/README.md) menambahkan
perbandingan baseline, pruned, dan recovered pada direktori eksperimen terpisah.
Workflow target GFLOPs saat ini memiliki notebook Kaggle dan Filkom untuk pruning N/S serta recovery per metode. Lihat [panduan bersama](../../docs/lamp_gflops_recovery_workflow.md) untuk fingerprint Poltrada, publikasi raw, konfigurasi fine-tuning/KD, dan pemisahan hasil menurut perangkat.

| Model / versi | Bukti yang tersedia | Panduan |
| --- | --- | --- |
| EDA seluruh recovery 01–05 | [Verifikasi dan status 8 November](recovery/recovery_custom_eda_verification_20261108.json), [indeks report W&B](https://wandb.ai/ilhamrafiqin-brawijaya-university/yolom26-kd-indotraffic/reports/EDA--VmlldzoxODI3MDUxNg==) | [Baseline → raw → recovery dan diagram per metrik](../../docs/lamp_gflops_recovery_workflow.md#5-baca-perbandingan-tiga-tahap) |
| YOLO26n V1 | [CPU 10%](yolo26n/v1/lamp_yolo26n_cpu_smoke_20261022.json), [CPU 30%](yolo26n/v1/lamp_ratio030_cpu_smoke_20261022.json) | [Notebook 01 dan protokol](../../docs/lamp_yolo26n_no_recovery.md) |
| YOLO26s V1 | [CPU 30%](yolo26s/v1/lamp_yolo26s_ratio030_cpu_smoke_20261022.json) | [Notebook 02 dan protokol](../../docs/lamp_yolo26s_no_recovery.md) |
| YOLO26n/S V2 | [Bukti N](yolo26n/v2/README.md), [bukti S](yolo26s/v2/README.md), dan [perbandingan N/S](comparisons/README.md); extended 0,10 / 0,20 / 0,30 / 0,40 / 0,50 secara default | [Notebook 03, EDA, dan protokol](../../docs/lamp_yolo26n_extended_no_recovery.md) |
| YOLO26n/S target GFLOPs | [Audit P5 dan metode](lamp_gflops_audit.md), [tabel historis dari pengguna](lamp_gflops_historical_unverified.csv), [verifikasi analisis N/S](lamp_gflops_eda_verification_20261105.json), [uji notebook terpisah 5 November](lamp_gflops_split_ns_verification_20261105.json) | [Notebook dan protokol Kaggle/Filkom](../../docs/lamp_gflops_recovery_workflow.md) |
| Lima target dan recovery YOLO26s | [Verifikasi CPU 6 November](lamp_recovery_workflow_verification_20261106.json) | [Fine-tuning, Native KD, CrossKD, MGD](../../docs/lamp_gflops_recovery_workflow.md#3-pilih-notebook-recovery-per-metode) |
| Sembilan target notebook 05/06 | [Feasibility N/S dan smoke notebook 05](lamp_gflops_05_nine_targets_verification_20261106.json), [replay notebook 06](lamp_gflops_06_nine_targets_verification_20261106.json) | [Grid aktif dan checkpoint maksimum bersama](../../docs/lamp_gflops_recovery_workflow.md#2-jalankan-pruning-dan-pilih-raw-pruned) |
| YOLO26s multi-seed dengan baseline sesuai seed | [Smoke dua baseline, resume, dan simulasi tujuh publikasi](lamp_yolo26s_per_seed_verification_20261107.json) | [Pruning, publikasi raw, dan registry seed](../../docs/lamp_gflops_recovery_workflow.md#pruning-yolo26s-multi-seed) |
| Fine-tuning YOLO26s satu raw per sesi | [Tujuh smoke satu epoch, EDA/W&B online, resume dan proses paralel](recovery/yolo26s_selected_finetune_verification_20261107.json) | [Pemilihan student dan protokol](../../docs/lamp_gflops_recovery_workflow.md#3-pilih-notebook-recovery-per-metode) |
| KD YOLO26s satu raw 1,20× per sesi | [15 smoke dan verifikasi W&B selesai 8 November](recovery/recovery_custom_eda_verification_20261108.json), [progres historis 7 November](recovery/yolo26s_selected_kd_verification_20261107.json), [lima raw asli pulihan](lamp_yolo26s_raw_restored_20261107.json) | [Konfigurasi KD dan teacher YOLO26m](../../docs/lamp_gflops_recovery_workflow.md#kd-yolo26s-satu-raw-pilihan) |
| YOLO26s multi-seed historis dengan baseline bersama | [Smoke CPU dan resume 7 November](lamp_yolo26s_multiseed_cpu_20261107.json), [registry Kaggle `:v0` terverifikasi](lamp_yolo26s_wandb_v0_registry.json) | [Riwayat publikasi seed 42](../../docs/ARTEFAK_WANDB.md#riwayat-publikasi-seed-42) |

Setiap hasil memiliki batas klaim pada laporan masing-masing. Structural smoke CPU
tidak membuktikan mAP Poltrada V3 atau latency CUDA. Keduanya hanya dicatat bila evaluasi
dataset dan perangkat yang sesuai benar-benar dijalankan.

Output V2 N/S berada di `/kaggle/working/pruning/lamp_v2/{suite_id}/`,
atau `outputs/pruning/lamp_v2/{suite_id}/` untuk proses lokal. Subdirektori `yolo26n/` dan
`yolo26s/` masing-masing memiliki baseline dan konfigurasi policy/rasio dengan EDA;
ringkasan lintas model berada di `comparison/`. Baseline diukur/divalidasi satu kali per model.
Diagram channel tiga panel, heatmap indeks asli, importance LAMP, serta metrik hanya memakai
hasil yang dicatat. Dataset/GPU tidak tersedia tetap `NOT_RUN`; data lama tanpa ledger atau
skor tetap `UNAVAILABLE`. Direktori output dan laporan historis V1 maupun V2 N tetap di lokasi aslinya.

[Kembali ke indeks seluruh laporan](../README.md).
