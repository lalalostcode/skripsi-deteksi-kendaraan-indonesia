# Peta repositori skripsi YOLO26

Gunakan peta ini sebelum menyunting file, menjalankan eksperimen, atau mencari bukti penelitian. Mulai dari [dokumen proyek](#dokumen-proyek), [notebook](#notebook), [routing W&B](#routing-wb), [kode dan hasil](#kode-dan-hasil), atau [referensi](#referensi).

## Dokumen proyek

Lihat tabel [dokumen skripsi lokal](#dokumen-skripsi-lokal) untuk naskah, revisi, dan arsip.

| Kebutuhan | Dokumen | Kapan dipakai | Lokasi |
| --- | --- | --- | --- |
| Aturan kerja | Panduan kerja repositori skripsi YOLO26 | Baca sebelum menyunting repo atau mencatat temuan penelitian. | [AGENTS.md](AGENTS.md) |
| Setup proyek | Notebook Penelitian YOLO26: Baseline, KD, dan Pruning | Mulai dari sini untuk dependensi, dataset, dan pipeline modular. | [README.md](README.md) |
| Navigasi | Peta repositori skripsi YOLO26 | Cari jalur dokumen, notebook, hasil, sumber, atau run W&B. | [peta-repo.md](peta-repo.md) |
| Penamaan W&B | Penamaan eksperimen W&B YOLO26 | Pakai saat menamai run baru atau mencari run berdasarkan metode. | [docs/wandb_naming.md](docs/wandb_naming.md) |
| Inventaris W&B | Artefak baseline, teacher, dan raw student | Temukan versi checkpoint serta pemetaan seed untuk recovery. | [docs/ARTEFAK_WANDB.md](docs/ARTEFAK_WANDB.md) |
| Registry historis raw LAMP YOLO26s | Tiga versi W&B `:v0` terverifikasi | Telusuri raw dan SHA-256 dari desain dengan baseline bersama seed 42. | [Registry JSON historis](reports/pruning/lamp_yolo26s_wandb_v0_registry.json) |
| Audit PKD | Audit implementasi PKD YOLO26s → YOLO26n | Telusuri rumus Pearson, pemilihan tap, dan validasi implementasi. | [docs/pkd_yolo26n_implementation_audit.md](docs/pkd_yolo26n_implementation_audit.md) |
| LAMP V1 YOLO26n | LAMP structured pruning YOLO26n tanpa recovery | Jalankan pruning n tanpa recovery dan cek pemuatan ulang checkpoint. | [docs/lamp_yolo26n_no_recovery.md](docs/lamp_yolo26n_no_recovery.md) |
| LAMP V1 YOLO26s | LAMP structured pruning YOLO26s tanpa recovery | Jalankan pruning s tanpa recovery dengan sumber checkpoint varian s. | [docs/lamp_yolo26s_no_recovery.md](docs/lamp_yolo26s_no_recovery.md) |
| LAMP V2 n/s | LAMP structured pruning V2 YOLO26n/S tanpa recovery dan EDA | Pilih conservative atau extended, rasio, dan diagnostik channel. | [docs/lamp_yolo26n_extended_no_recovery.md](docs/lamp_yolo26n_extended_no_recovery.md) |
| LAMP V2 recovery | LAMP YOLO26n/S dengan recovery fine-tuning biasa | Bandingkan baseline, pruned, dan recovered dengan fine-tuning. | [docs/lamp_yolo26ns_finetune_recovery.md](docs/lamp_yolo26ns_finetune_recovery.md) |
| Audit LAMP target GFLOPs | Audit P5 dan seleksi student YOLO26n/S | Telusuri bukti historis, metode, dan batas klaim notebook 05/06. | [Laporan audit](reports/pruning/lamp_gflops_audit.md) |
| Pruning dan recovery per metode | Workflow YOLO26n/S pada Poltrada V3 | Jalankan grid target GFLOPs dan feasibility per model, lalu gunakan raw yang sama untuk fine-tuning, Native KD, CrossKD, serta MGD. | [Panduan bersama Kaggle/Filkom](docs/lamp_gflops_recovery_workflow.md) |
| Recovery YOLO26s | Fine-tuning, Native KD, CrossKD dan MGD dengan konfigurasi asal | Mulai semua metode dari raw yang sama. | [Panduan recovery](docs/lamp_gflops_recovery_workflow.md#3-pilih-notebook-recovery-per-metode) |
| Indeks notebook | Indeks Notebook Penelitian YOLO26 | Bandingkan susunan notebook antarplatform dan status arsip. | [notebooks/README.md](notebooks/README.md) |
| Panduan Kaggle | YOLO26 — Panduan Notebook Kaggle | Cari konfigurasi penelitian, dataset, artifact teacher, dan aturan eksekusi Kaggle. | [notebooks/kaggle/README.md](notebooks/kaggle/README.md) |
| Panduan Filkom | Notebook YOLO26 pada GPU Filkom UB | Siapkan kernel Jupyter lewat SSH serta path dataset, checkpoint, output, dan W&B. | [notebooks/filkom/README.md](notebooks/filkom/README.md) |
| Panduan Colab | YOLO26 research notebooks — Google Colab | Siapkan runtime Colab dan cari padanan notebook Kaggle. | [notebooks/colab/README.md](notebooks/colab/README.md) |
| Indeks laporan | Indeks laporan penelitian | Temukan bukti menurut baseline, EDA, KD, pruning, dan engineering. | [reports/README.md](reports/README.md) |
| Kesimpulan penelitian | Log kesimpulan penelitian YOLO26 | Baca temuan terdahulu sebelum mengulang atau mengubah metode. | [reports/log-penelitian.md](reports/log-penelitian.md) |
| Baseline AMD | Baseline YOLO26n pada AMD ROCm cloud dan Roboflow | Telusuri setup ROCm dan pengambilan Poltrada V3 melalui Roboflow. | [reports/baseline/amd/amd_baseline_roboflow_implementation.md](reports/baseline/amd/amd_baseline_roboflow_implementation.md) |
| EDA Poltrada V3 | Laporan implementasi EDA Poltrada V3 | Lihat diagnostik dataset dan keluaran yang mendukung analisis. | [reports/eda/eda_poltrada_v3_implementation.md](reports/eda/eda_poltrada_v3_implementation.md) |
| Multi-seed | Multi-seed serial: baseline n, Native s → n, CrossKD s → n | Periksa urutan run independen, konfigurasi, dan resume. | [reports/engineering/serial_multiseed_implementation.md](reports/engineering/serial_multiseed_implementation.md) |
| Indeks bukti eksekusi | Hasil percobaan GPU | Pilih bukti smoke test berdasarkan platform dan metode. | [reports/experiments/README.md](reports/experiments/README.md) |
| Validasi Colab | Colab collection — local verification, 22 October 2026 | Periksa cakupan uji lokal dan batas verifikasi koleksi Colab. | [reports/experiments/colab/colab_collection_validation_20261022.md](reports/experiments/colab/colab_collection_validation_20261022.md) |
| Smoke HEAD dan RM | HEAD/RM Kaggle smoke test — 21–22 Oktober 2026 | Telusuri uji Kaggle dan batas hasil kedua metode. | [reports/experiments/kaggle/head_rm_kaggle_smoke_20261021.md](reports/experiments/kaggle/head_rm_kaggle_smoke_20261021.md) |
| Smoke PKD | Pearson/PKD — smoke GPU 22 Oktober 2026 | Periksa jalur GPU PKD dan bukti eksekusi yang tersedia. | [reports/experiments/kaggle/pkd_kaggle_smoke_20261022.md](reports/experiments/kaggle/pkd_kaggle_smoke_20261022.md) |
| Smoke MGD | MGD: smoke pipeline lokal, 31 Oktober 2026 | Telusuri pipeline CPU dan teacher v62 yang dipakai. | [reports/experiments/local/mgd_cpu_smoke_20261031.md](reports/experiments/local/mgd_cpu_smoke_20261031.md) |
| Audit CSAKD | Audit dan perbaikan adaptasi CSAKD pada YOLO26 | Periksa kesetiaan loss, perbaikan hook, dan bukti OOM. | [reports/kd/csakd/csakd_audit.md](reports/kd/csakd/csakd_audit.md) |
| CSAKD teacher s | CSAKD teacher s → student n dan pengorganisasian W&B | Telusuri setup teacher s, verifikasi sintetis, dan konfigurasi tracking historis. | [reports/kd/csakd/csakd_teacher_s_and_tracking.md](reports/kd/csakd/csakd_teacher_s_and_tracking.md) |
| Implementasi MGD | MGD YOLO26s → YOLO26n — implementasi Kaggle | Telusuri loss, provenance, dan adaptasi YOLO26s → YOLO26n. | [reports/kd/mgd/mgd_yolo26n_implementation.md](reports/kd/mgd/mgd_yolo26n_implementation.md) |
| KD teacher s | KD YOLO26s ke YOLO26n — mode langsung 100 epoch | Bandingkan konfigurasi Native dan CrossKD teacher YOLO26s. | [reports/kd/teacher_s/teacher_s_kd_implementation.md](reports/kd/teacher_s/teacher_s_kd_implementation.md) |
| Indeks pruning | Laporan pruning LAMP | Pilih bukti LAMP berdasarkan versi, varian model, dan tahap recovery. | [reports/pruning/README.md](reports/pruning/README.md) |
| Perbandingan LAMP V2 | Perbandingan LAMP V2 YOLO26n dan YOLO26s | Cari bukti komparasi n/s pada konfigurasi yang didokumentasikan. | [reports/pruning/comparisons/README.md](reports/pruning/comparisons/README.md) |
| Bukti recovery | Bukti LAMP dengan recovery fine-tuning | Telusuri verifikasi CPU tahap pruning dan fine-tuning. | [reports/pruning/recovery/README.md](reports/pruning/recovery/README.md) |
| EDA semua recovery | Baseline asal → raw pruned → hasil recovery | Buka diagram per metrik dan halaman W&B dari recovery 01–05 Kaggle/Filkom, termasuk hasil parsial. | [Panduan EDA](docs/lamp_gflops_recovery_workflow.md#5-baca-perbandingan-tiga-tahap) · [Publisher hasil lama](scripts/publish_recovery_custom_eda.py) |
| Penyimpanan recovery | Best/latest dan pembersihan smoke | Periksa retensi checkpoint, resume, serta receipt penghapusan diagnostik lokal/W&B. | [Panduan retensi](docs/lamp_gflops_recovery_workflow.md#retensi-checkpoint-dan-pembersihan-smoke) · [Receipt](reports/pruning/recovery/smoke_cleanup_20261108/) |
| Bukti V2 YOLO26n | Bukti LAMP V2 YOLO26n | Cari verifikasi dan smoke test V2 untuk varian n. | [reports/pruning/yolo26n/v2/README.md](reports/pruning/yolo26n/v2/README.md) |
| Bukti V2 YOLO26s | Bukti LAMP V2 YOLO26s | Cari bukti V2 varian s dan rujukan komparasi n/s. | [reports/pruning/yolo26s/v2/README.md](reports/pruning/yolo26s/v2/README.md) |

## Dokumen skripsi lokal

Ada 17 dokumen dan simpanan web di `Jurnal/`. Folder ini diabaikan Git. Simpanan web di `Prunning/` memakai aset pendukung di folder yang sama.

| Kebutuhan | Dokumen | Kapan dipakai | Lokasi |
| --- | --- | --- | --- |
| Arsip | 235150300111012_ILHAMRAFIQIN_Rev1.pdf | Rujuk versi proposal atau naskah historis. | [Jurnal/Archieve/235150300111012_ILHAMRAFIQIN_Rev1.pdf](Jurnal/Archieve/235150300111012_ILHAMRAFIQIN_Rev1.pdf) |
| Arsip | PROPOSAL_SKRIPSI_ILHAM_RAFIQIN_235150300111012.pdf | Rujuk versi proposal atau naskah historis. | [Jurnal/Archieve/PROPOSAL_SKRIPSI_ILHAM_RAFIQIN_235150300111012.pdf](Jurnal/Archieve/PROPOSAL_SKRIPSI_ILHAM_RAFIQIN_235150300111012.pdf) |
| Arsip | Prapoposal_IlhamRafiqin_235150300111012.pdf | Rujuk versi proposal atau naskah historis. | [Jurnal/Archieve/Prapoposal_IlhamRafiqin_235150300111012.pdf](Jurnal/Archieve/Prapoposal_IlhamRafiqin_235150300111012.pdf) |
| Arsip | Proposal_Skripsi_Ilham Rafiqin_235150300111.pdf | Rujuk versi proposal atau naskah historis. | [Jurnal/Archieve/Proposal_Skripsi_Ilham Rafiqin_235150300111.pdf](Jurnal/Archieve/Proposal_Skripsi_Ilham%20Rafiqin_235150300111.pdf) |
| Arsip | SKRIPSI_P0_ILHAMRAFIQIN_235150300111012.pdf | Rujuk versi proposal atau naskah historis. | [Jurnal/Archieve/SKRIPSI_P0_ILHAMRAFIQIN_235150300111012.pdf](Jurnal/Archieve/SKRIPSI_P0_ILHAMRAFIQIN_235150300111012.pdf) |
| Naskah KD | ILHAMRAFIQIN_235150300111012_SKRIPSI_KD_BAB1-5.docx | Buka naskah KD bab 1–5. | [Jurnal/ILHAMRAFIQIN_235150300111012_SKRIPSI_KD_BAB1-5.docx](Jurnal/ILHAMRAFIQIN_235150300111012_SKRIPSI_KD_BAB1-5.docx) |
| Contoh skripsi | Contoh Skripsi Non Implementatif (1).pdf | Lihat contoh penulisan skripsi nonimplementatif. | [Jurnal/Jurnal Referensi/Contoh Skripsi Non Implementatif (1).pdf](Jurnal/Jurnal%20Referensi/Contoh%20Skripsi%20Non%20Implementatif%20%281%29.pdf) |
| Arsip web | ,  l.html | Buka salinan HTML dari folder yang sama dengan aset pendukungnya. | [Jurnal/Prunning/,  l.html](Jurnal/Prunning/,%20%20l.html) |
| Revisi KD | SKRIPSI_REVISI_KD_BAB1-3.docx | Gunakan naskah yang dapat diedit untuk bab 1-3. | [Jurnal/Revisi_KD/SKRIPSI_REVISI_KD_BAB1-3.docx](Jurnal/Revisi_KD/SKRIPSI_REVISI_KD_BAB1-3.docx) |
| Revisi KD | SKRIPSI_REVISI_KD_BAB1-3.pdf | Gunakan salinan untuk ditinjau untuk bab 1-3. | [Jurnal/Revisi_KD/SKRIPSI_REVISI_KD_BAB1-3.pdf](Jurnal/Revisi_KD/SKRIPSI_REVISI_KD_BAB1-3.pdf) |
| Revisi KD | SKRIPSI_REVISI_KD_BAB1-5.docx | Gunakan naskah yang dapat diedit untuk bab 1-5. | [Jurnal/Revisi_KD/SKRIPSI_REVISI_KD_BAB1-5.docx](Jurnal/Revisi_KD/SKRIPSI_REVISI_KD_BAB1-5.docx) |
| Revisi KD | SKRIPSI_REVISI_KD_BAB1-5.pdf | Gunakan salinan untuk ditinjau untuk bab 1-5. | [Jurnal/Revisi_KD/SKRIPSI_REVISI_KD_BAB1-5.pdf](Jurnal/Revisi_KD/SKRIPSI_REVISI_KD_BAB1-5.pdf) |
| Revisi KD | SKRIPSI_REVISI_KD_BAB3-5.docx | Gunakan naskah yang dapat diedit untuk bab 3-5. | [Jurnal/Revisi_KD/SKRIPSI_REVISI_KD_BAB3-5.docx](Jurnal/Revisi_KD/SKRIPSI_REVISI_KD_BAB3-5.docx) |
| Revisi KD | SKRIPSI_REVISI_KD_BAB3-5.pdf | Gunakan salinan untuk ditinjau untuk bab 3-5. | [Jurnal/Revisi_KD/SKRIPSI_REVISI_KD_BAB3-5.pdf](Jurnal/Revisi_KD/SKRIPSI_REVISI_KD_BAB3-5.pdf) |
| Panduan FILKOM | PanduanSkripsiFilkom-v3.0.pdf | Cek persyaratan dan format penulisan skripsi. | [Jurnal/Rules/PanduanSkripsiFilkom-v3.0.pdf](Jurnal/Rules/PanduanSkripsiFilkom-v3.0.pdf) |
| Naskah resampling | SKRIPSI_RESAMPLING_ILHAMRAFIQIN.docx | Rujuk naskah untuk topik resampling. | [Jurnal/SKRIPSI_RESAMPLING_ILHAMRAFIQIN.docx](Jurnal/SKRIPSI_RESAMPLING_ILHAMRAFIQIN.docx) |
| Template | Template-Skripsi-v3.0 (1).docx | Gunakan untuk format dokumen skripsi. | [Jurnal/Template-Skripsi-v3.0 (1).docx](Jurnal/Template-Skripsi-v3.0%20%281%29.docx) |

## Notebook

Ada 62 notebook aktif dan 10 notebook arsip. Lihat [indeks notebook](notebooks/README.md) dan panduan [Filkom](notebooks/filkom/README.md), [Kaggle](notebooks/kaggle/README.md), atau [Colab](notebooks/colab/README.md) untuk cara menjalankannya. Gunakan metode dan pasangan model untuk mencari padanannya.

### Kaggle

| Metode atau peran | Teacher | Model atau student | Notebook |
| --- | --- | --- | --- |
| Baseline pretrained | — | YOLO26s | [baseline/02_baseline_yolo26s.ipynb](notebooks/kaggle/baseline/02_baseline_yolo26s.ipynb) |
| Baseline pretrained | — | YOLO26n | [baseline/03_baseline_yolo26n.ipynb](notebooks/kaggle/baseline/03_baseline_yolo26n.ipynb) |
| Baseline scratch | — | YOLO26n | [baseline/03_baseline_yolo26n_scratch.ipynb](notebooks/kaggle/baseline/03_baseline_yolo26n_scratch.ipynb) |
| EDA Poltrada V3 | — | — | [eda/00_eda_poltrada_v3.ipynb](notebooks/kaggle/eda/00_eda_poltrada_v3.ipynb) |
| Hasil EDA Poltrada V3 tersimpan | — | — | [eda/00_results_eda-poltrada.ipynb](notebooks/kaggle/eda/00_results_eda-poltrada.ipynb) |
| Native KD | YOLO26m | YOLO26s | [kd/teacher_m/04_native_yolo26s.ipynb](notebooks/kaggle/kd/teacher_m/04_native_yolo26s.ipynb) |
| Native KD | YOLO26m | YOLO26n | [kd/teacher_m/05_native_yolo26n.ipynb](notebooks/kaggle/kd/teacher_m/05_native_yolo26n.ipynb) |
| CrossKD | YOLO26m | YOLO26s | [kd/teacher_m/06_crosskd_yolo26s.ipynb](notebooks/kaggle/kd/teacher_m/06_crosskd_yolo26s.ipynb) |
| CrossKD | YOLO26m | YOLO26n | [kd/teacher_m/07_crosskd_yolo26n.ipynb](notebooks/kaggle/kd/teacher_m/07_crosskd_yolo26n.ipynb) |
| CSAKD | YOLO26m | YOLO26s | [kd/teacher_m/08_csakd_yolo26s.ipynb](notebooks/kaggle/kd/teacher_m/08_csakd_yolo26s.ipynb) |
| CSAKD | YOLO26m | YOLO26n | [kd/teacher_m/09_csakd_yolo26n.ipynb](notebooks/kaggle/kd/teacher_m/09_csakd_yolo26n.ipynb) |
| Native KD | YOLO26s | YOLO26n | [kd/teacher_s/10_native_yolo26n_teacher_yolo26s.ipynb](notebooks/kaggle/kd/teacher_s/10_native_yolo26n_teacher_yolo26s.ipynb) |
| CrossKD | YOLO26s | YOLO26n | [kd/teacher_s/11_crosskd_yolo26n_teacher_yolo26s.ipynb](notebooks/kaggle/kd/teacher_s/11_crosskd_yolo26n_teacher_yolo26s.ipynb) |
| CSAKD | YOLO26s | YOLO26n | [kd/teacher_s/12_csakd_yolo26n_teacher_yolo26s.ipynb](notebooks/kaggle/kd/teacher_s/12_csakd_yolo26n_teacher_yolo26s.ipynb) |
| PKD | YOLO26s | YOLO26n | [kd/teacher_s/12_pkd_yolo26n.ipynb](notebooks/kaggle/kd/teacher_s/12_pkd_yolo26n.ipynb) |
| HEAD | YOLO26s | YOLO26n | [kd/teacher_s/13_head_yolo26n_teacher_yolo26s.ipynb](notebooks/kaggle/kd/teacher_s/13_head_yolo26n_teacher_yolo26s.ipynb) |
| Rank Mimicking | YOLO26s | YOLO26n | [kd/teacher_s/14_rm_yolo26n_teacher_yolo26s.ipynb](notebooks/kaggle/kd/teacher_s/14_rm_yolo26n_teacher_yolo26s.ipynb) |
| MGD | YOLO26s | YOLO26n | [kd/teacher_s/15_mgd_yolo26n_teacher_yolo26s.ipynb](notebooks/kaggle/kd/teacher_s/15_mgd_yolo26n_teacher_yolo26s.ipynb) |
| LAMP V1 tanpa recovery | — | YOLO26n | [pruning/01_lamp_yolo26n_no_recovery.ipynb](notebooks/kaggle/pruning/01_lamp_yolo26n_no_recovery.ipynb) |
| LAMP V1 tanpa recovery | — | YOLO26s | [pruning/02_lamp_yolo26s_no_recovery.ipynb](notebooks/kaggle/pruning/02_lamp_yolo26s_no_recovery.ipynb) |
| LAMP V2 tanpa recovery | — | YOLO26n/s | [pruning/03_lamp_yolo26n_extended_no_recovery.ipynb](notebooks/kaggle/pruning/03_lamp_yolo26n_extended_no_recovery.ipynb) |
| LAMP V2 + recovery | — | YOLO26n/s | [pruning/04_lamp_yolo26ns_finetune_recovery.ipynb](notebooks/kaggle/pruning/04_lamp_yolo26ns_finetune_recovery.ipynb) |
| LAMP sembilan target GFLOPs dan audit P5 | — | YOLO26n | [pruning/05_lamp_yolo26n_gflops_no_recovery.ipynb](notebooks/kaggle/pruning/05_lamp_yolo26n_gflops_no_recovery.ipynb) |
| LAMP sembilan target GFLOPs dan audit P5 | — | YOLO26s | [pruning/06_lamp_yolo26s_gflops_no_recovery.ipynb](notebooks/kaggle/pruning/06_lamp_yolo26s_gflops_no_recovery.ipynb) |
| LAMP tujuh perlakuan dan lima baseline per seed | — | YOLO26s | [06 multi-seed · Kaggle dan Filkom](notebooks/kaggle/pruning/06_lamp_yolo26s_multiseed_wandb.ipynb) |
| Fine-tuning raw pruned | — | YOLO26n | [recovery/01_yolo26n_standard_finetuning.ipynb](notebooks/kaggle/recovery/01_yolo26n_standard_finetuning.ipynb) |
| Fine-tuning raw pruned | — | YOLO26s | [recovery/02_yolo26s_standard_finetuning.ipynb](notebooks/kaggle/recovery/02_yolo26s_standard_finetuning.ipynb) |
| Native KD raw pruned | YOLO26m v50 | YOLO26s | [recovery/03_yolo26ns_native_kd.ipynb](notebooks/kaggle/recovery/03_yolo26ns_native_kd.ipynb) |
| CrossKD raw pruned | YOLO26m v50 | YOLO26s | [recovery/04_yolo26ns_crosskd.ipynb](notebooks/kaggle/recovery/04_yolo26ns_crosskd.ipynb) |
| MGD raw pruned | YOLO26m v50 | YOLO26s | [recovery/05_yolo26ns_mgd_kd.ipynb](notebooks/kaggle/recovery/05_yolo26ns_mgd_kd.ipynb) |
| Teacher supervised | — | YOLO26m | [teacher/01_teacher_yolo26m.ipynb](notebooks/kaggle/teacher/01_teacher_yolo26m.ipynb) |
| Teacher supervised lima seed | — | YOLO26m | [teacher/02_teacher_yolo26m_multiseed.ipynb](notebooks/kaggle/teacher/02_teacher_yolo26m_multiseed.ipynb) |

### Filkom

Untuk melatih teacher dengan lima seed, gunakan [teacher/02_teacher_yolo26m_multiseed.ipynb](notebooks/filkom/teacher/02_teacher_yolo26m_multiseed.ipynb). Untuk eksperimen pruning, jalankan notebook `pruning/05` atau `06`, lalu pilih `recovery/01`–`05` sesuai metode. [Matriks pruning dan recovery Filkom](docs/lamp_gflops_recovery_workflow.md) · [Setup SSH dan Jupyter Server](notebooks/filkom/README.md)

### Google Colab

| Metode atau peran | Teacher | Model atau student | Notebook |
| --- | --- | --- | --- |
| Baseline pretrained | — | YOLO26s | [baseline/02_baseline_yolo26s.ipynb](notebooks/colab/baseline/02_baseline_yolo26s.ipynb) |
| Baseline pretrained | — | YOLO26n | [baseline/03_baseline_yolo26n_pretrained.ipynb](notebooks/colab/baseline/03_baseline_yolo26n_pretrained.ipynb) |
| Baseline scratch | — | YOLO26n | [baseline/04_baseline_yolo26n_scratch.ipynb](notebooks/colab/baseline/04_baseline_yolo26n_scratch.ipynb) |
| EDA Poltrada V3 | — | — | [eda/00_eda_poltrada_v3.ipynb](notebooks/colab/eda/00_eda_poltrada_v3.ipynb) |
| Native KD | YOLO26m | YOLO26s | [kd/teacher_m/05_native_yolo26m_to_yolo26s.ipynb](notebooks/colab/kd/teacher_m/05_native_yolo26m_to_yolo26s.ipynb) |
| Native KD | YOLO26m | YOLO26n | [kd/teacher_m/06_native_yolo26m_to_yolo26n.ipynb](notebooks/colab/kd/teacher_m/06_native_yolo26m_to_yolo26n.ipynb) |
| CrossKD | YOLO26m | YOLO26s | [kd/teacher_m/07_crosskd_yolo26m_to_yolo26s.ipynb](notebooks/colab/kd/teacher_m/07_crosskd_yolo26m_to_yolo26s.ipynb) |
| CrossKD | YOLO26m | YOLO26n | [kd/teacher_m/08_crosskd_yolo26m_to_yolo26n.ipynb](notebooks/colab/kd/teacher_m/08_crosskd_yolo26m_to_yolo26n.ipynb) |
| CSAKD | YOLO26m | YOLO26s | [kd/teacher_m/09_csakd_yolo26m_to_yolo26s.ipynb](notebooks/colab/kd/teacher_m/09_csakd_yolo26m_to_yolo26s.ipynb) |
| CSAKD | YOLO26m | YOLO26n | [kd/teacher_m/10_csakd_yolo26m_to_yolo26n.ipynb](notebooks/colab/kd/teacher_m/10_csakd_yolo26m_to_yolo26n.ipynb) |
| Native KD | YOLO26s | YOLO26n | [kd/teacher_s/11_native_yolo26s_to_yolo26n.ipynb](notebooks/colab/kd/teacher_s/11_native_yolo26s_to_yolo26n.ipynb) |
| CrossKD | YOLO26s | YOLO26n | [kd/teacher_s/12_crosskd_yolo26s_to_yolo26n.ipynb](notebooks/colab/kd/teacher_s/12_crosskd_yolo26s_to_yolo26n.ipynb) |
| CSAKD | YOLO26s | YOLO26n | [kd/teacher_s/13_csakd_yolo26s_to_yolo26n.ipynb](notebooks/colab/kd/teacher_s/13_csakd_yolo26s_to_yolo26n.ipynb) |
| PKD | YOLO26s | YOLO26n | [kd/teacher_s/14_pearson_pkd_yolo26s_to_yolo26n.ipynb](notebooks/colab/kd/teacher_s/14_pearson_pkd_yolo26s_to_yolo26n.ipynb) |
| HEAD | YOLO26s | YOLO26n | [kd/teacher_s/15_head_yolo26s_to_yolo26n.ipynb](notebooks/colab/kd/teacher_s/15_head_yolo26s_to_yolo26n.ipynb) |
| Rank Mimicking | YOLO26s | YOLO26n | [kd/teacher_s/16_rank_mimicking_yolo26s_to_yolo26n.ipynb](notebooks/colab/kd/teacher_s/16_rank_mimicking_yolo26s_to_yolo26n.ipynb) |
| Teacher supervised | — | YOLO26m | [teacher/01_teacher_yolo26m.ipynb](notebooks/colab/teacher/01_teacher_yolo26m.ipynb) |

### AMD ROCm

| Metode atau peran | Teacher | Model atau student | Notebook |
| --- | --- | --- | --- |
| Baseline ROCm | — | YOLO26n | [baseline/03_baseline_yolo26n_amd_roboflow.ipynb](notebooks/amd/baseline/03_baseline_yolo26n_amd_roboflow.ipynb) |

### Lokal

| Metode atau peran | Teacher | Model atau student | Notebook |
| --- | --- | --- | --- |
| Analisis kegagalan baseline seed 42 | — | YOLO26n | [analysis/03_baseline_seed42_failure_analysis.ipynb](notebooks/local/analysis/03_baseline_seed42_failure_analysis.ipynb) |
| EDA dataset lokal | — | — | [eda/01_eda_dataset.ipynb](notebooks/local/eda/01_eda_dataset.ipynb) |

### Arsip

| Koleksi | Eksperimen historis | Notebook |
| --- | --- | --- |
| Feature Based Knowledge Distilation | Baseline Teacher YOLO26s — IndoTraffic — Seed 42 | [02a_yolo26s_teacher_baseline.ipynb](notebooks/archieve/Feature%20Based%20Knowledge%20Distilation/02a_yolo26s_teacher_baseline.ipynb) |
| Feature Based Knowledge Distilation | YOLO26n Student Baseline Seed 42 — Shared KD W&B Project | [02b-yolo26n-student-baseline-seed42-kd-project.ipynb](notebooks/archieve/Feature%20Based%20Knowledge%20Distilation/02b-yolo26n-student-baseline-seed42-kd-project.ipynb) |
| modular | YOLO26n FPN-PAN baseline — pipeline modular | [02_yolo26n_baseline.ipynb](notebooks/archieve/modular/02_yolo26n_baseline.ipynb) |
| modular | YOLO26n BIFPN — pipeline modular | [03_yolo26n_bifpn.ipynb](notebooks/archieve/modular/03_yolo26n_bifpn.ipynb) |
| modular | YOLO26n ASFF — pipeline modular | [04_yolo26n_asff.ipynb](notebooks/archieve/modular/04_yolo26n_asff.ipynb) |
| modular | YOLO26n AFPN — pipeline modular | [05_yolo26n_afpn.ipynb](notebooks/archieve/modular/05_yolo26n_afpn.ipynb) |
| multi_scale_feature_fusion | Baseline YOLO26n IndoTraffic — final research pipeline | [02_train_yolo26n_baseline_research_final (1).ipynb](notebooks/archieve/multi_scale_feature_fusion/02_train_yolo26n_baseline_research_final%20%281%29.ipynb) |
| multi_scale_feature_fusion | YOLO26n + BiFPN — Final Research Pipeline | [03_train_yolo26n_bifpn_research_clean_fair.ipynb](notebooks/archieve/multi_scale_feature_fusion/03_train_yolo26n_bifpn_research_clean_fair.ipynb) |
| multi_scale_feature_fusion | YOLO26n + ASFF — Clean Full-Training Research Pipeline | [04_train_yolo26n_asff_research_clean_full.ipynb](notebooks/archieve/multi_scale_feature_fusion/04_train_yolo26n_asff_research_clean_full.ipynb) |
| multi_scale_feature_fusion | YOLO26n + AFPN — Clean Full-Training Research Pipeline | [05_train_yolo26n_afpn_research_clean_full.ipynb](notebooks/archieve/multi_scale_feature_fusion/05_train_yolo26n_afpn_research_clean_full.ipynb) |

## Routing W&B

Gunakan project `yolom26-kd-indotraffic` pada entity `ilhamrafiqin-brawijaya-university`. [Standar penamaan W&B](docs/wandb_naming.md) berisi aturan lengkap untuk run baru. Tabel berikut mengikuti konfigurasi notebook aktif dan generator.

| Identitas | Cara memakainya |
| --- | --- |
| Nama tampilan | Cari berdasarkan metode, model, seed, atau rasio. Nama yang sama dapat dipakai oleh beberapa run, termasuk padanan Kaggle dan Colab. |
| ID eksperimen notebook | Cocokkan `EXPERIMENT_ID` di konfigurasi. Contohnya Colab `14_pearson_pkd_yolo26s_to_yolo26n.ipynb` memakai ID `12_pkd_yolo26n`, sama seperti padanan Kaggle. |
| ID run W&B | Gunakan ID atau URL run untuk memilih satu eksekusi. Jika nama tampilan sama, cocokkan tag `colab` atau `amd-rocm` bila relevan, lalu cek konfigurasi dan ID run. |
| Versi artifact | Gunakan referensi lengkap `entity/project/nama-artifact:vN` saat mengunduh checkpoint. Bagian `:vN` adalah versi artifact. |

Untuk KD, `<pasangan>` sekaligus menjadi group, yaitu `T26m_S26s`, `T26m_S26n`, atau `T26s_S26n`. Pada diagnosis training, cari akhiran `_smoke` dan group berawalan `diagnostic-`. Untuk pruning, gunakan group, tag, dan status verifikasi pada bukti run. EDA dan analisis lokal dirutekan lewat tabel notebook dan laporan. Run arsip memakai identitas historis yang tercatat pada notebook atau laporan masing-masing.

| Keluarga | Platform | Pola dan contoh nama tampilan | Group | Contoh tag | Artifact sumber |
| --- | --- | --- | --- | --- | --- |
| Teacher | Filkom, Kaggle, Colab | `base_teacher_<model>_seed<seed>` → `base_teacher_yolo26m_seed42` | `supervised` | `teacher`, `model-m`, `seed42` | — |
| Baseline pretrained dan scratch | Kaggle, Colab | `base_<model>[_scratch]_seed<seed>` → `base_yolo26n_scratch_seed42` | `supervised` | `baseline`, `model-n`, `model-s`, `from-scratch` | — |
| Baseline AMD | AMD ROCm | `base_yolo26n_amd_seed<seed>` → `base_yolo26n_amd_seed7` | `supervised` | `amd-rocm`, `roboflow-v3`, `seed7` | — |
| Native KD | Kaggle, Colab | `kd_native_<pasangan>_dis<bobot>_seed<seed>` → `kd_native_T26m_S26s_dis1_seed42` | `<pasangan>` | `native`, `dis1`, `seed42` | Teacher m / s sesuai pasangan |
| CrossKD | Kaggle, Colab | `kd_crosskd_<pasangan>_seed<seed>` → `kd_crosskd_T26m_S26n_seed42` | `<pasangan>` | `crosskd`, `cls1`, `box1` | Teacher m / s sesuai pasangan |
| CSAKD | Kaggle, Colab | `kd_csakd_<pasangan>_seed<seed>` → `kd_csakd_T26s_S26n_seed42` | `<pasangan>` | `csakd`, `agka1`, `cross1` | Teacher m / s sesuai pasangan |
| PKD | Kaggle, Colab | `kd_pkd_T26s_S26n_seed<seed>` → `kd_pkd_T26s_S26n_seed42` | `T26s_S26n` | `pkd`, `pkdw<bobot>`, `seed42` | Teacher s |
| HEAD | Kaggle, Colab | `kd_head_T26s_S26n_seed<seed>` → `kd_head_T26s_S26n_seed42` | `T26s_S26n` | `head`, `teacher-s`, `student-n` | Teacher s |
| Rank Mimicking | Kaggle, Colab | `kd_rm_T26s_S26n_seed<seed>` → `kd_rm_T26s_S26n_seed42` | `T26s_S26n` | `rm`, `teacher-s`, `student-n` | Teacher s |
| MGD | Kaggle | `kd_mgd_T26s_S26n_seed<seed>` → `kd_mgd_T26s_S26n_seed42` | `T26s_S26n` | `mgd`, `teacher-s`, `student-n`, `research` | Teacher s |
| LAMP V1 | Kaggle 01–02 | `prun_lamp1_<model>_pruned-pr<rasio>` → `prun_lamp1_yolo26n_pruned-pr0.3` | `lamp-v1` | `lamp`, `no-recovery`, `yolo26n`, `pr0.3` | Baseline n / s sesuai model |
| LAMP V2 conservative | Kaggle 03 | `prun_lamp2_<model>_c_pruned-pr<rasio>` → `prun_lamp2_yolo26n_c_pruned-pr0.3` | `lamp-v2-<suite_id>` | `v2`, `conservative`, `no-recovery`, `pr0.3` | Baseline n / s sesuai model |
| LAMP V2 extended | Kaggle 03 | `prun_lamp2_<model>_pruned-pr<rasio>` → `prun_lamp2_yolo26s_pruned-pr0.3` | `lamp-v2-<suite_id>` | `v2`, `extended`, `no-recovery`, `pr0.3` | Baseline n / s sesuai model |
| LAMP V2 recovery | Kaggle 04 | `prun_lamp2_<model>[_c]_ft<epoch>_pruned-pr<rasio>` → `prun_lamp2_yolo26n_ft5_pruned-pr0.3` | `lamp-finetune-<suite_id>` | `finetune`, `recovery`, `epochs5`, `pr0.3` | Baseline n / s sesuai model |
| LAMP target GFLOPs | Kaggle/Filkom pruning 05/06 | `prun_lamp_gflops_<model>_<platform>_su<target>_seed<seed>` | `lamp-gflops-<platform>-<suite>` | satu varian per suite, ekspor kandidat N/S untuk seleksi akhir | Baseline n `v84` dan s `v62` |
| LAMP YOLO26s multi-seed | Kaggle/Filkom | `yolo26s-lamp-su1.20x-seed42` | `yolo26s-lamp-multiseed-<suite>` | Artefak `-raw` per pasangan target dan seed | [Lima baseline s sesuai seed](docs/ARTEFAK_WANDB.md#suite-dengan-baseline-sesuai-seed) |
| Recovery raw pruned | Kaggle/Filkom recovery 01–05 | `prun_lamp_gflops_<model>_<platform>_<metode><epoch>_su<target>_seed<seed>` | `lamp-gflops-<platform>-<suite>` | `ft`, `native`, `crosskd` atau `mgd` | Raw terkonfirmasi `:vN` + SHA-256, serta teacher untuk KD |
| KD raw YOLO26s pilihan | Kaggle/Filkom recovery 03–05 | `yolo26s-lamp-su1.20x-seed<seed>-<metode>` | `yolo26s-lamp-kd-<suite-pruning>` | `native`, `crosskd`, atau `mgd` | Lima raw 1,20× pulihan `:v1` dengan hash asli; teacher YOLO26m `v50` |
| Fine-tuning raw YOLO26s pilihan | Kaggle/Filkom recovery 02 | `yolo26s-lamp-su1.20x-seed42-ft` | `yolo26s-lamp-ft-<suite-pruning>` | satu input melalui comment/uncomment, koleksi `-checkpoints`, `-results`, `-resume` | [Tujuh raw per seed](docs/ARTEFAK_WANDB.md#suite-dengan-baseline-sesuai-seed) |

### Artifact sumber yang dipatok

| Sumber | Referensi lengkap | Dipakai untuk |
| --- | --- | --- |
| Teacher m | `ilhamrafiqin-brawijaya-university/yolom26-kd-indotraffic/01_teacher_yolo26m-fux5wdcb-checkpoints:v50` | Sumber KD dengan teacher YOLO26m. |
| Teacher s / baseline s | `ilhamrafiqin-brawijaya-university/yolom26-kd-indotraffic/02_baseline_yolo26s-q5kp9k9d-checkpoints:v62` | Sumber KD dengan teacher YOLO26s dan pruning model s. |
| Baseline n | `ilhamrafiqin-brawijaya-university/yolom26-kd-indotraffic/03_baseline_yolo26n-9uu3meve-checkpoints:v84` | Sumber pruning model n. |

## Kode dan hasil

| Kebutuhan | Lokasi | Kapan dipakai |
| --- | --- | --- |
| Dependensi dan setup | [pyproject.toml](pyproject.toml), [uv.lock](uv.lock), [.python-version](.python-version), [.env.example](.env.example) | Gunakan manifest dan lockfile untuk lingkungan Python. Simpan nilai rahasia di `.env` lokal. |
| Konfigurasi dan dataset | [src/skripsi/config.py](src/skripsi/config.py), [src/skripsi/data.py](src/skripsi/data.py), [data/raw/](data/raw/), [data/processed/](data/processed/), [data/audited_v2/](data/audited_v2/) | Cek protokol pipeline, audit label, serta lokasi dataset. Isi dataset lokal bergantung pada setup. |
| Training dan evaluasi | [src/skripsi/pipeline.py](src/skripsi/pipeline.py), [src/skripsi/evaluation.py](src/skripsi/evaluation.py) | Telusuri preflight, training, pemilihan threshold, dan evaluasi pipeline modular. |
| Analisis model dan data | [src/skripsi/analysis.py](src/skripsi/analysis.py), [src/skripsi/diagnostics.py](src/skripsi/diagnostics.py), [src/skripsi/models/](src/skripsi/models/) | Cari diagnostik deteksi dan implementasi arsitektur FPN-PAN, BiFPN, ASFF, dan AFPN. |
| Struktur dan generator notebook | [scripts/notebook_paths.py](scripts/notebook_paths.py), [scripts/generate_colab_notebooks.py](scripts/generate_colab_notebooks.py), [scripts/colab_runtime.py](scripts/colab_runtime.py), [notebooks/colab/manifest.json](notebooks/colab/manifest.json), [scripts/generate_modular_notebooks.py](scripts/generate_modular_notebooks.py) | Cari peta notebook, pengaturan port Colab, dan sumber notebook modular arsip. |
| Sumber KD | [scripts/generate_mgd_notebook.py](scripts/generate_mgd_notebook.py), [scripts/mgd_method_source.py](scripts/mgd_method_source.py), [scripts/mgd_runtime_source.py](scripts/mgd_runtime_source.py), [scripts/rewrite_head_rm_kaggle.py](scripts/rewrite_head_rm_kaggle.py) | Cari sumber notebook MGD dan skrip penataan HEAD/RM. |
| Sumber teacher lima seed | [scripts/generate_teacher_yolo26m_notebooks.py](scripts/generate_teacher_yolo26m_notebooks.py) | Bangun padanan Kaggle dan Filkom dari alur supervised serial. |
| Sumber pruning multi-seed | [Generator 06 multi-seed](scripts/generate_lamp_multiseed_notebook.py) | Bangun satu notebook YOLO26s untuk Kaggle dan Filkom dengan registry checkpoint. |
| Sumber KD satu student pilihan | [Generator KD 03–05](scripts/generate_selected_kd_notebooks.py), [adapter KD](scripts/pruned_kd_selected_source.py), [autentikasi MGD FILKOM](scripts/filkom_mgd_auth_source.py), [dataset lokal MGD FILKOM](scripts/filkom_mgd_dataset_source.py), [pembaca label campuran FILKOM](scripts/filkom_mgd_labels_source.py), [uji eksekusi](scripts/verify_selected_kd_cpu.py) | Bangun enam notebook KD dengan lima raw 1,20× dan teacher YOLO26m `v50`; mekanisme comment/uncomment, W&B, EDA, serta resume. MGD FILKOM memuat key environment/`.env` sebelum unduhan, mencatat fingerprint lokal tanpa mewajibkan cocok dengan raw, serta membaca label box/polygon campuran tanpa mengedit dataset sumber. |
| Sumber pruning dan recovery | [generator V2](scripts/generate_lamp_v2_notebook.py), [generator recovery historis](scripts/generate_lamp_recovery_notebook.py), [generator target GFLOPs](scripts/generate_lamp_gflops_notebook.py), [generator recovery per metode](scripts/generate_standard_finetuning_notebooks.py), [analisis notebook 05/06](scripts/lamp_gflops_eda_source.py), [scripts/](scripts/) | Cari runtime, policy, validasi, EDA, dan training recovery yang disisipkan ke notebook. |
| Pemeriksaan CPU | [scripts/verify_mgd_cpu_smoke.py](scripts/verify_mgd_cpu_smoke.py), [scripts/verify_lamp_v2_cpu.py](scripts/verify_lamp_v2_cpu.py), [scripts/verify_lamp_recovery_cpu.py](scripts/verify_lamp_recovery_cpu.py), [pipeline notebook 05/06](scripts/verify_lamp_gflops_cpu.py), [replay EDA notebook 05/06](scripts/verify_lamp_gflops_eda.py) | Pilih smoke test MGD/pruning/recovery atau replay analisis dari bukti tersimpan. |
| Tes dan fixture | [tests/](tests/), [tests/fixtures/](tests/fixtures/), [tests/test_notebook_layout.py](tests/test_notebook_layout.py), [tests/test_report_layout.py](tests/test_report_layout.py), [tests/test_kd_notebook_tracking.py](tests/test_kd_notebook_tracking.py) | Pilih tes menurut metode atau platform. Cari fixture MGD dan PKD serta kontrak layout dan tracking. |
| Hasil baseline seed 42 | [reports/baseline/seed42_evaluation/](reports/baseline/seed42_evaluation/), [reports/baseline/seed42_evaluation/overall_validation/](reports/baseline/seed42_evaluation/overall_validation/) | Cari CSV metrik, grafik, prediksi, dan confusion matrix. |
| Bukti eksperimen platform | [reports/experiments/kaggle/](reports/experiments/kaggle/), [reports/experiments/colab/](reports/experiments/colab/), [reports/experiments/local/](reports/experiments/local/) | Cari laporan dan JSON smoke HEAD/RM, PKD, MGD, atau validasi koleksi Colab. |
| Bukti pruning | [reports/pruning/yolo26n/v1/](reports/pruning/yolo26n/v1/), [reports/pruning/yolo26s/v1/](reports/pruning/yolo26s/v1/), [reports/pruning/yolo26n/v2/](reports/pruning/yolo26n/v2/), [reports/pruning/yolo26s/v2/](reports/pruning/yolo26s/v2/), [reports/pruning/comparisons/](reports/pruning/comparisons/), [reports/pruning/recovery/](reports/pruning/recovery/) | Cari smoke test, perbandingan model n/s, dan verifikasi recovery berdasarkan versi. |
| Jejak penataan | [reports/engineering/notebook_reorganization_20261022.json](reports/engineering/notebook_reorganization_20261022.json), [reports/engineering/report_reorganization_20261022.json](reports/engineering/report_reorganization_20261022.json) | Cari pemetaan jalur dan bukti migrasi notebook serta laporan. |
| Output eksperimen lokal | [outputs/](outputs/), [outputs/pruning/](outputs/pruning/), [outputs/mgd_cpu_smoke/](outputs/mgd_cpu_smoke/), [wandb/](wandb/) | Cari checkpoint, manifest, dan log lokal. Gunakan ID run untuk membedakan eksekusi W&B. |
| Lingkungan dan file sementara | `.venv/`, `.pytest_cache/`, `.ruff_cache/`, `tmp/` | Area lokal untuk dependensi, cache, dan pekerjaan sementara. |

## Referensi

Folder [references/](references/) mengelompokkan sumber menurut topik. Git dapat melacak `.gitkeep` pada `code/` dan `paper/`. Isi clone dan file paper tetap lokal.

| Topik | Kode | Paper | Clone lokal | Jumlah PDF |
| --- | --- | --- | --- | --- |
| `crosskd` | [code/](references/crosskd/code/) | [paper/](references/crosskd/paper/) | [crosskd-official](references/crosskd/code/crosskd-official/) | 1 |
| `csakd` | [code/](references/csakd/code/) | [paper/](references/csakd/paper/) | [csakd-yolov8-kd](references/csakd/code/csakd-yolov8-kd/) | 0 |
| `deteksi-kendaraan` | [code/](references/deteksi-kendaraan/code/) | [paper/](references/deteksi-kendaraan/paper/) | — | 3 |
| `isomorphic-pruning` | [code/](references/isomorphic-pruning/code/) | [paper/](references/isomorphic-pruning/paper/) | [isomorphic-pruning](references/isomorphic-pruning/code/isomorphic-pruning/) | 0 |
| `knowledge-distillation` | [code/](references/knowledge-distillation/code/) | [paper/](references/knowledge-distillation/paper/) | — | 3 |
| `kompresi-model` | [code/](references/kompresi-model/code/) | [paper/](references/kompresi-model/paper/) | — | 1 |
| `lamp-pruning` | [code/](references/lamp-pruning/code/) | [paper/](references/lamp-pruning/paper/) | [lamp-original](references/lamp-pruning/code/lamp-original/), [mp-yolo](references/lamp-pruning/code/mp-yolo/) | 1 |
| `mdp` | [code/](references/mdp/code/) | [paper/](references/mdp/paper/) | [mdp](references/mdp/code/mdp/) | 0 |
| `mgd` | [code/](references/mgd/code/) | [paper/](references/mgd/paper/) | [MGD](references/mgd/code/MGD/) | 1 |
| `pkd` | [code/](references/pkd/code/) | [paper/](references/pkd/paper/) | [mmrazor-pkd](references/pkd/code/mmrazor-pkd/) | 1 |
| `torch-pruning` | [code/](references/torch-pruning/code/) | [paper/](references/torch-pruning/paper/) | [torch-pruning](references/torch-pruning/code/torch-pruning/) | 0 |
| `yolo26` | [code/](references/yolo26/code/) | [paper/](references/yolo26/paper/) | [ultralytics](references/yolo26/code/ultralytics/) | 3 |

### Pemetaan paper

Jalur asal memakai lokasi sebelum penataan pertama. Kedua file Ultralytics YOLO26 yang tercantum memiliki hash yang sama dan diwakili oleh satu PDF.

| Topik | Jalur asal | PDF lokal |
| --- | --- | --- |
| `crosskd` | `Jurnal/Jurnal Referensi/Wang_CrossKD_Cross-Head_Knowledge_Distillation_for_Object_Detection_CVPR_2024_paper.pdf` | [CrossKD - Cross-Head Knowledge Distillation for Object Detection.pdf](references/crosskd/paper/CrossKD%20-%20Cross-Head%20Knowledge%20Distillation%20for%20Object%20Detection.pdf) |
| `mgd` | `Referensi/MGD/paper/136710053.pdf` | [Masked Generative Distillation.pdf](references/mgd/paper/Masked%20Generative%20Distillation.pdf) |
| `lamp-pruning` | `Jurnal/Prunning/1-s2.0-S1047320325001749-main.pdf` | [MP-YOLO - multidimensional feature fusion based layer adaptive pruning YOLO for dense vehicle object detection algorithm.pdf](references/lamp-pruning/paper/MP-YOLO%20-%20multidimensional%20feature%20fusion%20based%20layer%20adaptive%20pruning%20YOLO%20for%20dense%20vehicle%20object%20detection%20algorithm.pdf) |
| `kompresi-model` | `Jurnal/Prunning/17056.docx (1).pdf` | [Kompresi Model YOLOv8 melalui Pruning, Sparsity, dan Knowledge Distillation untuk Deteksi Kendaraan dan Plat Nomor secara Real-Time pada Perangkat Edge.pdf](references/kompresi-model/paper/Kompresi%20Model%20YOLOv8%20melalui%20Pruning,%20Sparsity,%20dan%20Knowledge%20Distillation%20untuk%20Deteksi%20Kendaraan%20dan%20Plat%20Nomor%20secara%20Real-Time%20pada%20Perangkat%20Edge.pdf) |
| `knowledge-distillation` | `Jurnal/Jurnal Referensi/2006.05525v7.pdf` | [Knowledge Distillation - A Survey.pdf](references/knowledge-distillation/paper/Knowledge%20Distillation%20-%20A%20Survey.pdf) |
| `pkd` | `Jurnal/Jurnal Referensi/2207.02039v2.pdf` | [PKD - General Distillation Framework for Object Detectors via Pearson Correlation Coefficient.pdf](references/pkd/paper/PKD%20-%20General%20Distillation%20Framework%20for%20Object%20Detectors%20via%20Pearson%20Correlation%20Coefficient.pdf) |
| `knowledge-distillation` | `Jurnal/Jurnal Referensi/When Object Detection Meets Knowledge Distillation.pdf` | [When Object Detection Meets Knowledge Distillation - A Survey.pdf](references/knowledge-distillation/paper/When%20Object%20Detection%20Meets%20Knowledge%20Distillation%20-%20A%20Survey.pdf) |
| `knowledge-distillation` | `Jurnal/Jurnal Referensi/Yang_Focal_and_Global_Knowledge_Distillation_for_Detectors_CVPR_2022_paper.pdf` | [Focal and Global Knowledge Distillation for Detectors.pdf](references/knowledge-distillation/paper/Focal%20and%20Global%20Knowledge%20Distillation%20for%20Detectors.pdf) |
| `deteksi-kendaraan` | `Jurnal/Jurnal Referensi/MXT-YOLOv7t_An_Efficient_Real-Time_Object_Detection_for_Autonomous_Driving_in_Mixed_Traffic_Environments.pdf` | [MXT-YOLOv7t - An Efficient Real-Time Object Detection for Autonomous Driving in Mixed Traffic Environments.pdf](references/deteksi-kendaraan/paper/MXT-YOLOv7t%20-%20An%20Efficient%20Real-Time%20Object%20Detection%20for%20Autonomous%20Driving%20in%20Mixed%20Traffic%20Environments.pdf) |
| `deteksi-kendaraan` | `Jurnal/Jurnal Referensi/Object_Detection_in_Dense_and_Mixed_Traffic_for_Autonomous_Vehicles_With_Modified_Yolo.pdf` | [Object Detection in Dense and Mixed Traffic for Autonomous Vehicles With Modified Yolo.pdf](references/deteksi-kendaraan/paper/Object%20Detection%20in%20Dense%20and%20Mixed%20Traffic%20for%20Autonomous%20Vehicles%20With%20Modified%20Yolo.pdf) |
| `deteksi-kendaraan` | `Jurnal/Prunning/sensors-25-03399-v2.pdf` | [ZZ-YOLOv11 - A Lightweight Vehicle Detection Model Based on Improved YOLOv11.pdf](references/deteksi-kendaraan/paper/ZZ-YOLOv11%20-%20A%20Lightweight%20Vehicle%20Detection%20Model%20Based%20on%20Improved%20YOLOv11.pdf) |
| `yolo26` | `Jurnal/Jurnal Referensi/2606.03748v1.pdf` dan `Jurnal/Prunning/2606.03748v1 (1).pdf` | [Ultralytics YOLO26 - Unified Real-Time End-to-End Vision Models.pdf](references/yolo26/paper/Ultralytics%20YOLO26%20-%20Unified%20Real-Time%20End-to-End%20Vision%20Models.pdf) |
| `yolo26` | `Jurnal/Prunning/2601.12882v2.pdf` | [YOLO26 - An Analysis of NMS-Free End to End Framework for Real-Time Object Detection.pdf](references/yolo26/paper/YOLO26%20-%20An%20Analysis%20of%20NMS-Free%20End%20to%20End%20Framework%20for%20Real-Time%20Object%20Detection.pdf) |
| `yolo26` | `Jurnal/Prunning/1-s2.0-S0304423826005145-main.pdf` | [YOLO26-EGH - A lightweight knowledge-distilled detection model for blueberry ripeness grading in greenhouse.pdf](references/yolo26/paper/YOLO26-EGH%20-%20A%20lightweight%20knowledge-distilled%20detection%20model%20for%20blueberry%20ripeness%20grading%20in%20greenhouse.pdf) |
