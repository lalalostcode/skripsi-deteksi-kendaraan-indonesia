# Penamaan eksperimen W&B YOLO26

Gunakan project [yolom26-kd-indotraffic](https://wandb.ai/ilhamrafiqin-brawijaya-university/yolom26-kd-indotraffic) di entity `ilhamrafiqin-brawijaya-university` untuk semua eksperimen YOLO26 di Filkom, Kaggle, Colab, dan AMD.

| Keluarga | Format | Contoh |
| --- | --- | --- |
| Baseline | `base_<model>[_<varian>]_seed<seed>` | `base_yolo26n_seed42`, `base_yolo26n_scratch_seed42` |
| Teacher supervised | `base_teacher_<model>_seed<seed>` | `base_teacher_yolo26m_seed42` |
| Native KD | `kd_native_T26<teacher>_S26<student>_dis<bobot>_seed<seed>` | `kd_native_T26m_S26s_dis1_seed42` |
| KD lain | `kd_<metode>_T26<teacher>_S26<student>_seed<seed>` | `kd_crosskd_T26m_S26n_seed42`, `kd_pkd_T26s_S26n_seed42` |
| LAMP V1 | `prun_lamp1_<model>_pruned-pr<rasio>` | `prun_lamp1_yolo26n_pruned-pr0.3` |
| LAMP V2 | `prun_lamp2_<model>[_c][_ft<epoch>]_pruned-pr<rasio>` | `prun_lamp2_yolo26s_pruned-pr0.3`, `prun_lamp2_yolo26n_ft5_pruned-pr0.3` |
| LAMP target GFLOPs | `prun_lamp_gflops_<model>_<platform>_su<target>_seed<seed>` | `prun_lamp_gflops_yolo26n_filkom_su1p02_seed42` |
| LAMP YOLO26s multi-seed | `yolo26s-lamp-su<target>x-seed<seed>` | `yolo26s-lamp-su1.20x-seed42` |
| Fine-tuning raw YOLO26s pilihan | `yolo26s-lamp-su<target>x-seed<seed>-ft` | `yolo26s-lamp-su1.20x-seed0-ft` |
| KD raw YOLO26s pilihan | `yolo26s-lamp-su1.20x-seed<seed>-<metode>` | `yolo26s-lamp-su1.20x-seed0-native`, `yolo26s-lamp-su1.20x-seed7-crosskd`, `yolo26s-lamp-su1.20x-seed3407-mgd` |
| Fine-tuning raw pruned | `prun_lamp_gflops_<model>_<platform>_ft<epoch>_su<target>_seed<seed>` | `prun_lamp_gflops_yolo26n_filkom_ft100_su1p02_seed42` |
| Fine-tuning raw dengan protokol baseline | `prun_lamp_gflops_yolo26s_<platform>_ft-baseline<epoch>_su<target>_seed<seed>` | `prun_lamp_gflops_yolo26s_filkom_ft-baseline100_su1p02_seed42` |
| KD raw pruned | `prun_lamp_gflops_<model>_<platform>_<metode><epoch>_su<target>_seed<seed>` | `prun_lamp_gflops_yolo26s_kaggle_mgd100_su1p10_seed42` |

Recovery fine-tuning 02 memakai grup `yolo26s-lamp-ft-<suite-pruning>` dan protokol `selected-raw-finetune-v2`. Seed diwarisi dari raw yang dipilih. Platform dan budget epoch dicatat dalam config. Koleksi `<nama-run>-checkpoints` memuat best/last, `<nama-run>-results` memuat metrik dan EDA, serta `<nama-run>-resume` memuat snapshot training. Mode diagnostik menambahkan `-smoke` pada nama run dan grup.

Recovery KD 03–05 memakai grup bersama `yolo26s-lamp-kd-20261107T020459Z_45575a7f` dan protokol `selected-raw-kd-v1`. Tiap metode menjalankan satu raw 1,20× pilihan dari seed 42, 0, 2024, 7, atau 3407. Teacher bersama adalah YOLO26m `v50`; identitas seed student berasal dari manifest raw. Koleksi `<nama-run>-checkpoints`, `<nama-run>-results`, dan `<nama-run>-resume` memisahkan checkpoint student, evaluasi/EDA, dan snapshot lengkap. Mode diagnostik menambahkan `-smoke` pada nama run dan grup. Ringkasan menghitung 15 perlakuan dan memisahkan metode, platform, serta perangkat latency.

Recovery 01–05 mempertahankan **best dan latest** dengan kebijakan `best-and-latest-v1`.
`-checkpoints` berisi `best.pt`/`last.pt`. Satu versi `-resume` terbaru per run berisi state
terakhir beserta model epoch terbaik; versi yang telah digantikan dihapus sesudah hash unduhan
penggantinya terverifikasi. `best`, `last`, dan `latest` mengacu pada paket yang sama; epoch model
terbaik dan epoch snapshot dicatat terpisah. Seluruh metrik/EDA eksperimen penuh tetap disimpan.
Smoke yang lulus otomatis dibersihkan dari W&B dan salinan checkpoint lokalnya; registry memakai
`DELETED_AFTER_SMOKE`, dengan receipt hash serta log sebagai bukti historis.
[Retensi dan resume](lamp_gflops_recovery_workflow.md#retensi-checkpoint-dan-pembersihan-smoke).

Seluruh recovery 01–05 pada Kaggle/Filkom menambahkan paket `<nama-run>-eda:vN` bertipe `results`.
Paket berisi diagram per metrik, tabel tiga kondisi/per kelas, delta/gain, dan manifest/hash.
Summary run memuat `custom_eda_report_url`, `custom_eda_artifact_ref`, dan `custom_eda_status`.
Halaman `EDA · <nama-run> · <id-run>` menampilkan baseline asal → raw pruned → hasil recovery,
dengan filter ID run agar sesi paralel tidak tercampur. Gambar berada pada `custom_eda/01_accuracy/`,
`02_efficiency/`, `03_latency/`, `04_delta_gain/`, dan `05_per_class/`; Tables pada `custom_eda/tables/`.

Run KD terdahulu dengan protokol `source-matched-v1` memakai grup `lamp-source-matched-yolo26s-<platform>-<mode>`, artefak `fine-tuned` dan `fine-tuned-results`. Baris `ft-baseline` mendokumentasikan salinan historis. Pertahankan identitas protokol tersebut saat membaca hasil lama.

Pada LAMP YOLO26s multi-seed, `su1.20x` adalah target GFLOPs. Tiap pasangan target dan seed mendapat run dan koleksi Model Artifact dengan akhiran `-raw`, misalnya `yolo26s-lamp-su1.20x-seed0-raw`. Tujuh run suite memakai grup `yolo26s-lamp-multiseed-<suite_id>`. Metadata mencatat platform, capaian aktual, status target, versi dan hash baseline seed yang sama, serta metrik baseline dan raw. Ambil versi tetap dan SHA-256 raw serta baseline dari baris yang sama di `checkpoint_registry.json`. [Inventaris artefak](ARTEFAK_WANDB.md#student-raw-lamp-yolo26s-multi-seed)

- Pakai nama model lengkap, seperti `yolo26n` atau `yolo26s`. Notasi pasangan KD memakai `T26…` untuk teacher dan `S26…` untuk student. Varian supervised seperti `scratch` dan `amd` dicantumkan sebelum seed.
- Pada LAMP V2, tambahkan `_c` untuk conservative. Extended memakai nama dasar. Tambahkan `_ft<epoch>` untuk recovery. Angka rasio mengikuti konfigurasi, misalnya `pr0.3` untuk 30% channel eligible.
- Gunakan group `supervised` untuk baseline dan teacher, serta group pasangan seperti `T26s_S26n` untuk KD. V1 memakai group `lamp-v1`. V2 memakai `lamp-v2-<suite_id>` atau `lamp-finetune-<suite_id>` sesuai tahap recovery. Filter pruning memakai tag model, policy atau metode, rasio, dan seed. Budget recovery juga tersedia sebagai tag, misalnya `epochs5`.
- Tambahkan akhiran `_smoke` pada run diagnostik training dan awali group dengan `diagnostic-`. Gunakan group atau tag tersebut untuk memisahkan diagnosis dari hasil penelitian.

Teacher YOLO26m lima seed memakai satu run per seed dalam group `supervised` dengan tag platform. Koleksi `02_teacher_yolo26m_multiseed-<run_id>-checkpoints` berisi versi per epoch dengan alias `epoch-001`, `latest`, dan `best` saat sesuai. Untuk input KD, ambil pasangan `best_artifact.reference` dan `best_artifact.best_pt_sha256` dari `seed_result.json`. Hash tersebut merujuk pada `best.pt` dalam versi W&B itu. `best_pt_sha256` pada tingkat atas merujuk pada file lokal setelah training selesai.

Untuk workflow GFLOPs, `su1p02` menandai target yang diminta (1,02×). Speed-up aktual dan status pencapaian disimpan sebagai metadata. Recovery KD memakai `native`, `crosskd`, atau `mgd` dan budget total, misalnya `_native100`. Pruning dan recovery memiliki run terpisah. Referensi raw `:vN` dan SHA-256 mengikat lineage recovery, bersama versi teacher untuk KD. Simpan checkpoint epoch pada koleksi `training-state` dengan alias `epoch-001`, `latest`, serta `best` ketika sesuai. Model inference dan keluaran analisis memakai koleksi `recovered` dan `results`. [Panduan artefak dan resume](lamp_gflops_recovery_workflow.md#4-status-artefak-dan-resume)

Nama tampilan boleh sama ketika konfigurasi diulang. Eksekusi baru mendapat ID baru dengan `resume="never"`. Untuk melanjutkan workflow GFLOPs, gunakan ID yang tercatat pada suite dan konfigurasi yang telah lolos verifikasi kontrak resume. Cari run historis dengan ID run atau referensi versi artefak yang sudah dicatat. [Referensi SDK W&B](https://github.com/wandb/wandb/blob/v0.28.1/wandb/sdk/wandb_init.py#L1137-L1230)
