# Bukti LAMP dengan recovery fine-tuning

[Pembersihan smoke dan proteksi input — 8 November 2026](smoke_cleanup_20261108/)
memuat inventaris sebelum penghapusan, SHA-256 setiap salinan biner lokal yang dibersihkan,
receipt W&B, dan pemeriksaan raw/baseline/teacher yang dipertahankan. Referensi smoke pada bukti
historis berikut tidak menjamin artefaknya masih tersedia setelah pembersihan yang diminta pengguna.
[Retensi best/latest dan cleanup otomatis](../../../docs/lamp_gflops_recovery_workflow.md#retensi-checkpoint-dan-pembersihan-smoke).

[Verifikasi retensi recovery — 8 November 2026](recovery_storage_verification_20261108.json)
mencatat FT dua epoch dan MGD satu epoch online, Native/CrossKD satu epoch offline, resume
FT/MGD setelah interupsi, serta penghapusan bobot dan artefak setelah lolos verifikasi.
Metrik fixture CPU bukan hasil akurasi Poltrada V3. Riwayat sistem W&B yang tidak dapat
dihapus melalui SDK dilaporkan terpisah dari artefak model/hasil diagnostik.

[EDA seluruh recovery dan verifikasi lanjutan — 8 November 2026](recovery_custom_eda_verification_20261108.json)
mencatat notebook 01–05 Kaggle/Filkom, diagram batang per metrik, report W&B per run,
smoke KD dengan lima raw asli, resume, serta binding URI fine-tuning `:v1`.
Lihat `status`, jumlah kasus, dan batas bukti pada JSON; hasil yang tertunda tidak dianggap selesai.
[Indeks EDA W&B](https://wandb.ai/ilhamrafiqin-brawijaya-university/yolom26-kd-indotraffic/reports/EDA--VmlldzoxODI3MDUxNg==)
memisahkan hasil penelitian yang tersedia dari perlakuan `NOT_RUN` dan smoke diagnostik.

Empat run fine-tuning pengguna yang telah selesai diberi EDA dari tabel pengukuran tersimpan,
tanpa training tambahan. Jumlah epoch tetap sama. Report membandingkan baseline asal, raw,
dan hasil recovery; checkpoint lama tidak ditulis ulang. [Panduan membuka report dan menambahkan
EDA pada hasil lama](../../../docs/lamp_gflops_recovery_workflow.md#5-baca-perbandingan-tiga-tahap).

[KD satu raw 1,20× pilihan — 7 November 2026](yolo26s_selected_kd_verification_20261107.json)
mencatat status pengujian Native KD, CrossKD, dan MGD pada lima raw dan teacher YOLO26m v50
asli. Lihat field status dan jumlah kasus selesai untuk membedakan progres dari hasil lengkap.
[Lima paket raw pulihan `:v1`](../lamp_yolo26s_raw_restored_20261107.json) mempunyai hash yang
sama dengan paket asli; notebook fine-tuning 02 tetap memakai kontraknya sendiri.

[Fine-tuning satu raw pilihan — 7 November 2026](yolo26s_selected_finetune_verification_20261107.json)
mencatat smoke satu epoch untuk tujuh raw, EDA tiga kondisi, publikasi W&B, dan pemeriksaan
history serta media. Bukti resume dan uji proses paralel memakai CPU lokal. [Inventaris versi W&B](../../../docs/ARTEFAK_WANDB.md#fine-tuning-raw-yolo26s--smoke-satu-epoch)

[Notebook Kaggle 04](../../../notebooks/kaggle/pruning/04_lamp_yolo26ns_finetune_recovery.ipynb)
menjalankan LAMP N/S pada rasio 10-50%, kemudian recovery supervised biasa dari model
struktural hasil pruning. [Panduan konfigurasi dan metode](../../../docs/lamp_yolo26ns_finetune_recovery.md)
menjelaskan pemuatan struktur, optimizer baru, best EMA FP32, dan evaluasi tiga tahap.

Bandingkan baseline, pruned sebelum recovery, dan recovered pada protokol yang sama.
Status training/reload/evaluasi/tracking harus dilaporkan terpisah. Pengujian sintetis CPU
memeriksa implementasi dan tidak membuktikan mAP V3, latency CUDA, atau recovery pada
budget penuh. Metrik yang belum diukur tetap `NOT_RUN`.

Output baru memakai `pruning/lamp_v2_finetune/{suite_id}/`, dengan direktori per model,
policy/rasio, recovery/training, dan comparison. Checkpoint serta laporan notebook tanpa
recovery tetap berada pada direktori historisnya dan tidak ditulis ulang.

[Laporan pruning tanpa recovery](../README.md) tetap tersedia sebagai pembanding terpisah.

[Uji CPU notebook 04 — 3 November 2026](lamp_recovery_cpu_verification_20261103.json)
memakai checkpoint asli N/S pada pruning 50%, lalu satu epoch recovery native di dataset
sintetis 2 train/2 val, input 64, batch/nbs 2. Kedua konfigurasi lolos training, preservasi
struktur, FP32 best EMA, fresh reload/prediction, serta evaluasi tiga tahap pada fixture.
Masing-masing melakukan satu optimizer step; selisih terbesar raw/decoded outputs setelah
reload `1.9073486328125e-5`, dalam `rtol=1e-4`, `atol=1e-5`. Sebanyak 439 tes relevan lulus.
Hasil ini tidak mengukur recovery Poltrada V3 100 epoch, GPU latency, atau upload W&B nyata.

[Uji CPU fine-tuning YOLO26s dengan konfigurasi baseline — 6 November 2026](pruned_fine_tuned_yolo26s_cpu_20261106.json)
memverifikasi satu epoch dari raw YOLO26s 1,02×, lebar channel yang tetap, pembaruan bobot, reload `best.pt`, serta evaluasi dan diagram tiga tahap. [Panduan notebook](../../../docs/lamp_gflops_recovery_workflow.md#fine-tuning-yolo26s-dengan-protokol-baseline)

[Uji CPU recovery 02–05 YOLO26s — 6 November 2026](yolo26s_source_matched_cpu_20261106.json) memakai raw yang sama untuk fine-tuning, Native KD, CrossKD dan MGD. Bukti mencakup hash, reload pada proses baru, forward finite, dan EDA. Teacher KD memakai fixture sintetis dalam uji ini. [Panduan recovery](../../../docs/lamp_gflops_recovery_workflow.md#3-pilih-notebook-recovery-per-metode)
