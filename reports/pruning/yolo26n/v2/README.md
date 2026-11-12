# Bukti LAMP V2 YOLO26n

Laporan di folder ini merekam pengujian implementasi conservative dan extended.
Setiap laporan harus membedakan status struktur, reload/inference, evaluasi dataset,
dan latency CUDA; hasil yang tidak dijalankan tetap `NOT_RUN`.

Lihat [panduan Notebook 03](../../../../docs/lamp_yolo26n_extended_no_recovery.md)
dan [indeks laporan pruning](../../README.md).

[Verifikasi CPU N/S dan EDA — 24 Oktober 2026](../../comparisons/lamp_v2_ns_cpu_verification_20261024.json):
masing-masing model diuji pada extended 10%, 20%, dan 30%, dengan structural dan EDA PASS.
Lihat [perbandingan N/S](../../comparisons/README.md) untuk parameter/GFLOPs serta batas pengujian.
mAP Poltrada V3, latency CUDA, dan upload W&B nyata belum diuji pada eksekusi lokal ini.

[Verifikasi CPU checkpoint target](lamp_v2_cpu_verification_20261022.json): keenam konfigurasi
structural PASS, fresh reload/prediction konsisten; mAP target dan CUDA NOT_RUN. Sebanyak
301 tes relevan lulus, termasuk preservasi notebook/laporan dan mock upload W&B.

[Smoke lokal extended 10% — 23 Oktober 2026](lamp_v2_extended_local_smoke_20261023.json):
satu langkah pruning tanpa epoch training; structural, save/reload dan inference PASS.
Evaluasi V3 dan CUDA tetap NOT_RUN. Dataset lokal berlabel V2 tidak digunakan sebagai pengganti.
