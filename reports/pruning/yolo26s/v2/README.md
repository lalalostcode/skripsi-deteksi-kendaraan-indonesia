# Bukti LAMP V2 YOLO26s

[Notebook 03 dan protokol N/S](../../../../docs/lamp_yolo26n_extended_no_recovery.md)
memakai sumber `02_baseline_yolo26s-q5kp9k9d-checkpoints:v62` dalam project
`ilhamrafiqin-brawijaya-university/yolom26-kd-indotraffic`. Default extended 0,10/0,20/0,30/0,40/0,50
memulai ulang baseline S pada setiap rasio; hasil baseline N tidak digunakan untuk S.

Output S berada pada `pruning/lamp_v2/{suite_id}/yolo26s/`, dengan `baseline/`, direktori
policy/rasio, EDA, checkpoint FP32, provenance, laporan reload/prediction, dan metrik.
Diagram before/after/comparison, heatmap channel asli, tabel layer/stage, serta skor/mask
LAMP disimpan bersama data sumber. Nilai yang belum diukur tidak diganti dengan simulasi.

[Integrasi checkpoint asli CPU — 24 Oktober 2026](../../comparisons/lamp_v2_ns_cpu_verification_20261024.json)
memuat N dan S pada FP32 640 px. Ketiga konfigurasi S memperoleh structural dan EDA PASS,
termasuk save, fresh-process reload, prediction, dan parity raw/decoded outputs dengan selisih
maksimum nol. Hash sumber S sesuai `5bc3ae61f8da1c9f03916922de10146805c27c64a11e46edd766a00d0f6f18fd`.

| Policy / target ratio | Parameter | Reduksi parameter | GFLOPs | Reduksi GFLOPs |
| --- | ---: | ---: | ---: | ---: |
| Baseline | 9.952.508 | 0% | 22,765978 | 0% |
| Extended 0,10 | 8.858.156 | 11,00% | 21,470659 | 5,69% |
| Extended 0,20 | 7.936.472 | 20,26% | 19,522470 | 14,25% |
| Extended 0,30 | 7.075.142 | 28,91% | 17,348685 | 23,80% |

Channel independen S 4.112 menjadi 3.700/3.289/2.878. Ukuran checkpoint dibandingkan pada
format FP32 yang sama, dan estimasi GFLOPs memakai metode identik pada model unfused.
**mAP Poltrada V3 dan latency CUDA tetap `NOT_RUN`**. Tracking/upload W&B nyata tidak
dijalankan pada verifikasi CPU; hasil kompresi tidak menunjukkan kualitas deteksi.
[Indeks perbandingan N/S](../../comparisons/README.md) merangkum hasil kedua model.
Laporan V1 S dan bukti historis N tetap dipertahankan.

[Kembali ke indeks laporan pruning](../../README.md).
