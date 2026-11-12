# Perbandingan LAMP V2 YOLO26n dan YOLO26s

[Notebook 03](../../../notebooks/kaggle/pruning/03_lamp_yolo26n_extended_no_recovery.ipynb)
menjalankan sepuluh eksperimen extended secara default: N/S pada rasio 0,10/0,20/0,30/0,40/0,50.
Setiap model memiliki baseline FP32, provenance, class order, dan pengukuran sendiri;
konfigurasi tidak meneruskan pruning dari rasio sebelumnya.

Ringkasan lintas model berada di `pruning/lamp_v2/{suite_id}/comparison/`. Grafik mAP versus
parameter/GFLOPs/latency mencantumkan model dan rasio. Diagram channel tiga panel, indeks asli,
mask/skor LAMP, dan tabel layer/stage berada di direktori konfigurasi masing-masing model.
Pengukuran confusion matrix memakai confidence aktual validator dan IoU pencocokan 0,45
pada SDK terpasang; validation IoU 0,7 merupakan pengaturan berbeda.

## Perluasan 40-50% — 3 November 2026

[Verifikasi CPU checkpoint asli N/S 40-50%](lamp_v2_ns_ratio050_cpu_verification_20261103.json):
keempat konfigurasi extended memperoleh structural dan EDA PASS pada FP32 640 px,
termasuk save/reload proses baru dan prediction. Baseline dimuat ulang pada setiap rasio.

| Model | Rasio | Parameter | GFLOPs | Checkpoint FP32 (byte) |
| --- | ---: | ---: | ---: | ---: |
| YOLO26n | 0,40 | 1.625.418 | 3,961894 | 6.982.631 |
| YOLO26n | 0,50 | 1.536.092 | 3,372954 | 6.623.591 |
| YOLO26s | 0,40 | 6.426.986 | 15,131258 | 26.256.231 |
| YOLO26s | 0,50 | 6.077.564 | 12,792422 | 24.857.127 |

mAP V3 dan latency CUDA tetap `NOT_RUN`; W&B diuji melalui mock. Cap output group terjaga
dan seluruh skor LAMP finite. Pengujian ini membuktikan implementasi, bukan akurasi terjaga.

## Bukti historis 10-30% — 24 Oktober 2026

[Integrasi CPU checkpoint asli N/S — 24 Oktober 2026](lamp_v2_ns_cpu_verification_20261024.json):
enam konfigurasi extended 0,10/0,20/0,30 memperoleh structural dan EDA PASS pada FP32 640 px.
Save, fresh-process reload dan prediction berhasil; perbedaan maksimum raw/decoded outputs
setelah reload adalah nol. Guard melarang training, backward, optimizer step, dan HTTP.

| Model | Policy / target ratio | Parameter | Reduksi parameter | GFLOPs | Reduksi GFLOPs |
| --- | --- | ---: | ---: | ---: | ---: |
| YOLO26n | Baseline | 2.506.140 | 0% | 5,905408 | 0% |
| YOLO26n | Extended 0,10 | 2.236.276 | 10,77% | 5,571206 | 5,66% |
| YOLO26n | Extended 0,20 | 2.013.751 | 19,65% | 4,954506 | 16,10% |
| YOLO26n | Extended 0,30 | 1.800.137 | 28,17% | 4,411438 | 25,30% |
| YOLO26s | Baseline | 9.952.508 | 0% | 22,765978 | 0% |
| YOLO26s | Extended 0,10 | 8.858.156 | 11,00% | 21,470659 | 5,69% |
| YOLO26s | Extended 0,20 | 7.936.472 | 20,26% | 19,522470 | 14,25% |
| YOLO26s | Extended 0,30 | 7.075.142 | 28,91% | 17,348685 | 23,80% |

Counter GFLOPs identik pada model unfused, dan baseline pembanding/checkpoint pruned disimpan
dalam format FP32 yang sama. **mAP Poltrada V3 dan latency CUDA `NOT_RUN`**; kompresi ini
tidak membuktikan kualitas deteksi. Tracking/upload W&B pada eksekusi ini tidak dijalankan;
penanganan upload diuji melalui mock. Pengujian sintetis N/S atas kurva, support, dan false
negative confusion matrix merupakan bukti implementasi, bukan hasil akurasi Poltrada V3.
Ledger atau skor historis yang tidak pernah dicatat tetap `UNAVAILABLE`.

[Bukti N historis](../yolo26n/v2/README.md), [indeks S](../yolo26s/v2/README.md),
[panduan metode/EDA](../../../docs/lamp_yolo26n_extended_no_recovery.md), dan
[indeks laporan pruning](../README.md). Bukti historis dan notebook 01/02 tidak ditulis ulang.
