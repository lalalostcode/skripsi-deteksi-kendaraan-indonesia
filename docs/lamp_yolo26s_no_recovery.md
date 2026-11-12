# LAMP structured pruning YOLO26s tanpa recovery

Notebook standalone Kaggle: [02_lamp_yolo26s_no_recovery.ipynb](../notebooks/kaggle/pruning/02_lamp_yolo26s_no_recovery.ipynb).
Gunakan GPU dan Internet, tambahkan dataset Poltrada V3 yang dipakai baseline, aktifkan
Kaggle Secret `WANDB_API_KEY`, periksa lokasi `DATA_YAML`, lalu Run All pada kernel baru.

Sumber checkpoint adalah artifact
`ilhamrafiqin-brawijaya-university/yolom26-kd-indotraffic/02_baseline_yolo26s-q5kp9k9d-checkpoints:v62`.
Selector `best.pt` harus unik. Identitas kelas dibaca dari checkpoint dan dicocokkan dengan dataset.
Audit memeriksa skala S dan stem 32 channel, bukan guard nano 16 channel.

Default: `PRUNING_RATIO=0.30`, global LAMP structured, satu langkah, cap tiap group 50%,
FP32, input 640, seed 42, `RUN_FULL_VALIDATION=True`, batch 16, workers 2, dan latency CUDA aktif.
Tidak ada recovery, fine-tuning, training, KD, backward, atau optimizer step.
Rasio 30% berlaku pada hidden channel eligible; reduksi parameter/GFLOPs dilaporkan terpisah.

Kandidat hanya Conv pertama dalam Bottleneck backbone/neck, BN, dan input Conv konsumennya.
Head one-to-many/one-to-one, attention, dimensi CSP/residual, dan lebar fitur antartahap dilindungi.
Kapasitas group dibatasi sebelum ranking global tanpa mengubah skor LAMP kandidat.
Model disimpan sebagai objek struktural FP32 lalu wajib diuji dengan `YOLO()` dan prediction
pada proses Python baru. Evaluasi baseline dan pruned memakai salinan terpisah dan protokol identik.

Project W&B tetap `yolom26-kd-indotraffic`. Nama run
`prun_lamp1_yolo26s_pruned-pr0.3` mengikuti rasio konfigurasi. Group `lamp-v1` dan tag
metode, model, seed, serta rasio mengikuti [standar W&B](wandb_naming.md).
UUID baru dan `resume="never"` mencegah penggunaan ulang run baseline/KD.
Output lokal terpisah di `/kaggle/working/lamp_yolo26s/{run_unik}/`.

Format output, restore, dependency pins, dan protokol evaluasi mengikuti
[panduan LAMP](lamp_yolo26n_no_recovery.md). Hasil numerik historis YOLO26n pada panduan
tersebut tidak dianggap sebagai hasil YOLO26s. Pengujian sintetis hanya membuktikan implementasi;
mAP dataset V3 dan latency CUDA memerlukan run Kaggle.

## Verifikasi CPU pada artifact target

Artifact `:v62` berhasil diunduh dan seluruh code cell dijalankan pada `best.pt` asli,
CPU FP32 640 px, tanpa tracking W&B atau instalasi ulang dependency.
Guard menggagalkan pemanggilan training, backward, dan optimizer step.
Hasil: **structural smoke PASS**, channel eligible 880 → 616 (30%), parameter
9.952.508 → 9.279.212 (6,7651%), GFLOPs 22,7659776 → 22,0271616.
Head tetap utuh; direct `YOLO()` reload dan prediction pada proses baru lulus,
dengan selisih maksimum raw/decoded outputs nol. Prediksi kosong pada input sintetis
tidak dianggap sebagai bukti akurasi deteksi objek nyata.

Sebanyak 127 tes pruning/layout lulus, termasuk kedua skala N/S, penamaan W&B,
penolakan skala salah, dan validasi dataset sintetis. Tidak ada run W&B sungguhan dibuat.
Validasi target dan latency CUDA tetap **NOT_RUN** karena V3/GPU tidak tersedia lokal.
Detail: [laporan CPU YOLO26s](../reports/pruning/yolo26s/v1/lamp_yolo26s_ratio030_cpu_smoke_20261022.json).
