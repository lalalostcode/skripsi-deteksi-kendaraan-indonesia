# LAMP YOLO26n/S dengan recovery fine-tuning biasa

[Notebook Kaggle 04](../notebooks/kaggle/pruning/04_lamp_yolo26ns_finetune_recovery.ipynb)
menambahkan training supervised setelah LAMP structured pruning. Notebook 01-03 tetap
menyimpan eksperimen tanpa recovery. Semua helper runtime tertanam di notebook; tidak
memerlukan clone atau import package repository pada Kaggle.

Default N/S pada rasio 10%, 20%, 30%, 40%, dan 50% menghasilkan sepuluh konfigurasi
independen. Setiap konfigurasi memuat ulang baseline asli, memangkas satu kali, mengevaluasi
model pruned, lalu memulai recovery dari bobot dan struktur pruned tersebut. Hasil rasio
sebelumnya tidak menjadi bobot awal rasio berikutnya. Baseline dievaluasi sekali per model.

## Dasar implementasi

[Dokumentasi engine Ultralytics](https://docs.ultralytics.com/usage/engine/) menjelaskan
penggunaan `DetectionTrainer` serta kustomisasi pemuatan model melalui `get_model`.
[Train mode](https://docs.ultralytics.com/modes/train/) menyediakan training supervised,
pemilihan optimizer, learning rate, checkpoint, dan validation. Recovery ini memakai loss,
optimizer step, augmentasi, serta EMA bawaan Ultralytics. Tidak ada teacher, KD, sparse
training, atau objective tambahan.

Pemuatan objek model struktural perlu dipertahankan secara eksplisit: pada Ultralytics
8.4.155, loader training standar dapat membangun `DetectionModel` dari YAML lalu mentransfer
bobot yang cocok. YAML baseline tidak merepresentasikan lebar channel hasil pruning.
Loader recovery mempertahankan objek native `DetectionModel` hasil pruning, memeriksa
urutan kelas dan struktur sebelum training, dan memulai optimizer baru dengan `resume=False`.
Ini bukan melanjutkan optimizer atau epoch dari checkpoint baseline.

[Kode penulis MP-YOLO](https://github.com/Wang-jj-zs/MP-YOLO/blob/master/MP-YOLO/compress.py)
memisahkan pruning dan fine-tuning dengan contoh recovery 300 epoch.
[Compressor penulis](https://github.com/Wang-jj-zs/MP-YOLO/blob/master/MP-YOLO/ultralytics/models/yolo/detect/compress.py)
memakai `LAMPImportance(p=2)`; cabang LAMP tidak mengaktifkan sparse learning sehingga
`sl_epochs=500` pada contoh bukan otomatis 500 epoch sparse training untuk LAMP.
Notebook 04 merupakan adaptasi ke YOLO26n/S dan Poltrada V3. Budget recovery 5 epoch
merupakan pilihan eksperimen proyek; jadwal pruning satu langkah, cap 50%, arsitektur,
dataset, dan optimizer tidak mereplikasi seluruh protokol MP-YOLO.

## Konfigurasi dan Run All

Sumber checkpoint tetap pada project `ilhamrafiqin-brawijaya-university/yolom26-kd-indotraffic`:

| Model | Artifact baseline |
| --- | --- |
| YOLO26n | `03_baseline_yolo26n-9uu3meve-checkpoints:v84` |
| YOLO26s | `02_baseline_yolo26s-q5kp9k9d-checkpoints:v62` |

Selector default `best.pt` harus unik. Checkpoint lokal dan expected SHA-256 dapat dipilih
per model seperti notebook 03; sumber, hash, dan tingkat verifikasi provenance tetap dicatat.

| Pengaturan | Default |
| --- | --- |
| Model dan policy | N/S, `extended` |
| Rasio | `[0.10, 0.20, 0.30, 0.40, 0.50]` channel eligible |
| Pruning | LAMP global, satu langkah, cap per group 50%, tanpa rounding |
| Budget recovery | 5 epoch per konfigurasi, budget sama untuk setiap model/rasio |
| Optimizer | AdamW, `lr0=0.0001`, weight decay `0.0005` |
| Training | Input 640, batch 16, nominal batch `RECOVERY_NBS=64`, workers 2, seed 42, FP32, AMP mati |
| Schedule | Warm-up 3 epoch, Mosaic dinonaktifkan pada epoch terakhir, `patience=0` |
| Resume dan KD | `resume=False`, tanpa KD atau sparse training |

Rasio menghitung channel eligible, bukan seluruh parameter atau ukuran file. Cap 50%
per group tetap berlaku ketika rasio global 50%. Struktur head dan attention dilindungi
saat pruning; bobotnya tetap dapat belajar saat fine-tuning biasa. Pemeriksaan sesudah
recovery membandingkan tipe modul, lebar channel, bentuk tensor, dan jumlah parameter
dengan model pruned, bukan menuntut bobot head tetap identik.

Di Kaggle aktifkan GPU dan Internet, tambahkan dataset **Poltrada V3 YOLO26 Clean**, dan
aktifkan secret `WANDB_API_KEY`. Periksa konfigurasi, lalu Run All dengan kernel baru.
Sepuluh konfigurasi menjalankan training secara serial. Untuk percobaan singkat, pilih
satu `MODEL_VARIANTS` dan satu `PRUNING_RATIOS`, lalu kecilkan budget epoch pada konfigurasi.
Untuk uji satu epoch, atur juga `RECOVERY_CLOSE_MOSAIC=0`. Pengujian singkat tidak
membuktikan recovery pada budget penuh. `RECOVERY_NBS` mengatur batch efektif untuk
akumulasi gradien; fixture CPU kecil memakai nbs=2 agar satu batch benar-benar melakukan step.

## Evaluasi tiga tahap dan checkpoint

Bandingkan **baseline → pruned tanpa recovery → recovered** pada dataset dan urutan kelas
yang sama: split `val`, 640, batch 16, workers 2, confidence `0.001`, IoU `0.7`, maksimum
300 deteksi, `nms=None`, FP32, tanpa augmentasi saat evaluasi. Training memakai split train;
validation memilih checkpoint terbaik. Evaluasi pada split yang juga dipakai untuk memilih
checkpoint bukan pengukuran generalisasi pada test set independen.

Simpan mAP50-95, mAP50, precision/recall/F1, AP dan support per kelas, kurva, confusion
matrix beserta threshold aktual, parameter, GFLOPs, ukuran checkpoint FP32, dan latency
raw forward CUDA dengan p05-p95. Delta terhadap baseline serta peningkatan dari pruned
ke recovered dicatat terpisah. Hilangnya mAP tidak boleh ditutupi oleh status struktur PASS.

Checkpoint pembanding memakai objek native FP32. Recovery menyimpan EMA terbaik dalam
presisi FP32 ketika validation fitness memilihnya; mengubah checkpoint internal FP16 ke
float sesudah training tidak mengembalikan presisi yang telah hilang. File internal SDK
`best.pt`/`last.pt` dapat memakai format SDK; pembandingan memakai `recovered_fp32.pt`.
File ekspor berisi model struktural native, `ema=None`, tanpa optimizer atau class trainer
khusus, sehingga `YOLO(recovered_checkpoint)` dapat dimuat pada proses baru tanpa repository.
Fresh-process reload, inference, serta parity raw/decoded outputs tetap wajib diperiksa.

## Output dan tracking

Root Kaggle: `/kaggle/working/pruning/lamp_v2_finetune/{suite_id}/`.
Padanan lokal: `outputs/pruning/lamp_v2_finetune/{suite_id}/`.

```text
suite_manifest.json
yolo26n/ atau yolo26s/
  baseline/
  extended/pr0.1/          # juga pr0.2 sampai pr0.5
    pruned_fp32.pt         # hasil sebelum recovery tetap disimpan
    eda/                  # channel, indeks asli, mask, dan importance LAMP
    recovery/
      recovered_fp32.pt
      training/           # log training dan checkpoint internal SDK
comparison/               # tabel/grafik baseline, pruned, dan recovered
```

Konfigurasi, provenance, audit pruning, skor/mask, status, metrik, laporan reload, serta
log training disimpan bersama checkpoint. Struktur channel setelah recovery tetap sama
dengan setelah pruning; perubahan akurasi sesudah recovery dideskripsikan sebagai hasil
training pada struktur tersebut. Diagram channel bukan visualisasi aktivasi feature map.

W&B tetap memakai project yang sama dengan UUID baru dan `resume="never"` untuk setiap
konfigurasi. Contoh nama `prun_lamp2_yolo26n_ft5_pruned-pr0.3`, dengan tag fine-tuning,
budget recovery, rasio, dan seed. Suite tercatat di group sesuai [standar W&B](wandb_naming.md).
Logger otomatis SDK dinonaktifkan
untuk mencegah run tambahan atau logging ke run lama. Artifact model pruned, recovered,
dan laporan memiliki identitas baru serta provenance sumber. Kegagalan tracking/upload
dicatat terpisah dari hasil lokal; tidak diklaim berhasil tanpa konfirmasi upload.

## Status dan batas pengujian

Laporkan validitas struktur, training recovery, reload, evaluasi, dan upload secara terpisah.
Dataset yang tidak tersedia menghasilkan `NOT_RUN`; dataset tersedia tetapi invalid atau
kegagalan training menghasilkan laporan FAIL lalu exception diteruskan. Checkpoint pruned
dan hasil konfigurasi yang sudah selesai tetap tersimpan ketika recovery berikutnya gagal.
Tidak ada klaim keberhasilan ilmiah otomatis hanya karena training atau kompresi berhasil.

Pengujian implementasi lokal menggunakan dataset sintetis kecil dan budget singkat untuk
memeriksa loss finite, optimizer step, perubahan bobot, preservasi struktur, save/reload,
serta kegagalan yang tercatat. Angka mAP Poltrada V3, pemulihan akurasi pada budget penuh,
latency GPU, dan upload W&B nyata hanya dapat diklaim setelah benar-benar dijalankan.
Lihat [indeks bukti recovery](../reports/pruning/recovery/README.md).

Pada 3 November 2026, **439 tes relevan lulus**. Selain tes training model sintetis N/S,
seluruh code cell notebook dijalankan pada checkpoint asli N `v84` dan S `v62`, pruning 50%,
dengan dataset sintetis 2 citra train/2 citra val, input 64, satu epoch, batch/nbs 2 pada CPU.
Kedua model melakukan optimizer step, menghasilkan loss finite dan perubahan bobot, mempertahankan
struktur pruned, serta lolos best-EMA FP32 save dan fresh-process reload/prediction.
Parity raw/decoded outputs lolos pada toleransi yang dicatat; selisih absolut terbesar
`1.9073486328125e-5`. Pengujian sintetis mengukur satu epoch di CPU.
