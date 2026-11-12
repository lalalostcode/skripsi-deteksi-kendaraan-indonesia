# MGD: smoke pipeline lokal, 31 Oktober 2026

**PASS CPU SMOKE — GPU KAGGLE NOT TESTED.** Satu epoch benar-benar dijalankan dari implementasi notebook, bukan sekadar compile atau mock training. Bukti terstruktur: [JSON hasil eksekusi](mgd_cpu_smoke_20261031.json).

## Lingkup eksekusi

Notebook: [15_mgd_yolo26n_teacher_yolo26s.ipynb](../../../notebooks/kaggle/kd/teacher_s/15_mgd_yolo26n_teacher_yolo26s.ipynb). SHA-256 notebook sebelum/sesudah pengujian sama:

`2c5af14de0361f4963eff2196c60cddcd1be76385384a7034b5028ecca526e30`

| Bagian | Pengujian aktual |
|---|---|
| Runtime | Windows, Python 3.11, PyTorch 2.14.0+cpu; CUDA tidak tersedia |
| Teacher | Checkpoint YOLO26s v62 asli yang tersimpan lokal; checksum, tensor schema, nc=6 dan urutan kelas diperiksa |
| SHA-256 teacher | `5bc3ae61f8da1c9f03916922de10146805c27c64a11e46edd766a00d0f6f18fd` |
| Student | Pretrained YOLO26n resmi dari Ultralytics assets v8.4.0 |
| SHA-256 student awal | `9b09cc8bf347f0fc8a5f7657480587f25db09b34bf33b0652110fb03a8ad4fef` |
| Data | Sintetis: 5 gambar train, 4 gambar val; **bukan Poltrada V3** |
| Konfigurasi smoke | Seed 42; 1 epoch penuh; 64px, batch 2, workers 0, CPU, AMP False, nbs=2, plots False |
| Durasi fungsi run | 15,93 detik; tidak termasuk persiapan fixture dan pemeriksaan proses baru |
| Integrasi eksternal | Download/audit lokal teacher dan logging/upload W&B menggunakan adapter offline eksplisit |

Student publik diambil dari `https://github.com/ultralytics/assets/releases/download/v8.4.0/yolo26n.pt`. Teacher tidak diunggah atau diunduh ulang. Adapter lokal tetap memeriksa file artifact dan hash sebelum/sesudah `.wait()`, tetapi tidak membuktikan autentikasi, manifest retrieval atau upload W&B online.

## Hasil yang diperiksa

- Dispatch smoke notebook meminta dan menyelesaikan tepat satu epoch; full training tidak dipanggil.
- Tiga batch aktual berukuran **2, 2, 1** selesai dengan loss detection dan MGD finite.
- AdamW melakukan **3 update**; parameter student berubah. Gradien student, alignment dan generator terukur nonzero. Teacher beserta statistik BN tetap sama.
- Validasi selama training, validasi checkpoint best dan evaluasi akhir berhasil. mAP data sintetis bernilai nol dan tidak dipakai untuk menilai kualitas metode.
- Snapshot student-only `best.pt`/`last.pt`, state MGD/optimizer terpisah, provenance dan manifest tersimpan; adapter artifact selesai menunggu upload simulasi.
- `python -I` pada proses baru berhasil memuat `best.pt` sebagai YOLO26n biasa tanpa class notebook/teacher/projector dan menjalankan inferensi finite. Archive training-state juga berhasil dimuat; status tetap `RESUME_NOT_VERIFIED`.
- Notebook asli tidak ditulis ulang. Default tetap **100 epoch, 640px, batch 16, AMP True, full training True, smoke False**, seed 42 aktif.

## Reproduksi lokal

[Script verifikasi](../../../scripts/verify_mgd_cpu_smoke.py) memerlukan dependensi pengujian repository dan kedua checkpoint lokal. Notebook Kaggle tetap standalone dan tidak membutuhkan script ini.

```powershell
.\.venv\Scripts\python.exe -X utf8 scripts/verify_mgd_cpu_smoke.py --teacher tmp/head_rm_gpu_20261021/teacher_input/best.pt --student tmp/mgd_smoke_20261031/assets/yolo26n.pt --report reports/experiments/local/mgd_cpu_smoke_20261031.json
```

Script menjalankan loss/model/runtime notebook yang sebenarnya; adapter dibatasi pada data, konfigurasi CPU dan layanan eksternal. Tidak ada job GPU atau login W&B yang diluncurkan. Pengaturan Ultralytics untuk pengujian ditulis ke folder sementara workspace.

## Batas bukti

Hasil ini memverifikasi pipeline lokal dengan bobot sebenarnya. **VRAM, AMP GPU, dataset Poltrada V3 aktual di Kaggle, akses artifact privat, upload W&B online, resume dan stabilitas 100 epoch belum terverifikasi oleh smoke ini.** Batch 16/640px masih membutuhkan eksekusi GPU aktual. Untuk uji GPU satu epoch terlebih dahulu, ubah sementara `RUN_SMOKE_TEST=True` dan `RUN_FULL_TRAINING=False` pada notebook Kaggle; kembalikan setelah smoke berhasil.
