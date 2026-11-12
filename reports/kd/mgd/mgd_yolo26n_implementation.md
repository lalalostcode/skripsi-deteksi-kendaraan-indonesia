# MGD YOLO26s → YOLO26n — implementasi Kaggle

Notebook: [15_mgd_yolo26n_teacher_yolo26s.ipynb](../../../notebooks/kaggle/kd/teacher_s/15_mgd_yolo26n_teacher_yolo26s.ipynb).

Status **PASS VERIFIKASI LOKAL — GPU KAGGLE: NOT TESTED**, 31 Oktober 2026. Hasil CPU sintetis tidak merupakan hasil empiris Poltrada V3 atau jaminan 100 epoch.

## Referensi dan adaptasi

Referensi ialah *Masked Generative Distillation*, Yang dkk., ECCV 2022. Paper lokal berada pada `references/mgd/paper/Masked Generative Distillation.pdf`: hal.5–6, Persamaan 2–6; hal.8, §4.2; dan hal.13, §5.4. Kode deteksi penulis berasal dari [yzd-v/MGD](https://github.com/yzd-v/MGD/tree/2c9da0b28625eb948db57afc02c824452c3910fe), file `det/mmdet/distillation/losses/mgd.py`, SHA-256 `a91595b98b906363abff7374662638ccaa7cc1557b44ab61640e5d610b9649de`. Fixture pengujian mempertahankan byte sumber beserta salinan lisensi Apache-2.0.

Per level: alignment Conv1×1 dengan bias bila channel berbeda; mask spasial B×1×H×W; generator Conv3×3/ReLU/Conv3×3 dengan bias; rekonstruksi semua posisi menggunakan **SSE / actual batch**, dikalikan alpha sekali. Mask menggunakan CPU `torch.rand` dan threshold ketat `U > 1-lambda` seperti kode deteksi. Kondisi paper `R < lambda` setara distribusi, bukan mask identik untuk angka acak yang sama. Reduksi batch berasal dari implementasi kode, sedangkan Eq.5 paper menuliskan penjumlahan layer/channel/spasial.

Adaptasi YOLO memakai seluruh tiga input neck Detect, P3/P4/P5 stride8/16/32 (tap16/19/22 pada graph yang diuji). Student64/128/256 → teacher128/256/512; tidak ada interpolasi. Setiap level mempunyai alignment/generator tersendiri; tambahan **6.367.872 parameter training-only**. Jumlah loss level tidak dirata-rata. Default one-stage resmi: `MGD_ALPHA=2e-5`, `MGD_MASK_RATIO=0.65`; belum dianggap optimal untuk YOLO26.

Alignment/generation/SSE berjalan FP32 di luar autocast untuk menghindari overflow reduksi besar; detector tetap memakai AMP. Fitur teacher detached/frozen/eval, BN tidak diperbarui. Teacher forward berhenti sebelum Detect, sementara student tetap menggunakan detection loss resmi O2M+O2O. `MGDModel(DistillationModel)` mengimplementasikan loss khusus; loss Native score-weighted KD tidak dipanggil. Faktor actual batch ditambahkan ke loss MGD sekali untuk kontrak loss native.

Paper memakai SGD, umumnya 24 epoch, dan pewarisan bobot neck/head teacher bila struktur cocok. Notebook mempertahankan inisialisasi `yolo26n.pt` dan protokol Native/CrossKD sebelumnya, tanpa pewarisan bobot teacher. Ini adaptasi MGD ke YOLO26, bukan reproduksi seluruh setting COCO paper.

## Kesetaraan protokol

Pembanding utama: [Native KD 10](../../../notebooks/kaggle/kd/teacher_s/10_native_yolo26n_teacher_yolo26s.ipynb) dan [CrossKD 11](../../../notebooks/kaggle/kd/teacher_s/11_crosskd_yolo26n_teacher_yolo26s.ipynb).

| Komponen | Nilai |
|---|---|
| Student | `yolo26n.pt`, fresh pretrained initialization setiap seed |
| Teacher | `02_baseline_yolo26s-q5kp9k9d-checkpoints:v62`, producer q5kp9k9d |
| SHA-256 teacher | `5bc3ae61f8da1c9f03916922de10146805c27c64a11e46edd766a00d0f6f18fd` |
| Dataset | `/kaggle/input/datasets/ilhamrafiqin/poltrada-v3-yolo26-clean/data.yaml` |
| Kelas | Bus, Mobil Penumpang, Pejalan Kaki, Sepeda Motor, Truck, Unmotorized |
| Anggaran | 100 epoch per seed; 640px, batch16, workers2, device0 |
| Optimizer | AdamW lr0.001, weight decay0.0005 |
| Jadwal | warmup3, close mosaic10, patience100, native scheduler/accumulation |
| Numerik | AMP True; clipping/GradScaler native; tidak ada fallback otomatis |
| Default engine | Ultralytics8.4.155, nbs64, lrf0.01, cos_lrFalse, fraction1, augmentasi/gains native |
| Evaluasi | val untuk pemilihan checkpoint dan evaluasi akhir; test tidak dipakai tuning |
| Paket | ultralytics8.4.155, wandb0.28.1, PyTorch bawaan Kaggle |

`SEEDS_TO_RUN` berisi 42 aktif; 123, 2024, 7, 3407 dikomentari. Kelima seed dapat dijalankan serial dengan membuka komentar. Daftar Native/CrossKD saat ini mengomentari 42 karena run sebelumnya selesai; hal itu tidak diterapkan pada MGD baru. Setiap seed menggunakan student/MGD/optimizer/scaler, ID W&B dan output baru. Default full training True, smoke False. Smoke opsional satu epoch penuh memakai run dan model terpisah.

## Tracking dan checkpoint

Entity/project W&B tetap sama, group `T26s_S26n`; nama `mgd_T26s_S26n_seed<seed>`, fresh ID, `resume="never"`. Audit teacher mengunci versi, producer, manifest, ukuran/checksum, SHA-256, tensor schema dan urutan kelas. Konfigurasi efektif dan hash YAML dicatat bersama loss mentah/berbobot per level, mask rate, rasio MGD/detection, gradien student/alignment/generator, optimizer update/AMP skip, VRAM dan durasi.

Native EMA dipertahankan. Snapshot per epoch mengekspor `best.pt`/`last.pt` student-only melalui jalur resmi `strip_optimizer`; tidak memerlukan class notebook, teacher atau generator. Archive state terpisah menyimpan student, alignment/generation, EMA, optimizer/scheduler/scaler, RNG dan provenance. Upload W&B ditunggu; kegagalan menghentikan antrean dan menyimpan snapshot serta `upload_status.json` gagal. Seed berikutnya tidak mengambil model atau hasil seed sebelumnya. `RESUME_NOT_VERIFIED` tetap eksplisit dan `resume=True` ditolak.

Default `KEEP_LAST_LOCAL_EPOCHS=1`: snapshot sukses lama dibersihkan hanya setelah snapshot berikutnya selesai diunggah dan tercatat. Snapshot terbaru membawa best student dan salinan training-state best, sehingga state best tetap tersedia walaupun direktori epoch asal sudah dibersihkan. Snapshot gagal, belum terkonfirmasi, atau metadata epoch tidak cocok tidak dihapus. Path cleanup diperiksa tetap berada dalam direktori run. Versi semua epoch tetap di W&B. File epoch ditambahkan dengan `policy="immutable", skip_cache=True`, lalu ditunggu `.wait()` sebelum cleanup, untuk mencegah duplikasi besar di cache upload lokal.

Kebijakan ini diperlukan karena satu snapshot CPU dengan arsitektur penuh berukuran sekitar 283 MiB (dua archive state masing-masing sekitar136 MiB ditambah dua student checkpoint); menyimpan100 snapshot tanpa cleanup dapat melampaui27 GiB per seed sebelum cache/raw checkpoint. Ini pengukuran file lokal, bukan estimasi VRAM atau durasi GPU. Batas alpha0/lambda1 tetap valid untuk oracle loss, tetapi ditolak sebelum research run karena tidak memenuhi pemeriksaan gradien trainable nonzero.

## Pengujian lokal

| Suite | Hasil |
|---|---|
| MGD: loss/gradien referensi, kontrak model/protokol, isolasi seed, checkpoint, AMP dan retensi | **42 passed**; 41 kasus dalam102,17 detik, kemudian satu regresi best-state carry-forward tambahan lulus dalam5,91 detik |
| Regresi Native/CrossKD teacher S, multi-seed, PKD, HEAD/RM, tracking dan Colab | **213 passed, 1 skipped**,281,13 detik; baseline yang tidak memakai teacher melewati satu kasus khusus teacher |
| Layout/registry notebook | **93 passed**,12,86 detik |
| Layout/integritas/navigasi laporan | **70 passed**,1,27 detik |
| Ruff: generator, sumber loss/runtime dan tes MGD | PASS |
| `git diff --check` | PASS |

Tes numerik menggunakan fixture kode deteksi resmi dengan hash yang dikunci, bobot identik, dan RNG/mask terkontrol. Kasus mencakup channel sama/berbeda, mask0/0,65/1, threshold ketat, SSE/B pada semua posisi, sum level, faktor batch sekali, serta batch akhir kecil. Pemeriksaan model memastikan gradien student/alignment/generator, teacher/BN tetap, tidak ada pemanggilan Native KD/teacher Detect, optimizer coverage tepat sekali, dan cache/hook dibersihkan.

Integrasi CPU nyata menyelesaikan **dua epoch untuk seed42 dan dua epoch untuk seed123**, memakai dataset dan checkpoint sintetis kecil, bukan Poltrada. Student awal sama berasal dari file fixture pretrained tetap; kedua run/ID/direktori/state terpisah. Setiap upload disimulasikan melalui adapter yang memeriksa file tersedia sampai `.wait()` selesai. Snapshot001 dibersihkan setelah upload002 berhasil; snapshot002 memuat full state dan portable best student. Student-only checkpoint dan archive juga dimuat dengan `python -I` dalam proses baru dan menghasilkan inferensi finite. Satu tes tiga snapshot terpisah membuktikan state best epoch1 tetap terbawa sampai epoch3 setelah direktori asal dihapus.

Tes kegagalan upload menghentikan antrean sebelum seed berikutnya dan mempertahankan snapshot. Tes fused AdamW + CPU GradScaler nyata membuktikan overflow dilewati tanpa dihitung sebagai update, lalu update finite mengubah parameter. Batas pengaturan penelitian tidak valid ditolak sebelum membuat run/mengakses GPU/download teacher. Pemeriksaan retensi mencakup keep1/keep2, snapshot gagal/belum terkonfirmasi, metadata epoch salah, dan path di luar direktori run.

Perintah MGD lengkap (termasuk integrasi bertanda slow):

```powershell
.\.venv\Scripts\python.exe -m pytest tests/test_mgd_notebook.py -o addopts= -q --tb=short --show-capture=no
.\.venv\Scripts\ruff.exe check scripts/generate_mgd_notebook.py scripts/mgd_method_source.py scripts/mgd_runtime_source.py tests/test_mgd_notebook.py
```

Peringatan suite tersebut berasal dari deprecation THOP. Tidak ada autentikasi/upload W&B nyata atau peluncuran training GPU Kaggle dalam pekerjaan ini. Suite utama di atas memakai fixture sintetis; smoke tambahan dengan bobot aktual dilaporkan berikut.

Notebook dihasilkan ulang dengan `python scripts/generate_mgd_notebook.py`; cell loss/model dan runtime disematkan sehingga notebook sendiri tidak memerlukan file generator/reference saat di Kaggle. Notebook KD lama dan koleksi Colab tidak ditulis ulang.

## Smoke tambahan dengan bobot aktual — 31 Oktober 2026

[Smoke CPU satu epoch](../../experiments/local/mgd_cpu_smoke_20261031.md) **PASS**, memakai teacher YOLO26s v62 asli dengan checksum yang dikunci dan student pretrained YOLO26n resmi. Lima gambar train sintetis menghasilkan batch2,2,1 dan tiga update optimizer; teacher/BN tetap, student/alignment/generator menerima gradien. Validasi, ekspor student-only dan load checkpoint/arsip dalam proses Python baru berhasil. [Bukti JSON](../../experiments/local/mgd_cpu_smoke_20261031.json) mencatat loss, gradien, hash dan konfigurasi aktual.

Uji ini memakai CPU, 64px, batch2, AMP False dan adapter W&B lokal. **GPU Kaggle 640px/batch16/AMP, data Poltrada aktual, akses artifact privat dan upload W&B online tetap NOT TESTED.** Default notebook 100 epoch/full training True/smoke False tidak berubah.
