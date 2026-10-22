# Audit implementasi PKD YOLO26s → YOLO26n

Tanggal: **18 Oktober 2026**, Asia/Jakarta.

## Hasil dan batas klaim

Notebook: `notebooks/kaggle/kd/teacher_s/12_pkd_yolo26n.ipynb`.
Nama persis belum dipakai saat implementasi dimulai. File CSAKD bernomor 12 memiliki nama
berbeda dan tidak ditimpa. Seluruh **14 notebook yang sudah ada**, termasuk 01–11, EDA dan
CSAKD, dipertahankan byte-identik terhadap snapshot awal pekerjaan ini.

| Pemeriksaan | Status | Bukti/cakupan |
|---|---|---|
| File notebook, JSON/nbformat, Python | PASS | Notebook standalone, 36 cell, 18 heading berbahasa Indonesia |
| Fidelity rumus PKD | PASS | Dibandingkan dengan class PKDLoss resmi MMRazor yang diverifikasi hash |
| Numerical verification CPU | PASS | Loss dan gradien; random, variansi kecil, konstan, identik, anticorrelated, sum tiga level |
| Feature taps dan alignment | PASS | Graph s/n aktual; stride 8/16/32; bentuk 640 diuji CPU satu gambar |
| Student/projector backward | PASS | Backward dari komponen PKD saja, tanpa detection loss sebagai pengganti bukti |
| Teacher frozen dan BN tetap | PASS | Gradien None; excluded optimizer; digest parameter dan buffer sebelum/sesudah tetap pada CPU |
| Sumber teacher s konsisten | PASS | Referensi artifact dan producer sama persis dengan Native 10/CrossKD 11; schema YOLO26s dan 172.032 parameter projector diuji |
| Integrasi trainer/validation/checkpoint | PASS | Diagnostic pendek dan satu epoch training pada data sintetis CPU 64px |
| Student-only checkpoint | PASS | Load/predict pada interpreter Python terpisah tanpa class notebook |
| State archive dan upload failure | PASS | State terpisah tersedia; upload sinkron; file lokal tetap ada jika upload gagal |
| Kaggle CUDA FP16, Poltrada 640/batch16 | **NOT TESTED** | Lingkungan lokal PyTorch 2.14.0+cpu, CUDA tidak tersedia |
| Peak VRAM dan durasi GPU | **NOT TESTED** | Tidak ada angka VRAM atau estimasi jam GPU nyata yang dihasilkan lokal |
| Teacher artifact nyata dan SHA/epoch aslinya | **NOT TESTED** | W&B dimock pada pengujian; notebook mengaudit artifact nyata saat dijalankan di Kaggle |
| Full training penelitian 100 epoch | **NOT TESTED** | Default flag False; full hanya boleh berjalan setelah GPU diagnostic PASS |
| Resume training yang setara | **RESUME_NOT_VERIFIED** | Tidak diimplementasikan sebagai fitur resume; `resume=True` ditolak |

PASS selalu dibatasi pada cakupan bukti di atas. Data sintetis, model berbobot acak dan W&B
mock tidak digunakan sebagai bukti performa Poltrada, peningkatan mAP atau keberhasilan GPU.

## Metode dan keputusan adaptasi

Referensi utama: [Cao dkk., PKD, NeurIPS 2022](https://papers.neurips.cc/paper_files/paper/2022/file/631ad9ae3174bf4d6c0f6fdca77335a4-Paper-Conference.pdf)
dan [PKDLoss OpenMMLab](https://github.com/open-mmlab/mmrazor/blob/v1.0.0/mmrazor/models/losses/pkd_loss.py).

Untuk BCHW, statistik per channel dihitung atas **B,H,W**. Dengan (m=BHW), gunakan sample
standard deviation (`correction=1`) dan normalisasi (hat F=(F-mu)/(sigma+10^{-6})).
Loss yang diimplementasikan:

\[
L_{PKD}=\sum_{i\in\{P3,P4,P5\}}\frac12\operatorname{MSE}_{mean}(\hat F_{S,i},\hat F_{T,i}),
\qquad L=L_{det}+\lambda_{PKD}L_{PKD}.
\]

Ini mengikuti reduksi kode referensi. Hubungan terhadap (1-r) mengandung faktor finite-sample
((m-1)/m) dan pengaruh epsilon; tidak diklaim persis sama dengan (1-r) pada semua tensor.
Statistik bukan dihitung per gambar dan tidak dihitung pada chunk batch terpisah.

Native Ultralytics memakai score-weighted L2 dan projector dua Conv/ReLU; PKD di sini tidak
memanggil loss Native tersebut. Detection loss resmi YOLO26, termasuk one-to-many/one-to-one
dan pembaruan criterion-nya, tetap dipakai. Kontrak engine menambahkan `B * PKD_WEIGHT * L_PKD`
ke vektor detection yang sudah batch-scaled: faktor batch diterapkan pada PKD tepat sekali.
Rasio yang dicatat membandingkan weighted PKD dengan detection total per gambar.

Keputusan adaptasi yang dicatat:

- Tap ditemukan melalui koneksi Detect aktual, dengan pemeriksaan tiga input dan stride
  P3/P4/P5=8/16/32. Pada YOLO26s/n yang diuji, indeksnya **[16,19,22]**.
- Channel student/teacher adalah **64→128**, **128→256**, **256→512**. Masing-masing memakai
  satu Conv1×1 tanpa bias, tanpa ReLU/BN. Total projector pada graph ini 172.032 parameter.
  Identity dipakai jika channel sudah sama. Projector tidak masuk checkpoint inference.
- Ukuran spasial wajib sama; tidak ada interpolasi tersembunyi. MMRazor memiliki opsi resize,
  tetapi pasangan YOLO26 ini tidak membutuhkannya.
- Projector, statistik dan PKD dihitung FP32 di luar autocast. Ini menjaga rumus saat feature
  awal berasal dari AMP; bukan cosine similarity, raw MSE atau attention transfer.
- Satu traversal teacher sampai neck terakhir, tanpa Detect teacher. Student melakukan satu
  forward dengan head resmi. Dummy forward 64px hanya untuk konstruksi/alignment; bukan bukti
  batch16/640 aman. PKD tidak menambahkan cross-forward atau matriks attention spasial kuadrat;
  blok backbone YOLO26 tetap dipertahankan.
- Feature cache dibersihkan dalam `finally`, termasuk pada exception. Autograd menyimpan
  tensor yang memang diperlukan untuk backward; tidak ada cache tensor antarbatch.

Sumber Native: [dokumentasi Ultralytics](https://docs.ultralytics.com/guides/knowledge-distillation/).
API diaudit langsung dari paket terpasang **8.4.155** (`DistillationModel`, trainer, checkpoint,
EMA, GradScaler). Link source tag yang diminta tidak berhasil diambil oleh browser tool,
sehingga audit tidak mengandalkan dugaan API dari versi lain.

## Referensi numerik yang dapat direproduksi

Source `main` dan release v1.0.0 yang dibaca memiliki formulasi yang sama. Notebook memakai
URL release tetap dan memeriksa SHA-256 kanonik (hanya trailing whitespace/newline dinormalisasi):

```text
https://raw.githubusercontent.com/open-mmlab/mmrazor/v1.0.0/mmrazor/models/losses/pkd_loss.py
41d342f58c56efdcadb188dfc6d77a492d05e1ec2bee11240cd2cff9230ed68c
```

Hanya class `PKDLoss` dieksekusi setelah hash cocok; decorator registry dilepas, tanpa mengubah
metode numeriknya. Tidak diperlukan instalasi MMCV/MMEngine. Fixture offline mempertahankan
copyright sumber di `tests/fixtures/mmrazor_pkd_v1_reference.py.txt`.

Hasil CPU FP32 dengan generator tetap:

| Kasus | Loss | Selisih absolut terhadap MMRazor |
|---|---:|---:|
| Random | 0,9901576042 | 5,9604644775e-8 |
| Variansi kecil | 0,0001018351 | 0 |
| Konstan | 0 | 0 |
| Identik | 0 | Uji near-zero lulus |
| Anticorrelated | 1,9843707085 | Lebih besar daripada identik |

Selisih maksimum termasuk agregasi tiga level **2,3841857910e-7**. Toleransi loss:
`atol=1e-6, rtol=1e-5`; toleransi perbandingan gradien: `atol=rtol=1e-4`.
Normalisasi dengan shift antar gambar juga diuji untuk membedakan statistik batch-spasial
dari normalisasi per gambar yang keliru.

## Protokol, teacher dan preflight GPU

Teacher mengikuti Native 10 dan CrossKD 11: **YOLO26s baseline v62 → YOLO26n**.
Konfigurasi training mengikuti Native/baseline: Poltrada V3, `yolo26n.pt`, seed42,
100 epoch, 640, batch16, device0, workers2, AdamW, LR0.001, patience100, AMP True,
weight decay0.0005, warmup3, close mosaic10, val. Default engine lain, termasuk augmentasi
dan `nbs=64`, berasal dari versi yang sama. Perbandingan konfigurasi efektif Native 05
berbeda hanya pada direktori/nama output saat sumber teacher disamakan. Referensi Native 05
digunakan untuk pemeriksaan parameter training; artifact teacher disamakan dengan 10/11
dan diverifikasi tersendiri. Seed pilot PKD tetap 42.

Teacher tetap:

```text
ilhamrafiqin-brawijaya-university/yolom26-kd-indotraffic/02_baseline_yolo26s-q5kp9k9d-checkpoints:v62
```

Download ke `/kaggle/working/teacher_yolo26s_v62`. Audit meliputi koleksi/versi, producer
q5kp9k9d dan entity/project, manifest/ukuran/checksum, SHA-256, schema tensor arsitektur s,
enam kelas dan urutan label, serta deklarasi dataset/model. `v62` bukan nomor epoch; epoch
dicatat dari metadata. Teacher tidak dilatih ulang dan tidak diganti pretrained sebagai fallback.

Default Kaggle kini `RUN_FULL_TRAINING=True`, `RUN_SMOKE_TEST=False`: langsung training
100 epoch setelah audit teacher dan numerical verification, tanpa run diagnostic tambahan.
Jika `RUN_SMOKE_TEST=True`, diagnostic menjalankan 20 batch aktual pada 640/batch16 dengan data
Poltrada. Lima langkah pertama menjadi warmup pengukuran. Diagnostic memakai trainer,
optimizer, warmup, augmentasi dan GradScaler asli, kemudian validation/checkpoint. Epoch
diagnostic yang dihentikan setelah 20 batch diberi label **partial diagnostic epoch**.

Dicatat: GPU/total VRAM, peak allocated/reserved, pertumbuhan allocated setelah warmup,
batch/resolusi, durasi iterasi, detection/PKD per level/raw/weighted/total, rasio PKD/detection,
loss scale, optimizer-step skip/backoff, norm gradien student/projector setelah native clipping,
layout/bentuk feature dan kesamaan digest teacher termasuk BN. Overflow AMP awal boleh
ditangani skip/backoff standar; langkah pengukuran terakhir harus memiliki update finite.
Pertumbuhan allocated melebihi toleransi 256 MiB setelah warmup menggagalkan diagnostic.

Jika smoke opsional diaktifkan, full gate menolak hasil yang belum PASS atau fingerprint kode/protokol/GPU/data/teacher yang
berubah. Ketika smoke dilewati, hasilnya tetap None dan tidak diberi status PASS; audit teacher,
numerical verification dan pemeriksaan runtime tetap aktif. Fingerprint kode mencakup bytecode dan konstanta yang stabil. Mengubah flag full saja tetap diperbolehkan;
path teacher terverifikasi tidak dihapus ketika configuration cell dijalankan ulang.

OOM tidak menyebabkan retry batch8 atau resize512. Fase dicatat dalam `failure.json`:
initialization, teacher/student forward, PKD/detection loss, backward, optimizer step,
validation atau checkpoint. Opsi perubahan protokol ditampilkan tanpa mengubah nilainya.
Estimasi durasi baru dihitung dari iterasi GPU yang benar-benar diukur, dan diberi label
**compute-only**: tidak memasukkan validation/upload/setup atau menjamin durasi sesi Kaggle.

## Checkpoint dan W&B

Group: `teacher_s_student_n_pkd`; nama final `12_pkd_yolo26n_seed42_<timestamp>`.
Run diagnostic memakai `job_type=diagnostic` dan tag diagnostic; final memakai training.
ID selalu baru dengan `resume="never"`. Teacher artifact ditautkan lewat `use_artifact`.
Metrik per epoch dicatat sekali, terpisah dari metrik per batch diagnostic.

Setiap epoch memiliki direktori snapshot unik dan artifact model yang memuat:

- `best.pt`, `last.pt`: inference student-only; tidak memerlukan class PKD, teacher atau projector.
- `training_state_last.pt`, `training_state_best.pt`: archive state tensor/class-independent,
  termasuk live/EMA student dan projector, optimizer, scheduler, scaler, epoch, criterion,
  stopper, gradient tertunda, RNG Python/NumPy/PyTorch/CUDA dan generator loader.
- `results.csv`, `args.yaml`, provenance teacher dan manifest SHA-256/ukuran.

Upload ditunggu dengan `.wait()` sebelum epoch berikutnya. File snapshot tidak ditimpa atau
dihapus; `upload_status.json` lokal mencatat PASS/FAIL sesudah upload. Jika upload gagal,
run ditutup gagal dan full gate tetap gagal, sementara checkpoint lokal tetap tersedia.
Checkpoint wrapper engine mentah tidak diberi nama inference `best.pt` di artifact publik.

**Resume tidak didukung/terverifikasi.** Archive ini melindungi state yang telah dihitung,
tetapi belum menjamin pemulihan counter lokal loop/akumulasi dan worker prefetch secara setara.
Tidak ada klaim uji kesetaraan resume. `RESUME_NOT_VERIFIED` ditampilkan sebelum full training;
`resume=True` ditolak. Kehilangan epoch yang sedang berjalan masih mungkin jika proses/quota
terputus sebelum checkpoint epoch selesai diunggah.

## Menjalankan di Kaggle

1. Upload notebook baru; aktifkan satu GPU dan Internet, pasang dataset pada path konfigurasi,
   serta Secret WANDB_API_KEY dengan akses teacher artifact dan project.
2. Restart kernel lalu **Run All** dengan default `RUN_FULL_TRAINING=True`,
   `RUN_SMOKE_TEST=False`, `EPOCHS=100`. Audit/numerical verification diikuti langsung training.
3. Tinjau rasio PKD/detection, gradien, VRAM, estimasi dan keterbatasan resume. `PKD_WEIGHT=1.0`
   adalah pilot yang belum dianggap optimal. Jangan menyalin bobot Native 6.0.
4. Untuk diagnostic tambahan saja, set `RUN_SMOKE_TEST=True`, `RUN_FULL_TRAINING=False`.
   Untuk diagnostic lalu training, set keduanya True. Jika diagnostic diaktifkan, training
   tetap mensyaratkan diagnostic PASS dengan fingerprint yang cocok.
5. Full mode memulai student baru dari pretrained yang sama, bukan melanjutkan student diagnostic.
   Evaluasi final hanya val. Test dievaluasi pada tahap penelitian terpisah setelah protokol final.

Risiko yang masih perlu diukur: CUDA FP16 dan kapasitas batch16/640, validation native yang
dapat memakai batch lebih besar, biaya transfer checkpoint setiap epoch, dan kecukupan kuota
GPU. Diagnostic mengurangi risiko implementasi, bukan jaminan semua batch/100 epoch akan aman.

## Perintah dan hasil pengujian

**50 tes lulus** dalam verifikasi suite PKD/tracking, konsistensi teacher s dengan Native/CrossKD,
dan regression test pengubahan flag
full melalui configuration cell. Lint notebook runtime dan pengujian lulus. Peringatan yang
tersisa pada tes integrasi berasal dari deprecation dependensi THOP. Perintah berikut dapat
dipakai untuk mengulang seluruh suite saat ini, termasuk regression test tersebut.

```powershell
.\.venv\Scripts\python.exe -m pytest tests/test_pkd_notebook.py tests/test_kd_notebook_tracking.py -m '' -q
# 50 kasus pada suite saat ini; mencakup dua pengujian integrasi CPU bertanda slow

.\.venv\Scripts\ruff.exe check tests/test_pkd_notebook.py tests/test_kd_notebook_tracking.py
# All checks passed
```

Snapshot SHA-256 sebelum/sesudah implementasi memverifikasi 14 notebook lama tidak berubah.
Pada audit awal, pemeriksaan CUDA nyata masih **NOT TESTED**; hasil CPU tidak menggantikannya.

## Pembaruan smoke GPU — 22 Oktober 2026

Pearson/PKD telah dicoba pada Tesla T4 dengan teacher YOLO26s v62 yang sama dan protokol
640/batch16/AMP. **Belum PASS satu epoch**: batch terakhir yang tercatat 416/424, kemudian
job berhenti bertepatan dengan batas 75 menit. Validasi dan checkpoint akhir belum selesai.
Loss finite, memori reserved terakhir 3,22 GiB, dan tidak ditemukan traceback/OOM dalam log.
Numerical verification terhadap referensi MMRazor lulus di worker GPU (max abs 2,38e-7).
Rincian serta batas klaim tersedia pada [laporan aktual](../reports/hasil_percobaan/pkd_kaggle_smoke_20261022.md).

Callback diagnostic diperbaiki agar pre-hook fused AdamW mengabaikan update yang ditandai
`optimizer.found_inf != 0`; AMP dapat memanggil hook walaupun update dilewati. Rumus PKD
tidak berubah. Suite PKD saat pembaruan ini: **35 tes lulus**, termasuk integrasi CPU dan
regresi fused AdamW + GradScaler nyata. W&B online, resume dan 100 epoch belum diverifikasi.
