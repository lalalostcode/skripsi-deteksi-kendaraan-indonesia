# KD YOLO26s ke YOLO26n — mode langsung 100 epoch

Diperbarui: **17 Oktober 2026**, Asia/Jakarta.

Pembaruan terbaru: notebook 10/11 sekarang menjalankan seed `[123, 2024]` secara serial,
dengan satu run W&B per seed. Seed 42 yang sudah selesai tidak dijalankan ulang secara default.
Detail terkini ada di [laporan multi-seed](serial_multiseed_implementation.md).
Bagian berikut mencatat implementasi dan verifikasi sebelum penambahan antrean multi-seed.

Sesuai permintaan terbaru pengguna, notebook 10 dan 11 sekarang langsung menjalankan training
100 epoch saat Run All setelah verifikasi teacher. Tahap smoke test dan syarat hasil smoke telah
dihapus. Ini menggantikan mode diagnostik default pada implementasi 16 Oktober.

## Notebook dan konfigurasi

- `notebooks/yolo26-kd-indotraffic/notebooks/10_native_yolo26n_teacher_yolo26s.ipynb`
- `notebooks/yolo26-kd-indotraffic/notebooks/11_crosskd_yolo26n_teacher_yolo26s.ipynb`

Keduanya standalone, dengan `RUN_FULL_TRAINING=True` di konfigurasi terpusat. Pembaruan terbaru
hanya mengubah koefisien Native pada notebook lama 04/05; notebook lama lainnya, notebook EDA,
dan dataset tidak diubah. SHA-256 notebook referensi diperbarui sesuai sumber saat ini.

Konfigurasi tetap: Poltrada V3, student `yolo26n.pt`, EPOCHS 100, IMGSZ 640, BATCH 16,
SEED 42, DEVICE 0, WORKERS 2, AdamW, LR 0.001, PATIENCE 100, AMP True,
WEIGHT_DECAY 0.0005, WARMUP_EPOCHS 3.0, CLOSE_MOSAIC 10 dan EVAL_SPLIT val.
Default augmentasi dan `nbs=64` mengikuti Ultralytics 8.4.155 seperti notebook sumber.
Akumulasi nominal 4 memberi effective batch nominal 64; akumulasi aktual warmup dicatat terpisah.
Loader validation training mengikuti engine sumber (maksimum batch 32); evaluasi final memakai 16.

Native KD memakai `DistillationModel` resmi dengan `dis=1.0`, tanpa mengganti rumus loss.
Atas permintaan terbaru pengguna pada 17 Oktober, perubahan koefisien dibatasi khusus Native KD:
notebook 04/05/10 diubah dari 6.0 ke 1.0. Konfigurasi loss CrossKD dan CSAKD tidak diubah.
Nama eksperimen Native diberi akhiran `_dis1`: `04_native_yolo26s_dis1`,
`05_native_yolo26n_dis1`, dan `10_native_yolo26n_teacher_yolo26s_dis1`. Nama W&B notebook 10
adalah `10_native_yolo26n_teacher_s_dis1`; notebook 04/05 menambahkan seed/timestamp seperti sebelumnya.
Nama file notebook tetap. Perubahan identitas ini berlaku untuk run baru, tidak mengganti nama run W&B historis.
Perubahan koefisien ini berlaku untuk eksekusi baru, bukan untuk bobot atau hasil run terdahulu.
CrossKD mempertahankan fungsi dan class objective dari notebook 07 secara identik; kesamaan AST diuji.
Tidak ditambahkan CSAKD, AT, PKD atau loss baru.

## Konsistensi dengan eksperimen teacher YOLO26m sebelumnya

Notebook 10 dibandingkan dengan `05_native_yolo26n.ipynb`, dan notebook 11 dengan
`07_crosskd_yolo26n.ipynb`. Pengujian membandingkan bukan hanya konstanta yang ditulis di config,
tetapi seluruh konfigurasi efektif dari `ultralytics.cfg.get_cfg` pada versi yang sama.
Perbedaan argumen training dibatasi pada `distill_model`, `project`, dan `name`, yaitu sumber teacher
serta direktori/identitas eksperimen. Parameter training dan default augmentasi lainnya identik.

- Native: `dis=1.0`, fungsi loss resmi dan optimizer step resmi tetap sama.
- CrossKD: bobot classification/box masing-masing 1.0, gamma 1.0, serta fungsi loss dan
  mekanisme projector/alignment/cross-head identik dengan notebook 07.
- Dataset, pretrained student, 100 epoch, 640, batch 16, seed 42, workers, AdamW, LR,
  patience, AMP, weight decay, warmup, close mosaic, `nbs`, dan default engine dipertahankan.
- Teacher baru adalah YOLO26s baseline v62, dengan run/group/provenance terpisah dari teacher m.

Pemeriksaan ini membuktikan konsistensi implementasi/konfigurasi notebook referensi, bukan audit
konfigurasi efektif pada run historis W&B yang belum diakses. Pengaman baru tetap menghentikan
OOM/perubahan AMP atau batch agar konfigurasi penelitian tidak bergeser tanpa keputusan pengguna.

## Perubahan terkait error AMP

Traceback pengguna terjadi di pemeriksaan gradien smoke setelah `scaler.scale(loss).backward()`,
sebelum optimizer step standar. Pemeriksaan itu langsung menggagalkan proses pada scaled gradient
non-finite, sehingga tidak memberi GradScaler kesempatan melakukan skip/backoff scale.
Traceback tersebut saja belum membuktikan apakah penyebab numeriknya hanya initial loss-scale
ataupun masalah lain pada batch/model.

Pada versi langsung ini:

- Tidak ada `run_gpu_smoke`, `SMOKE_RESULT`, fingerprint smoke atau file `smoke_report.json` yang diwajibkan.
- `optimizer_step` tetap fungsi engine Ultralytics yang melakukan unscale, clipping, scaler step dan update.
  Tidak ditambahkan unscale kedua, tidak diubah initial scale secara manual, dan AMP tetap aktif.
- Overflow yang terdeteksi GradScaler ditangani melalui mekanisme skip/update standar. Loss non-finite,
  teacher gradien aktif, perubahan batch/resolusi/AMP efektif dan OOM tetap menghentikan run.
- `amp/grad_scale` ditambahkan pada metrik per epoch untuk membantu memantau skala AMP aktual.
- Batch, resolusi, LR, koefisien KD, augmentasi dan effective batch policy tidak diturunkan otomatis.

Melewati smoke bukan bukti bahwa semua batch GPU akan stabil. Error dari training sebenarnya tetap
harus ditinjau berdasarkan traceback dan metrik, tanpa menyatakan keberhasilan sebelum diukur.

## Teacher dan provenance

Satu-satunya sumber teacher:

```text
ilhamrafiqin-brawijaya-university/yolom26-kd-indotraffic/02_baseline_yolo26s-q5kp9k9d-checkpoints:v62
```

`run.use_artifact` tetap digunakan. Notebook memeriksa versi/koleksi, producer q5kp9k9d,
entity/project, manifest best.pt, ukuran/checksum, SHA-256, schema tensor YOLO26s,
jumlah/urutan label dan deklarasi dataset/model. Teacher tetap frozen/eval dan dikeluarkan dari optimizer.
Tidak ada fallback teacher pretrained.

Versi v62 tidak dianggap sebagai epoch 62. Artifact epoch dan best epoch dibaca dari metadata.
Pemeriksaan best global tetap membandingkan riwayat checkpoint lengkap pada source run selesai,
fitness maksimum, digest best final dan summary best epoch. Perbedaan atau bukti tidak lengkap
memblokir training; notebook tidak diam-diam mengganti teacher.

Pemeriksaan artifact langsung dari lingkungan lokal pada 16 Oktober gagal AuthenticationError.
Tidak ada SHA checkpoint nyata, best epoch nyata atau best-global status yang dikonfirmasi di sini.
Traceback GPU pengguna menunjukkan proses sudah mencapai tahap smoke, tetapi tanpa log provenance
lengkap, laporan ini tidak menyimpulkan detail checkpoint ataupun hasil GPU yang belum diberikan.

## Verifikasi lokal

Pengujian menggunakan Python 3.11, PyTorch 2.14.0 CPU dan Ultralytics 8.4.155.
Hasil akhir versi ini: **62 PASS**, 4 peringatan deprecation dependensi THOP, dan lint lulus.

```powershell
.\.venv\Scripts\python.exe -m pytest tests/test_teacher_s_notebooks.py -q
.\.venv\Scripts\ruff.exe check tests/test_teacher_s_notebooks.py
```

Suite mencakup validasi nbformat/sintaks, default training langsung, tidak adanya cell smoke,
provenance/hash/source/epoch dan best-global checks dengan mock, schema teacher s aktual,
KD-only backward model s/n pada CPU 64px, teacher frozen, parameter optimizer,
checkpoint student-only dan reload pada interpreter baru, serta snapshot/upload W&B sinkron.
Kedua metode juga diuji satu epoch kecil dengan bobot acak, data sintetis CPU dan W&B mock.
Ini tidak memakai Poltrada atau artifact teacher penelitian dan tidak menjalankan 100 epoch.

Regresi AMP memakai GradScaler CPU sungguhan: loss finite yang overflow setelah scaling harus
melewati update bobot dan menurunkan scale; backward finite berikutnya dapat memperbarui bobot.
Test memastikan `ResearchTrainer.optimizer_step` masih fungsi engine resmi. Ini memeriksa perilaku
scaler/integrasi, bukan membuktikan kompatibilitas CUDA FP16 pada GPU Kaggle.

Tidak ada hasil training GPU lokal, angka VRAM/latency GPU atau durasi 100 epoch yang dapat diklaim.
Konfigurasi notebook siap dieksekusi langsung; keberhasilan empiris batch 16/640/AMP belum diverifikasi di sini.

## Cara menjalankan di Kaggle

1. Upload salah satu notebook terbaru; jangan menimpa notebook lama yang sedang berjalan.
2. Aktifkan satu GPU dan Internet, pasang dataset Poltrada V3 pada path konfigurasi, serta aktifkan
   Secret WANDB_API_KEY yang memiliki akses artifact teacher dan project W&B.
3. Restart kernel lalu Run All dari awal. `RUN_FULL_TRAINING=True` sudah aktif; tidak perlu
   menjalankan smoke, membuat SMOKE_RESULT atau mengganti flag lagi.
4. Setelah teacher tervalidasi, training langsung berjalan dengan target 100 epoch. Jangan menjalankan
   Native dan CrossKD bersamaan pada GPU yang sama. OOM tidak ditangani dengan penurunan batch otomatis.
5. Validation memilih best checkpoint. Evaluasi akhir dan upload hasil dijalankan otomatis setelah training.
   Test split tidak digunakan untuk tuning.

Run W&B sekarang `10_native_yolo26n_teacher_s_dis1` dan `11_crosskd_yolo26n_teacher_s`, dalam group
`poltrada-v3-teacher-yolo26s-student-yolo26n-seed42`, terpisah dari pasangan teacher m.
Metrik detection/KD/validation, physical/effective batch, AMP scale, peak memory,
best epoch dan durasi training dicatat. Setiap epoch mengekspor best/last student-only ke snapshot
unik lalu menunggu `log_artifact(...).wait()` sebelum lanjut. Hasil final berisi mAP50–95,
mAP50, precision, recall, parameter/GFLOPs student, latency validation dan provenance teacher.

Referensi: notebook lokal 02/05/07,
[sumber Ultralytics 8.4.155](https://github.com/ultralytics/ultralytics/tree/v8.4.155),
[CrossKD penulis](https://github.com/jbwang1997/CrossKD), dan sumber GradScaler PyTorch terpasang.

## Perbaikan progres dan identitas run Native 10

Pengguna mengonfirmasi keterlambatan log terjadi pada notebook 10 (teacher YOLO26s).
Pembaruan ini hanya mengubah notebook 10; hash 11 notebook lainnya tetap sama.

- Nama tampilan W&B bukan identitas run. Notebook 10 kini memberi `id=wandb.util.generate_id()`
  dan `resume="never"` secara eksplisit, sehingga eksekusi baru tidak mencoba melanjutkan ID run
  yang pernah dihapus. ID dan URL run baru dicetak; batas init W&B 90 detik.
- Status `[ENV]`, `[DATA]`, `[W&B]`, `[TEACHER]`, `[AUDIT]`, dan `[TRAIN]` memakai output flush.
  Instalasi paket, login, download, checksum, pemuatan model, audit dan setup trainer kini terlihat.
- Audit tetap membaca metadata seluruh epoch untuk memeriksa kelengkapan/fitness maksimum,
  tetapi manifest intermediate tidak diambil. Hanya manifest teacher terpilih dan checkpoint
  terakhir diperlukan untuk membandingkan digest best final. Status audit muncul setiap 10 item.
  Ini mengurangi permintaan manifest; belum ada pengukuran percepatan jaringan Kaggle nyata.
- Metrik epoch dikirim setelah validation/checkpoint training ditulis, sebelum ekspor/upload
  artifact. Callback setelah upload tidak menggandakan log epoch. Upload artifact tetap sinkron
  dan selesai sebelum training melanjutkan epoch berikutnya.
- Target 100 epoch, batch 16, 640, AMP, optimizer dan objective Native `dis=1.0` tidak diubah.

Pengujian tambahan memeriksa ID baru/resume never, output flush, tidak adanya akses manifest
intermediate, serta urutan metrics → export → upload/wait tanpa duplikasi. Tidak ada training GPU
atau perubahan run W&B historis yang dilakukan oleh patch lokal.

Untuk memakai patch, upload ulang notebook 10, restart kernel lalu Run All sekali. Buka URL run
baru yang dicetak. Bila masih menunggu, label status terakhir membedakan login/init, download,
audit, setup GPU, dan upload checkpoint. Menghapus run lama tidak diperlukan untuk membuat ID baru.
