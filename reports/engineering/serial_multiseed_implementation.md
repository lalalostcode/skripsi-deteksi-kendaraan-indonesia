# Multi-seed serial: baseline n, Native s → n, CrossKD s → n

Tanggal: 17 Oktober 2026, Asia/Jakarta.
Pembaruan daftar seed: 19 Oktober 2026, Asia/Jakarta.

## Perilaku

Notebook baseline 03 menggunakan `SEEDS_TO_RUN=[7, 3407]`; seed 42, 123 dan 2024
sudah selesai dan ditampilkan sebagai komentar. Native 10 dan CrossKD 11 menggunakan
`SEEDS_TO_RUN=[123, 2024, 7, 3407]`; hanya seed 42 yang sudah selesai dan dikomentari.
`SEED` diatur otomatis ke seed aktif; pengguna cukup mengubah daftar tersebut.
Seed selesai tidak ditambahkan atau dijalankan ulang secara otomatis. Untuk satu training,
isi daftar dengan satu seed yang belum selesai, misalnya `[7]`. Setiap seed memakai target
100 epoch; target akhir ketiga metode tetap lima seed: 42, 123, 2024, 7 dan 3407.
Varian AMD tetap ditunda dan tidak diubah pada pembaruan ini.

Ketiga notebook tetap standalone. Environment/login dan dataset disiapkan sebelum antrean.
`run_seed_queue()` memvalidasi daftar seed sebelum membuka run. `run_one_seed()` menyelesaikan
seluruh siklus training/evaluasi/upload/finish satu seed sebelum kembali ke antrean.
Daftar kosong, duplikat, bool, non-integer atau nilai di luar rentang uint32 ditolak.

Untuk setiap seed:

- RNG diset ulang dan student dimuat baru dari `yolo26n.pt`; bobot seed sebelumnya tidak dipakai.
- W&B mendapat ID baru dari `wandb.util.generate_id()`, `resume="never"`, nama dan tag seed aktif.
- Group tetap `supervised` untuk baseline dan `T26s_S26n` untuk Native/CrossKD.
- Callback baru mengikat identitas seed, run dan state. Callback usang ditolak sebelum mencatat log.
- Set epoch uploaded/logged, best epoch dan durasi upload direset untuk run baru.
- Artifact model/results memakai run ID dalam nama dan seed dalam metadata.
- Upload artifact memakai `.wait()`. `finish()` selesai sebelum seed berikutnya boleh dibuka.
- Model training dan validasi dilepas dengan `finally`, GC dan pembersihan cache CUDA.

Jika ada run W&B aktif, antrean menolak memulai atau menutup paksa run tersebut.
Error training, teacher, evaluasi, upload, atau finish menghentikan antrean. Run yang gagal
dicoba ditutup dengan exit code 1; kegagalan finish juga tidak membolehkan seed berikutnya.

## Konsistensi penelitian

Default dataset Poltrada V3, model pretrained n, 100 epoch, 640, batch 16, device 0,
workers 2, AdamW, LR 0.001, patience 100, AMP True, weight decay 0.0005, warmup 3,
close mosaic 10 dan validation tetap sama. Native dis1 serta CrossKD cls1/box1/gamma1 tidak diubah.
Definisi objective CrossKD dan guards/trainer dipertahankan; Native memakai objective dan
optimizer step/GradScaler Ultralytics yang sama.

Teacher s tetap berasal dari:

```text
ilhamrafiqin-brawijaya-university/yolom26-kd-indotraffic/02_baseline_yolo26s-q5kp9k9d-checkpoints:v62
```

Setiap run KD menautkan teacher dengan `use_artifact`, memverifikasi provenance dan best global,
kemudian mengisi `KD_TRAIN_ARGS["distill_model"]`. State teacher sebelumnya dibersihkan terlebih
dahulu; kegagalan verifikasi tidak boleh menggunakan teacher dari seed sebelumnya.

Perbandingan hash snapshot sebelum perubahan memastikan seluruh notebook selain 03/10/11
tidak berubah, termasuk EDA dan CSAKD. Native/CrossKD juga dibandingkan dengan konfigurasi
efektif notebook referensi pada seed yang sama.

## Output dan pelanjutan setelah kegagalan

Direktori: `<PROJECT>/<EXPERIMENT_ID>/multiseed_<UTC>/seed-<seed>/`.
Setiap seed menyimpan checkpoint/output evaluasi, `training_result.json`, `metrics.json`,
`config.json` dan `seed_result.json` dengan ID/URL run. Antrean menyimpan
`multiseed_results.json` dan `multiseed_progress.json`; status completed baru dicatat setelah
upload dan finish sukses. Hasil seed sebelumnya tetap tersimpan jika seed selanjutnya gagal.

Tidak ada auto-resume atau penarikan otomatis hasil historis. Setelah kegagalan/sesi terputus,
lihat hasil/progress dan isi `SEEDS_TO_RUN` hanya dengan seed yang belum selesai sebelum Run All
ulang. Hasil historis baseline (42, 123, 2024) dan KD (42) digabung manual saat pelaporan
mean ± standar deviasi. Rekap lokal ini hanya berisi hasil yang benar-benar selesai pada
antrean baru; notebook tidak mengambil atau mengagregasi hasil lama.

## Verifikasi implementasi awal (17 Oktober 2026)

- Validasi JSON/nbformat/Python seluruh notebook eksperimen dan lint pengujian.
- Mock dua seed memeriksa ID, tag, nama, state, callback, output, metrik dan artifact terpisah;
  finish/upload seed pertama harus selesai sebelum init seed kedua.
- Injeksi error training, upload model, evaluasi, upload hasil dan finish memastikan antrean
  berhenti dan hasil seed sebelumnya tetap tersimpan. Kegagalan teacher menghentikan training.
- Daftar seed default, satu seed dan tahap final diuji.
- Dua training sintetis CPU berurutan dijalankan untuk masing-masing baseline/Native/CrossKD
  (enam training, satu epoch, resolusi 64). Kedua seed memuat checkpoint awal sintetis yang sama;
  parameter awal sama, objek student baru, log epoch mulai dari 1, artifact ditunggu, model dilepas
  dan checkpoint akhir dapat dibuka sebagai YOLO26n student-only.

Pengujian memakai W&B mock, data sintetis dan bobot acak; tidak mengakses akun/artifact W&B nyata
atau menjalankan Poltrada 100 epoch di GPU Kaggle. Durasi sesi dan kapasitas VRAM penelitian
belum diukur oleh pengujian ini.

Hasil akhir: **201 tes lulus** dalam dua perintah; lint dan `git diff --check` lulus.
Tiga skip pada suite utama terdiri dari dua pengujian GPU opt-in dan satu kasus teacher yang
tidak berlaku untuk baseline. Peringatan yang muncul berasal dari deprecation dependensi THOP.

```powershell
.\.venv\Scripts\python.exe -m pytest tests/test_teacher_s_notebooks.py tests/test_kd_notebook_tracking.py tests/test_serial_multiseed_notebooks.py tests/test_csakd_notebooks.py tests/test_csakd_provenance.py -q
# 198 passed, 3 skipped, 6 deselected, 4 warnings

.\.venv\Scripts\python.exe -m pytest tests/test_serial_multiseed_notebooks.py -m slow -q
# 3 passed, 45 deselected, 4 warnings

.\.venv\Scripts\ruff.exe check tests/test_serial_multiseed_notebooks.py tests/test_teacher_s_notebooks.py tests/test_kd_notebook_tracking.py
# All checks passed
```

Audit konfigurasi efektif baseline terhadap snapshot sebelum perubahan (pada seed yang sama)
menemukan perbedaan hanya pada `project`/`name` untuk output. Audit Native/CrossKD terhadap
referensi teacher m tetap menemukan perbedaan hanya pada output dan sumber teacher.

## Verifikasi pembaruan seed (19 Oktober 2026)

- JSON/nbformat dan sintaks seluruh cell Python pada 03/10/11 valid. Audit AST terhadap
  snapshot sebelum pembaruan hanya menemukan perubahan daftar seed dan dua pesan antrean;
  rumus loss, konfigurasi training, teacher v62 dan mekanisme serial tetap sama.
- Hash 15 notebook lain tetap identik, termasuk AMD, PKD, HEAD dan RM.
- Mock default memverifikasi dua run baseline dan empat run per notebook KD, tanpa seed
  historis: student/state/callback, ID, nama/tag, direktori, metrik dan artifact terpisah.
  Upload/finish selesai sebelum run berikutnya. Tes injeksi kegagalan tetap lulus.
- Fixture training sintetis mengatur `[123, 2024]` secara eksplisit, terpisah dari default
  notebook. Dua training CPU satu epoch per metode berhasil (enam training total).

```powershell
.\.venv\Scripts\python.exe -m pytest tests/test_serial_multiseed_notebooks.py tests/test_teacher_s_notebooks.py tests/test_kd_notebook_tracking.py -q
# 122 passed, 1 skipped, 3 deselected, 4 warnings

.\.venv\Scripts\python.exe -m pytest tests/test_serial_multiseed_notebooks.py -m slow -q
# 3 passed, 45 deselected, 4 warnings

.\.venv\Scripts\ruff.exe check tests/test_serial_multiseed_notebooks.py tests/test_teacher_s_notebooks.py
# All checks passed
```

Total 125 tes lulus; satu kasus teacher tidak berlaku untuk baseline dan dilewati.
Peringatan berasal dari deprecation THOP. Pemeriksaan diff lulus. Tidak ada training
Poltrada 100 epoch atau akses W&B nyata dalam pengujian pembaruan ini.
