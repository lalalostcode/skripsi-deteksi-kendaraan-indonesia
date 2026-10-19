# YOLO26 Knowledge Distillation — 12 standalone experiments + W&B per-epoch artifacts

Dua belas eksperimen, tiap file `.ipynb` standalone untuk Kaggle dengan 8 section. Tidak membutuhkan cloning GitHub atau file Python proyek tambahan. Notebook `00_eda_poltrada_v3.ipynb` terpisah untuk eksplorasi dataset.

## Notebook

| File | Kondisi | W&B group |
|---|---|---|
| `01_teacher_yolo26m.ipynb` | Teacher YOLO26m | `supervised` |
| `02_baseline_yolo26s.ipynb` | Baseline YOLO26s | `supervised` |
| `03_baseline_yolo26n.ipynb` | Baseline YOLO26n | `supervised` |
| `04_native_yolo26s.ipynb` | Native KD m → s, dis1 | `T26m_S26s` |
| `05_native_yolo26n.ipynb` | Native KD m → n, dis1 | `T26m_S26n` |
| `06_crosskd_yolo26s.ipynb` | CrossKD m → s | `T26m_S26s` |
| `07_crosskd_yolo26n.ipynb` | CrossKD m → n | `T26m_S26n` |
| `08_csakd_yolo26s.ipynb` | CSAKD m → s | `T26m_S26s` |
| `09_csakd_yolo26n.ipynb` | CSAKD m → n | `T26m_S26n` |
| `10_native_yolo26n_teacher_yolo26s.ipynb` | Native KD s → n, dis1 | `T26s_S26n` |
| `11_crosskd_yolo26n_teacher_yolo26s.ipynb` | CrossKD s → n | `T26s_S26n` |
| `12_csakd_yolo26n_teacher_yolo26s.ipynb` | CSAKD s → n | `T26s_S26n` |

## Cara menjalankan di Kaggle

1. Di Kaggle aktifkan **GPU** dan **Internet**. Tambahkan `WANDB_API_KEY` pada **Add-ons > Secrets** dan berikan akses ke notebook. Jangan tulis API key di sel Python.
2. Tambahkan dataset pada `DATA="/kaggle/input/datasets/ilhamrafiqin/poltrada-v3-yolo26-clean/data.yaml"`. Default: `EPOCHS=100`, `IMGSZ=640`, `BATCH=16`, `SEED=42`, AdamW, `LR=0.001`, AMP aktif, dan `EVAL_SPLIT="val"`.
3. Project W&B ialah `yolom26-kd-indotraffic`. Notebook KD mengunduh teacher secara otomatis; akses artifact diperlukan.
4. Notebook 04–09 memakai teacher m `01_teacher_yolo26m-fux5wdcb-checkpoints:v50`. Notebook 10–12 memakai baseline s `02_baseline_yolo26s-q5kp9k9d-checkpoints:v62` dari entity `ilhamrafiqin-brawijaya-university` dan project yang sama. Nomor versi artifact bukan nomor epoch.
5. Upload notebook terbaru, restart kernel, lalu **Run All**. Notebook 12 langsung menjalankan target 100 epoch setelah verifikasi teacher. Jalankan satu eksperimen pada satu GPU pada satu waktu.
6. Untuk perbandingan fair, pertahankan protokol yang sama dan catat konfigurasi efektif. Test tidak dipakai untuk tuning atau pemilihan checkpoint. Pengujian CPU sintetis tidak membuktikan kapasitas atau stabilitas GPU Kaggle pada batch 16/640.

**Catatan:** Teacher untuk KD harus dilatih pada **dataset dan urutan label kelas** yang sama. Poltrada V3 pada konfigurasi ini memiliki enam kelas. CSAKD 12 menyalin objective, enam tap, activation checkpointing, chunk size 2, dan bobot AGKA/Cross 1/1 dari notebook 09; yang diganti adalah teacher dan identitas eksperimen.

## Multi-seed baseline n, Native s → n, CrossKD s → n

Notebook **03, 10, dan 11** menjalankan seed secara serial dalam notebook masing-masing:

```python
SEEDS_TO_RUN = [123, 2024]
```

Seed 42 yang sudah selesai tidak ditambahkan otomatis. Tiap seed memakai target **100 epoch**, jadi default dua seed berarti dua training terpisah pada setiap notebook. Student selalu dimuat kembali dari pretrained `yolo26n.pt`; training seed berikutnya tidak melanjutkan bobot seed sebelumnya. Teacher s untuk KD tetap artifact v62.

Satu seed menyelesaikan training, evaluasi, semua upload dan `finish()` sebelum seed berikutnya dimulai. Setiap seed mendapat ID W&B baru (`resume="never"`), nama/tag sesuai seed, penghitung epoch baru, callback terikat pada run tersebut dan direktori output terpisah. Jangan menjalankan ketiga notebook bersamaan pada GPU yang sama.

Nama run: `baseline_yolo26n_seed123`, `native_T26s_S26n_dis1_seed123`, atau `crosskd_T26s_S26n_seed123` (dan seed 2024). Group tetap `supervised` / `T26s_S26n`. Setelah tiga seed awal lengkap, gunakan `[7, 3407]` untuk menuju lima seed total. Untuk satu run saja gunakan `[123]`.

Hasil berada di `<PROJECT>/<EXPERIMENT_ID>/multiseed_<UTC>/seed-<seed>/`, termasuk checkpoint, `training_result.json`, `metrics.json`, `config.json` dan `seed_result.json`. Direktori antrean menyimpan `multiseed_results.json` serta `multiseed_progress.json`.

Jika training, evaluasi, upload atau penutupan run gagal, antrean berhenti; hasil seed yang selesai tetap tersimpan. Sebelum Run All ulang, isi `SEEDS_TO_RUN` hanya dengan seed yang belum selesai. Tidak ada auto-resume atau auto-skip dari riwayat W&B. Hasil seed 42 yang sudah ada digabung ketika melaporkan mean ± standar deviasi, bukan dianggap sudah diimpor oleh notebook.

## Group dan tag W&B

Semua eksperimen 01–12 memakai `WANDB_GROUP` dan `WANDB_TAGS` pada konfigurasi. Group mengikuti tabel di atas. Tag memuat metode/role (`teacher`, `baseline`, `native`, `crosskd`, `csakd`), `teacher-m`/`teacher-s` dan `student-s`/`student-n` untuk KD, `seed42`, `poltrada-v3`, serta ID eksperimen. Run supervised memakai `model-m`/`model-s`/`model-n`.

Tag bobot dihitung dari konfigurasi: Native `dis1`, CrossKD `cls1`/`box1`, CSAKD `agka1`/`cross1`. Tag seed mengikuti seed aktif, misalnya `seed42`, `seed123` atau `seed2024`. Jika seed/bobot diubah, tag ikut berubah. Run diagnosis memakai awalan group `diagnostic-` dan tag `diagnostic` pada notebook yang mendukung dataset diagnosis; run penelitian memakai tag `research`.

Contoh notebook 12: nama W&B `csakd_T26s_S26n_seed42`, group `T26s_S26n`, tag `csakd`, `teacher-s`, `student-n`, `agka1`, `cross1`, `seed42`, `poltrada-v3`, ID eksperimen, `research`. Group/tag ini berlaku untuk eksekusi baru; run historis yang sudah dikelompokkan manual tidak diperbarui oleh notebook.

## Apa yang di-upload ke W&B?

- **Setiap epoch:** satu record metrik berupa `train/*` (termasuk `train/dis_loss` bila metode KD menyediakannya), `metrics/*` validasi, `lr/*`, dan `checkpoint/best_epoch_so_far`.
- **Setiap epoch:** satu **versi baru** dari Artifact tipe `model`, bernama `<experiment>-<wandb_run_id>-checkpoints`, memuat `last.pt` epoch tersebut, `best.pt` terbaik hingga saat itu, `results.csv` saat itu, dan `args.yaml`. Upload ditunggu sampai selesai sebelum training lanjut (`log_artifact(...).wait()`), agar checkpoint epoch N tidak tertukar dengan N+1.
- **Aliases:** `epoch-001`, `epoch-002`, ... adalah versi tetap per epoch. Alias `latest` mengarah ke versi terbaru dan `best` mengarah ke versi ketika fitness terbaik ditemukan. Metadata setiap versi berisi `epoch`, `fitness`, `best_fitness`, `is_best_epoch`, dan `best_epoch_so_far`.
- **Setelah selesai:** ringkasan evaluasi `final/*`, grafik PNG (training dan evaluasi) sebagai W&B Images, serta Artifact `results` yang berisi `metrics.json`, `config.json`, `results.csv`, `args.yaml`, dan grafik.
- **Lokal:** `weights/best.pt`, `weights/last.pt`, `results.csv`, `args.yaml`, `metrics.json`, `config.json` tetap disimpan oleh notebook.

### Mengambil epoch tertentu

Di W&B buka run > Artifacts > koleksi `<experiment>-<run_id>-checkpoints` > pilih alias **`epoch-003`** (misalnya) untuk mengunduh checkpoint tepat epoch 3. Pilih **`best`** untuk checkpoint milik epoch terbaik berdasarkan validasi. File `last.pt` di tiap versi adalah bobot akhir **epoch versi itu**, bukan bobot epoch final.

### Biaya dan keterbatasan

- **Upload setiap epoch menghabiskan bandwidth, waktu, dan storage W&B** (terutama YOLO26m). Ini memang disengaja atas permintaan dokumentasi seluruh epoch. Tiap versi juga berisi best-so-far.
- Kaggle harus online dan terautentikasi. Error W&B tidak dimatikan/diganti menjadi offline secara diam-diam; training akan gagal jika upload penting gagal.
- Kaggle GPU dan W&B online end-to-end **belum diuji di sini**; yang diuji adalah struktur notebook dan sintaks Python. Adaptasi CrossKD/CSAKD masih memerlukan smoke training GPU sebelum dianggap valid secara empiris.
- Integrasi W&B Ultralytics otomatis dinonaktifkan agar tidak membuat run ganda. Callback manual ringan digunakan khusus untuk memenuhi syarat **semua artifact setiap epoch**, bukan framework tracking berlapis.
