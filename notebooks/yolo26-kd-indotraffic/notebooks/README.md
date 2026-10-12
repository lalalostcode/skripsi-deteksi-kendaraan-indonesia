# YOLO26 Knowledge Distillation — 9 clean standalone notebooks + W&B full per-epoch artifacts

Sembilan eksperimen, tiap file `.ipynb` fully standalone untuk Kaggle/Colab, masing-masing dengan 8 section. Tidak membutuhkan cloning GitHub, framework eksperimen, audit dataset, atau file Python proyek tambahan.

## Notebook

| File | Kondisi |
|---|---|
| `01_teacher_yolo26m.ipynb` | Teacher YOLO26m |
| `02_baseline_yolo26s.ipynb` | Baseline YOLO26s |
| `03_baseline_yolo26n.ipynb` | Baseline YOLO26n |
| `04_native_yolo26s.ipynb` | Native KD YOLO26s |
| `05_native_yolo26n.ipynb` | Native KD YOLO26n |
| `06_crosskd_yolo26s.ipynb` | CrossKD YOLO26s (adaptasi pilot) |
| `07_crosskd_yolo26n.ipynb` | CrossKD YOLO26n (adaptasi pilot) |
| `08_csakd_yolo26s.ipynb` | CSAKD YOLO26s (adaptasi pilot) |
| `09_csakd_yolo26n.ipynb` | CSAKD YOLO26n (adaptasi pilot) |

## Cara menjalankan demo

1. Di Kaggle aktifkan **GPU** dan **Internet**. Tambahkan `WANDB_API_KEY` pada **Add-ons > Secrets** dan berikan akses ke notebook. Jangan tulis API key di sel Python.
2. Konfigurasi default `DATA="coco8.yaml"`, `EPOCHS=10`, `IMGSZ=320`, `BATCH=2`, `WANDB_ENABLED=True`, `WANDB_PROJECT="skripsi-yolo26-kd-demo"`.
3. Jalankan notebook `01_teacher_yolo26m` dengan **Run All** dahulu. Simpan teacher `weights/best.pt` melalui Kaggle Save Version.
4. Isi `TEACHER_CHECKPOINT` dalam notebook Native/CrossKD/CSAKD (`04`–`09`) ke file teacher yang sama; baseline (`02`, `03`) tidak memerlukan teacher.
5. Jalankan masing-masing notebook sekali. Tiap notebook menghasilkan satu W&B run.
6. Saat dataset penelitian sudah ditentukan, ganti `DATA` ke `data.yaml` yang telah bersih serta tetapkan protokol training final dan `EVAL_SPLIT="test"` bila tersedia. Dataset COCO8 hanya untuk diagnosis dan tidak boleh dijadikan bukti peningkatan mAP.

**Catatan:** Teacher untuk KD harus dilatih pada **dataset dan urutan label kelas** yang sama. Teacher dari COCO8 80 kelas bukan teacher untuk dataset penelitian 5 kelas.

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
