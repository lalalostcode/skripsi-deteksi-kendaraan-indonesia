# YOLO26 — Panduan Notebook Kaggle

## Navigasi kategori

| Notebook | Kategori |
|---|---|
| [00_eda_poltrada_v3.ipynb](eda/00_eda_poltrada_v3.ipynb) | `eda` |
| [00_results_eda-poltrada.ipynb](eda/00_results_eda-poltrada.ipynb) | `eda` |
| [01_teacher_yolo26m.ipynb](teacher/01_teacher_yolo26m.ipynb) | `teacher` |
| [02_baseline_yolo26s.ipynb](baseline/02_baseline_yolo26s.ipynb) | `baseline` |
| [03_baseline_yolo26n.ipynb](baseline/03_baseline_yolo26n.ipynb) | `baseline` |
| [03_baseline_yolo26n_scratch.ipynb](baseline/03_baseline_yolo26n_scratch.ipynb) | `baseline` |
| [04_native_yolo26s.ipynb](kd/teacher_m/04_native_yolo26s.ipynb) | `kd/teacher_m` |
| [05_native_yolo26n.ipynb](kd/teacher_m/05_native_yolo26n.ipynb) | `kd/teacher_m` |
| [06_crosskd_yolo26s.ipynb](kd/teacher_m/06_crosskd_yolo26s.ipynb) | `kd/teacher_m` |
| [07_crosskd_yolo26n.ipynb](kd/teacher_m/07_crosskd_yolo26n.ipynb) | `kd/teacher_m` |
| [08_csakd_yolo26s.ipynb](kd/teacher_m/08_csakd_yolo26s.ipynb) | `kd/teacher_m` |
| [09_csakd_yolo26n.ipynb](kd/teacher_m/09_csakd_yolo26n.ipynb) | `kd/teacher_m` |
| [10_native_yolo26n_teacher_yolo26s.ipynb](kd/teacher_s/10_native_yolo26n_teacher_yolo26s.ipynb) | `kd/teacher_s` |
| [11_crosskd_yolo26n_teacher_yolo26s.ipynb](kd/teacher_s/11_crosskd_yolo26n_teacher_yolo26s.ipynb) | `kd/teacher_s` |
| [12_csakd_yolo26n_teacher_yolo26s.ipynb](kd/teacher_s/12_csakd_yolo26n_teacher_yolo26s.ipynb) | `kd/teacher_s` |
| [12_pkd_yolo26n.ipynb](kd/teacher_s/12_pkd_yolo26n.ipynb) | `kd/teacher_s` |
| [13_head_yolo26n_teacher_yolo26s.ipynb](kd/teacher_s/13_head_yolo26n_teacher_yolo26s.ipynb) | `kd/teacher_s` |
| [14_rm_yolo26n_teacher_yolo26s.ipynb](kd/teacher_s/14_rm_yolo26n_teacher_yolo26s.ipynb) | `kd/teacher_s` |
| [01_lamp_yolo26n_no_recovery.ipynb](pruning/01_lamp_yolo26n_no_recovery.ipynb) | `pruning` |

Baseline AMD/Roboflow berada di [platform AMD](../amd/baseline/03_baseline_yolo26n_amd_roboflow.ipynb). Sumber checkpoint teacher S tetap [baseline YOLO26s](baseline/02_baseline_yolo26s.ipynb).


Notebook Kaggle dikelompokkan per metode. Semuanya standalone; jumlah section mengikuti workflow masing-masing. [Indeks semua platform dan pemetaan path lama](../README.md).

## Cara menjalankan di Kaggle

1. Di Kaggle aktifkan **GPU** dan **Internet**. Tambahkan `WANDB_API_KEY` pada **Add-ons > Secrets** dan berikan akses ke notebook. Jangan tulis API key di sel Python.
2. Tambahkan dataset pada `DATA="/kaggle/input/datasets/ilhamrafiqin/poltrada-v3-yolo26-clean/data.yaml"`. Default: `EPOCHS=100`, `IMGSZ=640`, `BATCH=16`, `SEED=42`, AdamW, `LR=0.001`, AMP aktif, dan `EVAL_SPLIT="val"`.
3. Project W&B ialah `yolom26-kd-indotraffic`. Notebook KD mengunduh teacher secara otomatis; akses artifact diperlukan.
4. Notebook 04–09 memakai teacher m `01_teacher_yolo26m-fux5wdcb-checkpoints:v50`. Notebook 10–12 memakai baseline s `02_baseline_yolo26s-q5kp9k9d-checkpoints:v62` dari entity `ilhamrafiqin-brawijaya-university` dan project yang sama. Nomor versi artifact bukan nomor epoch.
5. Upload notebook terbaru, restart kernel, lalu **Run All**. Notebook 12 langsung menjalankan target 100 epoch setelah verifikasi teacher. Jalankan satu eksperimen pada satu GPU pada satu waktu.
6. Untuk perbandingan fair, pertahankan protokol yang sama dan catat konfigurasi efektif. Test tidak dipakai untuk tuning atau pemilihan checkpoint. Pengujian CPU sintetis tidak membuktikan kapasitas atau stabilitas GPU Kaggle pada batch 16/640.

**Catatan:** Teacher untuk KD harus dilatih pada **dataset dan urutan label kelas** yang sama. Poltrada V3 pada konfigurasi ini memiliki enam kelas. CSAKD 12 menyalin objective, enam tap, activation checkpointing, chunk size 2, dan bobot AGKA/Cross 1/1 dari notebook 09; yang diganti adalah teacher dan identitas eksperimen.

## Baseline nano tanpa pretrained (scratch)

Gunakan **`03_baseline_yolo26n_scratch.ipynb`** di Kaggle untuk membandingkan inisialisasi bobot acak dengan baseline pretrained 03. Notebook lama tidak diubah. Arsitektur dibangun dari `MODEL_WEIGHTS="yolo26n.yaml"` dan training memakai `PRETRAINED=False`; tidak memuat checkpoint awal atau melanjutkan bobot seed sebelumnya.

Default `SEEDS_TO_RUN=[42]`. Seed 123, 2024, 7 dan 3407 tersedia sebagai komentar **belum dijalankan untuk scratch**. Hasil pretrained pada seed tersebut tidak dihitung sebagai hasil scratch. Setiap seed yang diaktifkan mendapat model acak baru, ID W&B baru dengan `resume="never"`, output dan state logging terpisah, lalu menyelesaikan evaluasi/upload/finish sebelum seed berikutnya. Kegagalan menghentikan antrean; hasil seed yang selesai tetap tersimpan.

Dataset Kaggle, versi paket, loss deteksi, semua parameter optimasi/augmentasi/evaluasi dan anggaran training tetap mengikuti baseline: 100 epoch, 640px, batch16, AdamW lr0.001, AMP True, device0, workers2. Tidak ada penyesuaian LR atau penambahan epoch khusus scratch. Validation tetap memilih `best.pt` dan menjadi split evaluasi akhir.

Run bernama `baseline_yolo26n_scratch_seed42`, project `yolom26-kd-indotraffic`, group `supervised`, dengan tag `from-scratch`. Output terpisah di `/kaggle/working/results/03_baseline_yolo26n_scratch/multiseed_<UTC>/seed-42/`. Konfigurasi, hasil dan metadata artifact mencatat `pretrained=False` serta `initialization="random"`.

Bandingkan hasil scratch **seed 42 dengan pretrained seed 42**, memakai split dan protokol yang sama, misalnya `final/map50_95`, precision, recall, dan waktu training. Jika seed lain ditambahkan, bandingkan seed yang sama dan hitung mean ± standar deviasi secara terpisah untuk tiap kondisi; jangan gabungkan hasil dua kondisi inisialisasi. Notebook tidak mengambil atau mengagregasi hasil pretrained secara otomatis.

Pemeriksaan AMP bawaan Ultralytics dipertahankan dan dapat mengunduh model diagnostik pretrained. Bobot diagnostik tersebut tidak mengisi bobot student. Evaluasi memuat `best.pt` hasil training scratch ini. Verifikasi lokal memakai mock serta satu training CPU sintetis singkat; training Poltrada 100 epoch dan W&B nyata belum dijalankan untuk varian ini.

## Baseline nano pada AMD cloud

Gunakan **`03_baseline_yolo26n_amd_roboflow.ipynb`** pada Linux cloud dengan image PyTorch ROCm dan torchvision yang sudah kompatibel. Isi `WANDB_API_KEY` dan `ROBOFLOW_API_KEY` langsung pada **Section 1**, periksa konfigurasi Section 2, lalu Run All. Tidak memakai Kaggle Secrets, environment variable untuk key, atau prompt getpass. Penyimpanan kode otomatis W&B dimatikan; key tidak dimasukkan ke metadata/artifact.

Seed default **7 dan 3407** berjalan serial; 42, 123 dan 2024 tersedia sebagai komentar. Protokol baseline tetap 100 epoch, 640px, batch16, AdamW lr0.001 dan AMP True. Runtime harus ROCm dengan `DEVICE=0`; ketidakcocokan GPU, perubahan batch/resolusi/AMP, atau kegagalan training/upload menghentikan antrean. Torch/torchvision image dikunci selama instalasi dependensi penelitian.

Setup AMD memakai **`ultralytics-opencv-headless==8.4.155`** dan **`opencv-python-headless==4.10.0.84`**, sesuai server tanpa display dan pin Roboflow 1.4.0. Installer membersihkan provider desktop/namespace yang bertumpuk lalu memeriksa import, GUI=NONE, PNG dan resize di subprocess. Ini menangani error seperti `ImportError: libxcb.so.1` dari OpenCV desktop tanpa memasang library GUI sistem. Jika paket yang sudah dimuat (termasuk NumPy) berubah, restart kernel dan Run All lagi; tidak perlu mengganti PyTorch ROCm.

Dataset diambil sekali dari `ilham-rafiqin/poltrada-original-raw-2`, versi 3, format `yolo26`, ke `~/skripsi-kd/datasets/poltrada-v3`. Output ada di `~/skripsi-kd/results/03_baseline_yolo26n_amd/`. Run bernama `baseline_yolo26n_amd_seed7` / `baseline_yolo26n_amd_seed3407`, tetap pada project `yolom26-kd-indotraffic`, group `supervised`, dengan tag `amd-rocm` dan `roboflow-v3`.

Cache lengkap dengan metadata Roboflow workspace/project/version yang cocok digunakan **tanpa request API**, mengikuti pola reuse notebook arsip. Jika `ProxyError` terjadi pada cloud, Section 4 menampilkan tahap request, kategori kegagalan dan `detail_redacted` dengan key serta kredensial proxy disamarkan. Notebook mempertahankan proxy/TLS cloud; tidak melakukan bypass otomatis. Jika akses outbound Roboflow tidak tersedia, ekspor **versi 3, format yolo26** dari project yang sama dapat diunggah dan diekstrak langsung ke `DATASET_DIR`, dengan `data.yaml` di root direktori tersebut. Format cache tidak dibuktikan oleh YAML dan dicatat terpisah; kelas dan identitas sumber tetap divalidasi. W&B tetap memerlukan koneksi online.

Identitas Roboflow, hash YAML, kelas dan lingkungan dicatat. Kesetaraan dengan dataset Kaggle bersih belum diverifikasi; tidak ada agregasi hasil lama otomatis. Pengujian lokal memakai mock dan dua training sintetis pendek di CPU, bukan bukti keberhasilan pada AMD cloud atau dataset Poltrada nyata.

## Multi-seed baseline n, Native s → n, CrossKD s → n

Notebook **03, 10, dan 11** menjalankan seed secara serial dalam notebook masing-masing.
Baseline **03** sudah selesai pada seed 42, 123, dan 2024; jalankan dua sisanya:

```python
SEEDS_TO_RUN = [
    # 42,    # Sudah selesai
    # 123,   # Sudah selesai
    # 2024,  # Sudah selesai
    7,
    3407,
]
```

Native **10** dan CrossKD **11** baru selesai pada seed 42; jalankan empat sisanya:

```python
SEEDS_TO_RUN = [
    # 42,    # Sudah selesai
    123,
    2024,
    7,
    3407,
]
```

Seed yang sudah selesai tersedia sebagai komentar dan tidak ditambahkan otomatis. Tiap seed memakai target **100 epoch**: dua training untuk baseline dan empat training pada masing-masing notebook KD. Student selalu dimuat kembali dari pretrained `yolo26n.pt`; training seed berikutnya tidak melanjutkan bobot seed sebelumnya. Teacher s untuk KD tetap artifact v62.

Satu seed menyelesaikan training, evaluasi, semua upload dan `finish()` sebelum seed berikutnya dimulai. Setiap seed mendapat ID W&B baru (`resume="never"`), nama/tag sesuai seed, penghitung epoch baru, callback terikat pada run tersebut dan direktori output terpisah. Jangan menjalankan ketiga notebook bersamaan pada GPU yang sama.

Nama run: `baseline_yolo26n_seed7`, `native_T26s_S26n_dis1_seed123`, atau `crosskd_T26s_S26n_seed123`; akhiran nama dan tag mengikuti seed aktif. Group tetap `supervised` / `T26s_S26n`. Target akhir kelima seed adalah 42, 123, 2024, 7 dan 3407 untuk ketiga metode. Untuk satu run saja isi daftar dengan satu seed yang belum selesai, misalnya `[7]`.

Hasil berada di `<PROJECT>/<EXPERIMENT_ID>/multiseed_<UTC>/seed-<seed>/`, termasuk checkpoint, `training_result.json`, `metrics.json`, `config.json` dan `seed_result.json`. Direktori antrean menyimpan `multiseed_results.json` serta `multiseed_progress.json`.

Jika training, evaluasi, upload atau penutupan run gagal, antrean berhenti; hasil seed yang selesai tetap tersimpan. Sebelum Run All ulang, isi `SEEDS_TO_RUN` hanya dengan seed yang belum selesai. Tidak ada auto-resume atau auto-skip dari riwayat W&B. Hasil historis baseline (42, 123, 2024) dan KD (42) digabung secara manual ketika melaporkan mean ± standar deviasi; notebook tidak mengambil atau mengagregasi hasil lama.

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
