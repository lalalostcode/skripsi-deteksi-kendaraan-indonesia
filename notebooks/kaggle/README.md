# YOLO26 — Panduan Notebook Kaggle

## Navigasi kategori

| Notebook | Kategori |
|---|---|
| [00_eda_poltrada_v3.ipynb](eda/00_eda_poltrada_v3.ipynb) | `eda` |
| [00_results_eda-poltrada.ipynb](eda/00_results_eda-poltrada.ipynb) | `eda` |
| [01_teacher_yolo26m.ipynb](teacher/01_teacher_yolo26m.ipynb) | `teacher` |
| [02_teacher_yolo26m_multiseed.ipynb](teacher/02_teacher_yolo26m_multiseed.ipynb) | `teacher` |
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
| [15_mgd_yolo26n_teacher_yolo26s.ipynb](kd/teacher_s/15_mgd_yolo26n_teacher_yolo26s.ipynb) | `kd/teacher_s` |
| [01_lamp_yolo26n_no_recovery.ipynb](pruning/01_lamp_yolo26n_no_recovery.ipynb) | `pruning` |
| [02_lamp_yolo26s_no_recovery.ipynb](pruning/02_lamp_yolo26s_no_recovery.ipynb) | `pruning` |
| [03_lamp_yolo26n_extended_no_recovery.ipynb](pruning/03_lamp_yolo26n_extended_no_recovery.ipynb) | `pruning` V2 YOLO26n/S, conservative/extended dan EDA channel |
| [04_lamp_yolo26ns_finetune_recovery.ipynb](pruning/04_lamp_yolo26ns_finetune_recovery.ipynb) | `pruning` N/S dan recovery fine-tuning biasa |
| [05_lamp_yolo26n_gflops_no_recovery.ipynb](pruning/05_lamp_yolo26n_gflops_no_recovery.ipynb) | `pruning` khusus YOLO26n, target GFLOPs dan kontrol P5 |
| [06_lamp_yolo26s_gflops_no_recovery.ipynb](pruning/06_lamp_yolo26s_gflops_no_recovery.ipynb) | `pruning` khusus YOLO26s, target GFLOPs dan kontrol P5 |
| [06_lamp_yolo26s_multiseed_wandb.ipynb](pruning/06_lamp_yolo26s_multiseed_wandb.ipynb) | `pruning` YOLO26s dengan baseline sesuai seed, target 1,20× lima seed, serta 1,10× dan 1,70× seed 42. Tujuh run dan artefak W&B per suite |
| [01_yolo26n_standard_finetuning.ipynb](recovery/01_yolo26n_standard_finetuning.ipynb) | `recovery` fine-tuning YOLO26n |
| [02_yolo26s_standard_finetuning.ipynb](recovery/02_yolo26s_standard_finetuning.ipynb) | `recovery` satu raw YOLO26s pilihan, seed diwarisi, EDA tiga kondisi dan W&B |
| [03_yolo26ns_native_kd.ipynb](recovery/03_yolo26ns_native_kd.ipynb) | `recovery` Native KD dari raw pruned YOLO26s |
| [04_yolo26ns_crosskd.ipynb](recovery/04_yolo26ns_crosskd.ipynb) | `recovery` CrossKD dari raw pruned YOLO26s |
| [05_yolo26ns_mgd_kd.ipynb](recovery/05_yolo26ns_mgd_kd.ipynb) | `recovery` MGD dari raw pruned YOLO26s |

Baseline AMD/Roboflow berada di [platform AMD](../amd/baseline/03_baseline_yolo26n_amd_roboflow.ipynb). Sumber checkpoint teacher S tetap [baseline YOLO26s](baseline/02_baseline_yolo26s.ipynb).

Notebook pruning 03 mempertahankan nama file lama dan kini memproses N/S pada rasio 0,10/0,20/0,30/0,40/0,50
dengan policy extended secara default: sepuluh eksperimen independen tanpa recovery. Diagram channel
tiga panel, heatmap, keputusan LAMP, dan metrik terukur disimpan per model/rasio.
[Panduan konfigurasi dan output N/S](../../docs/lamp_yolo26n_extended_no_recovery.md).

Notebook pruning 04 menambahkan recovery supervised biasa setelah pruning pada sepuluh
konfigurasi N/S 10-50%. Default recovery 5 epoch dengan AdamW lr0.0001, optimizer baru,
FP32, dan tanpa KD/sparse training. Struktur model pruned dipertahankan, sementara bobot
termasuk head dapat belajar. Bandingkan baseline, pruned, dan recovered; training PASS
tidak menjamin akurasi pulih. [Panduan recovery](../../docs/lamp_yolo26ns_finetune_recovery.md).


Notebook Kaggle dikelompokkan per metode. Semuanya standalone; jumlah section mengikuti workflow masing-masing. [Indeks semua platform dan pemetaan path lama](../README.md).

Pruning 05 khusus YOLO26n dan 06 khusus YOLO26s. Keduanya memakai `MODE="full_val"`, FP32 640 px, batch 16, dan sembilan target GFLOPs 1,02×, 1,05×, 1,10×, 1,20×, 1,30×, 1,40×, 1,50×, 1,60×, dan 1,70×. Pilih `MODE="smoke"` untuk pemeriksaan pada subset tetap maksimal 128 citra. Kontrol extended dan P5 opsional. Sel EDA mencakup channel, ledger, skor, efisiensi, AP per kelas, kurva, confusion matrix, dan grafik trade-off. [Panduan feasibility dan checkpoint bersama](../../docs/lamp_gflops_recovery_workflow.md#2-jalankan-pruning-dan-pilih-raw-pruned)

Untuk workflow ini, aktifkan GPU dan Internet, tambahkan **Poltrada V3 YOLO26 Clean**, lalu periksa `DATASET_ROOT`, `DATA_YAML`, dan secret `WANDB_API_KEY`. Untuk YOLO26n, recovery 01 menerima `RAW_ARTIFACT_REFS` dan suite yang mendukung resume. [Panduan pruning, recovery, dan perbandingan baseline–raw–recovered](../../docs/lamp_gflops_recovery_workflow.md)

Pada recovery 02–05, aktifkan satu `STUDENT_INPUTS` melalui comment/uncomment, pilih perangkat, dan Run All. Identitas serta bobot diwarisi dari raw. Fine-tuning 02 menyediakan tujuh perlakuan. Native KD 03, CrossKD 04, dan MGD 05 menyediakan lima seed target 1,20× dengan teacher bersama YOLO26m `v50`; totalnya 15 perlakuan KD. Tiap sesi memiliki output dan ID W&B tersendiri. [Panduan](../../docs/lamp_gflops_recovery_workflow.md#3-pilih-notebook-recovery-per-metode) · [Fine-tuning 02](recovery/02_yolo26s_standard_finetuning.ipynb) · [Salinan historis](baseline/fine-tunned/01_yolo26s_pruned_fine_tuned.ipynb)

## Cara menjalankan di Kaggle

Recovery 01–05 menyediakan EDA custom **baseline asal → raw pruned → hasil metode recovery**,
dengan satu diagram batang per metrik, tabel per kelas, delta/gain, serta seluruh diagnostik.
Buka `custom_eda_report_url` pada Summary run W&B atau [indeks EDA](https://wandb.ai/ilhamrafiqin-brawijaya-university/yolom26-kd-indotraffic/reports/EDA--VmlldzoxODI3MDUxNg==).
Gambar dan Tables dicatat pada run; paket `<nama-run>-eda:vN` diverifikasi melalui SHA-256 unduhan.
Nilai yang belum diukur tetap `NOT_RUN`. [Panduan lengkap](../../docs/lamp_gflops_recovery_workflow.md#5-baca-perbandingan-tiga-tahap).

1. Di Kaggle aktifkan **GPU** dan **Internet**. Tambahkan `WANDB_API_KEY` pada **Add-ons > Secrets** dan berikan akses ke notebook. Jangan tulis API key di sel Python.
2. Tambahkan dataset pada `DATA="/kaggle/input/datasets/ilhamrafiqin/poltrada-v3-yolo26-clean/data.yaml"`. Default: `EPOCHS=100`, `IMGSZ=640`, `BATCH=16`, `SEED=42`, AdamW, `LR=0.001`, AMP aktif, dan `EVAL_SPLIT="val"`.
3. Project W&B ialah `yolom26-kd-indotraffic`. Notebook KD mengunduh teacher secara otomatis; akses artifact diperlukan.
4. Notebook 04–09 memakai teacher m `01_teacher_yolo26m-fux5wdcb-checkpoints:v50`. Notebook 10–12 memakai baseline s `02_baseline_yolo26s-q5kp9k9d-checkpoints:v62` dari entity `ilhamrafiqin-brawijaya-university` dan project yang sama. Nomor versi artifact bukan nomor epoch.
5. Upload notebook terbaru, restart kernel, lalu **Run All**. Notebook 12 langsung menjalankan target 100 epoch setelah verifikasi teacher. Jalankan satu eksperimen pada satu GPU pada satu waktu.
6. Untuk perbandingan fair, pertahankan protokol yang sama dan catat konfigurasi efektif. Test tidak dipakai untuk tuning atau pemilihan checkpoint. Pengujian CPU sintetis tidak membuktikan kapasitas atau stabilitas GPU Kaggle pada batch 16/640.

**Catatan:** Teacher untuk KD harus dilatih pada **dataset dan urutan label kelas** yang sama. Poltrada V3 pada konfigurasi ini memiliki enam kelas. CSAKD 12 menyalin objective, enam tap, activation checkpointing, chunk size 2, dan bobot AGKA/Cross 1/1 dari notebook 09; yang diganti adalah teacher dan identitas eksperimen.

## MGD YOLO26s → YOLO26n

[Notebook MGD 15](kd/teacher_s/15_mgd_yolo26n_teacher_yolo26s.ipynb) menambahkan Masked Generative Distillation dari referensi lokal `references/mgd`. Teacher tetap baseline YOLO26s artifact `q5kp9k9d:v62`; student dimulai dari `yolo26n.pt` baru untuk setiap seed. Konfigurasi pembanding mengikuti Native/CrossKD: Poltrada V3 yang sama, 100 epoch, 640px, batch16, AdamW lr0.001, weight decay0.0005, warmup3, close mosaic10, AMP, workers2, dan validation untuk pemilihan checkpoint. Bobot serta masking khusus MGD dicatat terpisah dari protokol bersama.

Default antrean `SEEDS_TO_RUN=[42]`; seed 123, 2024, 7, dan 3407 tersedia sebagai komentar untuk perbandingan berpasangan lima seed. Run All langsung menjalankan training penelitian sesuai daftar aktif, dengan direktori/ID W&B baru per seed dan group `T26s_S26n`. Student tidak meneruskan bobot seed sebelumnya. Notebook MGD saat ini khusus Kaggle; koleksi Colab tetap 17 workflow yang sudah ada.

## Baseline nano tanpa pretrained (scratch)

Gunakan **`03_baseline_yolo26n_scratch.ipynb`** di Kaggle untuk membandingkan inisialisasi bobot acak dengan baseline pretrained 03. Notebook lama tidak diubah. Arsitektur dibangun dari `MODEL_WEIGHTS="yolo26n.yaml"` dan training memakai `PRETRAINED=False`; tidak memuat checkpoint awal atau melanjutkan bobot seed sebelumnya.

Default `SEEDS_TO_RUN=[42]`. Seed 123, 2024, 7 dan 3407 tersedia sebagai komentar **belum dijalankan untuk scratch**. Hasil pretrained pada seed tersebut tidak dihitung sebagai hasil scratch. Setiap seed yang diaktifkan mendapat model acak baru, ID W&B baru dengan `resume="never"`, output dan state logging terpisah, lalu menyelesaikan evaluasi/upload/finish sebelum seed berikutnya. Kegagalan menghentikan antrean; hasil seed yang selesai tetap tersimpan.

Dataset Kaggle, versi paket, loss deteksi, semua parameter optimasi/augmentasi/evaluasi dan anggaran training tetap mengikuti baseline: 100 epoch, 640px, batch16, AdamW lr0.001, AMP True, device0, workers2. Tidak ada penyesuaian LR atau penambahan epoch khusus scratch. Validation tetap memilih `best.pt` dan menjadi split evaluasi akhir.

Run bernama `base_yolo26n_scratch_seed42`, project `yolom26-kd-indotraffic`, group `supervised`, dengan tag `from-scratch`. Output terpisah di `/kaggle/working/results/03_baseline_yolo26n_scratch/multiseed_<UTC>/seed-42/`. Konfigurasi, hasil dan metadata artifact mencatat `pretrained=False` serta `initialization="random"`.

Bandingkan hasil scratch **seed 42 dengan pretrained seed 42**, memakai split dan protokol yang sama, misalnya `final/map50_95`, precision, recall, dan waktu training. Jika seed lain ditambahkan, bandingkan seed yang sama dan hitung mean ± standar deviasi secara terpisah untuk tiap kondisi; jangan gabungkan hasil dua kondisi inisialisasi. Notebook tidak mengambil atau mengagregasi hasil pretrained secara otomatis.

Pemeriksaan AMP bawaan Ultralytics dipertahankan dan dapat mengunduh model diagnostik pretrained. Bobot diagnostik tersebut tidak mengisi bobot student. Evaluasi memuat `best.pt` hasil training scratch ini. Verifikasi lokal memakai mock serta satu training CPU sintetis singkat; training Poltrada 100 epoch dan W&B nyata belum dijalankan untuk varian ini.

## Baseline nano pada AMD cloud

Gunakan **`03_baseline_yolo26n_amd_roboflow.ipynb`** pada Linux cloud dengan image PyTorch ROCm dan torchvision yang sudah kompatibel. Isi `WANDB_API_KEY` dan `ROBOFLOW_API_KEY` langsung pada **Section 1**, periksa konfigurasi Section 2, lalu Run All. Tidak memakai Kaggle Secrets, environment variable untuk key, atau prompt getpass. Penyimpanan kode otomatis W&B dimatikan; key tidak dimasukkan ke metadata/artifact.

Seed default **7 dan 3407** berjalan serial; 42, 123 dan 2024 tersedia sebagai komentar. Protokol baseline tetap 100 epoch, 640px, batch16, AdamW lr0.001 dan AMP True. Runtime harus ROCm dengan `DEVICE=0`; ketidakcocokan GPU, perubahan batch/resolusi/AMP, atau kegagalan training/upload menghentikan antrean. Torch/torchvision image dikunci selama instalasi dependensi penelitian.

Setup AMD memakai **`ultralytics-opencv-headless==8.4.155`** dan **`opencv-python-headless==4.10.0.84`**, sesuai server tanpa display dan pin Roboflow 1.4.0. Installer membersihkan provider desktop/namespace yang bertumpuk lalu memeriksa import, GUI=NONE, PNG dan resize di subprocess. Ini menangani error seperti `ImportError: libxcb.so.1` dari OpenCV desktop tanpa memasang library GUI sistem. Jika paket yang sudah dimuat (termasuk NumPy) berubah, restart kernel dan Run All lagi; tidak perlu mengganti PyTorch ROCm.

Dataset diambil sekali dari `ilham-rafiqin/poltrada-original-raw-2`, versi 3, format `yolo26`, ke `~/skripsi-kd/datasets/poltrada-v3`. Output ada di `~/skripsi-kd/results/03_baseline_yolo26n_amd/`. Run bernama `base_yolo26n_amd_seed7` / `base_yolo26n_amd_seed3407`, tetap pada project `yolom26-kd-indotraffic`, group `supervised`, dengan tag `amd-rocm` dan `roboflow-v3`.

Cache lengkap dengan metadata Roboflow workspace/project/version yang cocok digunakan **tanpa request API**, mengikuti pola reuse notebook arsip. Jika `ProxyError` terjadi pada cloud, Section 4 menampilkan tahap request, kategori kegagalan dan `detail_redacted` dengan key serta kredensial proxy disamarkan. Notebook mempertahankan proxy/TLS cloud; tidak melakukan bypass otomatis. Jika akses outbound Roboflow tidak tersedia, ekspor **versi 3, format yolo26** dari project yang sama dapat diunggah dan diekstrak langsung ke `DATASET_DIR`, dengan `data.yaml` di root direktori tersebut. Format cache tidak dibuktikan oleh YAML dan dicatat terpisah; kelas dan identitas sumber tetap divalidasi. W&B tetap memerlukan koneksi online.

Identitas Roboflow, hash YAML, kelas dan lingkungan dicatat. Kesetaraan dengan dataset Kaggle bersih belum diverifikasi; tidak ada agregasi hasil lama otomatis. Pengujian lokal memakai mock dan dua training sintetis pendek di CPU, bukan bukti keberhasilan pada AMD cloud atau dataset Poltrada nyata.

## Seed baseline pretrained YOLO26s dan YOLO26n

Kedua baseline menyediakan seed 42, 0, 2024, 7, dan 3407. Sebelum Run All, aktifkan satu baris `SEED` di notebook **02** (default 42) atau satu entri `SEEDS_TO_RUN` di notebook **03** (default 7).

Hasil historis YOLO26n pada seed 123 tetap tercatat dengan seed aslinya. Untuk membandingkan baseline dan KD secara berpasangan, cocokkan seed yang benar-benar dijalankan pada kedua metode.

Seed 3407 kembali tersedia untuk baseline dan teacher. Lihat [catatan penelitian](../../reports/log-penelitian.md#2026-11-05--observasi-baseline-seed-3407) untuk status hasil 3407 dan rencana perbandingan.

## Teacher YOLO26m lima seed

[Teacher 02](teacher/02_teacher_yolo26m_multiseed.ipynb) menjalankan `SEEDS_TO_RUN=[42, 0, 2024, 7, 3407]` secara serial. Setiap seed mulai dari `yolo26m.pt` dengan SHA-256 awal yang sama, memakai 100 epoch, 640 px, batch 16, AdamW LR 0,001, AMP, dan evaluasi `best.pt` pada validation Poltrada V3. Tambahkan dataset serta Kaggle Secret `WANDB_API_KEY`, lalu Run All. Pilih subset `SEEDS_TO_RUN` bila hanya ingin menjalankan seed tertentu.

Setiap seed memiliki ID W&B baru, checkpoint berversi per epoch, evaluasi dan direktori sendiri. `multiseed_summary.json` merekap mean dan simpangan baku sampel dari seed yang selesai. Jika antrean berhenti, periksa `multiseed_progress.json` dan jalankan seed yang masih diperlukan. `seed_result.json` mencatat `best_artifact.reference` dan `best_artifact.best_pt_sha256` untuk mengambil checkpoint W&B yang tepat. [Padanan Filkom](../filkom/teacher/02_teacher_yolo26m_multiseed.ipynb) · [Penamaan W&B](../../docs/wandb_naming.md)

## Multi-seed Native s → n dan CrossKD s → n

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

Seed yang sudah selesai tersedia sebagai komentar. Tiap seed memakai target **100 epoch**. Setiap run memuat student dari pretrained `yolo26n.pt`. Teacher s memakai artifact v62.

Satu seed menyelesaikan training, evaluasi, semua upload dan `finish()` sebelum seed berikutnya dimulai. Setiap seed mendapat ID W&B baru (`resume="never"`), nama/tag sesuai seed, penghitung epoch baru, callback terikat pada run tersebut dan direktori output terpisah.

Contoh nama run KD `kd_native_T26s_S26n_dis1_seed123` dan `kd_crosskd_T26s_S26n_seed123` berada di group `T26s_S26n`. Akhiran nama dan tag mengikuti seed aktif. Untuk satu run, isi daftar dengan satu seed, misalnya `[7]`.

Hasil berada di `<PROJECT>/<EXPERIMENT_ID>/multiseed_<UTC>/seed-<seed>/`, termasuk checkpoint, `training_result.json`, `metrics.json`, `config.json` dan `seed_result.json`. Direktori antrean menyimpan `multiseed_results.json` serta `multiseed_progress.json`.

Jika training, evaluasi, upload atau penutupan run gagal, antrean berhenti dan hasil seed yang selesai tetap tersimpan. Sebelum Run All ulang, isi `SEEDS_TO_RUN` hanya dengan seed yang belum selesai. Gabungkan hasil KD historis secara manual ketika melaporkan mean ± standar deviasi.

## Group dan tag W&B

Lihat [standar penamaan W&B](../../docs/wandb_naming.md) untuk semua eksperimen baru.

Semua eksperimen 01–12 memakai `WANDB_GROUP` dan `WANDB_TAGS` pada konfigurasi. Group mengikuti tabel di atas. Tag memuat metode/role (`teacher`, `baseline`, `native`, `crosskd`, `csakd`), `teacher-m`/`teacher-s` dan `student-s`/`student-n` untuk KD, `seed42`, `poltrada-v3`, serta ID eksperimen. Run supervised memakai `model-m`/`model-s`/`model-n`.

Tag bobot dihitung dari konfigurasi: Native `dis1`, CrossKD `cls1`/`box1`, CSAKD `agka1`/`cross1`. Tag seed mengikuti seed aktif, misalnya `seed42`, `seed123` atau `seed2024`. Jika seed/bobot diubah, tag ikut berubah. Run diagnosis memakai awalan group `diagnostic-` dan tag `diagnostic` pada notebook yang mendukung dataset diagnosis; run penelitian memakai tag `research`.

Contoh notebook 12: nama W&B `kd_csakd_T26s_S26n_seed42`, group `T26s_S26n`, tag `csakd`, `teacher-s`, `student-n`, `agka1`, `cross1`, `seed42`, `poltrada-v3`, ID eksperimen, `research`. Group/tag ini berlaku untuk eksekusi baru; run historis yang sudah dikelompokkan manual tidak diperbarui oleh notebook.

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
