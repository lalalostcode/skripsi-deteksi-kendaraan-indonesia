# Pruning dan recovery YOLO26n/S pada Poltrada V3

Filkom adalah platform utama untuk pengukuran penelitian. Kaggle menyediakan notebook dengan metode dan evaluator yang sama. Setiap notebook memuat kode runtime sendiri dan bisa dijalankan tanpa memasang paket dari repositori ini. Lihat [setup Filkom melalui SSH](../notebooks/filkom/README.md) atau [setup Kaggle](../notebooks/kaggle/README.md).

| Eksperimen | Kaggle | Filkom |
| --- | --- | --- |
| Pruning YOLO26n | [05](../notebooks/kaggle/pruning/05_lamp_yolo26n_gflops_no_recovery.ipynb) | [05](../notebooks/filkom/pruning/05_lamp_yolo26n_gflops_no_recovery.ipynb) |
| Pruning YOLO26s | [06](../notebooks/kaggle/pruning/06_lamp_yolo26s_gflops_no_recovery.ipynb) | [06](../notebooks/filkom/pruning/06_lamp_yolo26s_gflops_no_recovery.ipynb) |
| Pruning YOLO26s multi-seed | [06 multi-seed](../notebooks/kaggle/pruning/06_lamp_yolo26s_multiseed_wandb.ipynb) | Notebook 06 multi-seed yang sama |
| Fine-tuning YOLO26n | [01](../notebooks/kaggle/recovery/01_yolo26n_standard_finetuning.ipynb) | [01](../notebooks/filkom/recovery/01_yolo26n_standard_finetuning.ipynb) |
| Fine-tuning YOLO26s | [02](../notebooks/kaggle/recovery/02_yolo26s_standard_finetuning.ipynb) | [02](../notebooks/filkom/recovery/02_yolo26s_standard_finetuning.ipynb) |
| Native KD YOLO26s | [03](../notebooks/kaggle/recovery/03_yolo26ns_native_kd.ipynb) | [03](../notebooks/filkom/recovery/03_yolo26ns_native_kd.ipynb) |
| CrossKD YOLO26s | [04](../notebooks/kaggle/recovery/04_yolo26ns_crosskd.ipynb) | [04](../notebooks/filkom/recovery/04_yolo26ns_crosskd.ipynb) |
| MGD YOLO26s | [05](../notebooks/kaggle/recovery/05_yolo26ns_mgd_kd.ipynb) | [05](../notebooks/filkom/recovery/05_yolo26ns_mgd_kd.ipynb) |

## Fine-tuning YOLO26s dengan protokol baseline

Gunakan recovery 02 sebagai pembanding supervised untuk recovery 03–05. Salinan `fine-tunned/01` tetap tersedia untuk membaca uji historis. [Kaggle](../notebooks/kaggle/baseline/fine-tunned/01_yolo26s_pruned_fine_tuned.ipynb) · [Filkom](../notebooks/filkom/fine-tunned/01_yolo26s_pruned_fine_tuned.ipynb)

## 1. Identitas Poltrada V3 dan checkpoint baseline

Pakai **Poltrada V3 YOLO26 Clean**. `DATASET_ROOT` menunjuk root dataset dan `DATA_YAML` menunjuk deskripsi sumber yang memuat split `train`, `val`, serta nama kelas dalam urutan asli. Gunakan dataset dengan file gambar dan label sumber yang sama pada kedua platform. Simpan juga `test` asli bila tersedia.

1. Di Kaggle, tambahkan dataset resmi dan pilih `DATASET_MANIFEST_ONLY=True` di salah satu notebook pruning. Jalankan untuk menghasilkan `dataset_manifest.json` sebelum melakukan pruning. Periksa ID dataset, provenance `VERIFIED`, urutan kelas, jumlah citra, dan fingerprint train/val.
2. Salin seluruh dataset dan manifest itu ke Filkom. Pertahankan jalur relatif setiap file. Hindari mengekspor ulang citra atau mengubah label di tengah rangkaian eksperimen.
3. Isi `DATASET_MANIFEST_PATH` dengan path manifest sumber di Filkom. Notebook menghitung kembali SHA-256 isi citra dan label. Lokasi mount baru tidak mengubah identitas jika konten dan jalur relatif sama. Perbedaan fingerprint atau gambar yang berbagi konten antara train dan val perlu diperbaiki sebelum lanjut.
4. Tanpa bukti sumber yang cocok, hasil dari path pengguna tercatat `UNVERIFIED`. Pertahankan label ini ketika melaporkan hasil. Provenance `VERIFIED` diperlukan untuk memilih student.

Notebook pruning mengunci YOLO26n baseline `v84` dan YOLO26s `v62`. Aktifkan unduhan dari W&B atau isi `LOCAL_CHECKPOINT_PATHS` untuk menggunakan salinan baseline yang SHA-256-nya sama. Untuk recovery, sumber baseline pembanding ikut disediakan di paket raw.

## 2. Jalankan pruning dan pilih raw pruned

Setel `DATASET_MANIFEST_ONLY=False` saat siap memulai. Aktifkan `MODE="smoke"` untuk uji pipeline dengan maksimal 128 citra validasi tetap. Gunakan `MODE="full_val"` untuk mengukur seluruh split `val`. Kedua mode memakai folder suite berbeda dan hasil smoke terpisah dari seleksi.

- Notebook 05 dan 06 memakai sembilan target GFLOPs **1,02×, 1,05×, 1,10×, 1,20×, 1,30×, 1,40×, 1,50×, 1,60×, dan 1,70×**. Target 1,70× mengikuti contoh konfigurasi MP-YOLO. Setiap kandidat memulai dari baseline asli dan memakai proteksi yang sama. Gunakan folder suite baru untuk grid yang diperluas. [Kode penulis MP-YOLO](https://github.com/Wang-jj-zs/MP-YOLO/blob/master/MP-YOLO/compress.py)
- Kapasitas dihitung per model pada ketiga tier sebelum evaluasi kandidat target. Uji CPU FP32 pada 640 px mengukur maksimum 1,2238× pada N dan 1,2322× pada S. Target 1,30×–1,70× melampaui kapasitas yang terukur pada kedua model. [Bukti feasibility](../reports/pruning/lamp_gflops_05_nine_targets_verification_20261106.json)
- Baris `TARGET_NOT_REACHED` mencatat GFLOPs dan speed-up aktual kandidat maksimum. Di setiap suite 05 atau 06, baris yang berbagi plan maksimum merujuk ke `target-1.70x` melalui `artifact_configuration_id`, path dan SHA-256 checkpoint yang sama. Gunakan checkpoint ini satu kali per metode jika memilih menjalankan recovery diagnostik. Kontrol extended dan P5 aktif lewat `RUN_DIAGNOSTIC_CONTROLS=True`.
- Periksa `suite_manifest.json`, `metrics.csv`, `student_candidates.json`, `raw_pruned_manifest.json`, dan `artifact_publications.json`. Audit struktur, evaluasi, latency, EDA, dan publikasi dilaporkan terpisah. Hasil konfigurasi yang selesai tetap dapat dibuka saat suite berstatus `PARTIAL`.
- Jika perlu memilih satu student lintas N/S, isi `COMPARISON_CANDIDATE_PATHS` pada notebook terakhir dengan ekspor full validation varian lain. Cocokkan kontrak dataset, evaluator dan perangkat latency. Seleksi mensyaratkan target tercapai, metrik lengkap, dan kehilangan mAP50–95 maksimum dua poin persentase. Urutan seleksi adalah target tertinggi, mAP50–95 tertinggi, median latency terendah, lalu parameter paling sedikit. Simpan `student_lock.json` bila ada kandidat yang memenuhi syarat.

Pruning menghasilkan `pruned_fp32.pt`, manifest, evaluasi, dan artefak `raw-pruned`. Simpan referensi `entity/project/artifact:vN` dan SHA-256 checkpoint dari publikasi yang sudah dikonfirmasi. Gunakan raw yang sama untuk **fine-tuning, Native KD, CrossKD, dan MGD** pada setiap perbandingan. Gunakan test independen hanya setelah pilihan student dikunci. [Audit LAMP dan P5](../reports/pruning/lamp_gflops_audit.md)

### Pruning YOLO26s multi-seed

Gunakan [notebook 06 multi-seed](../notebooks/kaggle/pruning/06_lamp_yolo26s_multiseed_wandb.ipynb) dengan `MODE="full_val"` dan `RESUME_SUITE_DIR=None` untuk suite baru. Di Kaggle pilih Tesla T4, aktifkan internet, pasang dataset Poltrada V3 YOLO26 Clean, hubungkan Secret `WANDB_API_KEY`, lalu jalankan **Run All**. Antrean berjalan serial pada GPU pertama. Di Filkom, isi `DATASET_MANIFEST_PATH` dengan manifest dari mount Poltrada V3 resmi dan set `WANDB_API_KEY` pada environment Jupyter. `LOCAL_BASELINE_PATHS` menerima salinan `best.pt` per seed jika SHA-256 cocok dengan pin.

Mode `full_val` mengevaluasi setiap raw terhadap baseline seed yang sama pada seluruh validation. Target 1,20× memakai seed 42, 0, 2024, 7, dan 3407. Pembanding 1,10× dan 1,70× memakai seed 42. Ketujuh perlakuan memperoleh run dan Model Artifact sendiri dalam satu grup suite. Setelah validasi dan latency lulus, notebook mengunggah paket dan mengonfirmasi versi W&B dengan SHA-256 unduhan. [Pemetaan baseline dan koleksi](ARTEFAK_WANDB.md#student-raw-lamp-yolo26s-multi-seed)

Simpan folder suite beserta `checkpoint_registry.csv`, `checkpoint_registry.json`, audit, dan EDA. `RESUME_SUITE_DIR` menerima salinan suite per seed yang dapat ditulis. Versi dan SHA-256 kelima baseline, fingerprint dataset, protokol, dan runtime harus cocok dengan kontrak awal. Resume melanjutkan tahap yang tertunda dan mempertahankan ID run serta versi W&B yang sudah terkonfirmasi.

Recovery 02–05 memilih satu raw melalui `STUDENT_INPUTS`. Fine-tuning 02 menyediakan tujuh perlakuan, sedangkan KD 03–05 menyediakan lima seed target 1,20×. Loader registry di bagian 14 pruning tetap tersedia untuk audit serta workflow recovery historis. Contoh berikut mendokumentasikan antarmuka registry historis: target 1,20× dan seed 7. `allow_diagnostic=True` diperlukan untuk kandidat `TARGET_NOT_REACHED`.

```python
RAW_ARTIFACT_REF, EXPECTED_RAW_SHA256, SEED, REGISTRY_BASELINE_REF, REGISTRY_BASELINE_SHA256 = (
    recovery_config_from_registry("checkpoint_registry.json", seed=7, include_baseline=True)
)
RAW_MANIFEST_PATH = None
```

Jalankan sel berikut sesudah bootstrap recovery dan sebelum run agar lock baseline memakai baris registry yang sama.

```python
BASELINE_REF, BASELINE_SHA256 = REGISTRY_BASELINE_REF, REGISTRY_BASELINE_SHA256
```

Tiga versi seed 42 `:v0` dan pemetaan seed dari desain lama tersedia di [registry historis terverifikasi](../reports/pruning/lamp_yolo26s_wandb_v0_registry.json).

Bangkitkan ulang notebook ini dengan `python scripts/generate_lamp_multiseed_notebook.py` dari root repositori.

## 3. Pilih notebook recovery per metode

Notebook fine-tuning YOLO26s `recovery/02` menyediakan tujuh `STUDENT_INPUTS` dari suite pruning `20261107T020459Z_45575a7f`. Comment/uncomment hingga tepat satu baris aktif, pilih `DEVICE`, lalu Run All. Model dan seed berasal dari raw terpilih. Setiap eksekusi membuat folder dan ID run sendiri, sehingga beberapa kernel Filkom dapat menjalankan pilihan berbeda secara paralel. Gunakan GPU yang tersedia secara eksklusif saat mengukur latency acuan. [Tujuh input W&B](ARTEFAK_WANDB.md#suite-dengan-baseline-sesuai-seed)

Pada fine-tuning 02, `LOCAL_RAW_PACKAGES` dapat memetakan nama input ke salinan paket lengkap. SHA-256 manifest, raw, baseline, dan audit diperiksa terhadap pin. Metrik setiap epoch, EDA tiga kondisi, best/last, dan snapshot resume tercatat di W&B. Seluruh CSV EDA juga menjadi W&B Tables pada `tables/eda/`, termasuk ledger channel, importance LAMP, dan confusion matrix. Tabel memuat identitas seed, target, serta speed-up aktual. `LOAD_GROUP_RESULTS=True` membaca ringkasan hasil sukses terbaru per perlakuan, platform, dan perangkat dari grup yang sama. Registry per run tersedia sebagai `finetuning_registry.json` dan CSV.

### KD YOLO26s satu raw pilihan

Notebook Native KD 03, CrossKD 04, dan MGD 05 pada Kaggle/Filkom memakai lima student raw 1,20× dari seed 42, 0, 2024, 7, dan 3407. Aktifkan tepat satu `STUDENT_INPUTS`, atur `DEVICE` per kernel, lalu Run All. Student dimuat langsung dari checkpoint native pruned dan seed training mengikuti manifest. Teacher bersama adalah YOLO26m `01_teacher_yolo26m-fux5wdcb-checkpoints:v50` dengan SHA-256 terkunci. `TEACHER_LOCAL_PATH` menerima salinan `best.pt` yang hash-nya sama; kosongkan `TEACHER_ARTIFACT_REF` ketika memakai salinan lokal.

Protokol `selected-raw-kd-v1` mempertahankan loss dan konfigurasi training metode asal. Validasi memakai student EMA tanpa teacher atau adapter KD. EDA lengkap membandingkan baseline asal, raw, dan student setelah KD. W&B memuat loss deteksi/KD/total, seluruh tabel CSV, grafik, best/last native FP32, lineage raw/baseline/teacher, serta snapshot lengkap. Registry tersedia pada `kd_registry.json` dan CSV. Grup bersama memuat 15 perlakuan, dengan mean/std lima seed per metode dan pemisahan platform/perangkat latency.

Pada batas epoch, worker dan buffer augmentasi diperbarui; RNG Python/NumPy augmentasi menggunakan `(seed_manifest + epoch) mod 2**32`. Torch RNG serta gradien student/adapter ikut dipulihkan dari snapshot. Kebijakan ini dicatat sebagai penyesuaian reproduksibilitas terhadap alur sumber. Generator `python scripts/generate_selected_kd_notebooks.py` menulis hanya enam notebook KD 03–05. Notebook fine-tuning 02 tetap memakai generator dan kontraknya sendiri.

Antarmuka historis `source-matched-v1` menerima tepat satu dari `RAW_ARTIFACT_REF` atau `RAW_MANIFEST_PATH`; contoh berikut dipertahankan untuk membaca konfigurasi lama.

```python
RAW_ARTIFACT_REF = "entity/project/artifact-raw-pruned:v1"
# Alternatif input lokal: RAW_MANIFEST_PATH = Path(".../raw_pruned_manifest.json")
# Untuk mengunci student pembanding: EXPECTED_RAW_SHA256 = "<sha256-raw>"
```

Untuk input lokal, sediakan manifest notebook 06 beserta checkpoint baseline/raw dan file audit yang dirujuk. Pada run online, raw lokal yang belum terdaftar dipublikasikan sebelum training. Fingerprint train dan validation, varian, SHA-256 baseline/raw, struktur, dan status reload harus cocok. Mode penelitian memakai manifest `full_val` dengan Poltrada V3 `VERIFIED`. Raw kandidat maksimum yang gagal mencapai target tetap berlabel `TARGET_NOT_REACHED`.

Artefak W&B untuk recovery memerlukan `raw_pruned_manifest.json`, `pruned_fp32.pt`, `baseline_fp32.pt`, dan file audit yang tercantum pada manifest. Untuk artefak lama yang hanya memuat `.pt`, gunakan output suite pruning yang lengkap untuk registrasi. Notebook dapat dibuka sebelum referensi raw diisi. Bagian recovery melaporkan `NOT_RUN` hingga input tersedia.

Teacher ketiga KD adalah YOLO26m `v50` dengan SHA-256 yang dikunci. Untuk salinan lokal, isi `TEACHER_LOCAL_PATH` dan kosongkan `TEACHER_ARTIFACT_REF`. Fine-tuning memakai loss deteksi native tanpa teacher. MGD mengadaptasi pasangan YOLO26s → YOLO26n dari notebook asal menjadi YOLO26m → YOLO26s pruned. Gunakan seed training yang sama untuk semua metode pembanding. Seed pruning dicatat terpisah.

| Metode | Rujukan konfigurasi | Default bobot KD |
| --- | --- | --- |
| Fine-tuning | [Baseline YOLO26s 02](../notebooks/kaggle/baseline/02_baseline_yolo26s.ipynb) | — |
| Native KD | [Native YOLO26m → YOLO26s 04](../notebooks/kaggle/kd/teacher_m/04_native_yolo26s.ipynb) | `dis=1.0` |
| CrossKD | [CrossKD YOLO26m → YOLO26s 06](../notebooks/kaggle/kd/teacher_m/06_crosskd_yolo26s.ipynb) | `KD_CLS_WEIGHT=1.0`, `KD_BOX_WEIGHT=1.0`, `KD_CLS_GAMMA=1.0` |
| MGD | [MGD asal 15](../notebooks/kaggle/kd/teacher_s/15_mgd_yolo26n_teacher_yolo26s.ipynb) | `MGD_ALPHA=2e-5`, `MGD_MASK_RATIO=0.65`, P3/P4/P5, rekonstruksi FP32 |

Trainer mempertahankan objek student native dan memakai loop Ultralytics `8.4.155` untuk optimizer, scaler, scheduler, augmentasi, akumulasi, dan validation per epoch. Teacher dibekukan dan dikeluarkan dari optimizer. Checkpoint inference memuat student recovered.

| Pengaturan training pembanding | Default |
| --- | --- |
| Data dan pemilihan EMA terbaik | `train` asli Poltrada V3, dipilih berdasarkan fitness `val` |
| Budget | 100 epoch total, `patience=100` |
| Optimizer | AdamW, `lr0=0.001`, `weight_decay=0.0005`, `momentum=0.937` |
| Scheduler | Linear, `lrf=0.01`, warm-up 3 epoch |
| Ukuran dan batch | 640 px, batch 16, `nbs=64`, `close_mosaic=10` |
| Presisi dan seed | AMP aktif, satu seed per run, default 42 |

Mode penelitian memerlukan CUDA. Gunakan `MODE="smoke"` untuk pemeriksaan satu epoch dengan AMP nonaktif, `nbs=BATCH`, dan subset validation tetap. Hasil CPU fixture dipisahkan dari hasil penelitian. Penyesuaian batch otomatis atau AMP yang gagal aktif membuat run ditolak. Dataset training berasal dari split train Poltrada V3 asli.

## 4. Status, artefak, dan resume

| Tahap | Yang disimpan |
| --- | --- |
| Pruning | Suite dan rencana target, `pruned_fp32.pt`, SHA-256, audit struktur/EDA, artefak `raw-pruned` |
| Fine-tuning 02 per epoch | `epoch_metrics.json`, snapshot lengkap lokal, metrik W&B, dan snapshot W&B setiap lima epoch serta akhir training |
| Fine-tuning 02 selesai | `final_checkpoints/best.pt` dan `last.pt` FP32, manifest, registry, hasil evaluasi tiga kondisi, tabel dan grafik EDA di W&B |
| Recovery KD 03–05 | `training/weights/best.pt` native student-only, manifest dan SHA-256, hasil evaluasi, kurva per epoch, tabel dan grafik EDA |

Fine-tuning 02 memakai protokol `selected-raw-finetune-v2`. Untuk resume, pilih input raw yang sama dan isi tepat satu `RESUME_RUN_DIR` atau `RESUME_SNAPSHOT_REF` versi tetap. Kontrak mengikat input, fingerprint data, training, dan runtime. Resume memulihkan model aktif, EMA, optimizer, scheduler, AMP scaler, gradien tertunda, state acak, serta batas epoch dataloader. Dataloader menyegarkan worker dan buffer augmentasi tiap epoch agar resume dapat direproduksi. Snapshot lokal tersedia setiap epoch, sedangkan sesi baru dapat memakai snapshot W&B terakhir yang berhasil diunggah.

### Retensi checkpoint dan pembersihan smoke

Recovery 01–05 Kaggle/Filkom memakai `best-and-latest-v1`. Model inference disimpan sebagai
`best.pt` dan `last.pt`; snapshot training terbaru sudah mencakup model epoch terbaik dan state
terakhir untuk resume. Commit lokal dilakukan setiap akhir epoch secara atomik. Sesudah hash
commit baru valid, snapshot lokal yang telah digantikan dihapus. Publikasi W&B mempertahankan
satu versi resume terbaru per run; versi lamanya dihapus setelah versi pengganti selesai diunduh
dan SHA-256 cocok. Aliases `best`, `last`, dan `latest` menunjuk paket yang berisi kedua kondisi
tersebut, dengan `best_model_epoch` dan `snapshot_epoch` pada metadata. Checkpoint inference,
resume, dan hasil tetap berada pada koleksi terpisah. Versi dari run lain atau yang sedang dipakai
run lain dilindungi. Seluruh history epoch, loss, Tables, Images, evaluasi, dan EDA penelitian
tetap dicatat. Receipt retensi tersedia pada `checkpoint_retention.json`.

Default `AUTO_DELETE_SMOKE_ARTIFACTS=True` dan `AUTO_DELETE_LOCAL_SMOKE_WEIGHTS=True`.
Smoke yang telah lulus training, evaluasi, EDA, serta verifikasi logging dan hash akan menghapus
artefak diagnostik W&B dan salinan biner lokal di folder run tersebut. Raw kanonis, baseline,
teacher, output penelitian, JSON/CSV, grafik dan log verifikasi tetap disimpan. Status
`DELETED_AFTER_SMOKE` pada registry menunjukkan bahwa referensi itu hanya bukti historis.
Receipt sebelum/sesudah penghapusan adalah `smoke_verification_before_cleanup.json`,
`smoke_artifact_cleanup.json`, dan `smoke_local_cleanup.json`. Run yang terhenti sebelum selesai
tetap dapat di-resume. Untuk uji pemulihan snapshot, matikan dua opsi cleanup selama uji,
lalu bersihkan setelah seluruh pemeriksaan selesai.

Lapisan I/O mempunyai `STORAGE_SHA256` terpisah; `IMPLEMENTATION_SHA256` dan kontrak training
yang sudah tersimpan tetap berlaku. Kernel GPU yang sedang berjalan memakai kode lama sampai
notebook diperbarui dan sesi dilanjutkan dari commit terakhir. Perubahan ini tidak menghapus
riwayat artefak penelitian yang dibuat kernel lama. Penghapusan W&B bersifat soft delete;
pembaruan penggunaan storage mengikuti garbage collection server dan berbagi berkas antarversi.
[Semantik penghapusan W&B](https://github.com/wandb/docs/blob/main/models/artifacts/delete-artifacts.mdx).

Recovery KD 03–05 memakai protokol `selected-raw-kd-v1` dengan `RESUME_RUN_DIR` atau `RESUME_SNAPSHOT_REF` versi tetap. Snapshot menyimpan student aktif, EMA, optimizer, scheduler, AMP scaler, criterion, adapter beserta gradien tertunda, RNG, dan state dataloader; teacher direferensikan dengan versi dan hash terkunci. Simpan snapshot lokal setiap epoch dan unggah setiap lima epoch serta akhir training. Run memakai ID yang sama ketika dilanjutkan. Kontrak menolak perubahan metode, bobot KD, teacher, student, dataset, atau runtime.

Protokol historis `source-matched-v1` tetap berstatus `RESUME_NOT_VERIFIED`. `RESUME_SUITE_DIR` tersedia pada pruning dan recovery YOLO26n 01 sesuai kontrak masing-masing. Pada Kaggle, arsipkan `/kaggle/working` sebelum sesi berakhir. Pada Filkom, arahkan output ke direktori persisten.

Recovery YOLO26s mengunggah artefak model final dan hasil secara terpisah. Gunakan versi tetap `:vN` untuk lineage raw dan teacher. `PARTIAL` berarti sebagian tahap selesai dan hasil lokal masih tersedia. Gunakan field per tahap untuk membedakan error training, evaluasi, latency, EDA, dan upload. [Penamaan W&B](wandb_naming.md)

## 5. Baca perbandingan tiga tahap

Pada seluruh notebook recovery 01–05 di Kaggle dan Filkom, bagian EDA memulai dengan diagram
batang **satu metrik per grafik**, dengan baseline asal biru, raw pruned merah, dan hasil metode
recovery hijau. Dua belas grafik membandingkan mAP50–95, mAP50, precision, recall, parameter,
GFLOPs, channel, ukuran checkpoint, latency p50/p95, serta speed-up GFLOPs/latency. Panel lain
memperlihatkan delta/gain dan enam metrik per kelas (AP50–95/AP50/P/R/F1/support). Diagnostik
channel, ledger, heatmap, LAMP, PR/F1, confusion matrix, training, dan trade-off tetap lengkap.

Di W&B buka **`custom_eda_report_url` pada Summary run**. Halaman perbandingan sudah mempunyai
panel; kamu tidak perlu menyusun chart dari file artefak. Gambar dan Tables dicatat langsung pada
run, sedangkan `<nama-run>-eda:vN` menyimpan salinan yang diunduh ulang dan diperiksa SHA-256.
Nilai kosong mempertahankan `NOT_RUN` atau `NOT_COMPARABLE`. Mode, seed, target, capaian,
platform, dan perangkat latency tetap dibedakan. Report menggunakan `wandb-workspaces==0.4.5`
bersama W&B `0.28.1`; dependency report dipasang tanpa mengganti dependency trainer.

Untuk menambahkan panel pada run lama yang telah `PASS`, jalankan dari root repo:

```powershell
.venv/Scripts/python.exe -B -m scripts.publish_recovery_custom_eda --run-ids <ID_RUN>
```

Perintah membaca hasil evaluasi versi tetap, menambahkan EDA, dan mempertahankan checkpoint
serta jumlah epoch. Untuk run yang masih aktif gunakan `--reports-only`: hanya report dan tautan
summary yang ditambahkan, tanpa membuka sesi training kedua. Panel yang belum diukur tetap kosong.
Bagian EDA notebook juga dapat dijalankan ulang dari `RESULT`/`BUNDLE` serta folder hasil tersimpan.

Input FT 02 kini mengikuti tujuh versi yang diberikan pengguna: 1,10× seed 42 `:v0`, serta lima
raw 1,20× dan pembanding 1,70× seed 42 `:v1`. Resume memverifikasi bahwa binding lama/baru
mempunyai SHA-256 raw dan manifest yang sama; bobot serta kontrak training tidak diganti.

Baseline, raw pruned, dan recovered diukur pada split yang sama, FP32 640 px dengan batch validasi 16, one-to-many/NMS, confidence 0,001, IoU 0,7, dan `max_det=300`. Gunakan hasil full validation untuk laporan komparatif. Latency raw forward diukur dengan batch 1 pada perangkat yang sama di ketiga tahap. Simpan identitas GPU bersama angka median/p95. Laporkan Filkom dan Kaggle terpisah jika perangkat berbeda.

EDA menampilkan batang baseline/raw/recovered untuk metrik deteksi, AP per kelas, parameter, GFLOPs, ukuran checkpoint sesuai format penyimpanan, total channel, median/p95 latency, dua ukuran speed-up, dan delta. Recovery gain adalah selisih mAP terhadap raw dalam poin persentase. Periksa juga kurva loss/mAP per epoch, PR/F1, confusion matrix, dan audit channel/importance pruning. Data sumber tabel dan grafik tersedia sebagai CSV/JSON, dengan grafik PNG/SVG.

Bangkitkan ulang kedua notebook fine-tuning 02 dengan `python scripts/generate_pruned_fine_tuned_notebooks.py`. Default generator hanya memilih metode fine-tuning. Status pengujian dan batas klaim dicatat pada [laporan pruning](../reports/pruning/README.md) dan [log penelitian](../reports/log-penelitian.md).
