# Inventaris Artefak Eksperimen Weights & Biases (W&B)

Dokumentasi referensi *model checkpoint* untuk eksperimen YOLO26 pada dataset IndoTraffic, berdasarkan data dan kode `wandb.use_artifact()` yang dicatat selama eksperimen.

> Tujuh URI student raw LAMP suite per seed sudah dicocokkan dengan metadata W&B. [Student raw LAMP](#student-raw-lamp-yolo26s-multi-seed)

> **Ketersediaan smoke, 8 November 2026:** atas permintaan pengguna, artefak diagnostik smoke
> W&B dan salinan checkpoint smoke lokal dibersihkan. Referensi smoke dan SHA-256 di bagian
> historis berikut mencatat verifikasi saat pengujian, bukan jaminan berkas masih tersedia.
> Raw pruning kanonis, baseline, teacher, dan hasil penelitian tetap dilindungi.
> [Receipt pembersihan](../reports/pruning/recovery/smoke_cleanup_20261108/)
> · [Aturan best/latest dan cleanup otomatis](lamp_gflops_recovery_workflow.md#retensi-checkpoint-dan-pembersihan-smoke).

### Smoke kebijakan best/latest — 8 November 2026

Kedua run berikut telah diverifikasi melalui API dan unduhan SHA-256, lalu artefaknya dihapus
sesuai permintaan pengguna. Mode adalah **smoke CPU sintetis 64 px**, bukan hasil Poltrada V3.
URI, ID run dan hash di tabel merupakan bukti historis; status ketersediaan `DELETED_AFTER_SMOKE`.

| Metode / epoch | ID run | Referensi checkpoint historis | SHA-256 `best.pt` | SHA-256 `last.pt` |
|---|---|---|---|---|
| Fine-tuning / 2 | `10a3d29cb5fe46e18c7799fbfa102e2a` | `ilhamrafiqin-brawijaya-university/yolom26-kd-indotraffic/yolo26s-lamp-su1.20x-seed2024-ft-smoke-checkpoints:v0` | `b9be78711668412d0c5a4c4766b435ad46e8c2e19f9be7ace2fb3002b4dd3264` | `90c22ca830138d25e5d7f5c91458b5eeffc4b19ed63714a753e77e259d6fffdc` |
| MGD / 1 | `1c655fbf8114423498649beda51e31b2` | `ilhamrafiqin-brawijaya-university/yolom26-kd-indotraffic/yolo26s-lamp-su1.20x-seed2024-mgd-smoke-checkpoints:v0` | `536ab6c7ee438d9f9a998b1db9a949b25c3a92472e404acb5b42876c8a289adc` | `b9c1ee402bc47883760c8667d3bd4bafbe5db688a7749099a601ebbeab36e011` |

Koleksi resume FT menguji `:v0` → `:v1`: versi pertama dihapus setelah pengganti lolos hash;
versi kedua dibersihkan di akhir smoke. Resume MGD `:v0`, hasil `-results:v0`, dan EDA `-eda:v0`
dari kedua run juga sudah dihapus. Metadata/SHA-256 seluruh berkas, versi, ID artefak, pemetaan
seed 2024 ke raw `:v1` dan baseline `v41`, serta receipt sebelum/sesudah penghapusan ada pada
[bukti gabungan](../reports/pruning/recovery/recovery_storage_verification_20261108.json).
Raw pruning, baseline dan teacher pada bagian input berikut tetap tersedia.

## 1. Identitas proyek

| Field | Nilai |
|---|---|
| Platform | Weights & Biases, di-host pada CoreWeave Forge |
| Entity | `ilhamrafiqin-brawijaya-university` |
| Project | `yolom26-kd-indotraffic` |
| Namespace | `ilhamrafiqin-brawijaya-university/yolom26-kd-indotraffic` |
| Jenis artifact | `model` untuk checkpoint, `results` untuk evaluasi dan EDA |
| Referensi versi input yang telah didokumentasikan | **19**: 13 referensi awal dan 6 raw pulihan `:v1`; lihat status akses pada bagian pemulihan |
| Model teacher | YOLO26m |
| Model baseline student | YOLO26s |
| Seed baseline | `0`, `7`, `42`, `2024`, `3407` |

## 2. Artefak teacher dan baseline

| No. | Peran | Nama eksperimen | Model | Seed | Best epoch | Run ID | Versi artifact yang direferensikan | Tautan W&B |
|---:|---|---|---|---:|---:|---|---|---|
| 1 | Teacher | `teacher_yolo26m_seed42` | YOLO26m | 42 | **51** | `fux5wdcb` | `v50` | [Lihat artifact](https://forge.coreweave.com/wandb/ilhamrafiqin-brawijaya-university/yolom26-kd-indotraffic/artifacts/model/01_teacher_yolo26m-fux5wdcb-checkpoints/v50) |
| 2 | Baseline (tanpa KD) | `base_yolo26s_seed7` | YOLO26s | 7 | **50** | `9cq437l3` | `v49` | [Lihat artifact](https://forge.coreweave.com/wandb/ilhamrafiqin-brawijaya-university/yolom26-kd-indotraffic/artifacts/model/02_baseline_yolo26s-9cq437l3-checkpoints/v49) |
| 3 | Baseline (tanpa KD) | `base_yolo26s_seed42` | YOLO26s | 42 | **63** | `q5kp9k9d` | `v62` | [Lihat artifact](https://forge.coreweave.com/wandb/ilhamrafiqin-brawijaya-university/yolom26-kd-indotraffic/artifacts/model/02_baseline_yolo26s-q5kp9k9d-checkpoints/v62) |
| 4 | Baseline (tanpa KD) | `base_yolo26s_seed3407` | YOLO26s | 3407 | **44** | `bmshdzv9` | `v43` | [Lihat artifact](https://forge.coreweave.com/wandb/ilhamrafiqin-brawijaya-university/yolom26-kd-indotraffic/artifacts/model/02_baseline_yolo26s-bmshdzv9-checkpoints/v43) |
| 5 | Baseline (tanpa KD) | `base_yolo26s_seed2024` | YOLO26s | 2024 | **42** | `1o9ggnyw` | `v41` | [Lihat artifact](https://forge.coreweave.com/wandb/ilhamrafiqin-brawijaya-university/yolom26-kd-indotraffic/artifacts/model/02_baseline_yolo26s-1o9ggnyw-checkpoints/v41) |
| 6 | Baseline (tanpa KD) | `base_yolo26s_seed0` | YOLO26s | 0 | **72** | `7f5jmtwb` | `v71` | [Lihat artifact](https://forge.coreweave.com/wandb/ilhamrafiqin-brawijaya-university/yolom26-kd-indotraffic/artifacts/model/02_baseline_yolo26s-7f5jmtwb-checkpoints/v71) |

### Referensi artifact untuk Python API

Gunakan **nama lengkap artifact berikut** pada `run.use_artifact(..., type="model")`:

| Eksperimen | Referensi artifact (`entity/project/name:version`) |
|---|---|
| `teacher_yolo26m_seed42` | `ilhamrafiqin-brawijaya-university/yolom26-kd-indotraffic/01_teacher_yolo26m-fux5wdcb-checkpoints:v50` |
| `base_yolo26s_seed7` | `ilhamrafiqin-brawijaya-university/yolom26-kd-indotraffic/02_baseline_yolo26s-9cq437l3-checkpoints:v49` |
| `base_yolo26s_seed42` | `ilhamrafiqin-brawijaya-university/yolom26-kd-indotraffic/02_baseline_yolo26s-q5kp9k9d-checkpoints:v62` |
| `base_yolo26s_seed3407` | `ilhamrafiqin-brawijaya-university/yolom26-kd-indotraffic/02_baseline_yolo26s-bmshdzv9-checkpoints:v43` |
| `base_yolo26s_seed2024` | `ilhamrafiqin-brawijaya-university/yolom26-kd-indotraffic/02_baseline_yolo26s-1o9ggnyw-checkpoints:v41` |
| `base_yolo26s_seed0` | `ilhamrafiqin-brawijaya-university/yolom26-kd-indotraffic/02_baseline_yolo26s-7f5jmtwb-checkpoints:v71` |

## 3. Cara mengunduh artifact

### Persiapan

Install SDK dan autentikasi ke server W&B yang sesuai dengan akun/proyek:

```bash
pip install wandb
wandb login --host https://forge.coreweave.com/wandb
```

> Jika URL host atau mekanisme login W&B milikmu berbeda, gunakan konfigurasi `WANDB_BASE_URL` dan autentikasi yang berlaku untuk instance tersebut. Jangan menyimpan API key di repository.

### A. Mengunduh satu artifact

Contoh berikut menggunakan *teacher* YOLO26m, sesuai pola kode yang digunakan sebelumnya:

```python
import wandb

run = wandb.init(
    entity="ilhamrafiqin-brawijaya-university",
    project="yolom26-kd-indotraffic",
    job_type="download-artifact",
)

artifact = run.use_artifact(
    "ilhamrafiqin-brawijaya-university/yolom26-kd-indotraffic/"
    "01_teacher_yolo26m-fux5wdcb-checkpoints:v50",
    type="model",
)

artifact_dir = artifact.download(root="artifacts/teacher_yolo26m_seed42")
print("Artifact tersimpan di:", artifact_dir)
run.finish()
```

Untuk eksperimen lain, ganti string pada `run.use_artifact()` dengan referensi di tabel **Referensi artifact untuk Python API**.

### B. Mengunduh semua artifact sekaligus

```python
from pathlib import Path
import wandb

ENTITY = "ilhamrafiqin-brawijaya-university"
PROJECT = "yolom26-kd-indotraffic"
PREFIX = f"{ENTITY}/{PROJECT}/"

ARTIFACTS = {
    "teacher_yolo26m_seed42": "01_teacher_yolo26m-fux5wdcb-checkpoints:v50",
    "base_yolo26s_seed7": "02_baseline_yolo26s-9cq437l3-checkpoints:v49",
    "base_yolo26s_seed42": "02_baseline_yolo26s-q5kp9k9d-checkpoints:v62",
    "base_yolo26s_seed3407": "02_baseline_yolo26s-bmshdzv9-checkpoints:v43",
    "base_yolo26s_seed2024": "02_baseline_yolo26s-1o9ggnyw-checkpoints:v41",
    "base_yolo26s_seed0": "02_baseline_yolo26s-7f5jmtwb-checkpoints:v71",
}

run = wandb.init(entity=ENTITY, project=PROJECT, job_type="download-artifacts")

try:
    for experiment, name_and_version in ARTIFACTS.items():
        reference = PREFIX + name_and_version
        artifact = run.use_artifact(reference, type="model")
        target_dir = Path("artifacts") / experiment
        saved_dir = artifact.download(root=str(target_dir))
        print(f"{experiment}: {saved_dir}")
finally:
    run.finish()
```

Perintah tersebut mengunduh artifact berdasarkan **versi yang dipatok secara eksplisit** (`v50`, `v49`, dan seterusnya), bukan versi dinamis seperti `latest`. Bila ada kegagalan akses, pastikan sudah login, memiliki izin ke project, dan versinya memang tersedia di instance W&B.

## 4. Catatan interpretasi hasil eksperimen

1. **Best epoch dan artifact version adalah dua atribut berbeda.** Pada enam catatan ini tampak pola `best_epoch = nomor_versi + 1` (misalnya epoch 51 dengan `v50`). Pola tersebut konsisten dengan penomoran berbasis nol, **tetapi belum membuktikan** bahwa file pada versi itu adalah checkpoint dari epoch tersebut. Periksa metadata artifact atau file checkpoint untuk konfirmasi.
2. **Jangan memilih model terbaik berdasarkan angka epoch saja.** Gunakan metrik validasi yang ditentukan protokol penelitian untuk memilih *best checkpoint*. Nilai mAP, precision, recall, atau loss belum didokumentasikan pada inventaris ini.
3. **Teacher vs baseline:** Teacher YOLO26m didokumentasikan satu seed (`42`), sementara baseline YOLO26s didokumentasikan lima seed (`0`, `7`, `42`, `2024`, `3407`). Belum ada artifact CrossKD, CSAKD, atau YOLO26n dalam inventaris ini.
4. **Reproducibility:** Simpan versi dataset, konfigurasi pelatihan, versi library, dan metrik evaluasi bersama referensi artifact jika tersedia. Hindari memasukkan file checkpoint besar atau kredensial ke Git tanpa kebutuhan khusus.

## 5. Ringkasan status

| Kelompok | Jumlah versi | Seed | Referensi versi |
|---|---:|---|---|
| Teacher YOLO26m | 1 | `42` | 1/1 tercatat |
| Baseline YOLO26s | 5 | `0`, `7`, `42`, `2024`, `3407` | 5/5 `best.pt` terverifikasi untuk suite per seed |
| Raw LAMP YOLO26s per seed | 7 | `0`, `7`, `42`, `2024`, `3407` | 7/7 URI `:v0` dicocokkan dengan metadata W&B |
| Raw LAMP YOLO26s 1,20× pulihan | 5 | `0`, `7`, `42`, `2024`, `3407` | 5/5 `:v1` diunduh dan cocok SHA-256; input KD aktif |
| Fine-tuning YOLO26s smoke utama | 21 | `0`, `7`, `42`, `2024`, `3407` | 7 run, masing-masing checkpoint, hasil, dan snapshot; SHA-256 unduhan cocok |
| Riwayat tambahan smoke/resume fine-tuning | 20 | `42` | Versi diagnostik tambahan; kegagalan dan perbaikannya tercatat |
| **Total** | **59** | — | **59 versi tercatat, termasuk riwayat diagnostik dan pemulihan raw** |

## Student raw LAMP YOLO26s multi-seed

### Suite dengan baseline sesuai seed

Ketujuh artefak raw di bawah berasal dari suite `20261107T020459Z_45575a7f`. Gunakan URI versi tetap di tabel dan SHA-256 dari registry suite untuk recovery. Kelima `best.pt` sumber telah diperiksa melalui metadata seed dan SHA-256. [Notebook multi-seed](../notebooks/kaggle/pruning/06_lamp_yolo26s_multiseed_wandb.ipynb)

| Seed | Artefak baseline | SHA-256 `best.pt` |
|---:|---|---|
| 42 | `02_baseline_yolo26s-q5kp9k9d-checkpoints:v62` | `5bc3ae61f8da1c9f03916922de10146805c27c64a11e46edd766a00d0f6f18fd` |
| 0 | `02_baseline_yolo26s-7f5jmtwb-checkpoints:v71` | `d02d8d90d5000fc8379379acb0c0cb3cde0c01143b197d237b25be1bae99bf4f` |
| 2024 | `02_baseline_yolo26s-1o9ggnyw-checkpoints:v41` | `47cfab586889d2b464c882239c86224b645363a5a4fa3badc31c46991ad27d70` |
| 7 | `02_baseline_yolo26s-9cq437l3-checkpoints:v49` | `3e59d5f5302fb524da729e139ef7e9b8456c08d0952c19b08af19c7f39bce6ab` |
| 3407 | `02_baseline_yolo26s-bmshdzv9-checkpoints:v43` | `de9f06892953a343629a4b3a59df880d771973ae95160dee02cd96e37de59e46` |

| Target | Seed | Status target | `RAW_ARTIFACT_REF` |
|---|---:|---|---|
| 1,10× | 42 | `REACHED` | [`ilhamrafiqin-brawijaya-university/yolom26-kd-indotraffic/yolo26s-lamp-su1.10x-seed42-raw:v0`](https://wandb.ai/ilhamrafiqin-brawijaya-university/yolom26-kd-indotraffic/artifacts/model/yolo26s-lamp-su1.10x-seed42-raw/v0) |
| 1,20× | 42 | `REACHED` | [`ilhamrafiqin-brawijaya-university/yolom26-kd-indotraffic/yolo26s-lamp-su1.20x-seed42-raw:v0`](https://wandb.ai/ilhamrafiqin-brawijaya-university/yolom26-kd-indotraffic/artifacts/model/yolo26s-lamp-su1.20x-seed42-raw/v0) |
| 1,70× | 42 | `TARGET_NOT_REACHED` | [`ilhamrafiqin-brawijaya-university/yolom26-kd-indotraffic/yolo26s-lamp-su1.70x-seed42-raw:v0`](https://wandb.ai/ilhamrafiqin-brawijaya-university/yolom26-kd-indotraffic/artifacts/model/yolo26s-lamp-su1.70x-seed42-raw/v0) |
| 1,20× | 0 | `REACHED` | [`ilhamrafiqin-brawijaya-university/yolom26-kd-indotraffic/yolo26s-lamp-su1.20x-seed0-raw:v0`](https://wandb.ai/ilhamrafiqin-brawijaya-university/yolom26-kd-indotraffic/artifacts/model/yolo26s-lamp-su1.20x-seed0-raw/v0) |
| 1,20× | 2024 | `REACHED` | [`ilhamrafiqin-brawijaya-university/yolom26-kd-indotraffic/yolo26s-lamp-su1.20x-seed2024-raw:v0`](https://wandb.ai/ilhamrafiqin-brawijaya-university/yolom26-kd-indotraffic/artifacts/model/yolo26s-lamp-su1.20x-seed2024-raw/v0) |
| 1,20× | 7 | `REACHED` | [`ilhamrafiqin-brawijaya-university/yolom26-kd-indotraffic/yolo26s-lamp-su1.20x-seed7-raw:v0`](https://wandb.ai/ilhamrafiqin-brawijaya-university/yolom26-kd-indotraffic/artifacts/model/yolo26s-lamp-su1.20x-seed7-raw/v0) |
| 1,20× | 3407 | `REACHED` | [`ilhamrafiqin-brawijaya-university/yolom26-kd-indotraffic/yolo26s-lamp-su1.20x-seed3407-raw:v0`](https://wandb.ai/ilhamrafiqin-brawijaya-university/yolom26-kd-indotraffic/artifacts/model/yolo26s-lamp-su1.20x-seed3407-raw/v0) |

Ketujuh paket telah diunduh dan diverifikasi. Di dalam setiap artefak raw, file student adalah
`pruned_fp32.pt` dan baseline pembanding adalah `baseline_fp32.pt`. Seed recovery mengikuti seed
pada baris yang sama. SHA-256 berikut dipatok oleh notebook fine-tuning 02.

| Target | Seed pruning/recovery | SHA-256 raw pruned | SHA-256 baseline FP32 dalam paket |
|---|---:|---|---|
| 1,10× | 42 | `913672ebac25afdebc09957659006506562b11cc56b7771a81b473a0041fc427` | `48733cbdb927ea6dde0985ec7a006cbd8f92d1b84def45d927e3c6c86fab5d50` |
| 1,20× | 42 | `1c20b10f3a33de89377e1cf80e1ff244489fb9fc737b121735cd6d24618d7159` | `48733cbdb927ea6dde0985ec7a006cbd8f92d1b84def45d927e3c6c86fab5d50` |
| 1,70× | 42 | `e181881b8e5bbd4dc021f6dfad54c579361bcdc4a908724247087204a155f026` | `48733cbdb927ea6dde0985ec7a006cbd8f92d1b84def45d927e3c6c86fab5d50` |
| 1,20× | 0 | `99df514946a6a283e91d90ba22c8c159e6fd107baba739f8860a20c79af2e8aa` | `fb811a61b070580b13c5a7908b93898d4e88c96bd3e82df192227699e275bac1` |
| 1,20× | 2024 | `3187dde6e33139a1e4c1afd5a6fbc2235d05cd0cc676be1b494bc2d08e45fcf6` | `e3b464a21029166408bd2006eb388b2fc1399f0d25fc9172307df0aeef604742` |
| 1,20× | 7 | `69ed0b860c79626ab435fa06afa1fe4fcf57e4c1478aec498713b4970430cf4a` | `7c3224a2ecdf113c0a0279d59241082565029a1af76d23d78c155dc1e8c7c0cc` |
| 1,20× | 3407 | `9a6e64cdf17fb64dfb38082e7f5231a59cf4d6b32f01f05a14aaaf937667d455` | `f1dd45dcaec079d67c90ca1c60da50be272bda959184928851578076f2ebb967` |

### Pemulihan lima raw 1,20× untuk KD

Pada audit KD 7 November 2026, lima versi raw 1,20× `:v0` tidak dapat diakses melalui API W&B,
walaupun koleksinya masih terlihat. Atas instruksi pengguna, paket lokal asli yang sudah cocok
SHA-256 dipublikasikan kembali. Kelima versi berikut telah diunduh dan seluruh hash file cocok.
Bobot, arsitektur, manifest, dan suite pruning asal tetap sama; pemulihan tidak melakukan training.
Notebook KD 03–05 memilih versi `:v1` ini. Pada 8 November 2026, fine-tuning 02 juga diperbarui
mengikuti referensi pulihan dari pengguna. Binding URI diperiksa terhadap hash raw dan manifest
asli. Kontrak training lama tetap dapat dilanjutkan: identitas historis dipertahankan pada kontrak
resume, sedangkan unduhan dan lineage baru memakai salinan identik `:v1` yang terverifikasi.

| Seed | Input raw KD terkonfirmasi | SHA-256 raw asli/pulihan |
|---:|---|---|
| 42 | [`yolo26s-lamp-su1.20x-seed42-raw:v1`](https://wandb.ai/ilhamrafiqin-brawijaya-university/yolom26-kd-indotraffic/artifacts/model/yolo26s-lamp-su1.20x-seed42-raw/v1) | `1c20b10f3a33de89377e1cf80e1ff244489fb9fc737b121735cd6d24618d7159` |
| 0 | [`yolo26s-lamp-su1.20x-seed0-raw:v1`](https://wandb.ai/ilhamrafiqin-brawijaya-university/yolom26-kd-indotraffic/artifacts/model/yolo26s-lamp-su1.20x-seed0-raw/v1) | `99df514946a6a283e91d90ba22c8c159e6fd107baba739f8860a20c79af2e8aa` |
| 2024 | [`yolo26s-lamp-su1.20x-seed2024-raw:v1`](https://wandb.ai/ilhamrafiqin-brawijaya-university/yolom26-kd-indotraffic/artifacts/model/yolo26s-lamp-su1.20x-seed2024-raw/v1) | `3187dde6e33139a1e4c1afd5a6fbc2235d05cd0cc676be1b494bc2d08e45fcf6` |
| 7 | [`yolo26s-lamp-su1.20x-seed7-raw:v1`](https://wandb.ai/ilhamrafiqin-brawijaya-university/yolom26-kd-indotraffic/artifacts/model/yolo26s-lamp-su1.20x-seed7-raw/v1) | `69ed0b860c79626ab435fa06afa1fe4fcf57e4c1478aec498713b4970430cf4a` |
| 3407 | [`yolo26s-lamp-su1.20x-seed3407-raw:v1`](https://wandb.ai/ilhamrafiqin-brawijaya-university/yolom26-kd-indotraffic/artifacts/model/yolo26s-lamp-su1.20x-seed3407-raw/v1) | `9a6e64cdf17fb64dfb38082e7f5231a59cf4d6b32f01f05a14aaaf937667d455` |

[Bukti paket, provenance, dan hash seluruh file](../reports/pruning/lamp_yolo26s_raw_restored_20261107.json).
Teacher YOLO26m `v50` serta kelima baseline asal juga dapat diakses saat audit ini.

### Input aktif fine-tuning 02 · 8 November 2026

Ketujuh pilihan comment/uncomment berikut tersedia di kedua platform. Semua masih memakai
raw dari suite `20261107T020459Z_45575a7f`, dengan hash pada tabel suite per seed di atas.
Target 1,70× seed 42 tetap `TARGET_NOT_REACHED`, dengan capaian aktual pada manifest.
Pemulihan URI tidak menginisialisasi ulang student atau mengganti seed asalnya.

| Target | Seed | Versi input aktif |
|---|---:|---|
| 1,10× | 42 | `yolo26s-lamp-su1.10x-seed42-raw:v0` |
| 1,20× | 42 | `yolo26s-lamp-su1.20x-seed42-raw:v1` |
| 1,70× | 42 | `yolo26s-lamp-su1.70x-seed42-raw:v1` |
| 1,20× | 0 | `yolo26s-lamp-su1.20x-seed0-raw:v1` |
| 1,20× | 2024 | `yolo26s-lamp-su1.20x-seed2024-raw:v1` |
| 1,20× | 7 | `yolo26s-lamp-su1.20x-seed7-raw:v1` |
| 1,20× | 3407 | `yolo26s-lamp-su1.20x-seed3407-raw:v1` |

[Verifikasi enam binding pulihan dan SHA-256 seluruh file](../reports/pruning/lamp_yolo26s_ft_raw_bindings_20261108.json).
Seed 2024, 7, dan 3407 telah diuji satu epoch CPU FP32 64 px dengan input `:v1` asli;
hasilnya pada `outputs/pruning/finetune_binding_smoke_20261108/verification.json`.
Ini uji pipeline, bukan pengukuran akurasi Poltrada V3.

### Riwayat publikasi seed 42

Ketiga paket historis dari run Kaggle memakai sumber `base_yolo26s_seed42` versi `v62`. Pada audit awal 7 November 2026, versi, metadata, dan SHA-256 seluruh file unduhan diperiksa. Manifest mencatat full validation pada 1.790 citra Poltrada V3 dengan Tesla T4. Ketiga URI `:v0` seed 42 sekarang menunjuk ke suite per seed di atas dengan SHA-256 raw yang berbeda. Gunakan cadangan lokal untuk membaca paket historis. [Registry historis](../reports/pruning/lamp_yolo26s_wandb_v0_registry.json)

| Target | Speed-up GFLOPs aktual | Status target | Seed recovery | Versi tercatat |
|---|---:|---|---|---|
| 1,10× | 1,100131× | `REACHED` | 42 | `v0` |
| 1,20× | 1,200202× | `REACHED` | 42, 0, 2024, 7, 3407 | `v0` |
| 1,70× | 1,229175× | `TARGET_NOT_REACHED` | 42, diagnostik | `v0` |

| Target | URI tercatat saat audit lama | SHA-256 cadangan historis |
|---|---|---|
| 1,10× | `ilhamrafiqin-brawijaya-university/yolom26-kd-indotraffic/yolo26s-lamp-su1.10x-seed42-raw:v0` | `c174b3930dbb9f4cee817b2457c9e67f841cb88142bc2a3c583cea2c1f1aeb75` |
| 1,20× | `ilhamrafiqin-brawijaya-university/yolom26-kd-indotraffic/yolo26s-lamp-su1.20x-seed42-raw:v0` | `8c89184862d44fea1db3d9ede5577b9194f3fea9e845013ecc6e79098fde15c3` |
| 1,70× | `ilhamrafiqin-brawijaya-university/yolom26-kd-indotraffic/yolo26s-lamp-su1.70x-seed42-raw:v0` | `17cfa19746617aecefe7fc71770f365d6f6839b9ab4d8b5f1a0f48ff55f89934` |

Registry historis memetakan lima seed recovery target 1,20× ke versi dan SHA-256 raw seed 42 yang sama. Versi tersebut mendokumentasikan desain dengan satu baseline bersama. Target 1,70× memerlukan `allow_diagnostic=True`. [Protokol dan cara pakai registry](lamp_gflops_recovery_workflow.md#pruning-yolo26s-multi-seed)

Cadangan lokal ketiga paket ada di `outputs/pruning/wandb_yolo26s_lamp_v0/`. Untuk memuat salinan lokal, pilih `local_raw_manifest_path` dari registry sebagai `RAW_MANIFEST_PATH` dan kosongkan `RAW_ARTIFACT_REF`.

## Fine-tuning raw YOLO26s · smoke satu epoch

Ketujuh input raw telah menjalani satu epoch CPU FP32 pada fixture sintetis enam citra train dan
enam citra val, 64 px, batch 2. Seed serta bobot awal diwarisi dari masing-masing raw. Tabel,
history epoch, 103 gambar per run, dan seluruh versi paket berikut telah diperiksa di W&B.
Pengujian ini memakai varian notebook Filkom pada CPU lokal. Training 100 epoch Poltrada V3,
AMP CUDA, dan latency CUDA berstatus `NOT_RUN`. [Bukti verifikasi](../reports/pruning/recovery/yolo26s_selected_finetune_verification_20261107.json)

Grup diagnostik memakai `yolo26s-lamp-ft-20261107T020459Z_45575a7f-smoke`. Koleksi checkpoint
memuat `best.pt`, `last.pt`, dan manifest. Artefak hasil memuat konfigurasi, metrik, serta EDA
baseline asal → raw pruned → fine-tuned. Referensi lengkap tersedia pada laporan JSON.

Setiap run juga memiliki 17 W&B Tables tambahan pada `tables/eda/`, termasuk channel per layer,
ledger, indeks channel, skor LAMP, dan confusion matrix. Identitas seed, target, serta speed-up
aktual tersedia pada setiap tabel. Snapshot smoke mengikat konfigurasi CPU dan hash implementasi
pada kontraknya. Gunakan tujuh raw di atas sebagai input penelitian 100 epoch.

| Target | Seed | W&B run | Model Artifact best/last | Hasil | Snapshot |
|---|---:|---|---|---|---|
| 1,10× | 42 | [run](https://wandb.ai/ilhamrafiqin-brawijaya-university/yolom26-kd-indotraffic/runs/06d2cd2bf833441081570d348abd0225) | [`yolo26s-lamp-su1.10x-seed42-ft-smoke-checkpoints:v0`](https://wandb.ai/ilhamrafiqin-brawijaya-university/yolom26-kd-indotraffic/artifacts/model/yolo26s-lamp-su1.10x-seed42-ft-smoke-checkpoints/v0) | [hasil](https://wandb.ai/ilhamrafiqin-brawijaya-university/yolom26-kd-indotraffic/artifacts/results/yolo26s-lamp-su1.10x-seed42-ft-smoke-results/v0) | [resume](https://wandb.ai/ilhamrafiqin-brawijaya-university/yolom26-kd-indotraffic/artifacts/model/yolo26s-lamp-su1.10x-seed42-ft-smoke-resume/v0) |
| 1,20× | 42 | [run](https://wandb.ai/ilhamrafiqin-brawijaya-university/yolom26-kd-indotraffic/runs/5884d03c8cba458f8cf7f437d3fa73a0) | [`yolo26s-lamp-su1.20x-seed42-ft-smoke-checkpoints:v0`](https://wandb.ai/ilhamrafiqin-brawijaya-university/yolom26-kd-indotraffic/artifacts/model/yolo26s-lamp-su1.20x-seed42-ft-smoke-checkpoints/v0) | [hasil](https://wandb.ai/ilhamrafiqin-brawijaya-university/yolom26-kd-indotraffic/artifacts/results/yolo26s-lamp-su1.20x-seed42-ft-smoke-results/v0) | [resume](https://wandb.ai/ilhamrafiqin-brawijaya-university/yolom26-kd-indotraffic/artifacts/model/yolo26s-lamp-su1.20x-seed42-ft-smoke-resume/v0) |
| 1,70× | 42 | [run](https://wandb.ai/ilhamrafiqin-brawijaya-university/yolom26-kd-indotraffic/runs/d3846bc86bfa4a7d96aef4835ece8cb5) | [`yolo26s-lamp-su1.70x-seed42-ft-smoke-checkpoints:v0`](https://wandb.ai/ilhamrafiqin-brawijaya-university/yolom26-kd-indotraffic/artifacts/model/yolo26s-lamp-su1.70x-seed42-ft-smoke-checkpoints/v0) | [hasil](https://wandb.ai/ilhamrafiqin-brawijaya-university/yolom26-kd-indotraffic/artifacts/results/yolo26s-lamp-su1.70x-seed42-ft-smoke-results/v0) | [resume](https://wandb.ai/ilhamrafiqin-brawijaya-university/yolom26-kd-indotraffic/artifacts/model/yolo26s-lamp-su1.70x-seed42-ft-smoke-resume/v0) |
| 1,20× | 0 | [run](https://wandb.ai/ilhamrafiqin-brawijaya-university/yolom26-kd-indotraffic/runs/378bf1f9c9f04ffa9a830f2ac932ed23) | [`yolo26s-lamp-su1.20x-seed0-ft-smoke-checkpoints:v0`](https://wandb.ai/ilhamrafiqin-brawijaya-university/yolom26-kd-indotraffic/artifacts/model/yolo26s-lamp-su1.20x-seed0-ft-smoke-checkpoints/v0) | [hasil](https://wandb.ai/ilhamrafiqin-brawijaya-university/yolom26-kd-indotraffic/artifacts/results/yolo26s-lamp-su1.20x-seed0-ft-smoke-results/v0) | [resume](https://wandb.ai/ilhamrafiqin-brawijaya-university/yolom26-kd-indotraffic/artifacts/model/yolo26s-lamp-su1.20x-seed0-ft-smoke-resume/v0) |
| 1,20× | 2024 | [run](https://wandb.ai/ilhamrafiqin-brawijaya-university/yolom26-kd-indotraffic/runs/13411cc3388846b1b9016399deb946c3) | [`yolo26s-lamp-su1.20x-seed2024-ft-smoke-checkpoints:v0`](https://wandb.ai/ilhamrafiqin-brawijaya-university/yolom26-kd-indotraffic/artifacts/model/yolo26s-lamp-su1.20x-seed2024-ft-smoke-checkpoints/v0) | [hasil](https://wandb.ai/ilhamrafiqin-brawijaya-university/yolom26-kd-indotraffic/artifacts/results/yolo26s-lamp-su1.20x-seed2024-ft-smoke-results/v0) | [resume](https://wandb.ai/ilhamrafiqin-brawijaya-university/yolom26-kd-indotraffic/artifacts/model/yolo26s-lamp-su1.20x-seed2024-ft-smoke-resume/v0) |
| 1,20× | 7 | [run](https://wandb.ai/ilhamrafiqin-brawijaya-university/yolom26-kd-indotraffic/runs/57ac099470d440afa297081043a2e7b2) | [`yolo26s-lamp-su1.20x-seed7-ft-smoke-checkpoints:v0`](https://wandb.ai/ilhamrafiqin-brawijaya-university/yolom26-kd-indotraffic/artifacts/model/yolo26s-lamp-su1.20x-seed7-ft-smoke-checkpoints/v0) | [hasil](https://wandb.ai/ilhamrafiqin-brawijaya-university/yolom26-kd-indotraffic/artifacts/results/yolo26s-lamp-su1.20x-seed7-ft-smoke-results/v0) | [resume](https://wandb.ai/ilhamrafiqin-brawijaya-university/yolom26-kd-indotraffic/artifacts/model/yolo26s-lamp-su1.20x-seed7-ft-smoke-resume/v0) |
| 1,20× | 3407 | [run](https://wandb.ai/ilhamrafiqin-brawijaya-university/yolom26-kd-indotraffic/runs/bf837132872a4e5a9d4934a6c662f474) | [`yolo26s-lamp-su1.20x-seed3407-ft-smoke-checkpoints:v0`](https://wandb.ai/ilhamrafiqin-brawijaya-university/yolom26-kd-indotraffic/artifacts/model/yolo26s-lamp-su1.20x-seed3407-ft-smoke-checkpoints/v0) | [hasil](https://wandb.ai/ilhamrafiqin-brawijaya-university/yolom26-kd-indotraffic/artifacts/results/yolo26s-lamp-su1.20x-seed3407-ft-smoke-results/v0) | [resume](https://wandb.ai/ilhamrafiqin-brawijaya-university/yolom26-kd-indotraffic/artifacts/model/yolo26s-lamp-su1.20x-seed3407-ft-smoke-resume/v0) |

| Target | Seed | SHA-256 best.pt | SHA-256 last.pt |
|---|---:|---|---|
| 1,10× | 42 | `5de5a6f4e8013b3501547e4016f371da09cb7989e59773f6191a72a63cf6cec0` | `7ea1bad9e175d9d1a8c646442b473f62ef864a627766d8a50b2c57635384ef9a` |
| 1,20× | 42 | `71c9156eb22c639f0ca66a674acf2ff96e32b01c0b7ea08f91a5f2ad66f2d736` | `7c2106a9d90716fe556acefcf008378b27ae527adae9fc5e9211c022f9b0631b` |
| 1,70× | 42 | `40b62a5ffeabe8d52da2a6e04a0225ae21b06255e98ddad1e74b8cee4febb8e5` | `d946cf8bf2ba7f28231f3b2f1843ba694052ca925f5099d5d6b717f6aee18856` |
| 1,20× | 0 | `e6b87ca6b137027e39a9c42f702c87eb5b6fea2fff42ec5650a75b20cbc80818` | `83fdc63a5fcdda75a37493e7316b231912aef69e1da1472d3d41bb3853b4a581` |
| 1,20× | 2024 | `2b4dbb14edef55286896d0c138d4212c55bfa30d4a18501bcc5a19cc2cfa8ddf` | `2940c6ca50d333fdd710751502ff4379555bb7fd88380a55d176d2cccb763b93` |
| 1,20× | 7 | `abdfd5209f9c1aff033baed35066395d02bff8eb8790fd3becb436795ed23973` | `66a8250ae62f403ef403c1cb54fd79dee4742977bdba79b387b8fbf22f6e7fd3` |
| 1,20× | 3407 | `f94193e235194129c07cb8e52aa3e88c2e9c7addea9618d343c986ff9badb6b4` | `3c699fd67c0df27768eeb7def9e4bbf412fa4eb22d51da849f7c782d0b9a960d` |

### Riwayat diagnostik resume fine-tuning

Versi tambahan berikut berasal dari uji resume seed 42 target 1,20× pada varian notebook Kaggle dengan CPU lokal. Seluruh paket di tabel telah diunduh dan cocok SHA-256. Riwayat ini melengkapi tujuh smoke utama di atas. Hash lengkap setiap file dan ID run tersedia pada [laporan verifikasi](../reports/pruning/recovery/yolo26s_selected_finetune_verification_20261107.json).

Uji akhir memulihkan `resume:v5` ke folder baru dengan ID run, state training, hash best, dan satu epoch history yang sama. Versi awal `resume:v1` belum memuat training_config dan dipertahankan sebagai bukti kegagalan yang sudah diperbaiki. Snapshot v3/v4 ditolak oleh kontrak sesudah kode berubah selama pengujian. Semua versi ini memakai konfigurasi smoke CPU.

| Versi artefak diagnostik | File acuan SHA-256 | SHA-256 terkonfirmasi |
|---|---|---|
| [`yolo26s-lamp-su1.20x-seed42-ft-smoke-checkpoints:v1`](https://wandb.ai/ilhamrafiqin-brawijaya-university/yolom26-kd-indotraffic/artifacts/model/yolo26s-lamp-su1.20x-seed42-ft-smoke-checkpoints/v1) | `best.pt` | `99a276b5d0f8dd142b1ff4da293eaa11dedcf7c413afb1291aaf8b6c5d348a55` |
| [`yolo26s-lamp-su1.20x-seed42-ft-smoke-checkpoints:v2`](https://wandb.ai/ilhamrafiqin-brawijaya-university/yolom26-kd-indotraffic/artifacts/model/yolo26s-lamp-su1.20x-seed42-ft-smoke-checkpoints/v2) | `best.pt` | `99a276b5d0f8dd142b1ff4da293eaa11dedcf7c413afb1291aaf8b6c5d348a55` |
| [`yolo26s-lamp-su1.20x-seed42-ft-smoke-checkpoints:v3`](https://wandb.ai/ilhamrafiqin-brawijaya-university/yolom26-kd-indotraffic/artifacts/model/yolo26s-lamp-su1.20x-seed42-ft-smoke-checkpoints/v3) | `best.pt` | `7dec7d33ba4453ce8944304f0a83853f8504a6f83cf42cae8844ac0f60610d8f` |
| [`yolo26s-lamp-su1.20x-seed42-ft-smoke-checkpoints:v4`](https://wandb.ai/ilhamrafiqin-brawijaya-university/yolom26-kd-indotraffic/artifacts/model/yolo26s-lamp-su1.20x-seed42-ft-smoke-checkpoints/v4) | `best.pt` | `7dec7d33ba4453ce8944304f0a83853f8504a6f83cf42cae8844ac0f60610d8f` |
| [`yolo26s-lamp-su1.20x-seed42-ft-smoke-checkpoints:v5`](https://wandb.ai/ilhamrafiqin-brawijaya-university/yolom26-kd-indotraffic/artifacts/model/yolo26s-lamp-su1.20x-seed42-ft-smoke-checkpoints/v5) | `best.pt` | `14d328ca2f46e208660728bf9ee781ff60d911fb792cb739bacfc2985ca8a9e7` |
| [`yolo26s-lamp-su1.20x-seed42-ft-smoke-checkpoints:v6`](https://wandb.ai/ilhamrafiqin-brawijaya-university/yolom26-kd-indotraffic/artifacts/model/yolo26s-lamp-su1.20x-seed42-ft-smoke-checkpoints/v6) | `best.pt` | `e4d9c5b7506cde24d7de67d81b580739f6ad436c50d35b2b34c9d4bc281262eb` |
| [`yolo26s-lamp-su1.20x-seed42-ft-smoke-checkpoints:v7`](https://wandb.ai/ilhamrafiqin-brawijaya-university/yolom26-kd-indotraffic/artifacts/model/yolo26s-lamp-su1.20x-seed42-ft-smoke-checkpoints/v7) | `best.pt` | `fbf8bcf424316949a5711b1a9acc5598d01ef43cc70a65e58931c39c12bc03d4` |
| [`yolo26s-lamp-su1.20x-seed42-ft-smoke-checkpoints:v8`](https://wandb.ai/ilhamrafiqin-brawijaya-university/yolom26-kd-indotraffic/artifacts/model/yolo26s-lamp-su1.20x-seed42-ft-smoke-checkpoints/v8) | `best.pt` | `fbf8bcf424316949a5711b1a9acc5598d01ef43cc70a65e58931c39c12bc03d4` |
| [`yolo26s-lamp-su1.20x-seed42-ft-smoke-results:v1`](https://wandb.ai/ilhamrafiqin-brawijaya-university/yolom26-kd-indotraffic/artifacts/results/yolo26s-lamp-su1.20x-seed42-ft-smoke-results/v1) | `evaluation_manifest.json` | `7af75f419d02a0bdae7fa193cb886b7ab2140ef7f4c14395237fb897f88938ff` |
| [`yolo26s-lamp-su1.20x-seed42-ft-smoke-results:v2`](https://wandb.ai/ilhamrafiqin-brawijaya-university/yolom26-kd-indotraffic/artifacts/results/yolo26s-lamp-su1.20x-seed42-ft-smoke-results/v2) | `evaluation_manifest.json` | `2b7b01eeff4c488c5fc9363ec09a43a3f85f15c2673a18dec5bb8a68b0b83293` |
| [`yolo26s-lamp-su1.20x-seed42-ft-smoke-results:v3`](https://wandb.ai/ilhamrafiqin-brawijaya-university/yolom26-kd-indotraffic/artifacts/results/yolo26s-lamp-su1.20x-seed42-ft-smoke-results/v3) | `evaluation_manifest.json` | `9ae48a80e2af4d5fbccc2feef83f0af692b4d89e1e6af13f90ae773fe03b3add` |
| [`yolo26s-lamp-su1.20x-seed42-ft-smoke-results:v4`](https://wandb.ai/ilhamrafiqin-brawijaya-university/yolom26-kd-indotraffic/artifacts/results/yolo26s-lamp-su1.20x-seed42-ft-smoke-results/v4) | `evaluation_manifest.json` | `cc5f13ec55e3a9651dfd16348f78b4fd98dbe953fb75fd0f4d58603128aa4ee1` |
| [`yolo26s-lamp-su1.20x-seed42-ft-smoke-results:v5`](https://wandb.ai/ilhamrafiqin-brawijaya-university/yolom26-kd-indotraffic/artifacts/results/yolo26s-lamp-su1.20x-seed42-ft-smoke-results/v5) | `evaluation_manifest.json` | `1f36b42eb3ec32560f457d3c5f5a86f456775269f356a1a0258781a0c7bc5c4a` |
| [`yolo26s-lamp-su1.20x-seed42-ft-smoke-results:v6`](https://wandb.ai/ilhamrafiqin-brawijaya-university/yolom26-kd-indotraffic/artifacts/results/yolo26s-lamp-su1.20x-seed42-ft-smoke-results/v6) | `evaluation_manifest.json` | `8ecdbfd3422d0b830b5ad63870fb71e682d190f768a5c1935ca72465e7afb5e1` |
| [`yolo26s-lamp-su1.20x-seed42-ft-smoke-results:v7`](https://wandb.ai/ilhamrafiqin-brawijaya-university/yolom26-kd-indotraffic/artifacts/results/yolo26s-lamp-su1.20x-seed42-ft-smoke-results/v7) | `evaluation_manifest.json` | `0b79789b31a6e4dbf77dc1e285009ffbb7cc52641d8e8ab3094e727d767c8f2b` |
| [`yolo26s-lamp-su1.20x-seed42-ft-smoke-resume:v1`](https://wandb.ai/ilhamrafiqin-brawijaya-university/yolom26-kd-indotraffic/artifacts/model/yolo26s-lamp-su1.20x-seed42-ft-smoke-resume/v1) | `checkpoints/resume-001-02c22972.pt` | `997c2bf2cd64a3501d35edb1ad901db65f5351e3df8e7be2a0de1fbcca675e08` |
| [`yolo26s-lamp-su1.20x-seed42-ft-smoke-resume:v2`](https://wandb.ai/ilhamrafiqin-brawijaya-university/yolom26-kd-indotraffic/artifacts/model/yolo26s-lamp-su1.20x-seed42-ft-smoke-resume/v2) | `checkpoints/resume-001-fa0b0132.pt` | `ac6adc86dbec2ea757865f83bf5c6fe0e98063da2994d90f74ec7192852a4bee` |
| [`yolo26s-lamp-su1.20x-seed42-ft-smoke-resume:v3`](https://wandb.ai/ilhamrafiqin-brawijaya-university/yolom26-kd-indotraffic/artifacts/model/yolo26s-lamp-su1.20x-seed42-ft-smoke-resume/v3) | `checkpoints/resume-001-3c87e2b7.pt` | `b53469e4d41083b5535fc8133805ed0e1c90c083afb203f8d7ccac30996c2ac6` |
| [`yolo26s-lamp-su1.20x-seed42-ft-smoke-resume:v4`](https://wandb.ai/ilhamrafiqin-brawijaya-university/yolom26-kd-indotraffic/artifacts/model/yolo26s-lamp-su1.20x-seed42-ft-smoke-resume/v4) | `checkpoints/resume-001-53d4df5d.pt` | `3b659380615a06f9d56d79ca157441546135ed1af79b43a7fe05b29166c98034` |
| [`yolo26s-lamp-su1.20x-seed42-ft-smoke-resume:v5`](https://wandb.ai/ilhamrafiqin-brawijaya-university/yolom26-kd-indotraffic/artifacts/model/yolo26s-lamp-su1.20x-seed42-ft-smoke-resume/v5) | `checkpoints/resume-001-757526aa.pt` | `5058f0ee155c32f923d3f79aa0bfc87cc90f254a65c66753268f8b18c49a5455` |

## Recovery dan EDA custom · verifikasi 8 November 2026

Lima raw 1,20× `:v1` dan teacher YOLO26m `v50` asli dipakai pada 15 smoke diagnostik: Native KD, CrossKD, dan MGD × seed 42/0/2024/7/3407. Setiap kasus menyelesaikan satu epoch CPU FP32, 64 px, batch 2, dengan fixture enam train/enam val. Semua paket pada tabel berikut sudah diunduh dan SHA-256 cocok. Angka smoke tidak mengukur akurasi Poltrada V3 atau latency CUDA.

Nama seluruh koleksi mengikuti `yolo26s-lamp-su1.20x-seed<seed>-<metode>-smoke-<jenis>`. Best/last hanya berisi student native FP32 dengan struktur pruned; snapshot menyimpan state training/adapternya. [Bukti lengkap](../reports/pruning/recovery/recovery_custom_eda_verification_20261108.json) · [Semua versi dan hash file, termasuk riwayat restore](../reports/pruning/recovery/recovery_wandb_inventory_20261108.json).

| Metode | Seed | Model best/last | Hasil | Snapshot | EDA |
|---|---:|---|---|---|---|
| crosskd | 42 | [`yolo26s-lamp-su1.20x-seed42-crosskd-smoke-checkpoints:v2`](https://wandb.ai/ilhamrafiqin-brawijaya-university/yolom26-kd-indotraffic/artifacts/model/yolo26s-lamp-su1.20x-seed42-crosskd-smoke-checkpoints/v2) | [`yolo26s-lamp-su1.20x-seed42-crosskd-smoke-results:v2`](https://wandb.ai/ilhamrafiqin-brawijaya-university/yolom26-kd-indotraffic/artifacts/results/yolo26s-lamp-su1.20x-seed42-crosskd-smoke-results/v2) | [`yolo26s-lamp-su1.20x-seed42-crosskd-smoke-resume:v1`](https://wandb.ai/ilhamrafiqin-brawijaya-university/yolom26-kd-indotraffic/artifacts/model/yolo26s-lamp-su1.20x-seed42-crosskd-smoke-resume/v1) | [`yolo26s-lamp-su1.20x-seed42-crosskd-smoke-eda:v1`](https://wandb.ai/ilhamrafiqin-brawijaya-university/yolom26-kd-indotraffic/artifacts/results/yolo26s-lamp-su1.20x-seed42-crosskd-smoke-eda/v1) |
| crosskd | 0 | [`yolo26s-lamp-su1.20x-seed0-crosskd-smoke-checkpoints:v0`](https://wandb.ai/ilhamrafiqin-brawijaya-university/yolom26-kd-indotraffic/artifacts/model/yolo26s-lamp-su1.20x-seed0-crosskd-smoke-checkpoints/v0) | [`yolo26s-lamp-su1.20x-seed0-crosskd-smoke-results:v0`](https://wandb.ai/ilhamrafiqin-brawijaya-university/yolom26-kd-indotraffic/artifacts/results/yolo26s-lamp-su1.20x-seed0-crosskd-smoke-results/v0) | [`yolo26s-lamp-su1.20x-seed0-crosskd-smoke-resume:v1`](https://wandb.ai/ilhamrafiqin-brawijaya-university/yolom26-kd-indotraffic/artifacts/model/yolo26s-lamp-su1.20x-seed0-crosskd-smoke-resume/v1) | [`yolo26s-lamp-su1.20x-seed0-crosskd-smoke-eda:v1`](https://wandb.ai/ilhamrafiqin-brawijaya-university/yolom26-kd-indotraffic/artifacts/results/yolo26s-lamp-su1.20x-seed0-crosskd-smoke-eda/v1) |
| crosskd | 2024 | [`yolo26s-lamp-su1.20x-seed2024-crosskd-smoke-checkpoints:v0`](https://wandb.ai/ilhamrafiqin-brawijaya-university/yolom26-kd-indotraffic/artifacts/model/yolo26s-lamp-su1.20x-seed2024-crosskd-smoke-checkpoints/v0) | [`yolo26s-lamp-su1.20x-seed2024-crosskd-smoke-results:v0`](https://wandb.ai/ilhamrafiqin-brawijaya-university/yolom26-kd-indotraffic/artifacts/results/yolo26s-lamp-su1.20x-seed2024-crosskd-smoke-results/v0) | [`yolo26s-lamp-su1.20x-seed2024-crosskd-smoke-resume:v0`](https://wandb.ai/ilhamrafiqin-brawijaya-university/yolom26-kd-indotraffic/artifacts/model/yolo26s-lamp-su1.20x-seed2024-crosskd-smoke-resume/v0) | [`yolo26s-lamp-su1.20x-seed2024-crosskd-smoke-eda:v0`](https://wandb.ai/ilhamrafiqin-brawijaya-university/yolom26-kd-indotraffic/artifacts/results/yolo26s-lamp-su1.20x-seed2024-crosskd-smoke-eda/v0) |
| crosskd | 7 | [`yolo26s-lamp-su1.20x-seed7-crosskd-smoke-checkpoints:v0`](https://wandb.ai/ilhamrafiqin-brawijaya-university/yolom26-kd-indotraffic/artifacts/model/yolo26s-lamp-su1.20x-seed7-crosskd-smoke-checkpoints/v0) | [`yolo26s-lamp-su1.20x-seed7-crosskd-smoke-results:v0`](https://wandb.ai/ilhamrafiqin-brawijaya-university/yolom26-kd-indotraffic/artifacts/results/yolo26s-lamp-su1.20x-seed7-crosskd-smoke-results/v0) | [`yolo26s-lamp-su1.20x-seed7-crosskd-smoke-resume:v0`](https://wandb.ai/ilhamrafiqin-brawijaya-university/yolom26-kd-indotraffic/artifacts/model/yolo26s-lamp-su1.20x-seed7-crosskd-smoke-resume/v0) | [`yolo26s-lamp-su1.20x-seed7-crosskd-smoke-eda:v0`](https://wandb.ai/ilhamrafiqin-brawijaya-university/yolom26-kd-indotraffic/artifacts/results/yolo26s-lamp-su1.20x-seed7-crosskd-smoke-eda/v0) |
| crosskd | 3407 | [`yolo26s-lamp-su1.20x-seed3407-crosskd-smoke-checkpoints:v0`](https://wandb.ai/ilhamrafiqin-brawijaya-university/yolom26-kd-indotraffic/artifacts/model/yolo26s-lamp-su1.20x-seed3407-crosskd-smoke-checkpoints/v0) | [`yolo26s-lamp-su1.20x-seed3407-crosskd-smoke-results:v0`](https://wandb.ai/ilhamrafiqin-brawijaya-university/yolom26-kd-indotraffic/artifacts/results/yolo26s-lamp-su1.20x-seed3407-crosskd-smoke-results/v0) | [`yolo26s-lamp-su1.20x-seed3407-crosskd-smoke-resume:v0`](https://wandb.ai/ilhamrafiqin-brawijaya-university/yolom26-kd-indotraffic/artifacts/model/yolo26s-lamp-su1.20x-seed3407-crosskd-smoke-resume/v0) | [`yolo26s-lamp-su1.20x-seed3407-crosskd-smoke-eda:v0`](https://wandb.ai/ilhamrafiqin-brawijaya-university/yolom26-kd-indotraffic/artifacts/results/yolo26s-lamp-su1.20x-seed3407-crosskd-smoke-eda/v0) |
| mgd | 42 | [`yolo26s-lamp-su1.20x-seed42-mgd-smoke-checkpoints:v2`](https://wandb.ai/ilhamrafiqin-brawijaya-university/yolom26-kd-indotraffic/artifacts/model/yolo26s-lamp-su1.20x-seed42-mgd-smoke-checkpoints/v2) | [`yolo26s-lamp-su1.20x-seed42-mgd-smoke-results:v2`](https://wandb.ai/ilhamrafiqin-brawijaya-university/yolom26-kd-indotraffic/artifacts/results/yolo26s-lamp-su1.20x-seed42-mgd-smoke-results/v2) | [`yolo26s-lamp-su1.20x-seed42-mgd-smoke-resume:v1`](https://wandb.ai/ilhamrafiqin-brawijaya-university/yolom26-kd-indotraffic/artifacts/model/yolo26s-lamp-su1.20x-seed42-mgd-smoke-resume/v1) | [`yolo26s-lamp-su1.20x-seed42-mgd-smoke-eda:v1`](https://wandb.ai/ilhamrafiqin-brawijaya-university/yolom26-kd-indotraffic/artifacts/results/yolo26s-lamp-su1.20x-seed42-mgd-smoke-eda/v1) |
| mgd | 0 | [`yolo26s-lamp-su1.20x-seed0-mgd-smoke-checkpoints:v0`](https://wandb.ai/ilhamrafiqin-brawijaya-university/yolom26-kd-indotraffic/artifacts/model/yolo26s-lamp-su1.20x-seed0-mgd-smoke-checkpoints/v0) | [`yolo26s-lamp-su1.20x-seed0-mgd-smoke-results:v0`](https://wandb.ai/ilhamrafiqin-brawijaya-university/yolom26-kd-indotraffic/artifacts/results/yolo26s-lamp-su1.20x-seed0-mgd-smoke-results/v0) | [`yolo26s-lamp-su1.20x-seed0-mgd-smoke-resume:v0`](https://wandb.ai/ilhamrafiqin-brawijaya-university/yolom26-kd-indotraffic/artifacts/model/yolo26s-lamp-su1.20x-seed0-mgd-smoke-resume/v0) | [`yolo26s-lamp-su1.20x-seed0-mgd-smoke-eda:v0`](https://wandb.ai/ilhamrafiqin-brawijaya-university/yolom26-kd-indotraffic/artifacts/results/yolo26s-lamp-su1.20x-seed0-mgd-smoke-eda/v0) |
| mgd | 2024 | [`yolo26s-lamp-su1.20x-seed2024-mgd-smoke-checkpoints:v0`](https://wandb.ai/ilhamrafiqin-brawijaya-university/yolom26-kd-indotraffic/artifacts/model/yolo26s-lamp-su1.20x-seed2024-mgd-smoke-checkpoints/v0) | [`yolo26s-lamp-su1.20x-seed2024-mgd-smoke-results:v0`](https://wandb.ai/ilhamrafiqin-brawijaya-university/yolom26-kd-indotraffic/artifacts/results/yolo26s-lamp-su1.20x-seed2024-mgd-smoke-results/v0) | [`yolo26s-lamp-su1.20x-seed2024-mgd-smoke-resume:v0`](https://wandb.ai/ilhamrafiqin-brawijaya-university/yolom26-kd-indotraffic/artifacts/model/yolo26s-lamp-su1.20x-seed2024-mgd-smoke-resume/v0) | [`yolo26s-lamp-su1.20x-seed2024-mgd-smoke-eda:v0`](https://wandb.ai/ilhamrafiqin-brawijaya-university/yolom26-kd-indotraffic/artifacts/results/yolo26s-lamp-su1.20x-seed2024-mgd-smoke-eda/v0) |
| mgd | 7 | [`yolo26s-lamp-su1.20x-seed7-mgd-smoke-checkpoints:v0`](https://wandb.ai/ilhamrafiqin-brawijaya-university/yolom26-kd-indotraffic/artifacts/model/yolo26s-lamp-su1.20x-seed7-mgd-smoke-checkpoints/v0) | [`yolo26s-lamp-su1.20x-seed7-mgd-smoke-results:v0`](https://wandb.ai/ilhamrafiqin-brawijaya-university/yolom26-kd-indotraffic/artifacts/results/yolo26s-lamp-su1.20x-seed7-mgd-smoke-results/v0) | [`yolo26s-lamp-su1.20x-seed7-mgd-smoke-resume:v0`](https://wandb.ai/ilhamrafiqin-brawijaya-university/yolom26-kd-indotraffic/artifacts/model/yolo26s-lamp-su1.20x-seed7-mgd-smoke-resume/v0) | [`yolo26s-lamp-su1.20x-seed7-mgd-smoke-eda:v0`](https://wandb.ai/ilhamrafiqin-brawijaya-university/yolom26-kd-indotraffic/artifacts/results/yolo26s-lamp-su1.20x-seed7-mgd-smoke-eda/v0) |
| mgd | 3407 | [`yolo26s-lamp-su1.20x-seed3407-mgd-smoke-checkpoints:v0`](https://wandb.ai/ilhamrafiqin-brawijaya-university/yolom26-kd-indotraffic/artifacts/model/yolo26s-lamp-su1.20x-seed3407-mgd-smoke-checkpoints/v0) | [`yolo26s-lamp-su1.20x-seed3407-mgd-smoke-results:v0`](https://wandb.ai/ilhamrafiqin-brawijaya-university/yolom26-kd-indotraffic/artifacts/results/yolo26s-lamp-su1.20x-seed3407-mgd-smoke-results/v0) | [`yolo26s-lamp-su1.20x-seed3407-mgd-smoke-resume:v0`](https://wandb.ai/ilhamrafiqin-brawijaya-university/yolom26-kd-indotraffic/artifacts/model/yolo26s-lamp-su1.20x-seed3407-mgd-smoke-resume/v0) | [`yolo26s-lamp-su1.20x-seed3407-mgd-smoke-eda:v0`](https://wandb.ai/ilhamrafiqin-brawijaya-university/yolom26-kd-indotraffic/artifacts/results/yolo26s-lamp-su1.20x-seed3407-mgd-smoke-eda/v0) |
| native | 42 | [`yolo26s-lamp-su1.20x-seed42-native-smoke-checkpoints:v5`](https://wandb.ai/ilhamrafiqin-brawijaya-university/yolom26-kd-indotraffic/artifacts/model/yolo26s-lamp-su1.20x-seed42-native-smoke-checkpoints/v5) | [`yolo26s-lamp-su1.20x-seed42-native-smoke-results:v5`](https://wandb.ai/ilhamrafiqin-brawijaya-university/yolom26-kd-indotraffic/artifacts/results/yolo26s-lamp-su1.20x-seed42-native-smoke-results/v5) | [`yolo26s-lamp-su1.20x-seed42-native-smoke-resume:v2`](https://wandb.ai/ilhamrafiqin-brawijaya-university/yolom26-kd-indotraffic/artifacts/model/yolo26s-lamp-su1.20x-seed42-native-smoke-resume/v2) | [`yolo26s-lamp-su1.20x-seed42-native-smoke-eda:v1`](https://wandb.ai/ilhamrafiqin-brawijaya-university/yolom26-kd-indotraffic/artifacts/results/yolo26s-lamp-su1.20x-seed42-native-smoke-eda/v1) |
| native | 0 | [`yolo26s-lamp-su1.20x-seed0-native-smoke-checkpoints:v1`](https://wandb.ai/ilhamrafiqin-brawijaya-university/yolom26-kd-indotraffic/artifacts/model/yolo26s-lamp-su1.20x-seed0-native-smoke-checkpoints/v1) | [`yolo26s-lamp-su1.20x-seed0-native-smoke-results:v0`](https://wandb.ai/ilhamrafiqin-brawijaya-university/yolom26-kd-indotraffic/artifacts/results/yolo26s-lamp-su1.20x-seed0-native-smoke-results/v0) | [`yolo26s-lamp-su1.20x-seed0-native-smoke-resume:v2`](https://wandb.ai/ilhamrafiqin-brawijaya-university/yolom26-kd-indotraffic/artifacts/model/yolo26s-lamp-su1.20x-seed0-native-smoke-resume/v2) | [`yolo26s-lamp-su1.20x-seed0-native-smoke-eda:v1`](https://wandb.ai/ilhamrafiqin-brawijaya-university/yolom26-kd-indotraffic/artifacts/results/yolo26s-lamp-su1.20x-seed0-native-smoke-eda/v1) |
| native | 2024 | [`yolo26s-lamp-su1.20x-seed2024-native-smoke-checkpoints:v0`](https://wandb.ai/ilhamrafiqin-brawijaya-university/yolom26-kd-indotraffic/artifacts/model/yolo26s-lamp-su1.20x-seed2024-native-smoke-checkpoints/v0) | [`yolo26s-lamp-su1.20x-seed2024-native-smoke-results:v0`](https://wandb.ai/ilhamrafiqin-brawijaya-university/yolom26-kd-indotraffic/artifacts/results/yolo26s-lamp-su1.20x-seed2024-native-smoke-results/v0) | [`yolo26s-lamp-su1.20x-seed2024-native-smoke-resume:v0`](https://wandb.ai/ilhamrafiqin-brawijaya-university/yolom26-kd-indotraffic/artifacts/model/yolo26s-lamp-su1.20x-seed2024-native-smoke-resume/v0) | [`yolo26s-lamp-su1.20x-seed2024-native-smoke-eda:v0`](https://wandb.ai/ilhamrafiqin-brawijaya-university/yolom26-kd-indotraffic/artifacts/results/yolo26s-lamp-su1.20x-seed2024-native-smoke-eda/v0) |
| native | 7 | [`yolo26s-lamp-su1.20x-seed7-native-smoke-checkpoints:v0`](https://wandb.ai/ilhamrafiqin-brawijaya-university/yolom26-kd-indotraffic/artifacts/model/yolo26s-lamp-su1.20x-seed7-native-smoke-checkpoints/v0) | [`yolo26s-lamp-su1.20x-seed7-native-smoke-results:v0`](https://wandb.ai/ilhamrafiqin-brawijaya-university/yolom26-kd-indotraffic/artifacts/results/yolo26s-lamp-su1.20x-seed7-native-smoke-results/v0) | [`yolo26s-lamp-su1.20x-seed7-native-smoke-resume:v0`](https://wandb.ai/ilhamrafiqin-brawijaya-university/yolom26-kd-indotraffic/artifacts/model/yolo26s-lamp-su1.20x-seed7-native-smoke-resume/v0) | [`yolo26s-lamp-su1.20x-seed7-native-smoke-eda:v0`](https://wandb.ai/ilhamrafiqin-brawijaya-university/yolom26-kd-indotraffic/artifacts/results/yolo26s-lamp-su1.20x-seed7-native-smoke-eda/v0) |
| native | 3407 | [`yolo26s-lamp-su1.20x-seed3407-native-smoke-checkpoints:v0`](https://wandb.ai/ilhamrafiqin-brawijaya-university/yolom26-kd-indotraffic/artifacts/model/yolo26s-lamp-su1.20x-seed3407-native-smoke-checkpoints/v0) | [`yolo26s-lamp-su1.20x-seed3407-native-smoke-results:v0`](https://wandb.ai/ilhamrafiqin-brawijaya-university/yolom26-kd-indotraffic/artifacts/results/yolo26s-lamp-su1.20x-seed3407-native-smoke-results/v0) | [`yolo26s-lamp-su1.20x-seed3407-native-smoke-resume:v0`](https://wandb.ai/ilhamrafiqin-brawijaya-university/yolom26-kd-indotraffic/artifacts/model/yolo26s-lamp-su1.20x-seed3407-native-smoke-resume/v0) | [`yolo26s-lamp-su1.20x-seed3407-native-smoke-eda:v0`](https://wandb.ai/ilhamrafiqin-brawijaya-university/yolom26-kd-indotraffic/artifacts/results/yolo26s-lamp-su1.20x-seed3407-native-smoke-eda/v0) |

| Metode | Seed | SHA-256 best.pt | SHA-256 last.pt |
|---|---:|---|---|
| crosskd | 42 | `62d44d815654ffd855a0859dc15f9abf7a85f3e2fa15f11ab52aa9aa855c3fa1` | `4dfb7d2b2add0815ca40daa515f748b3325675f3bba625d7a5ee6a749e5ff012` |
| crosskd | 0 | `578363e61a867bd5c103c28a533af0a123eedb78459630e93a900e7af8dadbfc` | `ce56537acabaadfecbf4f0b53f2ac3e7590bf893033e4d85aee5f0d7d423b53f` |
| crosskd | 2024 | `3228a0baa2857171c085855775cb0943ce624051f53b2e74cfd989cfd1214848` | `9fdc8773dcccefd5b258bf0fa6eba8b7b8062ad4365fb784fe35b47f54409b58` |
| crosskd | 7 | `981d681771c874e2b27efbdd8ec97af7ae7a50c6fb5a4c6d0cc0235f026fe432` | `05ab23d7e55c6c1820c1061c038fb42ab2003141b11dd5264a3b93edd2b6642b` |
| crosskd | 3407 | `416ff72f447890901cfb26cdde5c9a044fb300e8483b43d2c16112903f6b8e8e` | `b98c32ea12db65be29db9e60855c8fad515e700bc1e911be04783cc1bba9d2e8` |
| mgd | 42 | `9df7811ea4e82efd8c808b45ab13fb6a06e22fb960ee87fd1549e0856e98c919` | `6f6968d4f72ebd16bb0f1bc0484709b1c678073ea87f15a6473571a9dc0f8d3b` |
| mgd | 0 | `306d4e611a8c4b6dc4d75343fad9db5989e2084c7a008ed660e81824d62ec748` | `7e938bfadba7e957b2076aef01853885ff8f76644928aaf5224a138bd47d9eeb` |
| mgd | 2024 | `3a5230856f751bf13685220b54316d0506037dde89128e8d6dfac9a01a515d5f` | `c332384b80a03376bbaeb03f576d16cf7d239046bbc634c749337c81041cf87d` |
| mgd | 7 | `33b917914df3bc41f6e1634654d2fa146f3cd8f28a504dd08015c15ddd973c4c` | `552738e12166f9c4ee571b6a669a5f0753d9f2a11445402a012cfb255cd6bf4e` |
| mgd | 3407 | `e68a8b97f8d45ce9ecae862a0ef6ca4ecdfe2ff19c7945df115588bdf80c170e` | `91336f8de73d921f67903717cba046932372d948188022a13c91198d0279a4ca` |
| native | 42 | `36d2000bdd715f9928c2b12135db01ad6f89fca1632b607aebae1d358ed7ece3` | `7019d01b3a25bcf4b9d0f26edcc6b7e6f96d8b6417448d9beabb3926a8d91455` |
| native | 0 | `d4500d78abfe82ba1cff5503bfc605dfbf889e37481954270b0caec7a9dc5fb7` | `ba62e58bcea28875ae020fc96ccce9ea8f0227229560146ae1a5036eab010bc2` |
| native | 2024 | `ae31c0160feaea8f84360046f09a094f96aaf8f911632a21bc33a6ccaf555cf2` | `e805724fa30833358647dcb5601519766be8fef566f6788462d4f44dc866a26d` |
| native | 7 | `7c58eac9b26d93a25baa503f4f6c1c6eb341894abef6ea30da43b3aafe484470` | `232c90ca275c61c4e3a05beed9330cd825688540b83f4aa1de2f87abb5a86818` |
| native | 3407 | `8a111796a6a3537db984ae4162280c17ef7cfbd6df10dc940ad6b0d16e17fbe5` | `6d1d76b79e0d40d072792600541bb5579da8b7fe246ce8644bcb140ea1e88192` |

Versi tambahan berikut berasal dari uji restore snapshot ke folder baru dengan ID run serta state satu epoch yang sama. Riwayat lama dipertahankan; tabel utama memakai hasil verifikasi terakhir.

| Versi model historis diagnostik | SHA-256 best.pt | SHA-256 last.pt |
|---|---|---|
| [`yolo26s-lamp-su1.20x-seed42-crosskd-smoke-checkpoints:v1`](https://wandb.ai/ilhamrafiqin-brawijaya-university/yolom26-kd-indotraffic/artifacts/model/yolo26s-lamp-su1.20x-seed42-crosskd-smoke-checkpoints/v1) | `62d44d815654ffd855a0859dc15f9abf7a85f3e2fa15f11ab52aa9aa855c3fa1` | `5f13d4e4915db1f3e75871542ece8e1ef87044f5a6ab04d2bb2448ace27c378e` |
| [`yolo26s-lamp-su1.20x-seed42-mgd-smoke-checkpoints:v1`](https://wandb.ai/ilhamrafiqin-brawijaya-university/yolom26-kd-indotraffic/artifacts/model/yolo26s-lamp-su1.20x-seed42-mgd-smoke-checkpoints/v1) | `9df7811ea4e82efd8c808b45ab13fb6a06e22fb960ee87fd1549e0856e98c919` | `2ae7b4d056b1736669523ea6f9c45e59e99a35bc8bf817fac11466b9a4fa5fbb` |
| [`yolo26s-lamp-su1.20x-seed42-native-smoke-checkpoints:v4`](https://wandb.ai/ilhamrafiqin-brawijaya-university/yolom26-kd-indotraffic/artifacts/model/yolo26s-lamp-su1.20x-seed42-native-smoke-checkpoints/v4) | `36d2000bdd715f9928c2b12135db01ad6f89fca1632b607aebae1d358ed7ece3` | `0940d062547c920a7264ca38709db5165a3d58bcd23a30ed5615588640866b60` |

### Fine-tuning varian Kaggle dengan input pulihan

Satu smoke online tambahan memakai raw seed 2024 `:v1` dan notebook Kaggle pada CPU lokal. Satu epoch, hash raw awal, pembaruan bobot, struktur native, 19 gambar custom, empat Tables custom, best/last, hasil, dan snapshot lolos verifikasi. EDA inti sebelumnya juga tetap dicatat.

- Model: [`yolo26s-lamp-su1.20x-seed2024-ft-smoke-checkpoints:v1`](https://wandb.ai/ilhamrafiqin-brawijaya-university/yolom26-kd-indotraffic/artifacts/model/yolo26s-lamp-su1.20x-seed2024-ft-smoke-checkpoints/v1); hasil: [`yolo26s-lamp-su1.20x-seed2024-ft-smoke-results:v1`](https://wandb.ai/ilhamrafiqin-brawijaya-university/yolom26-kd-indotraffic/artifacts/results/yolo26s-lamp-su1.20x-seed2024-ft-smoke-results/v1); snapshot: [`yolo26s-lamp-su1.20x-seed2024-ft-smoke-resume:v1`](https://wandb.ai/ilhamrafiqin-brawijaya-university/yolom26-kd-indotraffic/artifacts/model/yolo26s-lamp-su1.20x-seed2024-ft-smoke-resume/v1); EDA: [`yolo26s-lamp-su1.20x-seed2024-ft-smoke-eda:v0`](https://wandb.ai/ilhamrafiqin-brawijaya-university/yolom26-kd-indotraffic/artifacts/results/yolo26s-lamp-su1.20x-seed2024-ft-smoke-eda/v0).
- SHA-256 best: `f8294dc7d5db067b9c7938ab5ec9500b84143fc3cc5936c22cbd6aec217d32d2`; last: `51f8349a7d0077cc490391552d32b76b3fa6ab8f701a9dd857e86ef3ee9255bc`.

### EDA empat hasil fine-tuning penelitian yang sudah selesai

Keempat run pengguna berikut tetap mempunyai 100 epoch unik. Tabel sumber versi tetap diperiksa hash-nya, lalu ditambahkan 19 gambar dan empat Tables custom per run. Checkpoint dan training tidak diubah. Hash model berikut berasal dari metadata publikasi asli; audit ini tidak mengulang training/evaluasi penuh. Paket EDA diunduh ulang dan diverifikasi SHA-256.

| Run | Model versi tetap | EDA versi tetap | Report |
|---|---|---|---|
| `yolo26s-lamp-su1.20x-seed42-ft` | [`yolo26s-lamp-su1.20x-seed42-ft-checkpoints:v0`](https://wandb.ai/ilhamrafiqin-brawijaya-university/yolom26-kd-indotraffic/artifacts/model/yolo26s-lamp-su1.20x-seed42-ft-checkpoints/v0) | [`yolo26s-lamp-su1.20x-seed42-ft-eda:v3`](https://wandb.ai/ilhamrafiqin-brawijaya-university/yolom26-kd-indotraffic/artifacts/results/yolo26s-lamp-su1.20x-seed42-ft-eda/v3) | [tiga kondisi](https://wandb.ai/ilhamrafiqin-brawijaya-university/yolom26-kd-indotraffic/reports/EDA-·-yolo26s-lamp-su1.20x-seed42-ft-·-74b37d7f--VmlldzoxODI3MDQ2Mg==) |
| `yolo26s-lamp-su1.70x-seed42-ft` | [`yolo26s-lamp-su1.70x-seed42-ft-checkpoints:v0`](https://wandb.ai/ilhamrafiqin-brawijaya-university/yolom26-kd-indotraffic/artifacts/model/yolo26s-lamp-su1.70x-seed42-ft-checkpoints/v0) | [`yolo26s-lamp-su1.70x-seed42-ft-eda:v2`](https://wandb.ai/ilhamrafiqin-brawijaya-university/yolom26-kd-indotraffic/artifacts/results/yolo26s-lamp-su1.70x-seed42-ft-eda/v2) | [tiga kondisi](https://wandb.ai/ilhamrafiqin-brawijaya-university/yolom26-kd-indotraffic/reports/EDA-·-yolo26s-lamp-su1.70x-seed42-ft-·-74d5fa5a--VmlldzoxODI3MDQ2Ng==) |
| `yolo26s-lamp-su1.20x-seed0-ft` | [`yolo26s-lamp-su1.20x-seed0-ft-checkpoints:v0`](https://wandb.ai/ilhamrafiqin-brawijaya-university/yolom26-kd-indotraffic/artifacts/model/yolo26s-lamp-su1.20x-seed0-ft-checkpoints/v0) | [`yolo26s-lamp-su1.20x-seed0-ft-eda:v2`](https://wandb.ai/ilhamrafiqin-brawijaya-university/yolom26-kd-indotraffic/artifacts/results/yolo26s-lamp-su1.20x-seed0-ft-eda/v2) | [tiga kondisi](https://wandb.ai/ilhamrafiqin-brawijaya-university/yolom26-kd-indotraffic/reports/EDA-·-yolo26s-lamp-su1.20x-seed0-ft-·-72d597d7--VmlldzoxODI3MDQ2OA==) |
| `yolo26s-lamp-su1.10x-seed42-ft` | [`yolo26s-lamp-su1.10x-seed42-ft-checkpoints:v0`](https://wandb.ai/ilhamrafiqin-brawijaya-university/yolom26-kd-indotraffic/artifacts/model/yolo26s-lamp-su1.10x-seed42-ft-checkpoints/v0) | [`yolo26s-lamp-su1.10x-seed42-ft-eda:v2`](https://wandb.ai/ilhamrafiqin-brawijaya-university/yolom26-kd-indotraffic/artifacts/results/yolo26s-lamp-su1.10x-seed42-ft-eda/v2) | [tiga kondisi](https://wandb.ai/ilhamrafiqin-brawijaya-university/yolom26-kd-indotraffic/reports/EDA-·-yolo26s-lamp-su1.10x-seed42-ft-·-37210e0d--VmlldzoxODI3MDQ2OQ==) |

| Model penelitian pengguna | SHA-256 best.pt dari metadata | SHA-256 last.pt dari metadata |
|---|---|---|
| [`yolo26s-lamp-su1.20x-seed42-ft-checkpoints:v0`](https://wandb.ai/ilhamrafiqin-brawijaya-university/yolom26-kd-indotraffic/artifacts/model/yolo26s-lamp-su1.20x-seed42-ft-checkpoints/v0) | `b42324565225cf3df4d05896ccdb47535abfefef3faed65e40d3a3fdf3a16fd9` | `3e65163ce7c61c7c66a2136ab42942feff638b6f848432d935aa6ed76d1e89af` |
| [`yolo26s-lamp-su1.70x-seed42-ft-checkpoints:v0`](https://wandb.ai/ilhamrafiqin-brawijaya-university/yolom26-kd-indotraffic/artifacts/model/yolo26s-lamp-su1.70x-seed42-ft-checkpoints/v0) | `92bfd25cc722d3966c379b7451678722ac0df15a4983e0fc39063ee17e0e7c39` | `a50aa325fd1cdaae1010a471a3c669f7f039f0b91b7409a314d060f6d1ad33d8` |
| [`yolo26s-lamp-su1.20x-seed0-ft-checkpoints:v0`](https://wandb.ai/ilhamrafiqin-brawijaya-university/yolom26-kd-indotraffic/artifacts/model/yolo26s-lamp-su1.20x-seed0-ft-checkpoints/v0) | `a67c817b3a8b03a3db106cd759d2657fe558855b8ed291bf614e05aa0ac7b555` | `53451d8af0429609255ecc23db0454bf68d1371e8d6e61341ee378228e2221f3` |
| [`yolo26s-lamp-su1.10x-seed42-ft-checkpoints:v0`](https://wandb.ai/ilhamrafiqin-brawijaya-university/yolom26-kd-indotraffic/artifacts/model/yolo26s-lamp-su1.10x-seed42-ft-checkpoints/v0) | `e9f72915d7d2b60636c3c68d12ff5795ecb468f8e639f7b5ff2804dad4ecf61d` | `411e708215e2128b15a9e9040dfde73561576ad43ca280b85064dc67cfe55828` |

Buka [indeks EDA recovery](https://wandb.ai/ilhamrafiqin-brawijaya-university/yolom26-kd-indotraffic/reports/EDA--VmlldzoxODI3MDUxNg==) atau `custom_eda_report_url` pada Summary tiap run. Lima Native KD penelitian yang masih aktif hanya diberi report/tautan melalui API; tidak ada sesi training kedua. Batang final yang belum diukur tetap `NOT_RUN`. Source dan batas bukti tersedia pada [audit empat run selesai](../reports/pruning/recovery/finished_recovery_eda_audit_20261108.json).
