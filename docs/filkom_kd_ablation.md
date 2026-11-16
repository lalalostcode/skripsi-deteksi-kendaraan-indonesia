# Ablation bobot KD pada server FILKOM

Pertanyaan penelitian: bagaimana perubahan satu bobot distillation memengaruhi validation mAP50–95, dengan raw student, teacher, data, seed dan protokol tetap? Pembandingnya nilai default setiap metode. Satu seed mengukur sensitivitas konfigurasi, bukan signifikansi statistik atau variasi antar seed.

## Menjalankan satu kandidat

Gunakan [CrossKD 06](../notebooks/filkom/recovery/06_yolo26s_crosskd_kdscale_ablation.ipynb) atau [MGD 07](../notebooks/filkom/recovery/07_yolo26s_mgd_alpha_ablation.ipynb). Restart Kernel, ubah satu nilai di sel konfigurasi, lalu Run All. Setiap sesi menjalankan satu nilai, tanpa loop tiga training otomatis.

| Metode | Grid | Pilihan awal | Parameter lain |
| --- | --- | --- | --- |
| CrossKD | `KD_SCALE_VALUES = [0.5, 1.0, 2.0]` | `KD_SCALE = 1.0` | `KD_CLS_WEIGHT = KD_BOX_WEIGHT = KD_SCALE`, gamma 1 |
| MGD | `MGD_ALPHA_VALUES = [1e-5, 2e-5, 4e-5]` | `MGD_ALPHA = 2e-5` | mask ratio 0,65 |

CrossKD mempertahankan quality focal loss, confidence-weighted GIoU, normalisasi fitur dan interaksi cross-head dari recovery 04. MGD mempertahankan alignment 1×1, mask spasial dengan RNG CPU, generator conv3×3–ReLU–conv3×3 dan SSE/batch dari recovery 05. Hanya pengali loss berubah. Rujukan: [CrossKD penulis](https://github.com/jbwang1997/CrossKD), [MGD detection penulis](https://github.com/yzd-v/MGD/blob/2c9da0b28625eb948db57afc02c824452c3910fe/det/mmdet/distillation/losses/mgd.py). Adaptasi YOLO26 memakai graph dual-head dan tap P3/P4/P5 yang sudah ada; arsitektur standar student tidak dibangun ulang.

## Input dan environment terkunci

- Student seed **42**: `ilhamrafiqin-brawijaya-university/yolom26-kd-indotraffic/yolo26s-lamp-su1.20x-seed42-raw:v1`, SHA-256 `1c20b10f3a33de89377e1cf80e1ff244489fb9fc737b121735cd6d24618d7159`.
- Teacher: `ilhamrafiqin-brawijaya-university/yolom26-kd-indotraffic/01_teacher_yolo26m-fux5wdcb-checkpoints:v50`, SHA-256 `d749ef0baa0d6c3ae94e83ded1d9999d83c804b76ad5a841fc2ebec953af1428`.
- Training MGD FILKOM: 100 epoch, batch 16, 640 px, AdamW 0,001, weight decay 0,0005, warmup 3 epoch, close mosaic 10, patience 100, AMP aktif. Scheduler, loss detection, akumulasi, augmentasi dan default lain mengikuti Ultralytics 8.4.155. Pemilihan best mengikuti native validation fitness/EMA, termasuk aturan tie native.
- Python >=3.11, Ultralytics 8.4.155, W&B 0.28.1. Torch/torchvision CUDA harus tersedia di kernel dan dipertahankan saat bootstrap. Versi aktual, CUDA dan GPU masuk kontrak perbandingan. Tidak memaksakan versi Torch yang hanya dilaporkan tanpa inspeksi server.

Path mengikuti MGD FILKOM: workspace `~/skripsi-kd`, repo `repo`, data `datasets/roboflow-poltrada-v3/data.yaml`. Override: `SKRIPSI_ROOT`, `SKRIPSI_REPO_ROOT`, `SKRIPSI_DATASET_ROOT`, `SKRIPSI_DATA_YAML`. Key W&B dibaca dari environment lalu `REPO_ROOT/.env`; nilai rahasia tidak dicetak. [Setup kernel server](../notebooks/filkom/README.md).

Fingerprint train/validation aktual boleh berbeda dari snapshot raw pruning. Perbedaan tetap berstatus eksplisit pada manifest; provenance tidak dipaksa VERIFIED. Urutan kelas dan pemisahan train/val tetap diperiksa. Adapter label campuran menulis salinan polygon-ke-box per run, mempertahankan file sumber. Seluruh kandidat memakai label policy yang sama. Tidak menjalankan test split.

Run pertama menyimpan `research/protocol.json` di output ablation, mengunci fingerprint train/val, kelas, label policy, checkpoint, konfigurasi efektif dan environment. Kandidat berikutnya harus cocok. Memindahkan mount absolut diperbolehkan; perubahan isi dataset atau protokol memerlukan output suite baru melalui override, tanpa menulis ulang bukti lama. Smoke memakai lock `smoke/protocol.json` terpisah.

## Output, resume dan provenance

Default output: `~/skripsi-kd/outputs/filkom/recovery/ablation/<metode>/<parameter>/<execution-id>/`. Override `SKRIPSI_ABLATION_OUTPUT_DIR`; `SKRIPSI_OUTPUT_DIR` recovery utama tidak digunakan.

Nama run berbentuk `yolo26s-su1p20-seed42-crosskd-scale0p5-ablation-<execution-id>` atau `yolo26s-su1p20-seed42-mgd-alpha1e-5-ablation-<execution-id>`. Group: `crosskd-kdscale-ablation` dan `mgd-alpha-ablation`; smoke mendapat suffix terpisah. Pengulangan memakai execution ID baru; resume memakai ID run yang tersimpan.

Isi run mencakup best/last native pruned FP32, manifest, epoch history, audit smoke, konfigurasi efektif, kontrak resume, dataset/label manifest, EDA tiga kondisi dan resource segments. Artifact `<nama-run>-checkpoints`, `-results`, `-resume`, `-eda` dan `-ablation-record` memakai referensi `:vN`. Upload dan history/media W&B diperiksa ulang. Retensi best/latest hanya berlaku di eksekusi itu; cleanup smoke otomatis dinonaktifkan agar bukti diagnostik dapat diperiksa.

Set `RESUME_RUN_DIR` ke folder eksekusi lama atau `RESUME_SNAPSHOT_REF` ke versi artifact resume. Jangan mengubah kandidat/metode/input/protokol saat resume. Sel run memunculkan error bila status FAIL/PARTIAL, dengan alasan dan lokasi manifest. Checkpoint raw serta teacher tidak berubah. Smoke sementara memakai salinan model, lalu RNG dikembalikan; training normal tetap dimulai dari raw.

Default lama diaudit ketika `REUSE_EXISTING_DEFAULT=True`. Notebook mencari run recovery utama seed 42, mengunduh results/resume versi tetap, dan memeriksa hash serta metadata dataset, label, metode, protokol, environment, kelengkapan epoch dan best. Run yang identik boleh digunakan tanpa training ulang, dengan `default_reuse_audit.json` berisi URL dan URI asal. Metadata kurang, seed 0, atau perbedaan kondisi menyebabkan default dijalankan sebagai ablation baru. Paket yang tidak lagi tersedia tidak dianggap bukti identik.

## Membaca rekap

`research/ablation_registry.csv` dan `.json` memetakan semua percobaan dan statusnya. Setelah training selesai, aktifkan `LOAD_GROUP_RESULTS` dan jalankan sel terakhir. Sel ini juga dapat memuat registry artifact W&B yang terverifikasi. Grafik/tabel rekap dipublikasikan sebagai run analisis terpisah agar paket training final tetap immutable.

Rekap menampilkan tiga kandidat, termasuk NOT_RUN/FAIL/PARTIAL. Hanya hasil PASS, publikasi CONFIRMED, logging terverifikasi, 100 epoch dan signature setara yang ikut seleksi akhir. Untuk pengulangan nilai yang sama, gunakan eksekusi terverifikasi terbaru, bukan skor tertinggi dari beberapa percobaan.

Metrik utama berasal dari **evaluasi ulang best pada full validation** (`final_validation/*`), terpisah dari `metrics/mAP50-95(B)` pada kurva Ultralytics per epoch. Detection/KD loss diambil dari epoch best; komponen mentah dan berbobot disimpan. Waktu training adalah jumlah `epoch_seconds` termasuk validation selama training. Peak allocated/reserved CUDA dicatat per segmen resume; CPU dan resource historis tanpa bukti tetap kosong. Angka memory bukan kebutuhan inference.

Pilihan akhir menunggu ketiga hasil lengkap. Nilai seri dilaporkan semuanya. Bila pemenang berada di ujung grid, optimum global belum terbukti. Kegagalan menjalankan pipeline tidak sama dengan hasil evaluasi buruk.

## Verifikasi dan status penelitian

Generator: `python -m scripts.generate_kd_ablation_notebooks`. Uji: `python -m pytest tests/test_kd_ablation_notebooks.py`. Smoke enam kandidat dan resume: `python -m scripts.verify_kd_ablation_cpu --output outputs/kd_ablation_cpu_<id>`.

Verifier offline membutuhkan paket raw/teacher terverifikasi dan fixture CPU yang sudah tersedia lokal. Pengujian memakai CPU FP32 64 px, batch 2 dan citra sintetis; itu bukti kompatibilitas dan resume, bukan akurasi Poltrada. GPU training 100 epoch, full validation Poltrada serta publikasi online baru tetap **NOT_RUN** sampai dijalankan di server. Lihat [log penelitian](../reports/log-penelitian.md) untuk bukti dan batas hasil aktual.

[Bukti 16 November 2026](../reports/pruning/recovery/filkom_kd_ablation_verification_20261116.json): enam kandidat menyelesaikan smoke satu epoch, reload native FP32, validation fixture dan EDA. CrossKD dan MGD lulus perbandingan resume dua epoch, termasuk student/EMA, optimizer, scheduler, scaler, criterion, gradien dan adapter. Dua puluh tes kontrak ablation lulus; suite lintas modul memperoleh 279 lulus, satu skip dan satu kegagalan fixture retensi historis MGD yang sudah berbeda dari notebook HEAD tanpa perubahan. Semua 72 notebook lama dipertahankan. Tidak ada pemilihan hyperparameter penelitian dari hasil fixture ini.
