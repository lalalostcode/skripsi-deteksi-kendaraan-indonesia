# LAMP structured pruning V2 YOLO26n/S tanpa recovery dan EDA

[Notebook Kaggle 03](../notebooks/kaggle/pruning/03_lamp_yolo26n_extended_no_recovery.ipynb)
berdiri sendiri, dengan 14 bagian dan satu cell konfigurasi. Notebook 01/02 tetap dipertahankan.
Default `MODEL_VARIANTS=["n", "s"]` menjalankan **sepuluh eksperimen extended**: masing-masing
YOLO26n dan YOLO26s pada rasio 0,10/0,20/0,30/0,40/0,50. Setiap konfigurasi memulai dari baseline model
tersebut yang identik, bukan pruning kumulatif. `PRUNING_POLICY="both"` menambahkan conservative
untuk kedua model, sehingga totalnya 20 eksperimen. Tidak ada training, sparse training, backward,
optimizer, recovery, fine-tuning, atau KD.

Perluasan 3 November 2026 menambahkan rasio 40% dan 50% pada notebook yang sama.
Rasio adalah bagian channel eligible yang dihapus; ini tidak menyatakan persentase
pengurangan parameter, GFLOPs, atau ukuran file. Cap setiap group tetap 50%, dan setiap
rasio memuat ulang baseline asli. Rasio tinggi merupakan eksperimen kompresi tanpa recovery;
structural PASS tidak menyatakan akurasi terjaga. Laporan 10-30% di bawah tetap historis.

## Kaggle Run All

1. Impor notebook 03, aktifkan GPU dan Internet, lalu tambahkan **Poltrada V3 YOLO26 Clean**.
2. Aktifkan Kaggle Secret `WANDB_API_KEY` atau session W&B yang sudah login.
3. `MODEL_ARTIFACTS` memilih baseline per model dalam project
   `ilhamrafiqin-brawijaya-university/yolom26-kd-indotraffic`:
   N memakai `03_baseline_yolo26n-9uu3meve-checkpoints:v84`, sedangkan S memakai
   `02_baseline_yolo26s-q5kp9k9d-checkpoints:v62`. `best.pt` harus unik; selector ambigu
   menghentikan proses. Versi artifact bukan nomor epoch.
4. Periksa `DATASET_ROOT`/`DATA_YAML`, lalu Run All pada kernel baru. Uji cepat dapat memakai
   `MODEL_VARIANTS=["n"]` atau `["s"]`, `PRUNING_POLICY="extended"`, dan `PRUNING_RATIOS=[0.10]`.
   Full run sudah disetel ke `DEVICE="auto"`, input 640, batch 16, workers 2, validasi penuh,
   latency CUDA, tracking W&B, serta upload model/laporan. Default checkpoint lokal kosong
   sehingga artifact baseline diunduh langsung pada sesi Kaggle. Untuk checkpoint lokal,
   isi `LOCAL_CHECKPOINT_PATHS={"n": ..., "s": ...}` dan opsional hash yang telah diketahui
   pada `LOCAL_EXPECTED_SHA256S`; model yang dipilih tetap mencatat identitas artifact sumber.
   Hash salah atau checkpoint salah skala menghentikan konfigurasi.
5. Simpan seluruh direktori suite. Dataset tidak tersedia menghasilkan `NOT_RUN`;
   dataset tersedia tetapi invalid menghasilkan `FAIL`, tanpa penggantian dataset.

Pin: Ultralytics 8.4.155, Torch-Pruning 1.6.1, W&B 0.28.1. Bootstrap mempertahankan
Torch/torchvision runtime dan berhenti pada konflik. Pengukuran memakai model unfused FP32;
validator dan prediction memakai salinan terpisah. Baseline diukur/divalidasi satu kali
**per model**; hasil N tidak digunakan sebagai baseline S. Audit memeriksa skala dan stem
aktual (N: 16, S: 32), class order, dan kedua cabang head dari checkpoint.

## Policy dan importance

Conservative mereplikasi hidden Conv Bottleneck V1. Extended menyaring dependency group
backbone/neck melalui graph: group yang konsisten pada Conv/BN, residual addition,
concatenation, dan hidden SPPF dapat diterima. Tidak ada daftar indeks layer hardcoded.
Seluruh head one-to-many/one-to-one dan envelope attention tetap utuh. Output split CSP,
reshape, mapping invalid, operasi tidak dikenal, dan dimensi terlindungi ditolak eksplisit.
Group yang terikat pada dimensi output sama dikanonisasi sebelum ranking dan penghitungan.

Scorer tetap `LAMPImportance(p=2, group_reduction="mean", bias=False)` dari Torch-Pruning,
dipakai oleh `BasePruner` dengan global ranking, satu langkah, cap 50%, tanpa rounding.
Kandidat di luar kapasitas group diberi sentinel `+inf` sebelum threshold; skor LAMP
kandidat yang masih eligible tetap asli. Setiap proposal diperiksa lagi sebelum penghapusan.
Laporan membedakan rasio channel independen, perubahan fisik Conv-output, parameter,
GFLOPs, dan ukuran checkpoint FP32. Rasio sama pada kedua policy tidak berarti kompresi sama.

[LAMP asli](https://arxiv.org/abs/2010.07611) memakai magnitude bobot; normalisasi pada
dependency group merupakan adaptasi structured pruning. Source versi terpasang ikut dicatat.
Studi Gabriel Winasta Sinisuka, Rizal Setya Perdana, dan Irawati Nurmala Sari pada PDF lokal
`references/kompresi-model/paper/Kompresi Model YOLOv8 melalui Pruning, Sparsity, dan Knowledge Distillation untuk Deteksi Kendaraan dan Plat Nomor secara Real-Time pada Perangkat Edge.pdf` (hlm. 3–5) memakai **BNScale**, sparse training,
pruning YOLOv8 termasuk hidden head, dan KD recovery. Akurasi akhir setelah recovery
tidak menjadi target pembuktian LAMP tanpa recovery. Perubahan C2f_v2 YOLOv8 tidak disalin.

## Output dan status

Output Kaggle: `/kaggle/working/pruning/lamp_v2/{suite_id}/`;
padanan lokal: `outputs/pruning/lamp_v2/{suite_id}/`.

```text
suite_manifest.json, config.json, environment.json
yolo26n/
  baseline/               # pembanding FP32, provenance, audit, evaluasi N satu kali
  extended/pr0.1/         # juga pr0.2/pr0.3/pr0.4/pr0.5; checkpoint, laporan dan eda/
  conservative/pr0.1/     # opsional; juga pr0.2/pr0.3/pr0.4/pr0.5
yolo26s/
  baseline/               # pembanding FP32, provenance, audit, evaluasi S satu kali
  extended/pr0.1/         # juga pr0.2/pr0.3/pr0.4/pr0.5; checkpoint, laporan dan eda/
  conservative/pr0.1/     # opsional; juga pr0.2/pr0.3/pr0.4/pr0.5
comparison/               # ringkasan N/S, tabel, grafik dan reproducibility
```

Setiap konfigurasi menyimpan audit accepted/rejected group, inventory Conv/BN dengan
estimasi operasi per layer, CSV before/after, manifest, metrik, checkpoint pruned FP32,
`RESTORE.md`, fresh-reload reference/output/report, smoke status, dan tracking report.
Checkpoint memakai objek DetectionModel, `ema=None`, `optimizer=None`; tidak memakai YOLO.save().
Fresh Python tanpa PYTHONPATH repository memeriksa YOLO(checkpoint), prediction dan parity
kedua cabang pada `rtol=1e-4`, `atol=1e-5`. Checkpoint siap untuk studi recovery berikutnya;
perbandingan fine-tuning vs KD nanti harus memakai checkpoint dan budget sama.

Structural PASS memerlukan penghapusan fisik, parameter turun, head/attention tetap utuh,
output finite, save/reload dan prediction berhasil. Evaluation PASS hanya jika baseline dan
pruned benar-benar divalidasi pada dataset/protokol sama. Trade-off tidak memperoleh PASS
otomatis; hasil negatif tetap dilaporkan. Delta mAP fraction, poin persentase, dan perubahan
relatif dibedakan. Seed tunggal tidak membuktikan generalisasi statistik.

Angka Kaggle V1 pada brief tetap `UNVERIFIED`, ratio `UNKNOWN`, sampai manifest sumber
tersedia. Angka tersebut tidak menjadi titik pada grafik eksperimen terverifikasi.
Laporan historis dipertahankan di [indeks laporan](../reports/README.md).

## EDA channel, efisiensi, dan akurasi

Analisis disusun menurut identitas/cakupan, perubahan channel, keputusan LAMP,
efisiensi, lalu akurasi dan trade-off. Untuk setiap model/rasio, diagram channel memakai
**seluruh output Conv**, termasuk layer terlindungi yang tetap sama. Tiga panel memakai
urutan modul dan batas sumbu Y yang identik: **Before Pruning** biru, **After Pruning** merah,
dan **Channel Comparison** overlay biru/merah, sesuai contoh diagram pada brief.
Overview seluruh layer dilengkapi halaman detail maksimum `EDA_MAX_LAYERS_PER_PAGE=32`.
PNG `EDA_DPI=300`, SVG, dan CSV sumber disimpan; notebook menampilkan ringkasan semua rasio
dan detail `EDA_DISPLAY_RATIO=0.50` secara default.

Heatmap memakai indeks channel asli. Ledger mencatat pemetaan indeks asli ke indeks saat ini
untuk Conv-output, consumer-input, dan BN **sebelum** penghapusan, termasuk group yang terikat
residual atau concat. Jumlah channel independen dan perubahan fisik Conv-output dilaporkan
terpisah. Skor LAMP finite asli, mask kapasitas, dan keputusan terhapus/bertahan disimpan;
sentinel `+inf` dari cap tidak ditampilkan sebagai skor importance. EDA ini tidak mengukur
aktivasi feature map.

Tabel layer/stage merangkum channel, parameter, dan estimasi operasi; ringkasan model memuat
GFLOPs, ukuran checkpoint FP32, latency raw model forward dengan p05–p95, serta delta mAP
dalam poin persentase dan perubahan relatif. Grafik perbandingan N/S menandai model dan rasio
pada mAP versus parameter/GFLOPs/latency. Latency tidak mencakup preprocessing, NMS, atau I/O.

Evaluator memakai protokol yang sama: `val`, input 640, batch 16, workers 2, FP32,
confidence `0.001`, validation IoU `0.7`, maksimum 300, `nms=None`, tanpa augmentasi.
`plots=True` pada baseline dan pruned diperlukan agar SDK benar-benar mengisi confusion matrix.
Instance validator aktual ditangkap melalui callback `on_val_end`, karena Ultralytics 8.4.155
tidak menyimpan instance tersebut pada atribut `YOLO.validator`.

Ekspor `per_class_metrics.csv`, `curves.json`, `confusion_matrix.json`, dan
`metric_semantics.json` berada di direktori label validasi masing-masing. Hasil mencakup
precision/recall/F1/support per kelas, AP, dan kurva PR/F1/confidence dengan ID/nama kelas SDK.
Precision/recall/F1 memakai titik confidence terbaik menurut smoothed mean F1 SDK pada IoU
0,50; ini berbeda dari threshold confusion matrix. Matrix numerik berorientasi **baris predicted,
kolom true**, termasuk background. Confidence matrix dibaca dari validator aktual (`0.001` pada
konfigurasi ini), sedangkan IoU pencocokan berasal dari default `process_batch` aktual (`0.45`)
dan berbeda dari validation IoU `0.7`. Normalisasi dilakukan per kolom true; kolom kosong tetap
nol dan ditandai eksplisit. Support nol berarti tidak ada ground truth untuk kelas tersebut;
metrik tanpa baris SDK tetap `null`, bukan skor nol buatan.

`eda_status.json` membedakan grafik yang terukur, parsial, `NOT_RUN`, dan `UNAVAILABLE`.
Sampel kurva NaN/Inf menjadi `null`. Dataset/GPU yang tidak tersedia tidak menghasilkan angka
akurasi/latency; mapping atau skor dari laporan lama yang tidak pernah dicatat tetap
`UNAVAILABLE`. Hubungan perubahan channel dan mAP bersifat deskriptif, bukan pembuktian bahwa
suatu layer menyebabkan penurunan akurasi. Subdirektori EDA ikut artifact laporan W&B dengan
path relatif.

## Tracking

Project tetap `yolom26-kd-indotraffic`, satu UUID/run baru per konfigurasi, `resume="never"`.
Contoh nama extended `prun_lamp2_yolo26n_pruned-pr0.3`. Tambahkan `_c` setelah model
untuk conservative. Suite tetap tercatat di group. [Standar W&B](wandb_naming.md)
W&B mencatat baseline/pruned mAP, parameter, GFLOPs, delta, policy, rasio, seed, source hash,
dataset YAML hash, versi, dan protokol. Default mengunggah model serta laporan sebagai artifact
baru; upload dinyatakan berhasil setelah konfirmasi. Kegagalan jaringan tersimpan sebagai
tracking FAIL tanpa menghapus hasil lokal atau mengubah structural PASS yang valid.
Kredensial tidak masuk config/provenance/report.

## Verifikasi perluasan 40-50% — 3 November 2026

[Laporan CPU checkpoint asli N/S 40-50%](../reports/pruning/comparisons/lamp_v2_ns_ratio050_cpu_verification_20261103.json)
merekam empat konfigurasi extended FP32 pada input 640. Structural, save, fresh-process reload,
prediction, dan EDA memperoleh PASS; cap per group tetap 50%. Matriks sintetis menguji kedua
policy pada kelima rasio. Suite regresi menghasilkan 401 tes lulus pada run awal; satu asersi
jumlah job lama diperbarui dan lulus pada uji ulang, sehingga seluruh 402 kasus terselesaikan.
mAP Poltrada V3, latency CUDA, dan upload W&B nyata tidak dijalankan pada verifikasi CPU ini.
Notebook default tetap mengaktifkan evaluasi penuh, latency, serta tracking/upload untuk Kaggle.

## Status perluasan N/S — 24 Oktober 2026

Default N/S dan EDA merupakan versi baru notebook 03. Kedua checkpoint baseline asli telah
dijalankan pada CPU, FP32, 640 px, masing-masing dengan extended 0,10/0,20/0,30. Enam
konfigurasi memperoleh **structural PASS dan EDA PASS**: channel dihapus secara fisik,
parameter berkurang, head terlindungi tetap sama, output finite, save FP32, fresh-process
YOLO reload, dan prediction berhasil. Selisih maksimum seluruh raw/decoded outputs setelah
reload adalah nol. Setiap rasio memulai dari baseline model yang sama dan hash sumber
dicocokkan. Tidak ada training, backward, optimizer step, atau permintaan HTTP pada eksekusi ini.

| Model | Policy / target ratio | Parameter | Reduksi parameter | GFLOPs | Reduksi GFLOPs |
| --- | --- | ---: | ---: | ---: | ---: |
| YOLO26n | Baseline | 2.506.140 | 0% | 5,905408 | 0% |
| YOLO26n | Extended 0,10 | 2.236.276 | 10,77% | 5,571206 | 5,66% |
| YOLO26n | Extended 0,20 | 2.013.751 | 19,65% | 4,954506 | 16,10% |
| YOLO26n | Extended 0,30 | 1.800.137 | 28,17% | 4,411438 | 25,30% |
| YOLO26s | Baseline | 9.952.508 | 0% | 22,765978 | 0% |
| YOLO26s | Extended 0,10 | 8.858.156 | 11,00% | 21,470659 | 5,69% |
| YOLO26s | Extended 0,20 | 7.936.472 | 20,26% | 19,522470 | 14,25% |
| YOLO26s | Extended 0,30 | 7.075.142 | 28,91% | 17,348685 | 23,80% |

Denominator channel independen adalah 2.056 untuk N dan 4.112 untuk S. Pada 30%, masing-masing
menjadi 1.439 dan 2.878; integer channel dan group yang terikat membuat rasio aktual sedikit
berbeda dari target. Estimasi GFLOPs menggunakan counter yang sama pada baseline/pruned
unfused; ukuran checkpoint dalam laporan memakai format FP32 yang sama.

[Laporan integrasi CPU N/S 24 Oktober 2026](../reports/pruning/comparisons/lamp_v2_ns_cpu_verification_20261024.json)
memuat hash, versi, hasil per konfigurasi, status EDA, dan lokasi output aktual.
**mAP Poltrada V3 dan latency CUDA tetap `NOT_RUN`**, serta upload W&B nyata tidak dijalankan.
Tabel hanya membuktikan kompresi struktural dan pipeline, bukan kualitas deteksi atau trade-off
akurasi. Plot akurasi/latency tidak memperoleh nilai buatan.

Sebanyak **365 tes relevan lulus** untuk pruning V1/V2, N/S, evaluator, EDA, layout laporan/notebook,
dan generator Colab. Termasuk regresi conservative terhadap V1, ledger residual/concat,
mask kapasitas, guard tanpa learning, serta mock identitas run dan kegagalan upload W&B.

Pengujian evaluator terpisah pada dataset sintetis satu citra CPU memeriksa N dan S,
support kelas, empat kurva, serta false negative confusion matrix saat tidak ada deteksi.
Ini merupakan pengujian implementasi dan bukan pengukuran akurasi Poltrada V3. Angka historis
N di bawah tidak dipakai sebagai bukti EDA atau hasil S versi baru.

Indeks terpisah tersedia untuk [YOLO26n V2](../reports/pruning/yolo26n/v2/README.md) dan
[YOLO26s V2](../reports/pruning/yolo26s/v2/README.md). Notebook 01/02 dan bukti historis tetap
dipertahankan; hash maupun direktori eksekusi lama tidak ditulis ulang.

## Verifikasi historis YOLO26n — 22–23 Oktober 2026

Pada 23 Oktober 2026, default Kaggle disetel ke extended saja pada ketiga rasio.
Laporan verifikasi historis di bawah tetap merekam versi notebook dan konfigurasi `both`
yang benar-benar dieksekusi; hash serta hasilnya tidak ditulis ulang mengikuti perubahan default.

Pada 22 Oktober 2026, seluruh code cell notebook 03 dijalankan pada `best.pt` artifact target
yang tersedia lokal (SHA-256 `380a087fd621e8a43fe2b9b4d6b1bb6fb00a4d0443cc5afe81ef3efdf0775f0f`),
CPU FP32 640 px, tanpa network, instalasi, atau run W&B. Guard melarang training, backward,
optimizer step, dan permintaan HTTP. Semua enam konfigurasi **structural PASS**, termasuk
head/attention utuh, save FP32, fresh-process reload, prediction, serta parity kedua cabang.
Selisih maksimum raw/decoded outputs setelah reload adalah nol.

| Policy | Target ratio | Channel independen sebelum | Rasio aktual | Parameter | Reduksi parameter | GFLOPs | Reduksi GFLOPs |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Baseline | 0 | — | 0 | 2.506.140 | 0% | 5,905408 | 0% |
| Conservative | 0,10 | 440 | 0,100000 | 2.439.236 | 2,67% | 5,851814 | 0,91% |
| Conservative | 0,20 | 440 | 0,200000 | 2.372.332 | 5,34% | 5,795437 | 1,86% |
| Conservative | 0,30 | 440 | 0,300000 | 2.337.684 | 6,72% | 5,720320 | 3,13% |
| Extended | 0,10 | 2.056 | 0,100195 | 2.236.276 | 10,77% | 5,571206 | 5,66% |
| Extended | 0,20 | 2.056 | 0,200389 | 2.013.751 | 19,65% | 4,954506 | 16,10% |
| Extended | 0,30 | 2.056 | 0,300097 | 1.800.137 | 28,17% | 4,411438 | 25,30% |

Conservative memiliki 13 group, extended 35 group; denominator rasio menghitung dimensi
independen, bukan penjumlahan ulang output Conv yang terikat residual. Perbedaan rasio aktual
extended berasal dari integer channel dan threshold global.

**mAP target dan latency CUDA NOT_RUN**: Poltrada V3 dan GPU belum tersedia lokal. Tabel ini
menunjukkan perluasan kompresi yang valid secara struktural, bukan perbaikan trade-off akurasi.
GFLOPs merupakan estimasi dengan metode runtime tercatat; jangan menyamakan nilainya dengan
hasil environment Kaggle lain tanpa memeriksa versi/counter. Prediksi kosong pada citra sintetis
tidak menunjukkan akurasi deteksi kendaraan nyata.

Sebanyak **301 tes relevan lulus**, termasuk konservatif identik dengan V1 hingga indeks/tensor,
kedua mode output, enam konfigurasi CPU sintetis, class/dataset, partial pruning failure,
W&B upload failure, penamaan UUID, enam grafik, Colab generator, dan integritas migrasi laporan.
Seluruh 50 notebook lama tetap identik. Generator V2 membangun notebook yang identik dengan
file yang dikirim; notebook tidak mengimpor authoring source repository.

[Laporan verifikasi CPU](../reports/pruning/yolo26n/v2/lamp_v2_cpu_verification_20261022.json)
memuat versi, hash notebook/source, hasil setiap konfigurasi, lokasi output, dan bukti tes.
Ulangi verifikasi tanpa network dengan:

```powershell
.venv\Scripts\python.exe -B scripts/verify_lamp_v2_cpu.py --checkpoint outputs/lamp_target_artifact/artifact/best.pt --imgsz 640
```

Urutan eksperimen Kaggle yang layak: bandingkan hasil aktual kedua policy pada ketiga rasio;
prioritaskan pemeriksaan extended 10% sebelum menafsirkan extended 20%/30% yang lebih agresif.
Rekomendasi policy/rasio akhir harus menunggu mAP V3 dan latency GPU, bukan kompresi CPU saja.
