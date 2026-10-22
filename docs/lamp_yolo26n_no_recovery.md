# LAMP structured pruning YOLO26n tanpa recovery

Notebook: `notebooks/kaggle/pruning/01_lamp_yolo26n_no_recovery.ipynb`.
Notebook berdiri sendiri: tidak memerlukan clone repository atau import `skripsi`/`inatrc_kd`.
Tidak ada perubahan pada notebook KD, generator, pipeline training, `pyproject.toml`, atau `uv.lock`.

## Menjalankan di Kaggle

1. Impor notebook, aktifkan Internet dan GPU, lalu tambahkan **Poltrada V3 YOLO26 Clean** yang dipakai baseline.
2. Aktifkan Kaggle Secret `WANDB_API_KEY`, atau gunakan session W&B yang sudah login. Tidak perlu menulis key ke cell.
3. Periksa cell konfigurasi. Artifact default adalah
   `ilhamrafiqin-brawijaya-university/yolom26-kd-indotraffic/03_baseline_yolo26n-9uu3meve-checkpoints:v84`.
   Selector eksplisit `best.pt` harus unik; isi path relatif lengkap bila ada beberapa file bernama sama.
4. Sesuaikan `DATASET_ROOT`, `DATA_YAML`, dan `OUTPUT_DIR`, kemudian **Run All** pada kernel baru.
5. Baca `structural_smoke_status` dan `validation_status` secara terpisah, lalu simpan folder output run.

Versi yang dikunci: Ultralytics 8.4.155, W&B 0.28.1, Torch-Pruning 1.6.1.
Torch/torchvision bawaan Kaggle dikunci ke versi yang sudah terpasang selama instalasi tambahan.
Konflik dependency menghentikan proses; notebook tidak mengganti pasangan Torch/torchvision secara otomatis.

Untuk salinan artifact yang sudah diunduh, isi `LOCAL_CHECKPOINT_PATH` dengan file atau direktorinya.
Isi `LOCAL_EXPECTED_SHA256` dari provenance download sebelumnya bila tersedia.
`WANDB_TRACKING=False` mengizinkan pengujian lokal tanpa login; asal salinan tanpa pembanding hash
dicatat sebagai `user_asserted_local_copy`. Notebook tidak menganggap nama file sebagai bukti asal artifact.

Default evaluasi: `val`, 640 px, batch 16, workers 2, confidence 0.001, IoU 0.7,
max_det 300, FP32, dan `nms=None`. Pada versi ini `nms=False` memilih head NMS-free;
nilainya tidak dapat dipertukarkan dengan `None`. Evaluasi memakai wrapper terpisah untuk kedua model,
sehingga fusion internal validator tidak mengubah model untuk pengukuran struktur.

## Metode dan perlindungan arsitektur

Scorer adalah `LAMPImportance(p=2, group_reduction="mean", bias=False)` bawaan Torch-Pruning,
digunakan oleh `BasePruner` dengan global ranking, target 10%, satu langkah, cap 50%, dan tanpa rounding.
Magnitude channel dari anggota dependency group direduksi dan dinormalisasi dengan LAMP.
Ini adaptasi **structured channel pruning** dari [LAMP asli](https://arxiv.org/abs/2010.07611),
bukan implementasi masking unstructured yang identik dengan paper tersebut.

Root eligible hanya hidden Conv pertama pada modul `Bottleneck` backbone/neck.
Penghapusan mencakup output Conv tersebut, BN terkait, serta input Conv konsumennya.
Seluruh dependency group diperiksa sebelum `group.prune()`; cap juga diperiksa eksplisit.
Graph dibangun pada `eval()` dengan autograd aktif dan raw outputs kedua cabang.
Tidak ada backward, optimizer, training, KD, teacher, atau recovery.

Yang dilindungi:

| Komponen | Perlindungan dan alasan |
| --- | --- |
| Detect one-to-many dan one-to-one | Seluruh parameter, buffer, serta dimensi output tetap identik. |
| CSP/C3k2 split dan output blok | Lebar split, concatenation, dan output residual tidak dipangkas. |
| Attention/C2PSA | Dimensi head, reshape, dan hidden split tidak diubah. |
| Antarmuka backbone/neck/head | Lebar fitur antartahap tetap; Conv konsumen internal hanya berubah pada input hidden channel. |

Terdapat 13 root eligible pada checkpoint target. Ranking LAMP memilih tiga group untuk diubah:

| Root | Hidden channel sebelum | Sesudah |
| --- | ---: | ---: |
| `model.22.m.0.0.cv1.conv` | 64 | 50 |
| `model.8.m.0.m.1.cv1.conv` | 64 | 49 |
| `model.8.m.0.m.0.cv1.conv` | 64 | 49 |

Total seluruh root eligible: 440 → 396 channel. Group lain tetap utuh karena ranking global,
bukan karena setiap layer harus kehilangan 10% channel. Laporan CSV mencakup semua Conv/BN,
termasuk perubahan input consumer dan alasan perlindungan.

GFLOPs diukur pada input resolusi penuh menggunakan THOP untuk Conv/BN dan counter matmul
attention Ultralytics secara terpisah. Cara ini menjaga child Conv attention tetap terhitung pada
THOP klasik. Digunakan konvensi dua operasi per multiply-add, identik sebelum/sesudah.
Versi distribution THOP dan signature API aktual dicatat; angka tersebut adalah estimasi operasi,
bukan pengukuran latency GPU.

## Save, restore, dan output

Checkpoint menyimpan objek `DetectionModel` dengan arsitektur hasil pruning dalam FP32,
`ema=None`, serta `optimizer=None`. Tidak memakai `YOLO.save()` yang mengonversi salinan model ke FP16.
State dict hasil pruning tidak dimuat ke arsitektur YOLO26n standar.

```python
from ultralytics import YOLO

model = YOLO("pruned_fp32.pt", task="detect")
results = model.predict("image.jpg", imgsz=640, nms=None, quantize=None)
```

Notebook mewajibkan restore pada proses Python baru, tanpa `PYTHONPATH` repository, dilanjutkan
forward kedua cabang dan `predict()`. Raw outputs dan decoded predictions dibandingkan dengan
`rtol=1e-4`, `atol=1e-5`. Prediksi kosong pada citra sintetis diperbolehkan jika kontrak tensor valid;
hasil tersebut tidak membuktikan deteksi kendaraan nyata.

Folder run baru menyimpan model pruned dan baseline pembanding FP32, `RESTORE.md`,
`verify_reload.py`, referensi/output reload, `config.json`, `environment.json`,
`checkpoint_provenance.json`, `architecture_audit.json`, `pruning_manifest.json`,
`layer_pruning_report.csv`, `before_after_metrics.json`, `smoke_test_report.json`,
ringkasan CSV, dan gambar before/after. Ukuran checkpoint sumber dicatat terpisah.

Laporan parsial disimpan saat gagal, lalu exception diteruskan. Dataset yang tidak tersedia
menghasilkan `validation_status=NOT_RUN`; dataset yang ada tetapi invalid menghasilkan `FAIL`.
Tidak ada metrik mAP buatan maupun fallback dataset.

## Verifikasi yang benar-benar dijalankan

Pada 22 Oktober 2026, artifact target diunduh melalui W&B API memakai kredensial environment proyek,
lalu **seluruh code cell notebook** dieksekusi pada salinan lokal `best.pt` dengan SHA-256 terverifikasi:

`380a087fd621e8a43fe2b9b4d6b1bb6fb00a4d0443cc5afe81ef3efdf0775f0f`.

Hasil CPU FP32 640 px: **structural smoke PASS**. Parameter berkurang dari 2.506.140 menjadi
2.439.236 (2,6696%); channel eligible berkurang tepat 10%. Kedua head tetap identik,
dan selisih absolut maksimum seluruh tensor pembanding setelah reload adalah nol.
Angka GFLOPs, provenance W&B, status lengkap, dan lokasi artifact hasil ada pada
`reports/lamp_yolo26n_cpu_smoke_20261022.json`.

Runtime: Python 3.11.14, Torch 2.14.0 CPU, torchvision 0.29.0, Ultralytics 8.4.155,
Torch-Pruning 1.6.1, W&B 0.28.1. Driver verifikasi menonaktifkan tracking W&B,
memakai batch satu/workers nol, dan memasang guard pada training, backward, serta optimizer step.
Jalur download `run.use_artifact()` di notebook diperiksa dengan mock; download asli memakai `wandb.Api`.

Checkpoint memiliki kelas `Bus`, `Mobil Penumpang`, `Pejalan Kaki`, `Sepeda Motor`, `Truck`,
dan `Unmotorized`, dibaca dari checkpoint. Dataset lokal `data/audited_v2` masih berlabel versi 2;
dataset Poltrada V3 yang tercatat pada training belum tersedia di environment lokal.
Karena itu **mAP target NOT_RUN**, bukan metrik dari dataset versi lain. CUDA dan latency GPU
juga **NOT_RUN**. Keduanya masih memerlukan pengujian Kaggle dengan dataset V3 yang benar.

Tes baru meliputi nbformat/sintaks, checkpoint ambiguity, SHA-256/provenance, identitas artifact,
urutan kelas, propagation error, finite/shape output, kedua mode output YOLO26, dependency group
terlindungi, penurunan parameter/GFLOPs, save/reload interpreter baru, serta evaluasi sintetis lokal.
Tes memakai model sintetis hanya untuk validasi implementasi dan tidak mengganti artifact eksperimen.

```powershell
# Install tambahan ini hanya diperlukan untuk menjalankan tes pruning opsional secara lokal.
uv pip install --python .venv\Scripts\python.exe --no-deps torch-pruning==1.6.1
$env:PYTHONPATH='src'
.venv\Scripts\python.exe -B -m pytest tests\test_lamp_pruning_notebook.py -m 'slow or not slow' -q -p no:cacheprovider
```

Tes lama yang diperiksa: `test_modular_pipeline.py`, `test_kd_notebook_tracking.py`, dan
`test_serial_multiseed_notebooks.py`: 71 lulus, satu skipped, tujuh slow deselected, dua gagal.
Kedua kegagalan berasal dari referensi lama ke `notebooks/modular`, yang sudah berada di
`notebooks/archieve/modular` sebelum pekerjaan ini. File lama tidak dipindahkan atau diperbaiki
oleh eksperimen pruning ini.

Setelah penataan folder notebook, lokasi aktif menjadi `notebooks/kaggle/pruning/`.
Isi notebook dan bukti pengujian CPU di atas tetap sama; laporan historis mempertahankan path saat pengujian.
Referensi dua tes modular kemudian diarahkan ke folder arsip yang sebenarnya. Lihat
`notebooks/README.md` untuk pemetaan lokasi lama ke baru.
