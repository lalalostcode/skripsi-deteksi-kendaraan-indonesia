# Eksperimen YOLO26n modular

Paket `skripsi` memisahkan arsitektur YOLO26n FPN-PAN, BiFPN, ASFF, dan AFPN dari notebook. Empat notebook baru ada di [`notebooks/modular/`](notebooks/modular/); notebook lama dan EDA tetap tersedia sebagai referensi. Import paket, `preflight()`, dan evaluasi **tidak memulai training**. Training hanya berjalan saat Anda memanggil `run_training()` secara eksplisit.

Notebook baseline lama masih memakai dataset v1. Untuk perbandingan empat arsitektur, gunakan hanya notebook modular yang mengunci dataset v2.

## Lokal

Gunakan Python 3.11 dan pasang proyek serta dependensinya:

```powershell
uv sync
```

Jalankan salah satu notebook baru dari checkout proyek. Jika dataset Roboflow v2 tersedia di `data/raw/poltrada-original-raw-2-v2` atau `data/audited_v2`, notebook akan menyalinnya ke area kerja sebelum normalisasi label. Kedua lokasi dataset lokal diabaikan oleh Git. Jika belum tersedia, isi `ROBOFLOW_API_KEY` di `.env` atau environment agar `preflight()` dapat mengunduhnya. `WANDB_API_KEY` opsional. Output lokal ada di `outputs/research/`.

## Kaggle

Setelah Anda commit dan push perubahan ini ke repository GitHub publik, impor salah satu notebook modular ke Kaggle. Aktifkan Internet dan GPU, lalu masukkan `ROBOFLOW_API_KEY` dan, bila dipakai, `WANDB_API_KEY` ke Kaggle Secrets. Sel bootstrap meng-clone branch `main`, memasang versi utama `ultralytics==8.4.155`, `wandb==0.28.1`, dan `roboflow==1.4.0`, serta mencatat commit yang benar-benar dipakai. Output Kaggle ada di `/kaggle/working/yolo26n_research/`.

Urutan sel: bootstrap → konfigurasi → `preflight(config)` → aktifkan `RUN_TRAINING` saat Anda hendak melakukan training → aktifkan `RUN_EVALUATION` setelah `best.pt` tersedia. Default kedua flag adalah `False`. Ubah hanya `active_seed` menjadi `42`, `123`, atau `2026` untuk run berikutnya. Satu seed memakai satu direktori run; run yang sudah berisi hasil tidak ditimpa.

Jika `WANDB_API_KEY` tersedia, setiap epoch mencatat seluruh loss train (`train/*`), loss dan metrik validasi (`val/*`, termasuk `val/metrics/*`), learning rate tiap grup parameter (`lr/*`), fitness validasi, nomor epoch, serta durasi. Metrik precision, recall, dan mAP pada fase training berasal dari validasi; Ultralytics tidak menghitung metrik tersebut pada citra train selama loop training. Validasi akhir checkpoint terbaik tidak dihitung sebagai epoch training tambahan.

## Protokol bersama

Semua arsitektur memakai dataset `poltrada-original-raw-2` **versi 2**, urutan enam kelas yang sama, audit dan fingerprint yang sama, serta pengaturan training yang sama. Normalisasi polygon dilakukan pada salinan kerja, disertai backup dan laporan. Audit menghentikan proses bila ada label/citra bermasalah, duplikasi citra lintas split, jumlah data yang berubah, atau fingerprint yang tidak cocok dengan snapshot v2.

Validation memilih confidence threshold berdasarkan macro F1. Test memakai threshold tersebut untuk Precision/Recall/F1; mAP50 dan mAP50-95 dihitung oleh Ultralytics. AP-S/AP-M/AP-L memakai evaluasi COCO pada kanvas letterbox 640×640 untuk **semua** arsitektur. Benchmark CUDA memakai 50 warmup dan 200 pengukuran; bila tidak ada CUDA, tabel efisiensi tetap menyimpan parameter/GFLOPs dan menandai latency sebagai tidak diukur.

Evaluasi juga menyimpan tabel 3.5–3.8, confusion matrix Ultralytics, kurva pemilihan threshold, dan diagnostik bobot fusi untuk BiFPN/ASFF/AFPN.

Setelah keempat arsitektur masing-masing memiliki hasil untuk seed 42, 123, dan 2026, panggil `summarize_experiments(output_root)` untuk membuat tabel perbandingan dan uji Friedman. Fungsi ini menulis status `not_run` bila masih ada run yang belum lengkap dan menolak fingerprint atau lingkungan pengukuran yang tidak konsisten.

Modul neck dan pembuatan model berada di [`src/skripsi/models/`](src/skripsi/models/). Fungsi data, training, dan evaluasi berada di [`src/skripsi/`](src/skripsi/). Checkpoint custom harus dibuka pada lingkungan yang sudah memasang paket `skripsi` pada commit yang tercatat di metadata run. Kode modul disimpan sebagai file Python biasa agar PyTorch dapat menemukan kelas custom saat memuat checkpoint.

## Pemeriksaan tanpa training

```powershell
$env:PYTHONPATH='src'
.venv\Scripts\python.exe -m pytest tests\test_modular_pipeline.py -q -p no:cacheprovider
```

Tes memakai dataset dan checkpoint sintetis untuk memeriksa salinan data, evaluator, forward pass keempat arsitektur, pemuatan checkpoint, dan format notebook. Tes tidak memanggil `train()`.

Uji integrasi evaluasi (juga tanpa training) dapat dijalankan terpisah dengan `-m slow`. Uji ini memanggil `model.val()` pada satu citra sintetis; pada Windows, Ultralytics memerlukan izin membuat pipe untuk pekerja pemindai dataset.
