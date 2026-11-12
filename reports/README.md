# Indeks laporan penelitian

Laporan dikelompokkan berdasarkan eksperimen. File hasil historis mempertahankan angka,
hash, status pengujian, dan path eksekusi saat laporan dibuat. Direktori `outputs/`
tidak dipindahkan; tautan navigasi mengikuti lokasi laporan saat ini.

| Kategori | Isi dan lokasi |
| --- | --- |
| Baseline seed 42 | [Bundle evaluasi](baseline/seed42_evaluation/): tabel CSV, grafik, confusion matrix, dan visualisasi prediksi. |
| Baseline AMD | [Implementasi ROCm/Roboflow](baseline/amd/amd_baseline_roboflow_implementation.md). |
| EDA | [Implementasi Poltrada V3](eda/eda_poltrada_v3_implementation.md). |
| KD CSAKD | [Audit metode dan memori](kd/csakd/csakd_audit.md), [teacher S dan tracking](kd/csakd/csakd_teacher_s_and_tracking.md). |
| KD teacher S | [Implementasi metode teacher S](kd/teacher_s/teacher_s_kd_implementation.md). |
| KD MGD | [Implementasi MGD YOLO26s → YOLO26n](kd/mgd/mgd_yolo26n_implementation.md), protokol Native/CrossKD dan verifikasi lokal. |
| Pruning | [Indeks LAMP YOLO26n/YOLO26s](pruning/README.md), dipisahkan menurut model dan versi metode. |
| Bukti eksperimen | [Indeks Kaggle dan verifikasi Colab](experiments/README.md), [smoke MGD CPU dengan teacher v62 aktual](experiments/local/mgd_cpu_smoke_20261031.md). |
| Engineering | [Multi-seed serial](engineering/serial_multiseed_implementation.md), [penataan notebook](engineering/notebook_reorganization_20261022.json), [manifest migrasi laporan](engineering/report_reorganization_20261022.json). |

Manifest migrasi memuat 65 file asal, pemetaan lokasi lama ke baru, SHA-256 sebelum/sesudah,
dan perubahan href Markdown. Sebanyak 62 file dipertahankan identik; pada tiga Markdown
hanya tujuan tautan yang berubah. Path dan hash di dalam JSON/CSV historis tidak ditulis ulang.
Laporan baru, termasuk verifikasi LAMP V2, ditambahkan terpisah dari inventaris migrasi.

Lihat [indeks notebook](../notebooks/README.md) untuk menjalankan eksperimen.
