# Laporan implementasi EDA Poltrada V3

Tanggal dokumentasi: 14 Oktober 2026 (Asia/Jakarta).

## A. Ringkasan implementasi

Notebook mandiri dibuat pada `notebooks/yolo26-kd-indotraffic/notebooks/00_eda_poltrada_v3.ipynb`.
Notebook memuat 65 sel: 37 Markdown dan 28 kode. Seluruh keluaran empiris sengaja
kosong pada berkas yang diserahkan; angka penelitian baru dihitung setelah dataset
final tersedia dan notebook dijalankan. Tidak diperlukan modul Python proyek,
GPU, model YOLO, checkpoint, jaringan, atau W&B.

| Bagian | Fungsi |
|---|---|
| 0 | Tujuan, ruang lingkup deskriptif, verifikasi konfigurasi eksperimen, dan pemetaan acuan |
| 1 | Konfigurasi terpusat, versi lingkungan, fungsi statistik, audit, dan penyimpanan grafik |
| 2 | Validasi YAML, `nc`, `names`, partisi, resolusi path, dan sumber image/label |
| 3 | Parser YOLO, audit setiap label, pasangan file, metadata dan integritas gambar |
| 4 | Jumlah gambar, bbox valid, label lengkap, background dan status integritas per split |
| 5 | Instance, jumlah gambar per kelas, proporsi, dan rasio ketidakseimbangan |
| 6 | Distribusi jumlah objek: histogram, boxplot, ECDF dan statistik deskriptif |
| 7 | Resolusi, aspect ratio citra, ukuran file, brightness dan cakupan pengukuran |
| 8 | Dimensi bbox, luas, aspect ratio piksel, serta kategori ukuran relatif eksplisit |
| 9 | Heatmap pusat bbox gabungan dan per kelas |
| 10 | Jumlah objek versus luas, seluruh pasangan IoU pada cakupan biaya, jarak pusat terdekat |
| 11 | Ko-kemunculan kehadiran kelas per gambar dan Jaccard, gabungan/per split |
| 12 | Contoh ground truth: ekstrem, kelas jarang, contoh split dan sampel deterministik |
| 13 | Identitas path, SHA-256 opsional, konsistensi byte label, kandidat dHash opsional |
| 14 | Ringkasan audit dengan severity, tabel indikator dan interpretasi runtime |
| 15 | CSV mentah/deskriptif, PNG/SVG, laporan Markdown/JSON, provenance dan manifest |
| 16 | Kesimpulan runtime dan implikasi deskriptif bagi penelitian KD |

Konfigurasi kode notebook eksperimen 01–09 telah diperiksa: seluruhnya mengarah ke
`/kaggle/input/datasets/ilhamrafiqin/poltrada-v3-yolo26-clean/data.yaml`, memakai
`IMGSZ=640`, dan menggunakan loader Ultralytics. Nama kelas dan struktur partisi
final tidak diasumsikan dari versi dataset lain. YAML final Kaggle tidak tersedia
di lingkungan lokal; verifikasi isi YAML/direktori dilakukan pada runtime Bagian 2–3.
Kesembilan notebook eksperimen dibandingkan SHA-256 sebelum dan sesudah pekerjaan:
**PASS, seluruh byte tetap sama**. Perubahan pengguna yang sebelumnya ada pada
notebook 08–09 dan file audit/test CSAKD tetap dipertahankan.

## B. Pemetaan referensi

Dokumen lengkap skripsi Eldad Vikorian dan laporan Crowd Counting/KD tidak
disertakan. Pemetaan berikut mengikuti bagian dan halaman yang dijelaskan pengguna;
nomor halaman tersebut belum diverifikasi terhadap dokumen asli.

| Acuan yang disebut pengguna | Bagian EDA | Penyesuaian untuk object detection |
|---|---|---|
| Eldad, Bab III, Tabel 3.1 halaman 31 | 5 | Instance bbox dan gambar yang memuat kelas dihitung terpisah |
| Eldad, Bab III, Tabel 3.2 halaman 32 | 2, 4 | Partisi berasal dari YAML; test tidak dibuat jika tidak tercantum |
| Eldad, Bab V | 3, 7–12, 15 | Pembacaan label YOLO, audit, contoh bbox dan provenance |
| Eldad, Bab VI | 13–16 | Test deskriptif dipisahkan dari seleksi model dan metrik performa |
| Laporan KD, 3.8.1, halaman 13–19 | 2–3, 13 | Inventaris file, validitas bbox, konsistensi label dan duplikasi |
| Laporan KD, 3.8.2–3.8.3 | 6–7 | Jumlah objek beranotasi, resolusi dan brightness citra |
| Laporan KD, 3.8.4 | 8–9 | Pusat bbox ternormalisasi menggantikan titik kepala |
| Laporan KD, 3.8.5 | 10 | Jumlah objek, luas relatif, IoU dan jarak antarpusat menggantikan kepadatan crowd |
| Laporan KD, 3.8.6 | 12 | Ground truth bbox dengan nama kelas menggantikan titik anotasi |
| Laporan KD, 3.8.8 | 16 | Karakteristik data menjadi konteks pembahasan KD tanpa menyimpulkan performa |

Format teknis mengikuti [dokumentasi dataset deteksi Ultralytics](https://docs.ultralytics.com/datasets/detect/)
dan inspeksi loader pada instalasi proyek. Tidak ada angka acuan yang disalin atau
density map Gaussian yang dibuat.

## C. Daftar visualisasi dan sumber data

Status **IMPLEMENTED / PASS sintetik** menyatakan kode tersedia dan sudah dijalankan
pada fixture. Semua analisis dataset asli berstatus **NOT TESTED** sampai Run All
dijalankan pada Poltrada V3 final. EDA-04 berupa tabel sesuai spesifikasi pengguna.

| Kode | Grafik/tabel dan fungsi | Sumber runtime | Status |
|---|---|---|---|
| EDA-01 | Bar jumlah gambar dan tabel pembagian split | `image_inventory`, `bounding_boxes` | IMPLEMENTED / PASS sintetik |
| EDA-02 | Bar instance keseluruhan per kelas | `class_distribution` dari bbox valid | IMPLEMENTED / PASS sintetik |
| EDA-03 | Grouped bar kelas per split | `class_distribution` | IMPLEMENTED / PASS sintetik |
| EDA-04 | Tabel proporsi dan rasio ketidakseimbangan | `class_proportions`, `class_imbalance` | IMPLEMENTED / PASS sintetik |
| EDA-05 | Histogram, boxplot, ECDF objek per gambar | `image_inventory`, hanya label lengkap | IMPLEMENTED / PASS sintetik |
| EDA-06 | Scatter resolusi, histogram aspect ratio/ukuran file, tabel frekuensi | Metadata file dan citra terverifikasi | IMPLEMENTED / PASS sintetik |
| EDA-07 | Histogram brightness tiap split dan coverage | Brightness thumbnail citra sumber | IMPLEMENTED / PASS sintetik |
| EDA-08 | Scatter lebar/tinggi bbox ternormalisasi | `bounding_boxes` | IMPLEMENTED / PASS sintetik |
| EDA-09 | Histogram dan boxplot luas relatif | `bounding_boxes.area_relative` | IMPLEMENTED / PASS sintetik |
| EDA-10 | Histogram aspect ratio piksel per kelas | Dimensi bbox dan gambar asli | IMPLEMENTED / PASS sintetik |
| EDA-11 | Heatmap pusat bbox gabungan dan per kelas | `bounding_boxes`, `spatial_bins` | IMPLEMENTED / PASS sintetik |
| EDA-12 | Boxplot luas relatif per kelas dan tabel kategori | `bounding_boxes`, `relative_size_by_class` | IMPLEMENTED / PASS sintetik |
| EDA-13 | Scatter jumlah objek versus median luas dan Spearman | `density_per_image` dari label lengkap | IMPLEMENTED / PASS sintetik |
| EDA-14 | Heatmap jumlah ko-kemunculan dan Jaccard | Matriks kehadiran gambar×kelas berlabel lengkap | IMPLEMENTED / PASS sintetik |
| EDA-15 | Grid gambar asli dengan ground truth dan alasan pemilihan | Citra sumber, bbox valid, seed | IMPLEMENTED / PASS sintetik |
| EDA-16 | Stacked bar status label dan tabel audit severity | `image_inventory`, `annotation_issues` | IMPLEMENTED / PASS sintetik |

Tambahan: histogram IoU maksimum per gambar dan jarak pusat terdekat per bbox;
tabel SHA-256 dan dHash beserta cakupan; statistik Count, Mean, Median, SD sampel,
Min/Max, Q1/Q3/IQR, P5/P95. Grafik diekspor PNG 300 DPI dan SVG. Urutan dan warna
kelas mengikuti YAML. Figure Matplotlib ditutup setelah penyimpanan.

## D. Hasil audit dan pengujian

Pengujian tersimpan di `tests/test_eda_poltrada_v3.py`. Perintah pengujian lokal:

```powershell
.\.venv\Scripts\python.exe -m pytest tests/test_eda_poltrada_v3.py -q --basetemp tmp/pytest-eda-local
```

Gunakan direktori basetemp khusus pengujian yang belum memuat file penting;
pytest mengelola direktori tersebut. Pada verifikasi digunakan direktori unik
di dalam workspace.

| Cakupan | Status | Bukti/batas |
|---|---|---|
| Schema notebook nbformat dan sintaks seluruh 28 code cell | PASS | Seluruh sel valid; output dan execution_count berkas akhir kosong |
| Parser label valid/kosong/duplikat | PASS | Kotak valid identik tetap dihitung dan diberi warning |
| Kolom salah, nonnumerik, NaN/Inf, ID noninteger/di luar rentang | PASS | Baris invalid dilaporkan dan tidak masuk bbox analitis |
| Normalisasi, dimensi nol/negatif, tepi melewati gambar | PASS | Alasan invalid tersimpan tanpa clipping |
| xywh→xyxy piksel, luas, aspect ratio piksel/normalisasi | PASS | Gambar tidak persegi menguji perbedaan kedua rasio |
| IoU identik/terpisah dan jarak pusat | PASS | Seluruh pasangan dihitung pada gambar di bawah batas |
| Count/Median/SD/kuartil/IQR/P5 dan kehadiran/ko-kemunculan kelas | PASS | Nilai dibandingkan dengan hasil yang diketahui pada fixture |
| Pasangan image–label, label yatim, gambar rusak/label hilang | PASS | Status unknown tidak diperlakukan sebagai nol objek |
| YAML list/dictionary, ID kontigu, nama ganda, key duplikat | PASS | Schema gagal menghasilkan laporan kritis atau exception informatif |
| Split direktori/daftar, referensi berulang, fallback `../` | PASS | Path dan resolusi dicatat |
| Test tidak tercantum dan test dideklarasikan tetapi hilang | PASS | Tidak membuat split baru; sumber hilang menghasilkan error |
| Semua sel berurutan, mode standard tanpa test | PASS | Fixture memiliki background, invalid label, citra rusak, dan kelas nol |
| Semua sel berurutan, mode full dengan test/dHash | PASS | SHA-256 lintas split dan label byte berbeda terdeteksi; truncation dHash diuji |
| Semua sel berurutan, seluruh label kosong | PASS | Tidak mengarang bbox; grafik kosong menyatakan keterbatasan |
| PNG/SVG, CSV, Markdown, JSON standar dan manifest SHA-256 | PASS | Seluruh artefak run diverifikasi; PNG minimal 299 DPI terukur |
| Reproduksibilitas dengan input dan seed sama | PASS | Tabel inti, sampling, spatial bins dan fingerprint sama antar-run |
| Source fixture tidak berubah | PASS | Semua file sumber dibandingkan hash sebelum/sesudah eksekusi |
| Notebook eksperimen 01–09 tidak berubah | PASS | Perbandingan SHA-256 dengan baseline sebelum pekerjaan |
| Inspeksi visual keluaran fixture | PASS | Count/ECDF, ko-kemunculan, pusat per kelas, status label dan grid bbox diperiksa; tidak ditemukan label terpotong pada grafik yang diperiksa |
| Kernel Jupyter lokal melalui nbclient | NOT TESTED | Startup/eksekusi tidak selesai di lingkungan lokal; pengujian dibatasi pada eksekusi seluruh sel Python berurutan |
| YAML/names/partisi Poltrada V3 final | NOT TESTED | Dataset Kaggle final tidak tersedia lokal |
| Statistik, audit dan seluruh gambar Poltrada V3 asli | NOT TESTED | Tidak ada statistik Poltrada V3 yang dinyatakan dari fixture |
| Kaggle CPU dengan dataset asli | NOT TESTED | Belum ada eksekusi sesi Kaggle |

Hasil suite terakhir: **26 passed**, tanpa kegagalan yang tersisa. Pemeriksaan Ruff
kode notebook dan pengujian juga PASS; impor antarcell (`E402`) dan eksekusi
kode notebook tepercaya dalam test (`S102`) dikecualikan dalam pemeriksaan gabungan.

## E. Lokasi output

Output penelitian setelah Run All berada pada:

```text
/kaggle/working/eda_poltrada_v3/
├── figures/       # PNG 300 DPI dan SVG; 17 kelompok grafik pada fixture lengkap
├── tables/        # 42 CSV mentah/deskriptif pada fixture lengkap
├── reports/
│   ├── eda_summary.md
│   ├── eda_summary.json
│   ├── dataset_provenance.json
│   └── audit_report.csv
└── manifest.json
```

Jumlah artefak pada fixture lengkap adalah 80 file dalam manifest, ditambah
manifest itu sendiri; angka ini merupakan informasi pengujian perangkat lunak,
bukan jumlah data atau temuan Poltrada V3. Tabel mentah utama:
`image_inventory.csv`, `label_inventory.csv`, `raw_label_rows.csv`,
`bounding_boxes.csv`, dan `annotation_issues.csv`. Seluruh statistik dapat ditelusuri
ke tabel-tabel tersebut.

Output pengujian lokal terpisah pada
`outputs/eda_poltrada_v3_validation/`; nama dataset di laporan fixture secara
eksplisit menyatakan **FIXTURE SINTETIK — UJI PERANGKAT LUNAK**. Direktori tersebut
diabaikan Git. Tidak ada gambar sintetik yang disisipkan ke notebook penelitian.

## F. Menjalankan notebook di Kaggle CPU

1. Unggah atau impor `00_eda_poltrada_v3.ipynb` sebagai Kaggle Notebook.
2. Tambahkan dataset final **Poltrada V3 YOLO26 Clean** melalui Input dan pastikan
   `data.yaml` berada pada path konfigurasi. Bila lokasi mount berbeda, ubah
   `DATA_YAML` pada sel konfigurasi saja.
3. Pilih akselerator **None/CPU**. Internet tidak diperlukan; gunakan library
   NumPy, Pandas, Matplotlib, Pillow, PyYAML, dan IPython bawaan lingkungan.
4. Mulai dengan `EDA_MODE = "standard"`, kemudian jalankan **Run All** dari awal.
   Periksa tabel YAML, kelas, split, dan temuan kritis sebelum menggunakan hasil.
5. Untuk decoding dan SHA-256 seluruh citra, pilih `EDA_MODE = "full"` lalu
   jalankan ulang dari awal. dHash tetap opsional melalui `RUN_NEAR_DUPLICATES`.
6. Ambil grafik/tabel/laporan dari folder Output. Gunakan `manifest.json` untuk
   mengetahui artefak yang benar-benar dibuat pada run tersebut.

Notebook tidak memerlukan perubahan konfigurasi training atau akses sesi Kaggle
eksperimen lain. Bila `nc`, `names`, atau path gagal validasi, laporan kritis
disimpan dan analisis dihentikan agar tidak menghasilkan deskripsi keliru.

## G. Keterbatasan dan tindak lanjut

- Dataset final Poltrada V3 belum dieksekusi; nama kelas, jumlah gambar/instance,
  keberadaan test, dan hasil audit empiris masih harus diverifikasi di Kaggle.
- Pengujian tambahan melalui kernel Jupyter lokal tidak selesai. Klaim PASS Run All
  pada fixture mengacu pada eksekusi seluruh code cell Python secara berurutan;
  belum merupakan verifikasi sesi Kaggle atau kernel Jupyter yang berhasil.
- Mode standard mengambil maksimum 500 gambar terverifikasi **per split** untuk
  brightness. Mean/median gabungan sampel tidak diberi bobot ulang untuk menaksir
  populasi. Brightness thumbnail merupakan proksi grayscale 0–255.
- Standard menggunakan verifikasi struktur file, sehingga semua pixel JPEG belum
  tentu telah didekode. Full melakukan decoding lengkap satu citra per waktu.
- Hash gambar lengkap secara default hanya aktif pada full. Tanpa SHA-256, audit
  byte duplikat belum lengkap; path audit tetap berjalan. Kesamaan byte tidak
  membuktikan semua kemungkinan leakage telah diperiksa.
- dHash secara default mati. Bila aktif, maksimal 1.500 path unik diambil dengan
  seed tetap, ambang Hamming 4, dan batas 10.000 pasangan kandidat; truncation dan
  coverage dilaporkan. Kandidat perlu inspeksi dan bukan bukti duplikasi semantik.
- Geometri pasangan melewati gambar di atas 300 objek agar RAM dan biaya tetap
  terbatas; gambar itu tidak dianggap telah diperiksa untuk IoU/jarak pusat.
- Statistik instance memakai bbox valid dari pasangan unik dan mempertahankan
  duplikat valid. Count dan ko-kemunculan memakai label lengkap. Label hilang
  adalah unknown, bukan background yang terbukti.
- Background label kosong merupakan background dalam **anotasi** yang terbaca;
  EDA tidak membuktikan tidak adanya objek nyata yang belum dianotasi.
- Luas relatif memakai citra asli dan ambang operasional 0,01/0,05. Kategori ini
  bukan COCO dan bukan ukuran setelah resize/letterbox input 640.
- EXIF nonstandar, pasangan ambigu, label invalid, citra rusak, dan kandidat
  kemiripan memerlukan inspeksi sumber. Notebook tidak melakukan perbaikan atau
  menyatakan semua eksperimen otomatis tidak valid.
- Resolusi root relatif terhadap direktori YAML dicatat transparan. Untuk YAML
  dengan root absolut yang tidak tersedia atau konvensi root berbeda, relokasi
  harus diberikan secara eksplisit melalui `DATA_ROOT_OVERRIDE`; tidak ada
  penggantian diam-diam dengan dataset lain.
- Nomor halaman dan detail dokumen akademik belum diperiksa karena dokumen asli
  kedua acuan tidak disertakan. Pemetaan mengikuti instruksi pengguna.
