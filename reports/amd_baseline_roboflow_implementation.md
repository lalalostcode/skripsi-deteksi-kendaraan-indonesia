# Baseline YOLO26n pada AMD ROCm cloud dan Roboflow

Tanggal: 19 Oktober 2026, Asia/Jakarta.

Notebook baru: `notebooks/yolo26-kd-indotraffic/notebooks/03_baseline_yolo26n_amd_roboflow.ipynb`.
Clone baseline 03, standalone, 19 cell, tanpa output eksekusi tersimpan. Seluruh 15 notebook
yang sudah ada dipertahankan byte-identik terhadap snapshot SHA-256 sebelum pekerjaan ini.
SHA-256 baseline sumber: `807b8928bd7d49b4feb2fe5e6dc2e50f48ae8e3f4f7860237b726d61491bca98`.

## Pemakaian

1. Pilih kernel Linux cloud dengan PyTorch ROCm dan torchvision kompatibel yang sudah terpasang.
   Notebook memeriksa HIP, GPU AMD pada DEVICE=0 dan operasi tensor GPU. Tidak memasang driver,
   ROCm atau PyTorch/torchvision pengganti; resolver pip memakai constraints versi stack image.
2. Isi `WANDB_API_KEY` dan `ROBOFLOW_API_KEY` pada cell tersendiri di **Section 1**.
   Keduanya wajib nonkosong. Tidak menggunakan environment/getpass untuk mengambil key.
3. Periksa **Section 2** lalu Run All. Daftar seed menampilkan 42, 123, 2024 sebagai komentar,
   dengan **7 dan 3407** aktif. Satu seed menyelesaikan training/evaluasi/upload/finish sebelum
   seed berikutnya dimulai. Setelah gagal, aktifkan hanya seed yang belum selesai; tidak ada auto-resume.

Protokol tetap: `yolo26n.pt`, 100 epoch, 640px, batch16, AdamW lr0.001, patience100,
AMP True, workers2, weight decay0.0005, warmup3, close mosaic10, evaluasi val.
Parameter efektif engine, termasuk default yang tidak ditulis eksplisit, dibandingkan dengan
baseline sumber pada seed/sumber data/lokasi output yang disamakan dan hasilnya identik.
Callback pengaman menolak perubahan batch/resolusi/AMP, loss non-finite dan recovery OOM
yang akan menurunkan batch. Math loss, optimizer dan langkah AMP engine tetap bawaan Ultralytics.

## Dataset, hasil dan logging

- Roboflow: workspace `ilham-rafiqin`, project `poltrada-original-raw-2`, version 3,
  format `yolo26`, lokasi `~/skripsi-kd/datasets/poltrada-v3`.
- SDK download dipanggil satu kali sebelum antrean seed jika belum ada cache valid; cache
  lengkap dengan metadata sumber yang cocok tidak menghubungi API. `DATA` berasal dari
  `Path(dataset.location) / "data.yaml"`. Loader resmi memakai `autodownload=False`,
  termasuk penyelesaian path relatif `../train/images` dan `../valid/images`.
- Cache tanpa YAML/split, direktori split tanpa gambar, metadata sumber berbeda, atau download
  gagal menghentikan proses. Cache tidak dihapus atau ditimpa otomatis. Tidak ada cleaning,
  resplit, penggantian label atau tambahan augmentasi di luar protokol baseline.
- Metadata mencatat identitas Roboflow, hash YAML, kelas, resolved split dan lingkungan GPU/HIP.
  Kesetaraan isi terhadap dataset Kaggle bersih berstatus **NOT_VERIFIED**; tidak ada agregasi
  hasil sebelumnya secara otomatis. Hash YAML bukan bukti kesamaan seluruh citra/label.
- Output: `~/skripsi-kd/results/03_baseline_yolo26n_amd/multiseed_<UTC>/seed-<seed>/`.
- W&B: project `yolom26-kd-indotraffic`, group `supervised`, run
  `baseline_yolo26n_amd_seed7` / `baseline_yolo26n_amd_seed3407`, tag `amd-rocm` dan `roboflow-v3`.
- Tiap seed memuat student pretrained baru, mereset RNG dan status callback/checkpoint,
  membuat ID W&B baru dengan `resume="never"`. Upload checkpoint per epoch dan results akhir
  ditunggu selesai. Kegagalan training/evaluasi/upload/finish menghentikan antrean dan
  mempertahankan checkpoint serta hasil seed yang sudah selesai.
- Key tidak dimasukkan ke config, artifact atau output notebook. Auto code upload W&B
  dinonaktifkan melalui `save_code=False` dan `disable_code=True`. Output/exception mentah
  SDK Roboflow yang dapat memuat URL bertoken tidak diteruskan ke console.

## Bukti pengujian dan batas klaim

### Perbaikan import OpenCV pada cloud tanpa display

Traceback cloud berhenti pada `cv2` dengan `ImportError: libxcb.so.1`, sebelum training.
Bootstrap awal memungkinkan paket `opencv-python` desktop dari dependency Ultralytics
bertumpuk dengan headless dari Roboflow. Varian AMD sekarang menggunakan distribusi resmi
`ultralytics-opencv-headless==8.4.155` serta `opencv-python-headless==4.10.0.84`, sesuai pin
Roboflow 1.4.0. Import Python tetap `from ultralytics import YOLO, settings` dan protokol
training tidak berubah.

Installer menghapus provider standar/desktop yang berbagi namespace, termasuk provider
headless lama dalam keluarga yang berkonflik, kemudian memasang ulang provider headless.
Torch/torchvision tetap dikunci. Pemeriksaan subprocess menolak cv2 yang gagal import,
versi modul salah, GUI aktif, atau codec/resize tidak berfungsi, sebelum autentikasi W&B
dan download dataset. Restart kernel diwajibkan jika paket yang sudah dimuat atau NumPy
yang sebelumnya di-import torchvision berubah. Rerun dengan stack benar tidak memasang
ulang dependency GUI. Versi/distribusi OpenCV dan NumPy juga dicatat dalam metadata.

Dokumentasi resmi diperiksa melalui **MCP Search**:

- [Instalasi PyTorch AMD](https://rocm.docs.amd.com/projects/ai-ecosystem/en/latest/frameworks/pytorch/install.html).
- [Matriks kompatibilitas AMD](https://rocm.docs.amd.com/en/latest/compatibility/compatibility-matrix.html).
- [Ultralytics headless server](https://docs.ultralytics.com/quickstart#headless-server-installation).
- [Metadata versi headless 8.4.155](https://pypi.org/pypi/ultralytics-opencv-headless/8.4.155/json),
  yang mendeklarasikan dependency `opencv-python-headless>=4.7.0,!=4.13.0.90` sehingga pin 4.10 kompatibel.

Pemeriksaan HIP/GPU AMD tetap sama. Python 3.14 pada traceback sendiri bukan bukti
ketidakcocokan ROCm; dukungannya bergantung pada kombinasi versi di matriks AMD. Tidak ada
driver/ROCm/PyTorch yang diubah atau keberhasilan cloud yang diklaim tanpa eksekusi nyata.

| Pemeriksaan | Hasil |
|---|---|
| JSON/nbformat dan Python seluruh cell | PASS |
| Semua parameter training/evaluasi sesuai baseline sumber | PASS |
| Key kosong menghentikan download/antrean sebelum run dibuka | PASS |
| ROCm valid, non-Linux, CUDA/CPU build, GPU tidak tersedia, device salah, probe non-finite | PASS dengan mock |
| Constraints torch/torchvision, paket sudah sesuai, perubahan stack, restart kernel | PASS dengan mock installer |
| Roboflow source/format/path, cache parsial, YAML/split hilang/kosong, metadata berbeda, error tidak membocorkan key | PASS dengan mock SDK dan loader YAML nyata |
| Dua seed dengan run/model/state/metrik/artifact terpisah, callback lama ditolak | PASS dengan mock W&B |
| Kegagalan training/evaluasi/upload/finish menghentikan antrean dan melindungi seed pertama | PASS dengan mock W&B |
| Dua training baseline nyata secara serial | PASS pada data sintetis CPU, masing-masing satu epoch, 64px, batch2 |
| Student awal sama bobotnya dan dibuat baru setiap seed; checkpoint dapat dimuat kembali | PASS pada fixture CPU |
| Keutuhan 15 notebook lama | PASS dengan SHA-256 sebelum/sesudah |
| GPU AMD cloud, dataset Roboflow nyata, W&B online | **NOT TESTED** |
| Training penelitian 100 epoch dan performa Poltrada | **NOT TESTED** |

Suite khusus AMD dan tracking setelah perbaikan headless/proxy-diagnostic/cache: **72 passed**, termasuk satu tes integrasi yang benar-benar
menjalankan dua training sintetis CPU. Empat warning berasal dari deprecation THOP.
Harness CPU hanya menonaktifkan pemeriksaan HIP/device dan memakai parameter sintetis eksplisit;
runtime notebook yang dikirim tetap mewajibkan ROCm, 100 epoch, 640px, batch16, AMP True.

```powershell
.\.venv\Scripts\python.exe -m pytest tests/test_amd_baseline_notebook.py tests/test_kd_notebook_tracking.py -m '' -q --tb=short --show-capture=no
.\.venv\Scripts\ruff.exe check tests/test_amd_baseline_notebook.py tests/test_kd_notebook_tracking.py notebooks/yolo26-kd-indotraffic/notebooks/03_baseline_yolo26n_amd_roboflow.ipynb
```

Ruff: **All checks passed**. Varian AMD dikecualikan dari assertion path Kaggle pada suite
tracking lama dan memiliki pemeriksaan konfigurasi/logging/data sendiri. Tidak ada run cloud
atau download/upload API nyata yang dilakukan dalam implementasi lokal ini.

## ProxyError Roboflow pada AMD Radeon Cloud / AnRuiCloud

Screenshot menunjukkan notebook dijalankan melalui `jupyter.anruicloud.com`. Error yang
tersedia adalah `ProxyError`; pesan awal menutupi detail URL sehingga tidak membuktikan apakah
penyebabnya proxy tidak tersedia, autentikasi proxy, penolakan tunnel, atau koneksi lain.
Baris `captured.close()` adalah cleanup output SDK, bukan bukti kerusakan buffer atau GPU.

Notebook arsip `02b-yolo26n-student-baseline-seed42-kd-project.ipynb` menggunakan alur SDK
workspace → project → version → download yang sama, dengan ekspor yolov8. Notebook arsip
multi-scale juga memiliki reuse snapshot lokal. Tidak ditemukan pengaturan `trust_env`,
HTTP_PROXY/HTTPS_PROXY/ALL_PROXY khusus pada kode notebook arsip yang diperiksa.
Versi AMD baru tetap memakai **Roboflow 1.4.0, dataset version 3, format yolo26**;
format lama, cleaning/resplit dan penghapusan cache otomatis dari arsip tidak disalin.

Perbaikan download mempertahankan tahapan API secara terpisah dan menampilkan JSON diagnostik:
stage, jenis error, kategori, nama variabel proxy yang hadir, serta `detail_redacted`.
API key plaintext/URL-encoded, URL bertoken, dan kredensial proxy disamarkan. Kategori
membedakan tunnel-denied 403, proxy authentication 407, koneksi proxy, timeout dan DNS.
Proxy environment dan verifikasi TLS tidak diubah. Dokumentasi Requests menjelaskan
[penggunaan proxy environment](https://requests.readthedocs.io/en/latest/user/advanced/#proxies).
Kebijakan outbound AnRuiCloud untuk Roboflow belum berhasil dikonfirmasi dari sumber publik.

Cache lokal yang lolos validasi isi split dan metadata workspace/project/version digunakan
tanpa membuat objek SDK atau melakukan request API. Cache parsial/identitas tidak lengkap
tetap ditolak. Untuk upload manual, gunakan ekspor versi 3 format yolo26 dan letakkan
data.yaml langsung di DATASET_DIR. Format yang diminta tetap dicatat, dengan
`format_evidence=not-recorded-in-cache-yaml` pada cache dan `sdk-request` pada download baru;
YAML tidak dianggap bukti format ekspor. Jalur ini menghindari request dataset, tetapi
tidak menggantikan kebutuhan koneksi W&B online atau membuktikan jaringan cloud telah pulih.

Kasus tambahan diuji tanpa jaringan nyata: cache valid tidak memanggil SDK, identitas cache
wajib lengkap, ProxyError pada kelima tahap API, sanitasi key/proxy, environment proxy tidak
berubah, dan klasifikasi 403/407/timeout/DNS yang tidak menyamakan semua error dengan key salah.
Suite tracking lama dibatasi pada notebook 01–12 sesuai skemanya; notebook baru 13/14 dengan
skema berbeda tidak dimodifikasi atau dianggap tercakup oleh tes Kaggle lama tersebut.
