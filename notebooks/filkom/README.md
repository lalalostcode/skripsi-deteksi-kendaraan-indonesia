# Notebook YOLO26 pada GPU Filkom UB

Pilih notebook pruning atau recovery dari [matriks Kaggle dan Filkom](../../docs/lamp_gflops_recovery_workflow.md). Notebook ini memakai GPU dan kernel Jupyter yang berjalan pada server Filkom. Gunakan direktori dataset, checkpoint, dan output Linux yang dapat dibaca atau ditulis oleh akun sendiri.

Untuk recovery 02–05 YOLO26s, aktifkan tepat satu `STUDENT_INPUTS` melalui comment/uncomment. Fine-tuning 02 menyediakan tujuh perlakuan; Native KD 03, CrossKD 04, dan MGD 05 memakai lima seed target 1,20× dengan teacher bersama YOLO26m `v50`. Atur `DEVICE`, `DATASET_ROOT`, `DATA_YAML`, dan `OUTPUT_DIR` persisten. Beberapa kernel dapat menjalankan pilihan raw berbeda secara paralel, dengan folder serta ID run masing-masing. Evaluasi latency acuan memakai GPU yang tersedia secara eksklusif. [Panduan](../../docs/lamp_gflops_recovery_workflow.md#3-pilih-notebook-recovery-per-metode) · [Fine-tuning 02](recovery/02_yolo26s_standard_finetuning.ipynb) · [Salinan historis](fine-tunned/01_yolo26s_pruned_fine_tuned.ipynb)

## Kernel dan Jupyter melalui SSH

Seluruh recovery 01–05 memiliki diagram batang per metrik untuk **baseline asal → raw pruned →
hasil recovery**, dengan warna biru/merah/hijau dan identitas seed/target. Report terstruktur tersedia
pada `custom_eda_report_url` di Summary run W&B dan [indeks EDA](https://wandb.ai/ilhamrafiqin-brawijaya-university/yolom26-kd-indotraffic/reports/EDA--VmlldzoxODI3MDUxNg==).
Gambar, Tables, dan paket `<nama-run>-eda:vN` disimpan terverifikasi; metrik yang belum diukur
tetap `NOT_RUN`. [Cara membaca dan menambahkan EDA pada run lama](../../docs/lamp_gflops_recovery_workflow.md#5-baca-perbandingan-tiga-tahap).

Untuk pruning YOLO26s lima seed, buka [notebook 06 multi-seed](../kaggle/pruning/06_lamp_yolo26s_multiseed_wandb.ipynb) dengan kernel GPU Filkom. Isi path dataset, manifest terverifikasi, dan output persisten di sel konfigurasi. [Protokol dan registry](../../docs/lamp_gflops_recovery_workflow.md#pruning-yolo26s-multi-seed)

Sambungkan ke akun Filkom dan aktifkan environment GPU yang disediakan. Ganti `USER@HOST` dan path pada contoh dengan akun dan lokasi aktual. Jalankan perintah berikut dari terminal server. Jika server sudah menyediakan Jupyter, gunakan kernel terdaftar pada server yang tersedia.

```bash
export SKRIPSI_ROOT=/path/ke/skripsi
cd "$SKRIPSI_ROOT"
nvidia-smi
python -c 'import torch, torchvision; print(torch.__version__, torchvision.__version__, torch.version.cuda); assert torch.cuda.is_available()'
python -m ipykernel install --user --name skripsi-filkom --display-name 'Python (Skripsi GPU Filkom)'
python -m jupyterlab --no-browser --ServerApp.ip=127.0.0.1 --ServerApp.port=8888 --ServerApp.port_retries=0
```

Apabila paket `ipykernel` belum tersedia, pasang `ipykernel` dalam environment yang dipilih. Jika Anda memakai server sendiri dan belum memiliki JupyterLab, pasang juga `jupyterlab`. Bootstrap notebook mengunci Ultralytics, Torch-Pruning, dan W&B serta mempertahankan Torch dan torchvision dari environment GPU. Periksa preflight sebelum menjalankan tahap eksperimen. [Registrasi kernel IPython](https://ipython.readthedocs.io/en/stable/install/kernel_install.html) · [Konfigurasi Jupyter Server](https://jupyter-server.readthedocs.io/en/stable/users/configuration.html)

Dari terminal laptop, teruskan port server melalui SSH. Buka URL `http://127.0.0.1:8888` dengan token yang ditampilkan Jupyter pada terminal server. Gunakan URL yang sama bila IDE menyediakan koneksi ke Jupyter Server, lalu pilih kernel **Python (Skripsi GPU Filkom)**. [Autentikasi token Jupyter](https://jupyter-server.readthedocs.io/en/latest/operators/security.html)

```bash
ssh -N -L 8888:127.0.0.1:8888 USER@HOST
```

## Path dataset, output, dan W&B

Salin Poltrada V3 YOLO26 Clean beserta folder split/label asli dan `dataset_manifest.json` dari sumber terverifikasi. Contoh penyalinan dari terminal Linux atau WSL, setelah membuat direktori `datasets` di server:

```bash
scp -r /path/lokal/poltrada-v3 USER@HOST:/path/ke/skripsi/datasets/
scp /path/lokal/dataset_manifest.json USER@HOST:/path/ke/skripsi/datasets/
```

Pada sel konfigurasi, periksa `WORKSPACE_ROOT`, `DATASET_ROOT`, `DATA_YAML`, `DATASET_MANIFEST_PATH`, `OUTPUT_DIR`, dan `DEVICE`. `WORKSPACE_ROOT` memakai `SKRIPSI_ROOT` atau direktori kernel saat ini. Secara default data dicari di `datasets/poltrada-v3` dan output ditulis di `outputs/filkom`. `DEVICE="auto"` memilih `cuda:0` ketika CUDA tersedia. Pastikan nama GPU dan versi CUDA pada preflight sesuai dengan perangkat yang hendak diukur.

Khusus [MGD recovery 05](recovery/05_yolo26ns_mgd_kd.ipynb), default HPC memakai
`Path.home() / "skripsi-kd"` sebagai `WORKSPACE_ROOT`, dengan repository di `repo`,
dataset di `datasets/roboflow-poltrada-v3`, YAML di folder dataset tersebut, dan output
akhir di `outputs/filkom/recovery/selected/yolo26s/mgd`. Lokasi ini tidak bergantung
pada direktori kernel. Override opsional memakai `SKRIPSI_ROOT` untuk workspace,
`SKRIPSI_REPO_ROOT`, `SKRIPSI_DATASET_ROOT`, `SKRIPSI_DATA_YAML`, serta
`SKRIPSI_OUTPUT_DIR`. Nilai `SKRIPSI_OUTPUT_DIR` adalah output dasar sampai `yolo26s`;
generator menambahkan `mgd` satu kali. Jika Jupyter mewarisi `SKRIPSI_ROOT` lama,
sesuaikan ke `~/skripsi-kd` atau hapus override agar default baru berlaku.

MGD 05 memeriksa Python **3.11 atau lebih baru**, sesuai `pyproject.toml`, sebelum
membuat folder run atau memasang dependency. Import UTC portabel tidak menambahkan
dukungan Python 3.8. Aktifkan environment HPC dengan Python yang memenuhi syarat
dan Torch/torchvision CUDA yang sesuai sebelum mendaftarkan kernel Jupyter memakai
`python -m ipykernel install` seperti contoh di atas. Bootstrap mempertahankan
Torch/torchvision yang telah tersedia; kedua paket tersebut harus dipasang di
environment kernel, termasuk ketika Python sudah 3.11. Seed aktif MGD FILKOM tetap
0 dari raw terpilih; loss, teacher, hyperparameter dan protokol evaluasi sama dengan
MGD recovery Kaggle untuk input raw/seed yang sama.

Sediakan `WANDB_API_KEY` di environment proses Jupyter atau isi prompt tersembunyi ketika notebook meminta login. Pastikan akun dapat membaca raw `:vN` serta teacher `:vN` yang dipakai dan menulis ke project penelitian. File notebook cukup menyimpan referensi artefak dan hash. Panduan menjalankan pruning, menyiapkan input recovery, dan meneruskan suite ada di [panduan eksperimen](../../docs/lamp_gflops_recovery_workflow.md).

## Teacher YOLO26m lima seed

Buka [Teacher 02](teacher/02_teacher_yolo26m_multiseed.ipynb) dengan kernel GPU di atas. Periksa `DATA` untuk YAML lokal Poltrada V3, `MODEL_WEIGHTS` untuk bobot pretrained, `DEVICE` untuk indeks GPU pada kernel, dan `PROJECT` untuk output persisten. Pastikan path split di YAML tersedia di server. Default `DATA` adalah `datasets/poltrada-v3/data.yaml` dan `PROJECT` adalah `outputs/filkom/teacher` di bawah `SKRIPSI_ROOT`.

Run All menjalankan lima seed `42, 0, 2024, 7, 3407` secara serial dengan 100 epoch per seed. Run W&B, file evaluasi, dan checkpoint terpisah per seed. Lihat `multiseed_summary.json` untuk mean dan simpangan baku dan `seed_result.json` untuk referensi versi dan SHA-256 artefak terbaik. Jalankan satu antrean lengkap pada GPU yang sama. [Padanan dan protokol Kaggle](../kaggle/README.md#teacher-yolo26m-lima-seed)
