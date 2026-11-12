# HEAD/RM Kaggle smoke test — 21–22 Oktober 2026

Notebook utama sudah ditulis ulang mengikuti alur notebook penelitian sebelumnya: konfigurasi, setup Kaggle, dataset/teacher, loss/model, trainer, W&B, smoke satu epoch, training, hasil.

- HEAD: 31.526 karakter kode (sebelumnya 70.530), 23 cell terpisah.
- RM: 33.623 karakter kode (sebelumnya 72.624), 23 cell terpisah.
- Objective/kelas model dipertahankan identik secara AST dengan implementasi yang diuji sebelumnya.
- Default Run All: satu epoch smoke di Kaggle; training 100 epoch tetap memerlukan `RUN_FULL_TRAINING=True`.
- Oracle numerik CPU dieksekusi otomatis sebelum smoke; hasilnya juga diuji lokal. Cell model/loss, trainer, logging, dan training final cocok persis dengan cell yang dipakai job GPU.
- CPU: 5 pengujian lulus, termasuk training nyata satu epoch sintetis, callback artifact simulasi, validasi, load student di proses Python baru, dan overflow GradScaler dengan fused AdamW nyata. Ini bukan hasil GPU.

GPU: **HEAD PASS; RM PASS**. Keduanya menyelesaikan satu epoch penuh di Tesla T4, validasi, pemeriksaan teacher/gradien, dan load student-only checkpoint di proses Python baru dengan inference finite. Mesin lokal mempunyai Intel GPU dan PyTorch CPU; uji CUDA dijalankan di Kaggle privat dengan izin pengguna untuk upload teacher dan tes GPU. RM dikirim setelah HEAD selesai dan hasilnya diverifikasi.

- [Job HEAD privat](https://www.kaggle.com/code/ilhamrafiqin/head-gpu-smoke-1ep-20261021-f42278)
- [Job RM privat](https://www.kaggle.com/code/ilhamrafiqin/rm-gpu-smoke-1ep-20261021-f42278)
- Input teacher privat: `ilhamrafiqin/head-rm-teacher-smoke-20261021-f42278`.

Teacher yang digunakan tetap `best.pt` artifact v62 dengan SHA256 `5bc3ae61f8da1c9f03916922de10146805c27c64a11e46edd766a00d0f6f18fd`. Salinan byte-identik dan provenance dipersiapkan sebagai input privat. Poltrada V3 memakai dataset Kaggle yang sudah ada, tanpa perubahan split/anotasi atau upload ulang.

Konfigurasi bersama, fingerprint YAML dan urutan kelas kedua job cocok. Fingerprint yang direkam mencakup SHA-256 YAML `aab6dda729d6dc18c293d2f13a117d73074ae3ebb1cbb8134f46183a4a7a9ae3` dan urutan enam kelas: Bus, Mobil Penumpang, Pejalan Kaki, Sepeda Motor, Truck, Unmotorized.

Cell teacher pada notebook bersih juga dieksekusi langsung secara baca-saja terhadap W&B asli: **PASS**, termasuk producer/ref v62, download `best.pt`, SHA-256, skala YOLO26s, schema parameter dan urutan enam kelas. Tidak ada run W&B yang dibuat. Ini menguji retrieval teacher; upload logging/artifact training online tetap belum diuji.

Job masing-masing memakai satu epoch penuh, seed 42, imgsz 640, batch 16, AMP dan device 0. HEAD lalu RM dijalankan bergantian. Uji remote memakai adapter log/artifact lokal agar tidak menyimpan secret W&B dalam kode yang diunggah; **login/upload W&B online tidak termasuk hasil uji GPU ini**. Tidak ada token yang dimasukkan ke source job.

Nilai konfigurasi `warmup_epochs=3` tetap; trainer native membatasi warmup ke `epochs-1`, sehingga smoke satu epoch tidak menjalankan fase warmup tiga epoch. Jadwal penuh 100 epoch dan resume belum diverifikasi.

## Percobaan GPU pertama dan perbaikan

HEAD percobaan pertama berjalan di **Tesla T4**, lolos pemeriksaan AMP, dan membaca **6.775 gambar train / 1.790 gambar val** (424 batch train per epoch). Sebelum update pertama, pemeriksaan gradien menghentikan run. Bukan OOM; log menunjukkan sekitar 4,32 GiB reserved saat itu.

Penyebab pada callback: fused AdamW memanggil optimizer pre-hook juga ketika `GradScaler` menandai update untuk dilewati (`optimizer.found_inf != 0`). Callback sebelumnya memeriksa gradien seolah update pasti dilaksanakan. Callback sekarang menghitung `amp_skipped_steps` dan memeriksa/menghitung gradien hanya untuk update yang tidak dilewati. Objective, optimizer, batch, resolusi, AMP dan pembagian data tidak diubah. Uji regresi fused AdamW + CPU GradScaler membuktikan overflow tidak mengubah parameter atau menambah hitungan update, lalu update finite berikutnya benar-benar mengubah parameter.

HEAD diulang dengan callback yang diperbaiki: **PASS GPU**. RM juga **PASS GPU**. Hasil mentah: [HEAD](head_gpu_smoke_20261022.json), [RM](rm_gpu_smoke_20261022.json).

| Pemeriksaan | HEAD | RM |
|---|---:|---:|
| GPU | Tesla T4 | Tesla T4 |
| Epoch / batch train | 1 / 424 | 1 / 424 |
| Optimizer step nyata | 99 | 101 |
| Update awal dilewati AMP | 7 | 5 |
| VRAM allocated / reserved maksimum | 4,172 / 4,615 GiB | 2,843 / 3,494 GiB |
| Durasi train + validasi + pemeriksaan akhir, di luar setup dataset | 1.920 detik | 3.271 detik |
| Validation mAP50 / mAP50–95 | 0,37631 / 0,24015 | 0,41912 / 0,25848 |
| Detection / KD / total loss, rerata batch | 3,58595 / 9,87563 / 13,46158 | 3,45977 / 0,07627 / 3,53603 |
| Teacher parameter/BN tetap dan gradien finite/nonzero | PASS | PASS |
| Load student-only checkpoint di proses Python baru + inference finite | PASS | PASS |

Progress bar native HEAD mencatat training **25 menit 11 detik**, sekitar **3,56 detik/batch termasuk menunggu data**. Kolom `GPU_mem` pada paruh kedua epoch berada pada **4,56–4,62 G**, dengan kenaikan sekitar 0,06 G. Ini pengamatan satu epoch, bukan jaminan stabilitas memori 100 epoch.

RM mencatat training **41 menit 40 detik**, sekitar **5,90 detik/batch termasuk menunggu data**. Kolom `GPU_mem` paruh kedua epoch berada pada **3,46–3,49 G**. Sinyal ranking aktif: rerata `rm_valid_groups=405,88` per batch; loss ranking kelas `0,013425` dan IoU `0,005641`, dengan bobot RM 4. Metrik di atas berasal dari smoke satu epoch, bukan hasil eksperimen 100 epoch.

SHA-256 source job HEAD yang diunduh ulang dari Kaggle cocok byte-per-byte dengan source lokal: `c8fb4f8b08eeb1bad83193287f9342ba978fd13ff3de43b35b137e2e21b2d2de`. Source RM yang sudah dikirim juga cocok: `6820f13a1e0275f5b6c85b64d551a05f58553690f8ec0c3af9944524c92b7af0`. Source remote memakai cell definisi notebook yang sama, adapter tracking lokal yang dijelaskan di atas, serta heartbeat log khusus harness RM. Heartbeat tidak mengubah model atau protokol training.
