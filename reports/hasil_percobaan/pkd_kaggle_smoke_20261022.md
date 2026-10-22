# Pearson/PKD — smoke GPU 22 Oktober 2026

Status GPU: **INCOMPLETE — BELUM PASS SATU EPOCH**. Job berhenti pada batas waktu yang dikonfigurasi, dengan batch terakhir yang tercatat **416/424 (98,1%)**. Validasi dan pemeriksaan checkpoint akhir belum selesai. Uji menggunakan implementasi pada `12_pkd_yolo26n.ipynb`, tanpa menjalankan training 100 epoch.

- [Job Kaggle privat](https://www.kaggle.com/code/ilhamrafiqin/pkd-gpu-smoke-1ep-20261022-32f96e).
- Poltrada V3: dataset Kaggle yang sama dengan HEAD/RM; seluruh train dan val.
- Teacher YOLO26s v62: memakai input privat yang telah diizinkan dan tersedia dari tes sebelumnya, tanpa upload checkpoint baru. SHA-256 wajib `5bc3ae61f8da1c9f03916922de10146805c27c64a11e46edd766a00d0f6f18fd`.
- Student awal `yolo26n.pt`; imgsz 640, batch 16, seed 42, AMP, workers 2, AdamW, lr 0,001, weight decay 0,0005, nbs native 64. Bobot PKD 1, epsilon 1e-6.
- Harness menetapkan EPOCHS=1 dan plots=False; flag full pada notebook tetap False. Target uji adalah satu epoch penuh, bukan diagnostic parsial 20 batch. Konfigurasi warmup 3 tetap mengikuti pembatasan native untuk epochs=1.
- Rumus PKD, projector, feature taps, trainer dan callback checkpoint diekstrak dari notebook. Harness menambahkan pengukuran update/gradien/memori, heartbeat, dan adapter tracking lokal; **logging/upload W&B online belum diuji**.
- Tidak ada penurunan batch/resolusi/AMP otomatis. Batas job 75 menit, dalam kuota gratis yang tersedia saat pengiriman; mode berbayar tidak aktif.

Pemeriksaan lokal: **35 tes PKD lulus**, termasuk loss/gradien terhadap referensi MMRazor yang dikunci SHA-256, integrasi trainer CPU satu epoch, state archive, checkpoint student-only di proses Python baru, dan regresi fused AdamW/GradScaler. Ruff juga lulus.

Perbaikan sebelum uji GPU: optimizer pre-hook diagnostic sekarang mengenali `optimizer.found_inf` pada fused AdamW dan tidak menganggap update yang dilewati AMP sebagai update dengan gradien valid. Loss/objective PKD tidak diubah. Regresi memakai fused AdamW + CPU GradScaler nyata, membuktikan overflow melewati update lalu gradien finite menghasilkan perubahan parameter.

Source job yang diunduh ulang cocok byte-per-byte dengan source lokal; SHA-256 `2648fed0be7694e43f07f22cfaa7864b317aa87cb5a9c68d5d9ddd54086c000d`. Job privat dan GPU aktif terverifikasi.

## Hasil aktual GPU

[Bukti terstruktur](pkd_gpu_smoke_20261022.json) memuat konfigurasi efektif, hasil numerik, baris progres terakhir, dan SHA-256 log asli. Tidak ada `gpu_smoke_result.json` sukses dari worker; bukti parsial disusun dari log dan konfigurasi yang diunduh, bukan dianggap sebagai hasil akhir harness.

| Pemeriksaan | Hasil |
|---|---|
| GPU / protokol | Tesla T4; 640 / batch16 / AMP; seluruh 6.775 train dan 1.790 val ditetapkan |
| Numerik PKD vs MMRazor, dijalankan di worker GPU | PASS; selisih maksimum 2,384185791015625e-7 |
| Batch train terakhir di log | 416 dari 424; **0 epoch penuh selesai** |
| Detection / PKD / total loss pada progres terakhir | 3,466 / 1,678 / 5,144; rerata berjalan yang dibulatkan native |
| PKD P3 / P4 / P5 | 0,6639 / 0,6050 / 0,4095 |
| Memori reserved terakhir yang ditampilkan | 3,22 GiB; bukan pengukuran peak allocated |
| Traceback / OOM dalam log yang tersedia | Tidak ditemukan |
| Validation mAP, checkpoint student-only, digest teacher akhir, cleanup hook akhir | **BELUM TERVERIFIKASI** |

Setup/scan dataset menghabiskan sekitar **31 menit 31 detik** sebelum training mulai. Progres terakhir mencatat training **43 menit 39 detik**, termasuk menunggu I/O. Log terakhir berada pada detik **4512,48** sejak worker dimulai. API sesudahnya menampilkan `CANCEL_ACKNOWLEDGED` tanpa `failure_message`. Penghentian bertepatan dengan timeout 4500 detik yang dikirim; **tidak ada perintah cancel manual**. Jadi penyebab timeout dinilai dari konfigurasi dan waktunya, bukan pesan alasan eksplisit dari Kaggle.

Hitungan optimizer-step, peak memory dan hasil pemeriksaan akhir tersimpan dalam memori harness dan belum sempat ditulis sebelum penghentian; tidak diklaim lulus. Loss yang menurun bukan pengganti validasi satu epoch atau bukti kualitas model.

Kuota gratis setelah job: terpakai **29 jam 55 menit 44,97 detik dari 30 jam**, tersisa sekitar **4 menit 15 detik**. `pay_to_scale=False`. Tidak ada pengulangan GPU, penurunan batch/resolusi, atau pemotongan dataset untuk mendapatkan status hijau. Uji penuh berikutnya memerlukan kuota baru/cukup; staging data ke disk kerja sebelum mengaktifkan GPU dapat mengurangi waktu tunggu input, dengan isi gambar/label tetap sama.

Full training 100 epoch dan resume belum diverifikasi (`RESUME_NOT_VERIFIED`).
