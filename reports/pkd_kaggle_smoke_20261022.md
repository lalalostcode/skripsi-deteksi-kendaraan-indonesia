# Pearson/PKD — smoke GPU 22 Oktober 2026

Status GPU: **PENDING**. Uji satu epoch penuh menggunakan implementasi pada `12_pkd_yolo26n.ipynb`, tanpa menjalankan training 100 epoch.

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

Full training 100 epoch dan resume belum diverifikasi (`RESUME_NOT_VERIFIED`).
