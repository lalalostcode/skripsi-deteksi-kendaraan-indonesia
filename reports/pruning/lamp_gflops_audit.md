# Audit awal LAMP YOLO26n/S dengan target GFLOPs

## Workflow Kaggle dan Filkom — 5–6 November 2026

Pengembangan berikut melanjutkan sumber notebook 05/06, dengan target tepat 1,02×/1,05×/1,10×/1,20×/1,70×.
Angka 1,70× berasal dari konfigurasi penulis MP-YOLO lokal; empat target lebih kecil merupakan eksplorasi
adaptasi YOLO26. LAMP + DepGraph, tier cap, baseline v84/v62, dan proteksi arsitektur dipertahankan.
Hipotesisnya adalah bahwa target komputasi ringan memberi pilihan raw student yang lebih layak sebelum
recovery. Pembandingnya adalah baseline asli dan standard fine-tuning 100 epoch dari raw yang sama.
Recovery dipisah per metode. Fine-tuning adalah kontrol untuk Native KD, CrossKD, dan MGD yang memakai raw serta SHA-256 identik. Metrik utama tetap mAP50–95 full validation, dengan parameter, GFLOPs dan latency dilaporkan terpisah. [Konfigurasi dan urutan eksekusi saat ini](../../docs/lamp_gflops_recovery_workflow.md)

Audit implementasi menemukan YAML evaluasi pruning menggunakan train=val sebagai placeholder yang tidak
boleh dipakai untuk training. Trainer recovery lama menolak resume, sementara checkpoint native SDK
menyimpan EMA/optimizer dalam FP16 dan loop SDK dapat menurunkan batch saat OOM. Implementasi baru perlu
descriptor train asli, resume FP32 lengkap termasuk accumulation, dan protokol batch tetap. Logging pruning
sebelumnya hanya tabel suite; publikasi raw harus dikonfirmasi sebelum recovery dan checkpoint per epoch.
Audit kode ini mendasari verifikasi lokal dan pengukuran GPU berikutnya. Bagian selanjutnya mendokumentasikan rancangan, target, dan bukti historis sebelum revisi lima target. Rujuk panduan bersama di atas untuk konfigurasi aktif.

Audit dicatat 4 November 2026 waktu Jakarta, sebelum implementasi notebook 05. Pertanyaan penelitian adalah apakah LAMP dengan proteksi jalur fitur dapat mencapai target komputasi dengan penurunan mAP50–95 maksimal dua poin persentase, dan seberapa sensitif jalur P5 terhadap pilihan channel pada pruning lama.

## Bukti awal dan batas provenance

[Tabel historis 12 baris](lamp_gflops_historical_unverified.csv) berasal dari pengguna. Angka dan nilai status dipertahankan. Provenance `UNVERIFIED` menandai bahwa ID run, artefak asli, fingerprint split, konfigurasi evaluator, versi counter, dan lingkungan pengukuran belum ditemukan. `validation_status=PASS` berasal dari tabel dan menyatakan status eksekusi yang dilaporkan. Tabel berdiri terpisah dari bukti struktur lokal. [Perbandingan CPU V2](comparisons/README.md)

Pada `extended` 10%, mAP50–95 yang dilaporkan turun dari 0,5261053 ke 0,1692238 pada YOLO26n dan dari 0,5611909 ke 0,2545885 pada YOLO26s. GFLOPs turun sekitar 5,5% pada kedua varian. Pada 20%, mAP50–95 yang dilaporkan sudah di bawah 0,003 untuk keduanya. Perubahan latency median tidak konsisten sepanjang sweep. Baseline N dan S dalam tabel masing-masing memakai 6,025830 dan 23,006822 GFLOPs, berbeda dari 5,905408 dan 22,765978 pada audit CPU. Konfirmasi versi counter, mode forward, dan artefak sebelum membandingkan angka antarsumber.

## Kontrol P5 dan integritas struktur

Ledger `extended` 10% historis menunjukkan bahwa `model.7.conv`, transisi backbone ke P5/32, kehilangan separuh channel output. Group ini menyumbang 128 dari 206 penghapusan channel independen pada N dan 256 dari 412 pada S, atau 62,1% pada kedua varian. Hitungan berasal dari `pruning_history` pada `dependency_audit.json` di `suite_dir` laporan CPU. Konsentrasi ini mendasari hipotesis sensitivitas P5. Besar pengaruh terhadap akurasi masih perlu diukur. [Laporan CPU dengan lokasi ledger](comparisons/lamp_v2_ns_cpu_verification_20261024.json)

- **P5-only.** Ambil indeks asli group `model.7.conv` dari pilihan `extended` 10% yang direproduksi. Pangkas hanya group tersebut pada salinan baseline dan ukur perubahan mAP, AP per kelas, serta GFLOPs terhadap baseline.
- **P5-protected.** Lindungi group P5 dan ulangi pemilihan LAMP dari kandidat lain dengan jumlah channel independen yang sama seperti `extended` 10%. Bandingkan keduanya pada split dan evaluator yang sama. Laporkan juga perbedaan GFLOPs dan lokasi pemangkasan karena distribusi kerja tiap layer berbeda.
- **Oracle mask.** Gunakan baseline dense dalam mode `eval`. Nolkan kolom bobot input Conv konsumen menggunakan indeks asli dari ledger. Pertahankan bias Conv, affine dan statistik BatchNorm, serta aktivasi. Channel yang dibuang dapat bernilai nonnol pada jalur dense tanpa menyumbang ke hasil Conv konsumen. Pemetaan residual memakai survivor yang sama pada kedua sisi penjumlahan, concat memakai offset per masukan, dan fan-out membutuhkan mask pada setiap konsumen yang terdampak. Periksa kecocokan irisan tensor, forward raw kedua cabang, invariansi head, dan pemuatan ulang di proses baru. Regresi dengan BN dan SiLU menguji kegagalan masking sebelum BatchNorm.

## Adaptasi MP-YOLO untuk notebook 05/06

Target utama adalah GFLOPs baseline dibagi GFLOPs pruned dengan sasaran 1,1×, 1,2×, dan 1,3×. Counter, resolusi 640, presisi FP32, dan mode forward disamakan pada tiap pasangan. MP-YOLO mendefinisikan speed-up komputasi dengan rasio FLOPs dan contoh konfigurasi memakai 1,7×. Notebook ini mengadaptasi target komputasi, ranking LAMP, dan penghapusan group DepGraph pada YOLO26n/S. Setiap kandidat berasal dari checkpoint N `v84` atau S `v62` yang sama untuk variannya. Protokol mencakup pruning dan evaluasi saja. [Paper MP-YOLO](https://www.sciencedirect.com/science/article/abs/pii/S1047320325001749) · [Konfigurasi penulis](https://github.com/Wang-jj-zs/MP-YOLO/blob/master/MP-YOLO/compress.py) · [Counter penulis](https://github.com/Wang-jj-zs/MP-YOLO/blob/master/MP-YOLO/ultralytics/models/yolo/detect/compress.py) · [LAMP](https://arxiv.org/abs/2010.07611) · [DepGraph](https://openaccess.thecvf.com/content/CVPR2023/html/Fang_DepGraph_Towards_Any_Structural_Pruning_CVPR_2023_paper.html)

Kedua cabang Detect, attention, split CSP, output downsampling, dan antarmuka fitur lateral serta pemasok Detect dilindungi saat mencari kandidat. Gunakan tier cap 25/10%, 35/20%, lalu 50/50% untuk hidden Bottleneck dan group eligible lain. Ranking menggunakan skor LAMP finite, nama root, dan indeks asli. Pencarian berlangkah satu poin persentase channel eligible, lalu memeriksa anggaran integer di sekitar batas. Kontrol `extended` mengikuti cakupan lama untuk diagnosis. Hasil awal CPU memperkirakan batas komputasi kebijakan sekitar 1,22× pada N dan 1,23× pada S. Ukur kembali saat eksekusi. Target di atas batas dilaporkan `UNREACHABLE_UNDER_POLICY`, dengan kandidat maksimum sebagai diagnostik. Isomorphic Pruning, MDP, dan HALP menjadi rujukan diagnostik topologi dan evaluasi efisiensi. [Isomorphic](https://www.ecva.net/papers/eccv_2024/papers_ECCV/html/4364_ECCV_2024_paper.php) · [MDP](https://mdp-cvpr.github.io/) · [HALP](https://github.com/NVlabs/HALP)

Smoke memakai hingga 128 citra `val` Poltrada V3 yang sama pada setiap konfigurasi. Catat fingerprint citra dan label, argumen evaluator efektif, mAP50–95, AP per kelas, GFLOPs, parameter, dan latency. Benchmark latency memakai raw forward pada GPU dan kondisi yang sama dengan p50 dan p95. Laporkan `latency speed-up = p50 baseline ÷ p50 pruned` terpisah dari speed-up komputasi. Hanya `full_val` yang dapat menghasilkan pilihan student. Dari kandidat N/S dengan metrik lengkap, target tercapai, dan penurunan mAP50–95 maksimal dua poin, urutkan berdasarkan target tertinggi, mAP50–95 tertinggi, median latency terendah, lalu parameter terendah. Jika tidak ada kandidat yang memenuhi syarat, catat `NO_QUALIFYING_STUDENT`. Kunci konfigurasi dan hash sebelum evaluasi pada split test independen jika tersedia.

## Menjalankan notebook 05 N dan 06 S

Impor [notebook 05 YOLO26n](../../notebooks/kaggle/pruning/05_lamp_yolo26n_gflops_no_recovery.ipynb)
atau [notebook 06 YOLO26s](../../notebooks/kaggle/pruning/06_lamp_yolo26s_gflops_no_recovery.ipynb)
ke Kaggle dengan GPU dan dataset Poltrada V3 YOLO26 Clean. Gunakan secret W&B untuk mengambil
checkpoint atau isi `LOCAL_CHECKPOINT_PATHS` untuk model notebook tersebut dengan hash yang dikunci.
Default eksperimen memakai `MODEL_VARIANTS=["n"]` atau `["s"]` sesuai notebook dan `MODE="full_val"`, seluruh split val,
FP32 640 px, batch validasi 16, serta target GFLOPs 1,1×/1,2×/1,3×. Pilih `MODE="smoke"` untuk
pemeriksaan awal Kaggle pada subset tetap maksimal 128 citra. Setiap full_val berjalan sendiri dengan
output terpisah di `lamp_gflops_yolo26n/` atau `lamp_gflops_yolo26s/`. Untuk menjalankan `test` di sesi lain, lampirkan lock dan kedua checkpoint.
Atur `STUDENT_LOCK_PATH` serta kedua path checkpoint test jika path asli telah berubah.

Simpan folder suite dengan `suite_manifest.json`, `metrics.csv`, manifest dataset, dan
`target_search.json` tiap varian. Setiap kandidat menyimpan checkpoint native FP32, manifest channel,
perbandingan oracle, hasil reload, dan folder EDA. Untuk KD, serahkan objek DetectionModel
yang sudah dipangkas ke loader trainer struktural dan periksa hash serta bentuk tensor.

Setiap notebook mengekspor `student_candidates.json`. Setelah kedua full_val selesai, lampirkan
seluruh folder output varian lain pada sesi salah satu notebook, isi `COMPARISON_CANDIDATE_PATHS`
dengan lokasi file kandidatnya, lalu jalankan ulang sel terakhir. Ekspor harus berasal dari full_val
VERIFIED berstatus PASS, fingerprint dataset sama, serta protokol, implementasi, versi paket, dan
perangkat latency yang sama. Checkpoint relatif terhadap folder ekspor mendukung pemindahan output
ke Kaggle Dataset. Hash checkpoint student dan baseline diperiksa sebelum lock dibuat. Satu hasil
varian saja tetap `NOT_RUN` untuk pemilihan akhir. Gabungan tanpa kandidat layak menghasilkan
`NO_QUALIFYING_STUDENT` sesuai aturan seleksi semula.

Pemisahan 5 November 2026 menggantikan notebook gabungan 05 dengan dua notebook di atas.
Laporan JSON historis mempertahankan nama/hash notebook yang benar-benar diuji saat itu. Salinan
notebook gabungan sebelum pemisahan tersimpan lokal pada `outputs/pruning/notebook_migrations/20261105_split_ns/`.

Kedua notebook terpisah telah menjalankan ulang enam konfigurasi representatif per varian pada
CPU FP32 64 px dan fixture enam citra. Struktur, oracle, replay, reload proses baru, validasi fixture,
dan ekspor kandidat relatif dengan SHA-256 lulus. EDA individual tiap notebook juga direplay dari
bukti tersimpan. Full validation Poltrada V3, Kaggle, dan latency CUDA tetap `NOT_RUN`.
[Bukti pemisahan dan uji N/S](lamp_gflops_split_ns_verification_20261105.json)

## Kelengkapan analisis mengikuti notebook 03

Peninjauan 5 November 2026 membaca kembali sel analisis notebook 03 serta sumber EDA yang disisipkan
generatornya. Notebook 05 telah memakai helper diagram channel, mask, skor, dan evaluasi yang sama.
Tampilan sebelumnya berfokus pada satu konfigurasi per model. Rekap AP lintas konfigurasi dan
grafik sweep seperti notebook 03 belum disajikan sebagai satu alur analisis. Perbaikan ini
memperluas penyajian dan ekspor tanpa mengganti pemilihan channel atau target pruning.

| Analisis notebook 03 | Penyajian notebook 05 |
| --- | --- |
| Identitas, arsitektur, eligible dan protected | Sumber/hash, baseline parity, inventory arsitektur, accepted/rejected roots, serta kapasitas policy |
| Diagram Before/After/Comparison | Overview seluruh konfigurasi N/S, halaman detail, tabel channel input/output dan reduksi |
| Mask dan skor LAMP | Ledger indeks asli Conv/BN/konsumen, statistik skor finite, kapasitas dan keputusan tiap group |
| Parameter/GFLOPs per layer/stage | Tabel lengkap serta grafik stage, ukuran FP32, GFLOPs speed-up dan latency p50/p95/speed-up |
| mAP/P/R dan AP per kelas | Rekap semua konfigurasi, AP/P/R/F1/support dan delta AP terhadap baseline per ID kelas |
| Kurva dan confusion matrix | PR/F1/confidence, threshold aktual, orientasi predicted/true dan protokol validator efektif |
| Sweep rasio dan trade-off | Kontrol extended dipisahkan dari pencarian target, kapasitas tier, grafik trade-off per varian dan N/S |
| Kesimpulan dan reproducibility | Kontras P5, kelayakan student full_val, status oracle/replay/reload/evaluasi/latency/EDA terpisah |

Sel 11–14 mengekspor 17 tabel CSV/JSON ke `comparison/analysis/`. Tabel historis 12 baris pengguna
ikut notebook standalone dan diekspor terpisah dengan provenance `UNVERIFIED`. Grafik suite tidak
menggabungkan angka historis tersebut. `EDA_CONFIGURATIONS=()` menampilkan overview semua konfigurasi.
`EDA_DETAIL_CONFIGURATIONS=("target-1.1x",)` menampilkan detail numerik/mask/kurva satu target per varian;
isi tuple kosong untuk seluruh detail. Pengaturan tampilan tidak membatasi ekspor.

Pemeriksaan analisis menggunakan hasil CPU sintetis tersimpan dan dapat diulang dengan
`python scripts/verify_lamp_gflops_eda.py --suite <folder-suite>`. Replay ini menjalankan sel EDA terbaru,
tanpa menjalankan ulang pruning atau validator. Regresi numerik menguji pairing baseline N/S,
urutan ID kelas, pemisahan denominator channel, nilai metrik yang hilang, dan penolakan angka
latency dari catatan `NOT_RUN`. [Bukti verifikasi EDA](lamp_gflops_eda_verification_20261105.json)

## Pemeriksaan kapasitas komputasi lokal

Counter notebook pada input 640 × 640, CPU FP32 dan forward dual head unfused mengukur kapasitas
berikut pada checkpoint asli dengan SHA-256 terkunci. Untuk probe kapasitas ini DepGraph ditelusuri
pada input 64 × 64, sementara seluruh penghitungan GFLOPs memakai 640 × 640.

| Varian | GFLOPs baseline | Maksimum cap 25/10% | Maksimum cap 35/20% | Maksimum cap 50/50% |
| --- | ---: | ---: | ---: | ---: |
| N | 5,905408 | 1,065145× | 1,103491× | 1,223767× |
| S | 22,765978 | 1,068209× | 1,111523× | 1,232162× |

Target 1,3× berada di atas batas terukur kedua varian dengan proteksi ini. Ukuran tersebut
berasal dari counter operasi, tanpa pengukuran latency atau akurasi. Kontrol P5-protected
mempertahankan group output P5. Pemangkasan di hulu tetap dapat mengubah fitur masuk ke P5.

Uji integrasi CPU memakai checkpoint asli dan enam citra sintetis pada input 64 px.
Jalankan ulang dengan `python scripts/verify_lamp_gflops_cpu.py --imgsz 64 --fixture --representative`.
Mode representatif mengevaluasi extended 10%, kedua probe P5, dan kandidat target tanpa merender EDA.
EDA lengkap diperiksa terpisah pada N extended 10%. [Bukti pengujian](lamp_gflops_cpu_verification_20261104.json)

Status awal pencatatan audit untuk implementasi notebook 05, uji notebook, smoke Kaggle, latency CUDA, `full_val`, dan test adalah `NOT_RUN`. Hasil eksekusi ditambahkan ke log terpisah menurut lingkungan dan cakupan bukti.
