# Audit dan perbaikan adaptasi CSAKD pada YOLO26

Tanggal audit: 13 Oktober 2026 (Asia/Jakarta). Lingkup: notebook eksperimen 08 dan 09; notebook 01–07 diperiksa sebagai pembanding protokol. Nomor cell pada laporan ini berbasis **1**, mencakup code dan markdown cell; bukan `execution_count`.

## A. Executive Summary

**Kode awal: FAIL.** Ada bug pada target teacher yang mengubah makna AGKA/AAWM: forward ulang modul teacher menulis output cross ke dictionary hook sebelum dictionary tersebut dibaca sebagai target teacher. Loss juga berbeda secara numerik dari kode resmi penulis pada arah attention, letak skala, normalisasi, residual guidance, dan reduksi. Checkpoint wrapper mentah per epoch tidak portabel ke proses Python baru tanpa class `CSAKDModel` dari notebook.

**Status setelah patch: PASS WITH WARNINGS untuk implementasi yang diperiksa pada CPU; belum siap untuk eksperimen final.** Suite gabungan mencatat **48 PASS, 2 SKIP (NOT TESTED), 0 FAIL**. Kedua student menyelesaikan satu epoch diagnostik dengan dua batch sintetis, validation, checkpoint per epoch, reload dalam proses baru, dan logging. Smoke pada teacher artifact v50, Poltrada V3, CUDA AMP, dan W&B online belum dilakukan; tidak boleh dianggap PASS berdasarkan fixture CPU.

Adaptasi ini mempertahankan enam modul `[4, 6, 10, 16, 19, 22]`, teacher YOLO26m, student YOLO26s/n, dan koefisien AGKA/Cross `1/1`. Klaim yang dapat dipertanggungjawabkan adalah **adaptasi CSAKD dengan enam modul dan penyelarasan channel pada YOLO26m→s/n**. Klaim reproduksi identik tidak tepat: paper menggunakan YOLOv8s dan student hasil hybrid quantization, sedangkan penelitian ini mengubah kapasitas model dan domain data. [Paper primer CSAKD, BMVC 2025](https://bmva-archive.org.uk/bmvc/2025/assets/papers/Paper_398/paper.pdf).

Konfigurasi bersama tidak diubah: Poltrada V3; 100 epoch; input 640; batch 16; seed 42; AdamW; lr 0.001; AMP aktif; weight decay 0.0005; warmup 3; close mosaic 10; patience 100; evaluasi akhir `val`. Tidak ada training 100 epoch yang dijalankan. Dataset lokal V2 tidak dipakai sebagai pengganti V3.

## B. Paper-to-Code Mapping

Acuan numerik adalah [kode penulis `OtherKD/CSAKD.py`](https://github.com/KefanZhan/YOLOv8-KD/blob/3433559ab6c9cfdef3b4ac099f1bb2feddd42dc8/OtherKD/CSAKD.py), dipatok pada commit `3433559ab6c9cfdef3b4ac099f1bb2feddd42dc8`. Persamaan paper dan implementasi kode penulis tidak identik. Pilihan mengikuti kode penulis sudah ditetapkan dalam rencana implementasi pengguna.

| Komponen / persamaan paper | Lokasi notebook 08 dan 09 | Implementasi yang dimaksud setelah patch | Klasifikasi |
|---|---|---|---|
| Objective gabungan, Eq.(1) | Cell12 `combined_loss`; cell13 `CSAKDModel.loss` | Detection bawaan + jumlah AGKA dan Cross enam tap; koefisien 1/1 | Prinsip sesuai; bobot/cakupan adaptasi |
| Cross feature, Eq.(2–4) | Cell13 `InputHook`, `CSAKDModel.kd_loss` | Input student aktual pada modul terpilih → projector input → modul teacher frozen | Adaptasi graph dan channel |
| Cross Supervising, Eq.(5),(10) | Cell13 `csakd_losses` | MSE mean fitur student/cross yang dinormalisasi L2 channel, dikali AAWM | Sesuai kode penulis; beda notasi norm paper |
| GAP/similarity/AAWM, Eq.(6–9) | Cell13 `csakd_losses` | Dot product GAP dibagi √C; mean batch sebelum sigmoid; satu skalar per tap | Sesuai kode penulis |
| Q/K/attention, Eq.(11–13) | Cell13 `csakd_losses` | `ReLU(cross) @ ReLU(target).transpose(-2,-1) / sqrt(C)` sebelum softmax | Sesuai kode penulis; Q/K dan skala beda paper |
| Guidance/AGKA, Eq.(14–15) | Cell13 `csakd_losses` | `(student + attention @ student)/2`; channel normalize; MSE mean terhadap target normalize | Sesuai kode penulis; residual/reduksi beda paper |
| Frozen teacher | Cell12 `combined_loss`; class dasar Ultralytics; optimizer CSAKD | Target gambar di `no_grad`; cross-forward tetap differentiable terhadap input | Sesuai prinsip; gradien teacher dihentikan sengaja |
| Alignment | Cell12 `align_channels`; cell13 constructor | Dua adapter1×1 per tap bila channel berbeda; identity bila sama; spatial harus sama | Diperlukan untuk m→s/n |
| Backbone/neck supervision | Cell13 `module_indices` | Enam modul, bukan seluruh modul backbone/neck | Penyederhanaan yang harus diungkapkan |

Paper menggunakan Q dari teacher dan K dari cross, menaruh √d sesudah softmax, serta menulis guidance tanpa pembagian 2. Kode penulis menukar Q/K, menaruh skala sebelum softmax, membagi guidance 2, dan memakai normalized MSE. Notebook mengikuti kode penulis secara eksplisit. [Persamaan CSAKD](https://bmva-archive.org.uk/bmvc/2025/assets/papers/Paper_398/paper.pdf).

### Arsitektur Ultralytics yang benar-benar diperiksa

Notebook memasang `ultralytics==8.4.155` dan `wandb==0.28.1`. Lingkungan diagnostik juga memakai Ultralytics 8.4.155; pemeriksaan hash RECORD terhadap enam sumber inti cocok. Audit membaca sumber terpasang `cfg/models/26/yolo26.yaml`, `nn/tasks.py`, `nn/modules/head.py`, `nn/distill_model.py`, `utils/loss.py`, `utils/torch_utils.py`, dan trainer deteksi. Kesimpulan berikut tidak diturunkan dari asumsi struktur YOLOv8/YOLO11.

Backbone berakhir pada modul 10 (`C2PSA`); modul 4/6 adalah `C3k2`. Neck menggunakan upsample/concat dan output `C3k2` pada16/19/22. Modul23 adalah `Detect` dengan input `[16,19,22]`, `end2end=True`, dan `reg_max=1`. Teacher dan student memiliki routing/resolusi sepadan, tetapi channel dan sebagian struktur internal berbeda. Input channel neck adalah hasil concat, sehingga alignment hanya pada output tap tidak cukup.

| Tap / lokasi | Spatial input/output pada640 | Teacher m: Cin→Cout | Student s: Cin→Cout | Student n: Cin→Cout |
|---|---:|---:|---:|---:|
| 4 / backbone P3 | 80×80 | 256→512 | 128→256 | 64→128 |
| 6 / backbone P4 | 40×40 | 512→512 | 256→256 | 128→128 |
| 10 / backbone P5 | 20×20 | 512→512 | 512→512 | 256→256 |
| 16 / neck P3 | 80×80 | 1024→256 | 512→128 | 256→64 |
| 19 / neck P4 | 40×40 | 768→512 | 384→256 | 192→128 |
| 22 / neck P5 | 20×20 | 1024→512 | 768→512 | 384→256 |

Channel di atas diperiksa melalui hook actual model acak. Forward640 CPU sebelum patch menghasilkan output evaluasi `(1,300,6)` finite untuk m/s/n; angka6 pada format ini adalah `xyxy,confidence,class`, bukan jumlah kelas. Uji constructor terpisah pada input 64 juga mengonfirmasi seluruh Cin/Cout. Ini bukti kompatibilitas graph, bukan kualitas checkpoint penelitian.

Detection criterion aktual adalah `E2ELoss`: dua criterion one-to-many/one-to-one dengan bobot awal 0.8/0.2, kemudian one-to-many turun hingga 0.1 mengikuti epoch. One-to-one detach input neck ketika training adalah perilaku resmi yang sengaja mencegah gradien branch itu menuju backbone; branch one-to-many dan KD tetap memberi gradien backbone. Dengan `reg_max=1`, komponen ketiga yang dicatat adalah **`l1_loss`**, yakni L1 ltrb ternormalisasi; bukan DFL distribusi. Nama internal `loss_dfl` pada `BboxLoss` tidak boleh disalahartikan sebagai penggunaan DFL. Tidak ada objectness loss terpisah yang ditambahkan oleh CSAKD.

## C. Temuan Audit

Tabel mencatat kode awal dan tindakan perbaikannya. Bukti runtime setelah patch dirinci pada F.

| Severity | Lokasi file/cell/fungsi | Masalah / kategori | Bukti teknis dan akar masalah | Dampak | Perbaikan |
|---|---|---|---|---|---|
| **Critical** | 08/09 cell13 `CSAKDModel.kd_loss` | Target teacher tertimpa / bug | `cross = teacher.model[index](incoming)` menjalankan `FeatureHook` yang menulis `_teacher_feats[index]`; `target = _teacher_feats[index]` baru dibaca sesudahnya. Target sama dengan cross, bukan fitur gambar teacher asli | AGKA membimbing terhadap cross; AAWM memakai cross sebagai teacher; hasil lama tidak mewakili metode yang dimaksud | Snapshot seluruh target `detach().clone()` sebelum cross-forward pertama |
| **High** | 08/09 cell13 `csakd_losses` | Formula menyimpang dari kode acuan / penyimpangan metode | Kode awal `t@cᵀ`, softmax/√C, residual tanpa /2, vector norm unnormalized, sigmoid per sampel. Acuan yang dipilih menghendaki `c@tᵀ`, scale sebelum softmax, normalize channel, MSE mean, mean similarity batch | Semantik attention, gradien, dan skala KD berbeda | Ganti hanya formula; koefisien 1/1 tetap; tambah oracle numerik dan gradien |
| **High** | 08/09 cell 15 `save_epoch_artifact` | Checkpoint per epoch tidak portabel / bug deployment | Checkpoint live mengandung `CSAKDModel` pickle; proses baru tanpa class gagal memuat. Final strip bawaan tidak menyelesaikan artifact epoch terdahulu | Artifact `best.pt`/`last.pt` dapat gagal inferensi di lingkungan bersih; deployment dapat membawa wrapper | Salinan student-only melalui `strip_optimizer(source,destination)`; checkpoint training asli tetap disimpan dengan nama berbeda |
| **Medium** | 08/09 cell13 `KDTrainer`, optimizer bawaan | Teacher frozen masih tercantum optimizer / hygiene integrasi | `requires_grad=False` menghentikan update teacher, tetapi parameter grup bawaan tidak memfilter semua frozen parameters | Bukan bukti teacher ter-update; grup optimizer tidak memenuhi invariant eksplisit penelitian | Filter menurut `requires_grad`; uji teacher tidak tercakup dan semua trainable parameters tercakup |
| **Medium** | 08/09 cell12 `combined_loss`; cell 15 logging | Item detection bukan objective detection aktual / diagnostik | `E2ELoss` mengembalikan weighted loss kedua branch, tetapi item deteksi hanya one-to-one | Menjumlahkan grafik box/cls/L1 dapat salah menilai dominasi KD | Tambah `base.sum()/B`, weighted KD, dan rasio; tidak memasukkan item diagnostik ke backward |
| **Medium** | 08/09 cell9 teacher loading | Provenance teacher belum lengkap / validitas eksperimen | Ref v50 dan pemeriksaan kelas tersedia; metadata artifact, producer, digest, SHA256 dan dataset belum seluruhnya direkam/diuji | Teacher yang berbeda dapat merusak komparasi | Verifikasi exact ref, arsitektur/kelas/dataset; catat provenance; fail bila mismatch; tanpa fallback pretrained |
| **Medium** | 08/09 cell13 + callback awal epoch | Batch/AMP efektif dapat berbeda dari konfigurasi / protokol | Recovery OOM Ultralytics dapat menurunkan batch; AMP efektif ditentukan trainer | Konfigurasi nominal16 belum membuktikan batch efektif16 | Catat batch/AMP aktual; hentikan run penelitian bila batch berubah |
| **Low** | 08/09 cell1, cell11, cell 15 narasi | Teks COCO8/10 epoch tidak sesuai konfigurasi utama / dokumentasi | Code config memakai Poltrada V3/100 epoch, markdown masih demo | Interpretasi laporan/hasil ambigu | Koreksi narasi penelitian dan label diagnostic fixture |
| **Medium** | 08/09 cell13 `module_indices`, adapters | Enam tap dan routing lokal / adaptasi | Cross dijalankan pada satu modul terhadap input student aktual, tidak mengganti stream graph teacher keseluruhan; ada dua adapter trainable per tap | Modul m/s/n tidak menerima input numerik identik tanpa projection; efek adapter ikut berkontribusi | Pertahankan pilihan yang disetujui; ungkapkan dalam skripsi dan catat parameter/biaya training |
| **Medium** | 08/09 cell3 bobot 1/1 | Bobot adaptasi belum divalidasi untuk Poltrada / risiko metodologi | Normalized mean MSE memiliki skala kecil; diagnostik oracle acak menunjukkan rasio KD/detection sekitar 0.0045–0.0049 | KD dapat memberi kontribusi kecil meskipun gradien nonzero | Catat rasio dan gradien; jangan mengganti bobot dari intuisi atau memakai test untuk tuning |

Cache dibersihkan sebelum batch dan melalui `finally` sejak kode awal; cleanup itu dipertahankan. Registrasi ulang pre-hook menghapus `InputHook` lama, sedangkan class dasar menghapus `FeatureHook` lama. Audit perlu membuktikan lifecycle saat register ulang/deepcopy/EMA, bukan menyebut memory leak hanya karena ada hook.

## D. Audit Matematika dan Gradient Flow

Untuk setiap tap (i), tulis (x_i^s,y_i^s,y_i^t) sebagai input student, output student, dan output teacher asli dari **gambar batch yang sama**. Dua alignment adalah (P_i^{in},P_i^{out}): identity atau conv1×1 tanpa bias. Setelah forward teacher, simpan (t_i=\mathrm{detach}(y_i^t).\mathrm{clone}()). Kemudian:

\[
s_i=P_i^{out}(y_i^s),\qquad c_i=M_i^t(P_i^{in}(x_i^s)).
\]

Cross memakai parameter teacher frozen, tetapi `c_i` tidak detach. Ini memungkinkan backward melalui operasi modul teacher menuju projector input dan fitur input student. Target (t_i) tidak mendapat gradien. Normalisasi dan loss yang dipakai:

\[
N_C(z)=\operatorname{normalize}(z,p=2,dim=1,eps=10^{-12}),
\]

\[
A_i=\operatorname{softmax}_{-1}\left(\operatorname{ReLU}(c_i)\operatorname{ReLU}(t_i)^T/\sqrt{C_i}\right),
\]

\[
L_{AGKA,i}=\operatorname{MSE}_{mean}\left(N_C((s_i+A_i s_i)/2),N_C(t_i)\right),
\]

\[
w_i=1-\sigma\left(\operatorname{mean}_B[(\operatorname{GAP}(t_i)\cdot\operatorname{GAP}(s_i))/\sqrt{C_i}]\right),
\]

\[
L_{Cross,i}=w_i\operatorname{MSE}_{mean}\left(N_C(s_i),N_C(c_i)\right).
\]

MSE `mean` membagi oleh seluruh (B C H W); jumlah enam tap tidak dibagi lagi oleh jumlah tap. AAWM adalah satu scalar per tap; mean batch terjadi **sebelum** sigmoid, sehingga tidak ekuivalen dengan rata-rata sigmoid per sampel. FP32 dipakai untuk projector cross dan loss KD di dalam context autocast disabled; detection training tetap mengikuti AMP trainer.

Matmul terakhir pada BCHW menghasilkan attention **B×C×H×H**, menerjemahkan operasi row-wise kode penulis; bukan attention `(HW)×(HW)`, bukan channel attention, dan bukan mask objek. Softmax per baris menghasilkan jumlah 1. Skala memakai √C mengikuti acuan, walaupun axis kontraksi matmul adalah W. `eps` melindungi pembagi normalisasi dari nol; ia tidak menjamin gradien kecil pada fitur yang hampir nol. Karena itu uji finite harus mencakup nol/mendekati nol, dan CUDA AMP masih perlu diuji.

Teacher, snapshot, student, dan backward:

1. Cache lama dibersihkan. Teacher `eval` + frozen menjalankan batch gambar dalam `no_grad`; enam target disalin sebelum cross.
2. Student forward mengisi input/output hook. Detection criterion bawaan menghitung weighted objective kedua branch.
3. Projector output dan input menghasilkan `s` dan input cross. Frozen module teacher membentuk `c`; hook boleh mengganti cache hidup, tetapi snapshot target tidak berubah.
4. AGKA membentuk attention dari cross dan target; Cross membandingkan student/cross dengan bobot AAWM. `c`, attention, dan AAWM tetap differentiable; hanya target teacher dihentikan.
5. Objective yang dikembalikan adalah concatenation detection vector dan **KD mean × batch size sekali**. Trainer menjumlahkan vector. Diagnostic items detach dan tidak dijumlahkan kembali ke objective.
6. Optimizer hanya memuat student/projector trainable. Teacher parameters dan statistik BatchNorm tidak boleh berubah; hook/cache dilepas atau dikosongkan sesuai lifecycle.

`base.sum()/B` adalah detection objective mean per gambar yang benar untuk dibandingkan dengan KD mean. Item `box_loss`, `cls_loss`, dan `l1_loss` tetap dipertahankan, tetapi menggambarkan loss one-to-one yang dikembalikan E2ELoss, bukan keseluruhan weighted objective. Tidak ada pembobotan batch tambahan pada logging.

Teacher `no_grad`, target `detach`, dan detach one-to-one dari neck adalah penghentian gradien yang disengaja. Menaruh **cross-forward** di `no_grad` atau detach cross/attention/AAWM akan menjadi bug karena memutus jalur yang disetujui. Gradien nol untuk dua fitur identik/fitur nol dapat benar secara matematis; kriteria nonzero diuji pada fixture nondegenerate.

## E. Perubahan Kode

Perubahan dibatasi pada notebook 08/09, `tests/test_csakd_notebooks.py`, `tests/test_csakd_provenance.py`, dan laporan ini. Notebook teacher, baseline, Native KD, dan CrossKD tidak diubah.

| Bagian | Patch / alasan |
|---|---|
| Cell12–13 / supervision dan loss | Snapshot detached cloned sebelum cross; formula numerik penulis; pemeriksaan shape/finite; cleanup exception; teacher tetap frozen dan cross differentiable |
| Cell13 / trainer | Filter optimizer trainable; pertahankan E2ELoss dan jadwal bawaan; audit invariant batch/AMP efektif |
| Cell15 / logging | Detection objective aktual, weighted KD dan rasio, config formula/taps/bobot/provenance; callback epoch tetap idempotent |
| Cell15 / artifact | `best.pt`/`last.pt` portabel student-only dari salinan; checkpoint training asli `training_state_best.pt`/`training_state_last.pt`; alias/`.wait()` tetap |
| Cell9 / teacher | Exact artifact v50, model/kelas/dataset, digest/producer/SHA256; tanpa fallback pretrained |
| Cell1,11,17,19 / dokumentasi dan evaluasi | Narasi V3/100 epoch, adaptasi numerik/cakupan, label validation, metadata run aktual; inferensi student-only |
| Tests | Ekstraksi definisi class/loss dari notebook tanpa instalasi/auth/training 100; oracle, target mutation, gradien, hooks/EMA, checkpoint proses baru, trainer satu epoch sintetis |

Checkpoint training asli yang diarsipkan sebagai `training_state_*` dalam artifact per epoch tetap membutuhkan definisi class notebook; CSAKD resume/compile tetap ditolak. Nama tersebut tidak menyiratkan bahwa resume telah didukung. Snapshot deployment menghapus teacher, projector, criterion training dan hook. `strip_optimizer(source,destination)` memakai tujuan berbeda agar tidak mengubah file live sebelum upload. Snapshot raw dan portabel sama-sama dapat diperiksa dalam artifact tiap epoch. Pada `final_eval`, Ultralytics tetap melakukan strip bawaan terhadap `weights/best.pt` dan `weights/last.pt`; karena itu file weights setelah training selesai dapat sudah student-only, sedangkan state asli per epoch dipertahankan di artifact.

Protokol notebook 01–09 yang diperiksa statik konsisten pada konfigurasi utama dan augmentasi bersama. `dis=1.0` pada CSAKD tidak menambahkan KD native kedua: wrapper hanya memakai objective custom dengan bobot AGKA/Cross. `KD_WEIGHT=6.0` Native dan bobot CrossKD1/1 tetap merupakan konfigurasi metode pembanding, bukan alasan menyamakan angka bobot lintas loss berbeda.

## F. Hasil Pengujian

Lingkungan diagnostik aktual: Windows; Python 3.11.14; PyTorch 2.14.0+cpu; Ultralytics 8.4.155; W&B 0.28.1; pytest 9.1.1. `torch.cuda.is_available()` adalah `False` dan `torch.version.cuda` adalah `None`. Versi ini dicatat sebagai lingkungan audit CPU, bukan lingkungan Kaggle eksperimen final.

### Bukti sebelum patch (diagnostik historis, bukan penerimaan patch)

| Uji / lingkungan | Status | Bukti / batas |
|---|---|---|
| Forward/backward/optimizer step notebook asli, CPU B2/64, nc=4, teacher/student acak | PASS eksekusi | m→s dan m→n berjalan; tidak membuktikan kesetiaan metode |
| Target teacher tidak berubah setelah cross pada kode asli | **FAIL** | Hook aktual mengganti `_teacher_feats[index]`; pembacaan target terjadi sesudahnya |
| Checkpoint wrapper mentah di proses baru | **FAIL** | Tidak menemukan `CSAKDModel` pada `__main__` |
| Checkpoint plain student di proses baru | PASS | Bukti feasibility snapshot deployment; bukan artifact penelitian |
| Oracle formula yang diperbaiki, CPU B2/64, nc=6 | PASS diagnostik formula | AGKA/Cross terpisah:234 tensor student mendapat gradien finite/nonzero pada kedua student; 9 adapter s, 12 adapter n; teacher parameter/BN tidak berubah |
| Forward640 m/s/n CPU, B1/nc=6, model acak | PASS | Format evaluasi `(1,300,6)` finite; tap/resolusi valid |

Pada oracle sementara tersebut, m→s memberi AGKA 0.0275687, Cross 0.0135896, detection mean 8.38115, KD/detection 0.00491081, dan rasio norma gradien KD/detection 0.000654. m→n memberi 0.0274113,0.0137183,9.08448,0.00452746, dan0.000925. Angka ini **bukan** hasil training atau ukuran performa Poltrada; bobot 1/1 tetap karena fixture acak tidak membenarkan retuning.

### Penerimaan setelah patch

Suite gabungan final selesai: **48 PASS, 2 SKIP, 0 FAIL**, durasi 46.13 detik. Sebanyak 16 PASS berasal dari regresi metode/trainer dan 32 PASS dari provenance mock pada kedua notebook. Assertion memeriksa setiap projector trainable dan keenam input cross, serta seluruh state teacher setelah optimizer step. Semua PASS di bawah terbatas pada fixture CPU/mock yang disebutkan, bukan teacher/data penelitian. Bukti mesin tersedia di `outputs/test_csakd/final_results.xml`, dan stdout lengkap di `outputs/test_csakd/final_test_output.txt`; kedua skip adalah smoke CUDA/Poltrada/teacher aktual untuk s dan n. Empat warning hanya deprecation `distutils.Version` dari dependency THOP.

Perintah reproduksi suite final dari root repositori:

```powershell
.venv/Scripts/python.exe -m pytest tests/test_csakd_notebooks.py tests/test_csakd_provenance.py -o addopts= -q --tb=short --junitxml=outputs/test_csakd/final_results.xml
```

| Smoke/regresi wajib | Status setelah patch | Bukti / lingkungan |
|---|---|---|
| Target immutable, teacher image sensitivity | PASS CPU | Keenam target sama dengan snapshot walaupun hook cross menimpa cache; perubahan target asli mengubah AGKA |
| Paritas loss/oracle dan reduksi | PASS CPU | Formula kode penulis termasuk batch mean AAWM; nilai loss dan gradien cocok |
| Loss/gradien ordinary, zero, near-zero finite | PASS CPU | FP32 pada skala 1, 0, 1e-14; zero gradient sah dibedakan dari bug |
| Forward dan shape kedua student | PASS CPU s dan n | Synthetic CPU 64 dan forward 640; alignment tap sesuai |
| Backward detection/AGKA/Cross terpisah | PASS CPU s dan n | Kedua head detection menerima gradien; student, **setiap projector trainable**, dan keenam input cross finite/nonzero untuk AGKA maupun Cross |
| Frozen teacher dan optimizer coverage | PASS CPU s dan n | Teacher tidak tercakup optimizer, grad=None; seluruh state teacher termasuk BN tetap setelah optimizer step |
| Student berubah setelah optimizer step | PASS CPU s dan n | Perbandingan parameter sebelum/sesudah |
| Cache/failure cleanup, hooks/deepcopy/EMA | PASS CPU | Tidak stale/duplikat; validation loss/EMA berjalan |
| Satu epoch beberapa batch + validation | **PASS CPU s dan n** | Empat gambar train/dua batch dan dua gambar val sintetis; nc=6, input 64, batch 2, nbs 2, AMP false |
| Checkpoint save/reload/inferensi di proses baru | **PASS CPU s dan n** | Raw source hash tetap; snapshot per epoch dan final best dimuat tanpa class notebook; prediksi finite |
| Logging/artifact callback idempotent | PASS CPU + mock | Komponen/metrik CSV ada; mock membuktikan raw+portable file names, upload `.wait()`, alias, dan penolakan duplicate epoch; tidak membuktikan server W&B |
| Recovery OOM / batch dan AMP guard | PASS mock | Perubahan batch retry ditolak; perubahan batch efektif dan penonaktifan CUDA AMP penelitian ditolak |
| Syntax/static consistency | PASS | AST seluruh 76 code cells notebook eksperimen valid; cell implementasi bersama s/n identik; lint kedua file test bersih |
| Teacher provenance guards | **32 PASS mock** | Kedua notebook: deklarasi cocok/absen, mismatch dataset/model/producer/version/kelas/keluarga/scale, manifest/checksum, metadata incomplete; tidak mengunduh checkpoint actual |
| Teacher artifact exact v50 metadata/manifest/best.pt | **NOT TESTED** | Permintaan API audit sebelumnya menghasilkan `AuthenticationError` |
| Dua batch V3 train + satu batch V3 val, batch 16/input 640 | **NOT TESTED** | Poltrada V3 Kaggle tidak tersedia lokal; tidak diganti dataset V2 |
| CUDA AMP, unscale gradien, memory batch 16 | **NOT TESTED** | GPU tidak tersedia pada lingkungan audit |
| W&B online actual upload/download artifact | **NOT TESTED** | Memerlukan akses account/artifact dan run nyata |
| Full 100 epoch | **NOT TESTED** | Sengaja tidak dijalankan, belum diinstruksikan |

Precision, recall, mAP50 dan mAP50-95 pada fixture validation acak bernilai 0. Nilai tersebut finite dan cukup membuktikan validator menjalankan alurnya; tidak menunjukkan kualitas detektor pada Poltrada. CSV fixture juga memuat detection `box_loss/cls_loss/l1_loss`, setiap KD component, `dis_loss`, `det_total_loss`, `kd_to_det_ratio`, validation metrics dan learning rate. Rasio KD/detection pada fixture satu epoch sekitar 0.0047–0.0054; ini konsisten dengan kontribusi objective yang kecil, tetapi tetap tidak membenarkan perubahan bobot dari data sintetis.

Smoke GPU yang tersedia di suite bersifat opt-in dan otomatis SKIP bila aset/GPU belum ada. `CSAKD_GPU_SMOKE=1` membutuhkan `CSAKD_TEACHER_CHECKPOINT`, `CSAKD_TEACHER_PROVENANCE_JSON` yang dihasilkan retrieval terverifikasi, kedua checkpoint pretrained student lokal (`CSAKD_STUDENT_S_CHECKPOINT`, `CSAKD_STUDENT_N_CHECKPOINT`), serta YAML V3 asli (`CSAKD_DATA_YAML` bila lokasi berbeda). Test mengambil 32 gambar **train asli** dan 16 gambar **val asli** ke fixture terpisah, menjalankan dua batch training 640/batch 16 dan satu batch validation; tidak mengganti atau menulis ulang split penelitian. Ini bukan instruksi untuk memalsukan provenance atau memakai V2.

Pada lingkungan GPU, jalankan cell 1–13 notebook untuk setup/verifikasi dan definisi metode, **tanpa cell 15 training 100 epoch**. Setelah retrieval berhasil, ekspor provenance terverifikasi:

```python
provenance_path = Path("/kaggle/working/teacher_v50_provenance.json")
provenance_path.write_text(json.dumps(TEACHER_PROVENANCE, indent=2), encoding="utf-8")
os.environ["CSAKD_GPU_SMOKE"] = "1"
os.environ["CSAKD_TEACHER_CHECKPOINT"] = str(teacher_path)
os.environ["CSAKD_TEACHER_PROVENANCE_JSON"] = str(provenance_path)
os.environ["CSAKD_DATA_YAML"] = DATA
# Isi dengan checkpoint pretrained student yang sudah tersedia lokal.
os.environ["CSAKD_STUDENT_S_CHECKPOINT"] = "/kaggle/working/yolo26s.pt"
os.environ["CSAKD_STUDENT_N_CHECKPOINT"] = "/kaggle/working/yolo26n.pt"
subprocess.run([
    sys.executable, "-m", "pytest", "tests/test_csakd_notebooks.py",
    "-o", "addopts=", "-q", "-k", "actual_poltrada",
], check=True)
```

Perintah memerlukan checkout repositori sebagai working directory dan paket pytest; path checkpoint student disesuaikan ke file lokal yang benar. Mode smoke tidak menjalankan cell 15 maupun training 100 epoch, dan test menolak provenance yang tidak lengkap atau hash checkpoint teacher yang berbeda.

Fixture diagnostik memakai subdirectory unik pada `outputs/test_csakd` untuk menghindari masalah ACL temporary directory Windows yang teridentifikasi saat pengujian. Perbaikan ini terbatas pada lingkungan test dan tidak mengubah notebook atau protokol eksperimen.

### Verifikasi teacher dan pemisahan validation/test

Ref yang harus diunduh tetap `ilhamrafiqin-brawijaya-university/yolom26-kd-indotraffic/01_teacher_yolo26m-fux5wdcb-checkpoints:v50`, dengan file `best.pt`. Alias versi artifact **v50 bukan epoch 50**; epoch harus dibaca dari metadata producer/checkpoint. Tidak ada bukti dalam audit ini untuk mengklaim file tersebut telah diperiksa byte-per-byte atau benar-benar teacher eksperimen yang dituju. Runtime checks memverifikasi keluarga YOLO26 dan skala m, urutan enam kelas, dataset, producer run, manifest, digest dan hash sebelum training; mismatch menghentikan eksperimen. Metadata tidak lengkap juga menghentikan run penelitian; mode diagnostic hanya dapat melanjutkan dengan label provenance belum terverifikasi.

Pemilihan best checkpoint memakai validation fitness trainer. Evaluasi akhir juga `split='val'`; precision/recall/mAP50/mAP50-95 pada output itu adalah **validation**, bukan testing. Test tidak dipakai dalam smoke CPU, training, tuning, atau pemilihan checkpoint. Penelitian yang memerlukan klaim generalisasi test masih membutuhkan evaluasi terpisah sekali setelah pilihan metode/checkpoint dikunci.

## G. Remaining Risks

- **Blocker eksperimen aktual:** teacher artifact v50 belum diverifikasi karena autentikasi; Poltrada V3 dan GPU/CUDA AMP tidak tersedia lokal. Uji CPU tidak membuktikan kompatibilitas seluruh stack Kaggle/driver atau penggunaan memori batch 16/input 640.
- **Fidelity yang harus diungkapkan:** metode memakai m→s/n, dua alignment per tap, enam sparse tap, local cross-forward, normalisasi/reduksi kode penulis, dan koefisien 1/1. Tidak memakai HQ/self-distillation identik paper. Ini batas klaim penelitian, bukan otomatis bug.
- **Bobot:** paper menampilkan kombinasi α=400/β=10 pada NWPU VHR-10; koefisien 1/1 penelitian ini belum dibuktikan optimal. [Tabel3 CSAKD](https://bmva-archive.org.uk/bmvc/2025/assets/papers/Paper_398/paper.pdf). Lakukan diagnostik skala pada train/val asli; jika retuning diputuskan kemudian, gunakan protokol pradaftar yang sama antar metode dan jangan memakai test.
- **Attention dan numerik:** B×C×H×H row-wise berbeda dari full spatial attention. √C mengikuti kode penulis, bukan inferensi dimensional baru. FP32 membantu overflow AMP tetapi fitur hampir nol masih dapat memiliki gradien besar setelah normalisasi; finite-gradient check pada CUDA diperlukan.
- **Training cost:** forward teacher utuh ditambah enam cross modules dan attention/adapter menambah memori/komputasi training. Parameter/GFLOPs inference harus dihitung dari snapshot student, sementara biaya training dilaporkan terpisah.
- **Reproducibility:** seed 42 saja tidak menjamin hasil bitwise sama lintas GPU/driver. Catat versi runtime, artifact hash, data version, AMP/batch aktual, serta early-stop/completed epochs. Tidak ada perubahan split yang dilakukan dalam patch.
- **Portability/resume:** file training-state tetap raw pickle notebook dan resume belum didukung. Snapshot student-only harus lolos proses baru sebelum artifact dipakai deployment.
- **Comparability:** konsistensi konfigurasi statik tidak membuktikan hasil eksperimen sah. Seluruh metode harus memakai teacher/data/init student/protokol validation sama dan melaporkan varian adaptasi secara transparan. Tidak ada sertifikasi metodologi Native/CrossKD dari audit CSAKD ini saja.
- **Hasil kode lama:** run yang sudah memakai target tertimpa atau formula lama tidak boleh digabung sebagai hasil implementasi sesudah patch; perlu diulang dengan hash kode/config baru.

## H. Keputusan Akhir

| Pertanyaan | Keputusan saat ini | Syarat pelepasan blocker |
|---|---|---|
| 1. CSAKD YOLO26s siap 100 epoch? | **Belum.** Regresi CPU PASS; smoke penelitian aktual belum selesai | Teacher v50 terverifikasi; V3 train/val+CUDA AMP batch 16 berhasil; checkpoint penelitian portabel valid |
| 2. CSAKD YOLO26n siap 100 epoch? | **Belum.** Syarat sama dan diperiksa terpisah untuk n | Tidak mewarisi PASS s; seluruh smoke n harus terbukti |
| 3. Cukup setia untuk klaim adaptasi penelitian? | **Bersyarat, ya sebagai adaptasi yang dijelaskan; bukan reproduksi identik** | Formula acuan/paritas dan snapshot target terbukti; perubahan domain, kapasitas, alignment/tap/bobot dinyatakan |
| 4. Layak dibandingkan Native KD/CrossKD/baseline? | **Bersyarat.** Protokol statik konsisten; hasil aktual belum disertifikasi | Teacher/data/config sama; evaluasi validation/test jelas; versi perbaikan dipakai semua run CSAKD |
| 5. Ada blocker sebelum eksperimen final? | **Ya** | Provenance teacher, smoke V3/CUDA AMP, validation/reload/inferensi dan W&B online aktual; regresi CPU sudah PASS |

Keputusan ini tidak mengizinkan full training sekarang. Setelah seluruh blocker yang relevan PASS, jalankan eksperimen final hanya atas instruksi eksplisit pengguna.
