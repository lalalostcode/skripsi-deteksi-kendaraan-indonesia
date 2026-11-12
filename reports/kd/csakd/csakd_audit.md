# Audit dan perbaikan adaptasi CSAKD pada YOLO26

Audit awal: 13 Oktober 2026. Pembaruan OOM: **14 Oktober 2026** (Asia/Jakarta). Lingkup: notebook eksperimen 08 dan 09; notebook 01–07 diperiksa sebagai pembanding protokol. Nomor cell pada laporan ini berbasis **1**, mencakup code dan markdown cell; bukan `execution_count`.

**Status terkini: PASS WITH WARNINGS untuk implementasi CPU setelah optimasi memori; belum siap 100 epoch.** Run Kaggle YOLO26s sebelum optimasi gagal pada cross-forward FP32 karena OOM pada GPU 0 T4/batch 16. Log pengguna mengonfirmasi download teacher v50, producer, hash dan AMP compatibility check; ini memperbarui beberapa `NOT TESTED` pada audit lama. Patch memori mencatat **55 PASS, 2 SKIP (NOT TESTED), 0 FAIL** pada suite CPU/mock, tetapi **belum ada bukti bahwa patch dapat memuat batch 16 pada T4**. Bagian A–H mempertahankan hasil historis 13 Oktober; bagian I memperbarui bukti, perbaikan, dan keputusan kesiapan 14 Oktober.

## A. Executive Summary

**Kode awal: FAIL.** Ada bug pada target teacher yang mengubah makna AGKA/AAWM: forward ulang modul teacher menulis output cross ke dictionary hook sebelum dictionary tersebut dibaca sebagai target teacher. Loss juga berbeda secara numerik dari kode resmi penulis pada arah attention, letak skala, normalisasi, residual guidance, dan reduksi. Checkpoint wrapper mentah per epoch tidak portabel ke proses Python baru tanpa class `CSAKDModel` dari notebook.

**Status pada 13 Oktober setelah patch: PASS WITH WARNINGS untuk implementasi yang diperiksa pada CPU; belum siap untuk eksperimen final.** Suite gabungan mencatat **48 PASS, 2 SKIP (NOT TESTED), 0 FAIL**. Kedua student menyelesaikan satu epoch diagnostik dengan dua batch sintetis, validation, checkpoint per epoch, reload dalam proses baru, dan logging. Pada saat itu smoke teacher artifact v50, Poltrada V3, CUDA AMP, dan W&B online belum dilakukan; tidak boleh dianggap PASS berdasarkan fixture CPU. Bukti baru 14 Oktober dan status OOM dijelaskan pada I.

Adaptasi ini mempertahankan enam modul `[4, 6, 10, 16, 19, 22]`, teacher YOLO26m, student YOLO26s/n, dan koefisien AGKA/Cross `1/1`. Klaim yang dapat dipertanggungjawabkan adalah **adaptasi CSAKD dengan enam modul dan penyelarasan channel pada YOLO26m→s/n**. Klaim reproduksi identik tidak tepat: paper menggunakan YOLOv8s dan student hasil hybrid quantization, sedangkan penelitian ini mengubah kapasitas model dan domain data. [Paper primer CSAKD, BMVC 2025](https://bmva-archive.org.uk/bmvc/2025/assets/papers/Paper_398/paper.pdf).

Konfigurasi bersama tidak diubah: Poltrada V3; 100 epoch; input 640; batch 16; seed 42; AdamW; lr 0.001; AMP aktif; weight decay 0.0005; warmup 3; close mosaic 10; patience 100; evaluasi akhir `val`. Tidak ada training 100 epoch yang dijalankan. Dataset lokal V2 tidak dipakai sebagai pengganti V3.

## B. Paper-to-Code Mapping

Acuan numerik adalah [kode penulis `OtherKD/CSAKD.py`](https://github.com/KefanZhan/YOLOv8-KD/blob/3433559ab6c9cfdef3b4ac099f1bb2feddd42dc8/OtherKD/CSAKD.py), dipatok pada commit `3433559ab6c9cfdef3b4ac099f1bb2feddd42dc8`. Persamaan paper dan implementasi kode penulis tidak identik. Pilihan mengikuti kode penulis sudah ditetapkan dalam rencana implementasi pengguna.

| Komponen / persamaan paper | Lokasi notebook 08 dan 09 | Implementasi yang dimaksud setelah patch | Klasifikasi |
|---|---|---|---|
| Objective gabungan, Eq.(1) | Cell 12 `combined_loss`; cell 13 `CSAKDModel.loss` | Detection bawaan + jumlah AGKA dan Cross enam tap; koefisien 1/1 | Prinsip sesuai; bobot/cakupan adaptasi |
| Cross feature, Eq.(2–4) | Cell 13 `InputHook`, `CSAKDModel.kd_loss` | Input student aktual pada modul terpilih → projector input → modul teacher frozen | Adaptasi graph dan channel |
| Cross Supervising, Eq.(5),(10) | Cell 13 `csakd_losses` | MSE mean fitur student/cross yang dinormalisasi L2 channel, dikali AAWM | Sesuai kode penulis; beda notasi norm paper |
| GAP/similarity/AAWM, Eq.(6–9) | Cell 13 `csakd_losses` | Dot product GAP dibagi √C; mean batch sebelum sigmoid; satu skalar per tap | Sesuai kode penulis |
| Q/K/attention, Eq.(11–13) | Cell 13 `csakd_losses` | `ReLU(cross) @ ReLU(target).transpose(-2,-1) / sqrt(C)` sebelum softmax | Sesuai kode penulis; Q/K dan skala beda paper |
| Guidance/AGKA, Eq.(14–15) | Cell 13 `csakd_losses` | `(student + attention @ student)/2`; channel normalize; MSE mean terhadap target normalize | Sesuai kode penulis; residual/reduksi beda paper |
| Frozen teacher | Cell 12 `combined_loss`; class dasar Ultralytics; optimizer CSAKD | Target gambar di `no_grad`; cross-forward tetap differentiable terhadap input | Sesuai prinsip; gradien teacher dihentikan sengaja |
| Alignment | Cell 12 `align_channels`; cell 13 constructor | Dua adapter1×1 per tap bila channel berbeda; identity bila sama; spatial harus sama | Diperlukan untuk m→s/n |
| Backbone/neck supervision | Cell 13 `module_indices` | Enam modul, bukan seluruh modul backbone/neck | Penyederhanaan yang harus diungkapkan |

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
| **Critical** | 08/09 cell 13 `CSAKDModel.kd_loss` | Target teacher tertimpa / bug | `cross = teacher.model[index](incoming)` menjalankan `FeatureHook` yang menulis `_teacher_feats[index]`; `target = _teacher_feats[index]` baru dibaca sesudahnya. Target sama dengan cross, bukan fitur gambar teacher asli | AGKA membimbing terhadap cross; AAWM memakai cross sebagai teacher; hasil lama tidak mewakili metode yang dimaksud | Snapshot seluruh target `detach().clone()` sebelum cross-forward pertama |
| **High** | 08/09 cell 13 `csakd_losses` | Formula menyimpang dari kode acuan / penyimpangan metode | Kode awal `t@cᵀ`, softmax/√C, residual tanpa /2, vector norm unnormalized, sigmoid per sampel. Acuan yang dipilih menghendaki `c@tᵀ`, scale sebelum softmax, normalize channel, MSE mean, mean similarity batch | Semantik attention, gradien, dan skala KD berbeda | Ganti hanya formula; koefisien 1/1 tetap; tambah oracle numerik dan gradien |
| **High** | 08/09 cell 15 `save_epoch_artifact` | Checkpoint per epoch tidak portabel / bug deployment | Checkpoint live mengandung `CSAKDModel` pickle; proses baru tanpa class gagal memuat. Final strip bawaan tidak menyelesaikan artifact epoch terdahulu | Artifact `best.pt`/`last.pt` dapat gagal inferensi di lingkungan bersih; deployment dapat membawa wrapper | Salinan student-only melalui `strip_optimizer(source,destination)`; checkpoint training asli tetap disimpan dengan nama berbeda |
| **Medium** | 08/09 cell 13 `KDTrainer`, optimizer bawaan | Teacher frozen masih tercantum optimizer / hygiene integrasi | `requires_grad=False` menghentikan update teacher, tetapi parameter grup bawaan tidak memfilter semua frozen parameters | Bukan bukti teacher ter-update; grup optimizer tidak memenuhi invariant eksplisit penelitian | Filter menurut `requires_grad`; uji teacher tidak tercakup dan semua trainable parameters tercakup |
| **Medium** | 08/09 cell 12 `combined_loss`; cell 15 logging | Item detection bukan objective detection aktual / diagnostik | `E2ELoss` mengembalikan weighted loss kedua branch, tetapi item deteksi hanya one-to-one | Menjumlahkan grafik box/cls/L1 dapat salah menilai dominasi KD | Tambah `base.sum()/B`, weighted KD, dan rasio; tidak memasukkan item diagnostik ke backward |
| **Medium** | 08/09 cell9 teacher loading | Provenance teacher belum lengkap / validitas eksperimen | Ref v50 dan pemeriksaan kelas tersedia; metadata artifact, producer, digest, SHA256 dan dataset belum seluruhnya direkam/diuji | Teacher yang berbeda dapat merusak komparasi | Verifikasi exact ref, arsitektur/kelas/dataset; catat provenance; fail bila mismatch; tanpa fallback pretrained |
| **Medium** | 08/09 cell 13 + callback awal epoch | Batch/AMP efektif dapat berbeda dari konfigurasi / protokol | Recovery OOM Ultralytics dapat menurunkan batch; AMP efektif ditentukan trainer | Konfigurasi nominal16 belum membuktikan batch efektif16 | Catat batch/AMP aktual; hentikan run penelitian bila batch berubah |
| **Low** | 08/09 cell1, cell11, cell 15 narasi | Teks COCO8/10 epoch tidak sesuai konfigurasi utama / dokumentasi | Code config memakai Poltrada V3/100 epoch, markdown masih demo | Interpretasi laporan/hasil ambigu | Koreksi narasi penelitian dan label diagnostic fixture |
| **Medium** | 08/09 cell 13 `module_indices`, adapters | Enam tap dan routing lokal / adaptasi | Cross dijalankan pada satu modul terhadap input student aktual, tidak mengganti stream graph teacher keseluruhan; ada dua adapter trainable per tap | Modul m/s/n tidak menerima input numerik identik tanpa projection; efek adapter ikut berkontribusi | Pertahankan pilihan yang disetujui; ungkapkan dalam skripsi dan catat parameter/biaya training |
| **Medium** | 08/09 cell 3 bobot 1/1 | Bobot adaptasi belum divalidasi untuk Poltrada / risiko metodologi | Normalized mean MSE memiliki skala kecil; diagnostik oracle acak menunjukkan rasio KD/detection sekitar 0.0045–0.0049 | KD dapat memberi kontribusi kecil meskipun gradien nonzero | Catat rasio dan gradien; jangan mengganti bobot dari intuisi atau memakai test untuk tuning |

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
| Cell 12–13 / supervision dan loss | Snapshot detached cloned sebelum cross; formula numerik penulis; pemeriksaan shape/finite; cleanup exception; teacher tetap frozen dan cross differentiable |
| Cell 13 / trainer | Filter optimizer trainable; pertahankan E2ELoss dan jadwal bawaan; audit invariant batch/AMP efektif |
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

Perintah yang digunakan untuk suite 13 Oktober dari root repositori (jumlah test dapat bertambah pada source versi 14 Oktober):

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

## I. Pembaruan OOM dan optimasi memori — 14 Oktober 2026

### Bukti Kaggle yang diberikan pengguna

Sumber bagian ini adalah log run Kaggle yang diberikan pengguna. Log tersebut tidak direproduksi oleh executor CPU audit. Status lama mengenai akses teacher/AMP diperbarui sebatas yang benar-benar ditunjukkan log; kegagalan sebelum backward tidak dihitung sebagai smoke lengkap yang berhasil.

| Tahap / fakta | Bukti baru | Status |
|---|---|---|
| Artifact teacher exact v50 | `01_teacher_yolo26m-fux5wdcb-checkpoints:v50` berhasil diunduh; producer run `fux5wdcb` | PASS menurut log |
| SHA256 `best.pt` teacher | `d749ef0baa0d6c3ae94e83ded1d9999d83c804b76ad5a841fc2ebec953af1428` | Teridentifikasi pada log; belum diunduh ulang executor lokal |
| Epoch teacher | Metadata melaporkan **51**; `v50` tetap versi artifact, bukan epoch 50 | Terverifikasi pada log |
| AMP compatibility check | Pemeriksaan AMP bawaan berhasil | PASS check; bukan bukti gradien KD CUDA finite |
| GPU yang digunakan run s | GPU 0 NVIDIA T4, total 14.56 GiB | Teridentifikasi pada log |
| Cross-forward FP32 run s | OOM pada batch training pertama; gagal meminta 50 MiB, allocated 14.19 GiB, reserved-unused 205.6 MiB, device-free 10.81 MiB | **FAIL pada run sebelum optimasi memori** |
| Recovery batch | Ultralytics mencoba 16→8; guard protokol menolak perubahan | PASS guard; bukan recovery training yang berhasil |
| Total loss, backward, optimizer step, validation, checkpoint penelitian | Run gagal sebelum rangkaian tersebut selesai | **NOT TESTED sampai selesai** |
| YOLO26n CUDA aktual | Tidak ada log smoke n yang berhasil | **NOT TESTED** |
| W&B checkpoint upload/inferensi artifact aktual | Download teacher berhasil; belum ada epoch training lengkap | **NOT TESTED** untuk artifact student |

Allocated memory hampir memenuhi device pada saat gagal. Log tidak membuktikan bahwa fragmentasi allocator adalah akar utama; mengubah allocator atau memanggil `empty_cache()` saja tidak memberikan bukti bahwa live activations akan muat. Detection AMP tetap aktif, sedangkan cabang KD sengaja FP32; keberhasilan AMP check tidak membuat cross-forward FP32 menjadi hemat seperti FP16.

Contoh ukuran tensor, **bukan peak memory terukur per layer**: attention tap4 `B16×C512×H80×H80` FP32 sendiri berukuran 200 MiB. Fitur BCHW pada tap itu juga 200 MiB. Chunk 2 mengubah satu tensor ekuivalen menjadi 25 MiB; total peak tetap mencakup student batch penuh, snapshot teacher, parameter/optimizer, operasi KD lain dan workspace CUDA. Hitungan ini menjelaskan sasaran optimasi, **tidak membuktikan muat pada T4**.

### Perbaikan yang mempertahankan metode dan protokol

Optimasi hanya pada cabang distillation: **activation checkpointing seluruh region KD per tap/chunk**, mencakup projector, cross-forward frozen teacher, attention dan normalized losses; chunk default 2. Student forward dan native detection criterion tetap menerima batch 16, input 640. Seed 42, AdamW, lr 0.001, bobot 1/1, enam tap, split dan evaluasi tetap sama. Ini bukan batch 8 dengan accumulation untuk meniru batch 16.

| Konfigurasi / interface baru | Nilai default / lokasi | Makna |
|---|---|---|
| `KD_CHECKPOINT` | `True`, cell 3 | Activation checkpoint KD; tidak berkaitan dengan file weights |
| `KD_CHUNK_SIZE` | `2`, cell 3 | Maksimum jumlah sampel per chunk KD; integer positif; tidak mengubah batch student |
| `CSAKDModel(..., checkpoint_kd=None, chunk_size=None)` | Cell 13 | Override eksplisit untuk diagnostik/paritas; attr model `kd_checkpoint` dan `kd_chunk_size` |
| `snapshot_teacher_targets()` | Cell 13; dipanggil cell 12 sebelum student forward | Clone detached enam target, lalu hapus cache input/output/head teacher yang tidak dipakai |
| `csakd_loss_terms()` | Cell 13 | Mengembalikan AGKA mean, discrepancy mean dan similarity mean; AAWM diterapkan setelah agregasi |
| `kd_loss(targets=None)` | Cell 13 | Menerima snapshot eksplisit; `csakd_losses(s,t,c)` tetap mengembalikan dua loss seperti sebelumnya |
| `KD_CONFIG["memory_strategy"]` | Cell 13, W&B/config/artifact metadata | `activation_checkpoint`, `use_reentrant=False`, `preserve_rng_state=True`, `kd_chunk_size`, `student_batch_unchanged=True`, `aawm_scope="full-batch-before-sigmoid"` |
| `memory/peak_allocated_gb`, `memory/peak_reserved_gb` | Callback W&B cell 15, CUDA saja | Peak proses sejak reset terakhir; **bukan** peak epoch yang di-reset setiap epoch |
| `peak_gpu_reserved_gb` | Results cell 17 | Tambahan peak reserved, bersama peak allocated yang sudah ada; bukan pengukuran CPU |

Activation checkpointing menukar penyimpanan intermediate activations dengan recomputation saat backward. Implementasi memakai `use_reentrant=False` secara eksplisit, dan forward/recompute harus menggunakan modul, tensor target, serta precision yang sama. [Dokumentasi resmi PyTorch 2.11](https://docs.pytorch.org/docs/2.11/checkpoint.html). Gagasan recomputation untuk menurunkan biaya memori juga dibahas pada [Chen dkk., *Training Deep Nets with Sublinear Memory Cost*](https://arxiv.org/html/1604.06174). Referensi tersebut menjadi dasar teknis strategi, bukan prediksi kapasitas memori run ini.

**AAWM tetap dihitung untuk batch asli, bukan per chunk.** Untuk chunk `k` dengan ukuran `n_k`, `B=Σ n_k`, similarity per sampel `a_b`, AGKA mean `g_k`, dan normalized cross-MSE mean `d_k`:

\[
\mu_i=\frac{1}{B}\sum_k\sum_{b\in k}a_b,\qquad
w_i=1-\sigma(\mu_i),\qquad
L_{AGKA,i}=\sum_k\frac{n_k}{B}g_k,\qquad
L_{Cross,i}=w_i\sum_k\frac{n_k}{B}d_k.
\]

Mean similarity diagregasi **sebelum sigmoid**, dengan gradien tetap aktif. Menghitung `sigmoid(mean(chunk))` lalu merata-ratakan weighted cross-loss akan mengubah objective; pendekatan itu ditolak. Pembobotan chunk berdasarkan jumlah sampel juga diperlukan jika chunk terakhir lebih pendek. Kesetaraan numerik/gradien diperiksa dengan toleransi floating point yang sesuai karena urutan penjumlahan berubah.

Teacher tetap eval/frozen, sehingga teacher BatchNorm tidak menghitung statistik training per chunk. Student BatchNorm tetap melihat batch 16 pada forward utamanya. Snapshot seluruh target teacher asli tetap disimpan sebelum cross. Function checkpoint harus menerima/bind referensi modul/projector/tensor secara stabil; dilarang mengambil loop index terakhir atau target dari cache hook yang dapat berubah saat recompute. Hook cross/recompute tidak boleh mempertahankan fitur besar atau mengisi cache batch lama sesudah `combined_loss` selesai.

Implementasi memakai `partial` untuk mengikat modul/adapters tiap tap dan melewatkan tensor slice sebagai argumen checkpoint. `ControlledFeatureHook` tetap kompatibel dengan cleanup `FeatureHook` bawaan; gate capture teacher bersifat lokal per wrapper dan diregistrasi ulang pada deepcopy/unpickle. `_suspend_teacher_capture()` mematikan output/pre-input capture selama seluruh chunk loss, lalu mengembalikan status melalui `finally`, termasuk saat recompute non-reentrant berhenti lebih awal. Native teacher original-image hooks tetap aktif untuk pengumpulan supervision. Pemeriksaan statik tidak menemukan perubahan formula; paritas loss/gradien dan lifecycle telah diuji pada CPU sesuai tabel berikut. Smoke CUDA pada patch ini tetap belum dilakukan.

### Verifikasi patch memori dan profil GPU

**Suite gabungan final setelah optimasi: 55 PASS, 2 SKIP, 0 FAIL dalam 66.02 detik.** Sebanyak 23 PASS memeriksa implementasi/trainer dan 32 PASS memeriksa provenance mock. Kedua student menyelesaikan satu epoch CPU pada implementasi checkpoint/chunk; dua smoke GPU di-skip karena lingkungan CPU. Empat warning hanya THOP `distutils.Version` deprecation. Angka 48 PASS historis 13 Oktober tidak dipakai sebagai bukti untuk kode memori baru. Bukti final: [JUnit OOM](<C:/Local D/Galeri Belajar/Skripsi/outputs/test_csakd/memory_final_results.xml>) dan [stdout OOM](<C:/Local D/Galeri Belajar/Skripsi/outputs/test_csakd/memory_final_test_output.txt>). Penerimaan patch mencakup:

| Pengujian optimasi | Status final audit lokal | Bukti / batas |
|---|---|---|
| Reference vs chunk/checkpoint: AGKA/Cross/total dan gradien | PASS CPU s dan n | Full batch 3 vs chunk 2+1, chunk 1/2/3 untuk oracle AAWM; global AAWM, setiap parameter trainable, semua adapter |
| Teacher frozen / gradient student / cache recompute | PASS CPU | Teacher full state tidak berubah; student/projector gradien finite nonzero; cache kosong setelah backward; capture teacher pulih |
| Saved activations dan repeated backward | PASS CPU proxy | Saved storage KD turun; cache tetap kosong setelah recompute; bukan pengukuran VRAM T4 |
| Satu epoch sintetis CPU, validation, reload/checkpoint/logger | PASS CPU s dan n | Implementasi checkpoint/chunk menjalankan dua batch, validation, snapshot per epoch dan final best reload dalam proses baru |
| Syntax, konsistensi s/n dan lint tests | PASS | Seluruh 18 code cells kedua notebook lolos AST; method/docs/config callback bersama identik; kedua file test lolos Ruff |
| Smoke CUDA V3/teacher aktual, input 640/batch 16/AMP | **NOT TESTED pada patch memori** | Dua batch training + satu batch validation dan checkpoint reload, terpisah s/n |
| Peak allocated/reserved/free CUDA dan waktu batch | **NOT TESTED** | Reset peak sebelum smoke, synchronize pada pengukuran; metric memory dan OOM tercatat |

Pada perbandingan **CPU B3/input 128 yang mengisolasi KD**, unique saved-tensor storages selain parameter adalah 74,032,456 bytes untuk reference langsung dan 5,578,824 bytes untuk checkpoint+chunk 2: rasio 0.075356, penurunan 92.46%. Ini ukuran storage yang dirujuk autograd pada region KD; **bukan** penurunan RAM total, peak CUDA training, atau prediksi muat pada T4. Bukti: [saved-tensor storage proxy](<C:/Local D/Galeri Belajar/Skripsi/outputs/test_csakd/memory_saved_tensor_storage_proxy.json>).

Paritas gradien diperiksa untuk seluruh parameter trainable dan state `grad=None` yang sah. Maksimum relative L2 pada gradien yang bermakna: s total 2.132e-6, AGKA 2.086e-6, Cross 2.294e-6; n total 2.473e-6, AGKA 2.448e-6, Cross 2.007e-6. Gradien BN bias yang mendekati nol diperiksa dengan absolute error, bukan rasio terhadap penyebut hampir nol: maksimum 6.15e-8 untuk combined objective dan 1.55e-11 untuk KD isolated. Pemeriksaan tersebut membuktikan paritas pada fixture/toleransi CPU yang diuji; tidak menggantikan finite-gradient smoke CUDA setelah unscale. Bukti per parameter/komponen: [paritas YOLO26s](<C:/Local D/Galeri Belajar/Skripsi/outputs/test_csakd/memory_gradient_parity_s.json>) dan [paritas YOLO26n](<C:/Local D/Galeri Belajar/Skripsi/outputs/test_csakd/memory_gradient_parity_n.json>).

Perintah regresi yang menghasilkan bukti 14 Oktober dari root repositori:

```powershell
.venv/Scripts/python.exe -m pytest tests/test_csakd_notebooks.py tests/test_csakd_provenance.py -o addopts= -q --tb=short --junitxml=outputs/test_csakd/memory_final_results.xml
```

Smoke GPU memakai aset/provenance aktual yang dijelaskan pada F, **tanpa cell 15 training 100 epoch**. Setelah environment variables F terisi, jalankan dari checkout pada host GPU:

```python
subprocess.run([
    sys.executable, "-m", "pytest", "tests/test_csakd_notebooks.py",
    "-o", "addopts=", "-q", "-k", "actual_poltrada",
], check=True)
```

Smoke memprofil dua batch total: batch pertama berlabel `warmup/setup`, batch kedua `measured`; setiap batch mereset allocator peak dan menyinkronkan GPU untuk pencatatan waktu. `gpu_memory_profile.json` disimpan pada directory fixture unik dalam `outputs/test_csakd`, termasuk bila run gagal melalui `finally`. JSON memuat GPU/device identity, effective batch/AMP, checkpoint/chunk config, allocated/reserved/peak bytes, free/total device bytes, waktu batch, serta status/error. Gradien student/projector diperiksa setelah unscale/clip; teacher tetap tanpa gradien. Peak seluruh run diagregasi dari snapshot setup/batch/final sehingga reset tiap batch tidak menghilangkan peak setup. Profil CPU atau saved-tensor storage tidak boleh diberi label sebagai VRAM CUDA. **Kode profil tersedia, tetapi hasil CUDA masih NOT TESTED.**

Jika masih OOM pada T4, batch guard tetap menghentikan perubahan 16→8. Jangan menyatakan 100 epoch siap; gunakan profil untuk menentukan langkah berikutnya secara eksplisit. Tidak ada persetujuan untuk mengubah shared protocol atau menjalankan full training sebagai bagian perbaikan ini.

### Keputusan kesiapan terbaru

1. **YOLO26s belum siap 100 epoch:** run CUDA sebelumnya gagal OOM; regresi patch CPU sudah PASS, tetapi smoke CUDA ulang belum dilakukan.
2. **YOLO26n belum siap 100 epoch:** tidak ada bukti smoke CUDA lengkap pada n; PASS s tidak diwariskan ke n.
3. **Klaim adaptasi CSAKD tetap dapat dipertanggungjawabkan dalam cakupan yang dinyatakan:** loss dan gradient parity checkpoint/chunk terbukti pada fixture CPU. Global AAWM dan native batch 16 dipertahankan; keterbatasan metodologi paper pada B/G tetap berlaku.
4. **Komparasi metodologis tetap bersyarat:** protokol dibekukan; run yang berhenti karena OOM bukan hasil eksperimen selesai dan tidak boleh dimasukkan sebagai mAP pembanding.
5. **Blocker terkini:** belum ada bukti patch memori menyelesaikan forward/backward/step/validation/reload pada V3 dan teacher teridentifikasi, batch 16/input 640/AMP di GPU target. Download teacher dan AMP check sudah ditunjukkan log, tetapi tidak menghilangkan blocker memori.
