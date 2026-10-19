# CSAKD teacher s → student n dan pengorganisasian W&B

Tanggal: 17 Oktober 2026, Asia/Jakarta.

## Implementasi

Notebook baru `notebooks/yolo26-kd-indotraffic/notebooks/12_csakd_yolo26n_teacher_yolo26s.ipynb`
merupakan salinan standalone notebook 09 dengan sumber teacher dan identitas eksperimen baru.
Teacher memakai artifact model tetap berikut, sama dengan notebook 10–11:

```text
ilhamrafiqin-brawijaya-university/yolom26-kd-indotraffic/02_baseline_yolo26s-q5kp9k9d-checkpoints:v62
```

Download ke `/kaggle/working/teacher_yolo26s_v62`; `best.pt` mengisi `TEACHER_CHECKPOINT`
dan diteruskan ke `KD_TRAIN_ARGS["distill_model"]`. Versi, producer `q5kp9k9d`, manifest,
ukuran/checksum, dataset, enam kelas beserta urutannya, dan arsitektur YOLO26 skala s diperiksa.
Metadata yang tidak lengkap menghentikan mode penelitian seperti notebook 09.
Download gagal atau best.pt yang hilang menghentikan proses sebelum model/training dilanjutkan.
Artifact sumber ditautkan ke run CSAKD dengan `use_artifact`.

Konfigurasi mengikuti CSAKD 09: Poltrada V3, student `yolo26n.pt`, 100 epoch, 640,
batch 16, seed 42, device 0, workers 2, AdamW, LR 0.001, patience 100, AMP True,
weight decay 0.0005, warmup 3, close mosaic 10, evaluasi val. AGKA/Cross masing-masing 1.0,
enam tap [4, 6, 10, 16, 19, 22], FP32 untuk komputasi KD, activation checkpointing dan
chunk size 2 tetap sama. Run All langsung training setelah verifikasi teacher.

Definisi loss, wrapper, trainer, dan ekspor checkpoint identik dengan notebook 09.
Teacher frozen/eval dan tidak masuk optimizer; checkpoint inference student-only.
Notebook ini tetap adaptasi CSAKD enam modul yang sama, bukan klaim reproduksi identik paper.
Nama W&B baru: `csakd_T26s_S26n_seed42`.

## Group dan tag semua eksperimen

Notebook 01–12 memakai `WANDB_GROUP`/`WANDB_TAGS` di konfigurasi:

| Group | Eksperimen |
|---|---|
| `supervised` | 01–03 |
| `T26m_S26s` | 04, 06, 08 |
| `T26m_S26n` | 05, 07, 09 |
| `T26s_S26n` | 10, 11, 12 |

Tag berisi metode/role, arsitektur teacher/student untuk KD atau model untuk supervised,
seed, `poltrada-v3`, ID eksperimen, dan `research`/`diagnostic`.
Native memuat tag dis, CrossKD cls/box, CSAKD agka/cross; nilai bobot dan seed dihitung dari
konfigurasi agar perubahan eksplisit tercermin pada tag. Diagnosis pada 01–09/12 diberi
awalan group `diagnostic-` agar tidak tercampur dengan run penelitian.

Perubahan notebook lama dibatasi pada group/tag dan pembaruan SHA-256 referensi notebook 10–11
karena byte konfigurasi referensi berubah. Perbandingan AST terhadap snapshot sebelum perubahan
memastikan seluruh kode model, objective, training, evaluasi, callback upload, dan nama eksperimen
lama tetap sama. EDA 00 tidak diubah. Tidak ada run historis W&B yang diubah.

## Verifikasi

- JSON, nbformat, dan kompilasi Python seluruh 12 notebook eksperimen lulus.
- Mock init W&B memastikan group/tag aktual diteruskan dengan benar, termasuk tag dinamis
  dan isolasi mode diagnosis. Tidak melakukan login atau update W&B nyata.
- Mock provenance menguji teacher m v50 dan teacher s v62, metadata/checksum/label/arsitektur,
  download gagal serta file best.pt hilang.
- Kesamaan kode CSAKD 08/09/12 dan argumen training 09/12 diperiksa.
- Model YOLO26s/n nyata dengan bobot acak diuji pada CPU: alignment enam tap di resolusi 640,
  backward KD pada resolusi kecil, gradien student/projector, teacher frozen dan tidak berubah.
- Satu epoch sintetis s → n pada CPU 64px lulus, termasuk optimizer, validation, penyimpanan
  dan reload/predict student-only pada interpreter baru.

Perintah utama:

```powershell
.\.venv\Scripts\python.exe -m pytest tests/test_kd_notebook_tracking.py tests/test_csakd_provenance.py tests/test_csakd_notebooks.py tests/test_teacher_s_notebooks.py -q
# 154 passed, 2 skipped (opt-in GPU), 3 deselected (slow), 4 THOP warnings

.\.venv\Scripts\python.exe -m pytest 'tests/test_csakd_notebooks.py::test_synthetic_one_epoch_cpu_validation_checkpoint_reload[s-n]' -m slow -q
# 1 passed, 4 THOP warnings

.\.venv\Scripts\ruff.exe check tests/test_kd_notebook_tracking.py tests/test_csakd_provenance.py tests/test_csakd_notebooks.py tests/test_teacher_s_notebooks.py
# All checks passed
```

Total 155 tes lulus dalam dua perintah. Training 100 epoch di GPU Kaggle, akses artifact nyata,
VRAM batch 16/640 dan hasil penelitian belum diuji pada perubahan ini. CPU/synthetic/mock
tidak digunakan sebagai bukti performa atau stabilitas CUDA AMP penelitian.
