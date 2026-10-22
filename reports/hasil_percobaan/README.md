# Hasil percobaan GPU

Semua laporan smoke test HEAD, Rank Mimicking (RM), dan Pearson/PKD beserta bukti JSON disimpan dalam folder ini.

| Metode | Hasil | Laporan | Bukti JSON |
|---|---|---|---|
| HEAD | PASS satu epoch, validasi, dan pemeriksaan checkpoint | [HEAD/RM](head_rm_kaggle_smoke_20261021.md) | [HEAD](head_gpu_smoke_20261022.json) |
| Rank Mimicking | PASS satu epoch, validasi, dan pemeriksaan checkpoint | [HEAD/RM](head_rm_kaggle_smoke_20261021.md) | [RM](rm_gpu_smoke_20261022.json) |
| Pearson/PKD | Belum PASS satu epoch; terakhir tercatat 416/424 batch, berhenti bertepatan dengan timeout | [Pearson/PKD](pkd_kaggle_smoke_20261022.md) | [PKD](pkd_gpu_smoke_20261022.json) |

Percobaan memakai Kaggle Tesla T4 dengan batch 16, resolusi 640, dan AMP. Hasil ini tidak memverifikasi training 100 epoch, resume, atau upload W&B online. Batas klaim dan rincian protokol tercantum pada masing-masing laporan.

## Verifikasi koleksi Colab

[Laporan verifikasi lokal 17 notebook Colab](colab_collection_validation_20261022.md): pemeriksaan adaptasi, dataset ZIP, provenance, dan dispatch training. Ini **bukan hasil training GPU Colab**.
