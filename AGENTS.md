# Panduan kerja repositori skripsi YOLO26

Aturan ini berlaku untuk seluruh repositori, termasuk naskah, referensi, notebook, kode, data, dan laporan.

## Wajib mulai dari router repo

- Sebelum membuat, mengubah, memindahkan, atau menghapus file, buka [peta repo](peta-repo.md). Ikuti tautannya ke indeks, panduan, atau laporan yang relevan agar perubahan masuk ke lokasi yang sesuai.
- Periksa perubahan yang sudah ada di area kerja dan baca implementasi serta pengujian yang terkait sebelum mengedit. Jika perubahan menggeser jalur penting, perbarui peta dan rujukannya pada pekerjaan yang sama.

## Bekerja sebagai peneliti visi komputer

- Untuk pekerjaan riset, rumuskan pertanyaan atau hipotesis, pembanding, dan metrik sebelum mengubah metode. Rujuk paper asli dan kode penulis yang relevan. Catat penyesuaian metode untuk YOLO26.
- Saat membandingkan baseline, knowledge distillation, atau pruning, gunakan dataset, split, seed, dan protokol evaluasi yang sebanding. Catat perbedaan yang disengaja dan dampaknya terhadap interpretasi hasil. Gunakan validation untuk memilih konfigurasi dan test untuk evaluasi akhir.
- Catat versi dataset, varian model, checkpoint, seed, konfigurasi, dan lingkungan yang relevan agar eksperimen bisa diulang. Untuk perubahan metode, lakukan pemeriksaan ringan sebelum run penuh. Labeli hasil sebagai smoke test atau eksperimen penuh sesuai bukti yang tersedia.
- Pisahkan hasil pengukuran dari dugaan atau interpretasi. Laporkan juga kegagalan, data yang belum tersedia, dan batasan bukti. Pertahankan hasil eksperimen historis agar dapat diaudit.

## Log kesimpulan penelitian wajib

- Baca [log kesimpulan penelitian](reports/log-penelitian.md) sebelum memilih, mengulang, atau mengubah metode agar temuan sebelumnya menjadi dasar keputusan.
- Sebelum menyelesaikan eksperimen atau audit yang menghasilkan temuan, tambahkan kesimpulannya ke log. Catat hasil positif, negatif, gagal dijalankan, dan hasil yang belum meyakinkan. Gunakan tanggal Jakarta dan pertahankan entri lama saat menambahkan bukti baru.
- Sertakan kondisi pengujian, pembanding dan metrik jika tersedia, tautan bukti, batas kesimpulan, serta tindak lanjut jika ada. Tandai observasi yang belum terverifikasi dan bedakan kegagalan menjalankan eksperimen dari hasil evaluasi yang buruk. Perubahan administratif dicatat hanya jika menghasilkan temuan penelitian.
