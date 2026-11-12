"""Attach checkpoint retention after the immutable training implementation hash."""
import hashlib


def attach_storage(book, root, *, selected):
    workflow = book["metadata"]["research_workflow"]
    if workflow.get("storage_sha256"):
        return book
    source = (root / "scripts/recovery_storage_source.py").read_text(encoding="utf-8")
    adapter = (root / f"scripts/recovery_storage_{'selected' if selected else 'suite'}_source.py").read_text(encoding="utf-8")
    digest = hashlib.sha256((source + "\n" + adapter).encode()).hexdigest()
    config = next(c for c in book["cells"] if "config" in c.get("metadata", {}).get("tags", []))
    config["source"] += ("\n# Kebijakan I/O; tidak mengubah kontrak training/resume.\n"
        'CHECKPOINT_RETENTION = "best-and-latest-v1"\n'
        "AUTO_DELETE_SMOKE_ARTIFACTS = True  # Sesudah evaluasi, EDA, upload dan hash terverifikasi.\n"
        "AUTO_DELETE_LOCAL_SMOKE_WEIGHTS = True  # Hanya salinan biner di folder smoke ini.\n").splitlines(keepends=True)
    code = f"STORAGE_SHA256 = {digest!r}\n" + source + "\n" + adapter
    if selected:
        cell = next(c for c in book["cells"] if "runtime" in c.get("metadata", {}).get("tags", []))
        cell["source"] = ("".join(cell["source"]) + "\n\n" + code).splitlines(keepends=True)
    else:
        position = next(i for i, c in enumerate(book["cells"]) if "run" in c.get("metadata", {}).get("tags", []))
        book["cells"].insert(position, {"cell_type": "code", "id": digest[:8], "source": code.splitlines(keepends=True),
            "metadata": {"tags": ["storage-runtime"]}, "execution_count": None, "outputs": []})
    for cell in book["cells"]:
        if cell["cell_type"] != "markdown":
            continue
        text = "".join(cell["source"])
        if "## 03" in text or (not selected and "## 05" in text):
            if not selected:
                text = text.replace("Epoch yang tersimpan diunggah dengan alias epoch, latest, dan best bila cocok.",
                    "Paket resume terbaru memuat state terakhir dan model dari epoch terbaik.")
            text += ("\n\n**Penyimpanan best/latest:** model inference hanya `best.pt` dan `last.pt`. "
                "Snapshot lokal mempertahankan model terbaik dan state resume terbaru. W&B menyimpan satu versi "
                "paket resume terbaru yang mencakup keduanya; versi dari run ini yang lebih lama dihapus hanya "
                "setelah penggantinya diunduh dan SHA-256 cocok. Run lain dan input raw/baseline/teacher tetap dilindungi. "
                "Loss, seluruh epoch history, Tables, Images, dan EDA eksperimen penuh tetap dicatat.\n\n"
                "**Smoke:** setelah training, evaluasi, EDA dan verifikasi publikasi selesai, artefak diagnostik W&B "
                "serta salinan checkpoint lokal di folder smoke ini otomatis dihapus. Receipt SHA-256, metrik, tabel, "
                "grafik dan log lokal tetap disimpan. Registry menandai `DELETED_AFTER_SMOKE`; referensinya menjadi "
                "bukti historis dan tidak dapat dipakai recovery. Interupsi sebelum selesai tetap dapat di-resume. "
                "Untuk uji restore snapshot, nonaktifkan dua opsi cleanup selama pengujian lalu bersihkan setelah verifikasi.\n")
            cell["source"] = text.splitlines(keepends=True)
    workflow["storage_sha256"] = digest
    workflow["checkpoint_retention"] = "best-and-latest-v1"
    return book
