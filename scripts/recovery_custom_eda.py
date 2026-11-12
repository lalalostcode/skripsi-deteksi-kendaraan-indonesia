"""Add visible three-condition comparisons without changing the training contract."""
import hashlib

if __package__:
    from .recovery_storage import attach_storage
else:
    from recovery_storage import attach_storage


def attach_custom_eda(book, root, *, selected=True):
    if book["metadata"]["research_workflow"].get("custom_eda_sha256"):
        return book
    source = (root / "scripts/recovery_custom_eda_source.py").read_text(encoding="utf-8")
    adapter = (root / "scripts/recovery_custom_eda_selected_source.py").read_text(encoding="utf-8") if selected else (
        root / "scripts/recovery_custom_eda_suite_source.py").read_text(encoding="utf-8")
    signature = hashlib.sha256((source + "\n" + adapter).encode()).hexdigest()
    code = f"CUSTOM_EDA_SHA256 = {signature!r}\n" + source + "\n" + adapter
    if selected:
        cell = next(c for c in book["cells"] if "runtime" in c.get("metadata", {}).get("tags", []))
        cell["source"] = ("".join(cell["source"]) + "\n\n" + code).splitlines(keepends=True)
    else:
        position = next(i for i, c in enumerate(book["cells"]) if "run" in c.get("metadata", {}).get("tags", []))
        book["cells"].insert(position, {"cell_type": "code", "id": signature[:8], "source": code.splitlines(keepends=True),
            "metadata": {"tags": ["custom-eda-runtime"]}, "execution_count": None, "outputs": []})
    for cell in book["cells"]:
        if cell["cell_type"] == "markdown" and "## 0" in "".join(cell["source"]) and "EDA" in "".join(cell["source"]).splitlines()[0]:
            cell["source"] += ("\n\nEDA custom memakai satu diagram batang per metrik: **baseline asal (biru) → raw pruned (merah) → "
                "hasil metode recovery ini (hijau)**. Tiap panel mencantumkan seed, target, speed-up aktual, metode, dan mode. "
                "Nilai yang belum terukur ditandai NOT_RUN, termasuk latency yang belum dijalankan.\n\n"
                "Buka tautan **custom_eda_report_url** pada summary run W&B untuk halaman perbandingan yang sudah disusun: "
                "metrik utama, efisiensi, delta/gain, AP/P/R/F1/support per kelas, channel/heatmap/LAMP, PR/F1, confusion matrix, "
                "training, dan trade-off. Tabel serta gambar dicatat langsung pada run; paket `<nama-run>-eda:vN` menyimpan "
                "salinan terverifikasi. Dependency halaman W&B: `wandb-workspaces==0.4.5`, tanpa mengganti versi W&B training.\n"
            ).splitlines(keepends=True)
    display = next(c for c in book["cells"] if "display" in c.get("metadata", {}).get("tags", []))
    if selected:
        extra = '\nceda_display(RESULT, RUN_DIR)\n'
    else:
        extra = '\nfor job in BUNDLE.get("jobs", []):\n    ceda_display(job, SUITE_DIR / "jobs" / job["configuration_id"])\n'
    # Put the concise comparison before the existing comprehensive diagnostics.
    display["source"] = (extra + "\n" + "".join(display["source"])).splitlines(keepends=True)
    book["metadata"]["research_workflow"]["custom_eda_sha256"] = signature
    return attach_storage(book, root, selected=selected)
