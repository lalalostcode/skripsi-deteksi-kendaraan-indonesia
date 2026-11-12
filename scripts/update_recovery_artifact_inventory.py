"""Append confirmed recovery versions and hashes to the user's W&B inventory."""
from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
NAMESPACE = "ilhamrafiqin-brawijaya-university/yolom26-kd-indotraffic"
HEADING = "## Recovery dan EDA custom · verifikasi 8 November 2026"


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def link(reference, kind="model"):
    name = reference.rsplit("/", 1)[-1]
    collection, version = name.rsplit(":", 1)
    return f"[`{name}`](https://wandb.ai/{NAMESPACE}/artifacts/{kind}/{collection}/{version})"


def main():
    report = read(ROOT / "reports/pruning/recovery/recovery_custom_eda_verification_20261108.json")
    assert report["status"] == "PASS", "Do not inventory unfinished cases as confirmed."
    cases = sorted(report["cases"], key=lambda r: (r["method"], [42, 0, 2024, 7, 3407].index(r["seed"])))
    history = {}
    suite = ROOT / "outputs/pruning/selected_kd_online_delivery_20261108"
    for journal in suite.rglob("artifact_publications.json"):
        for event, item in read(journal).items():
            if item.get("status") == "CONFIRMED":
                history[item["reference"]] = {**item, "event": event, "source_journal": str(journal.relative_to(ROOT)),
                                               "verification": "download and SHA-256"}
    for case in cases:
        item = case["custom_eda_publication"]
        history[item["reference"]] = {**item, "event": "custom_eda", "verification": "download and SHA-256"}
    ft = report["new_finetune_notebook_online_smoke"]
    for row in ft.values():
        folder = Path(row["run_dir"])
        for item in [*read(folder / "artifact_publications.json").values(), read(folder / "custom_eda_publication.json")]:
            assert item["status"] == "CONFIRMED"
            history[item["reference"]] = {**item, "verification": "download and SHA-256"}
    target = ROOT / "reports/pruning/recovery/recovery_wandb_inventory_20261108.json"
    target.write_text(json.dumps({"status": "PASS", "namespace": NAMESPACE, "artifacts": history,
        "finished_user_runs": report["finished_user_run_audit"]["rows"]}, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    lines = [HEADING, "", ("Lima raw 1,20× `:v1` dan teacher YOLO26m `v50` asli dipakai pada 15 smoke diagnostik: "
        "Native KD, CrossKD, dan MGD × seed 42/0/2024/7/3407. Setiap kasus menyelesaikan satu epoch CPU FP32, "
        "64 px, batch 2, dengan fixture enam train/enam val. Semua paket pada tabel berikut sudah diunduh dan SHA-256 cocok. "
        "Angka smoke tidak mengukur akurasi Poltrada V3 atau latency CUDA."), "",
        ("Nama seluruh koleksi mengikuti `yolo26s-lamp-su1.20x-seed<seed>-<metode>-smoke-<jenis>`. "
        "Best/last hanya berisi student native FP32 dengan struktur pruned; snapshot menyimpan state training/adapternya. "
        "[Bukti lengkap](../reports/pruning/recovery/recovery_custom_eda_verification_20261108.json) · "
        "[Semua versi dan hash file, termasuk riwayat restore](../reports/pruning/recovery/recovery_wandb_inventory_20261108.json)."), "",
        "| Metode | Seed | Model best/last | Hasil | Snapshot | EDA |", "|---|---:|---|---|---|---|"]
    for row in cases:
        pub = row["publications"]
        lines.append(f"| {row['method']} | {row['seed']} | {link(pub['checkpoints/final']['reference'])} | "
            f"{link(pub['results/final']['reference'], 'results')} | {link(pub['resume/epoch-001']['reference'])} | "
            f"{link(row['custom_eda_publication']['reference'], 'results')} |")
    lines += ["", "| Metode | Seed | SHA-256 best.pt | SHA-256 last.pt |", "|---|---:|---|---|"]
    for row in cases:
        hashes = row["publications"]["checkpoints/final"]["files_sha256"]
        lines.append(f"| {row['method']} | {row['seed']} | `{hashes['best.pt']}` | `{hashes['last.pt']}` |")
    current = {row["publications"]["checkpoints/final"]["reference"] for row in cases}
    extra = [item for ref, item in history.items() if item.get("event") == "checkpoints/final" and ref not in current]
    lines += ["", ("Versi tambahan berikut berasal dari uji restore snapshot ke folder baru dengan ID run serta "
        "state satu epoch yang sama. Riwayat lama dipertahankan; tabel utama memakai hasil verifikasi terakhir."), "",
        "| Versi model historis diagnostik | SHA-256 best.pt | SHA-256 last.pt |", "|---|---|---|"]
    for item in extra:
        hashes = item["files_sha256"]
        lines.append(f"| {link(item['reference'])} | `{hashes['best.pt']}` | `{hashes['last.pt']}` |")
    lines += ["", "### Fine-tuning varian Kaggle dengan input pulihan", "",
        ("Satu smoke online tambahan memakai raw seed 2024 `:v1` dan notebook Kaggle pada CPU lokal. "
        "Satu epoch, hash raw awal, pembaruan bobot, struktur native, 19 gambar custom, empat Tables custom, "
        "best/last, hasil, dan snapshot lolos verifikasi. EDA inti sebelumnya juga tetap dicatat."), ""]
    for row in ft.values():
        folder = Path(row["run_dir"])
        pub, eda = read(folder / "artifact_publications.json"), read(folder / "custom_eda_publication.json")
        hashes = pub["checkpoints/final"]["files_sha256"]
        lines += [(f"- Model: {link(pub['checkpoints/final']['reference'])}; hasil: {link(pub['results/final']['reference'], 'results')}; "
            f"snapshot: {link(pub['resume/epoch-001']['reference'])}; EDA: {link(eda['reference'], 'results')}."),
            f"- SHA-256 best: `{hashes['best.pt']}`; last: `{hashes['last.pt']}`."]
    lines += ["", "### EDA empat hasil fine-tuning penelitian yang sudah selesai", "",
        ("Keempat run pengguna berikut tetap mempunyai 100 epoch unik. Tabel sumber versi tetap diperiksa hash-nya, "
        "lalu ditambahkan 19 gambar dan empat Tables custom per run. Checkpoint dan training tidak diubah. "
        "Hash model berikut berasal dari metadata publikasi asli; audit ini tidak mengulang training/evaluasi penuh. "
        "Paket EDA diunduh ulang dan diverifikasi SHA-256."), "",
        "| Run | Model versi tetap | EDA versi tetap | Report |", "|---|---|---|---|"]
    for row in report["finished_user_run_audit"]["rows"].values():
        lines.append(f"| `{row['run_name']}` | {link(row['checkpoint_artifact_ref'])} | "
            f"{link(row['custom_eda_artifact_ref'], 'results')} | [tiga kondisi]({row['report']['url']}) |")
    lines += ["", "| Model penelitian pengguna | SHA-256 best.pt dari metadata | SHA-256 last.pt dari metadata |", "|---|---|---|"]
    for row in report["finished_user_run_audit"]["rows"].values():
        hashes = row["checkpoint_hashes_from_original_publication_metadata"]
        lines.append(f"| {link(row['checkpoint_artifact_ref'])} | `{hashes['best.pt']}` | `{hashes['last.pt']}` |")
    lines += ["", "Buka [indeks EDA recovery](https://wandb.ai/" + NAMESPACE + "/reports/EDA--VmlldzoxODI3MDUxNg==) "
        "atau `custom_eda_report_url` pada Summary tiap run. Lima Native KD penelitian yang masih aktif hanya diberi "
        "report/tautan melalui API; tidak ada sesi training kedua. Batang final yang belum diukur tetap `NOT_RUN`. "
        "Source dan batas bukti tersedia pada [audit empat run selesai](../reports/pruning/recovery/finished_recovery_eda_audit_20261108.json).", ""]
    inventory = ROOT / "docs/ARTEFAK_WANDB.md"
    existing = inventory.read_text(encoding="utf-8")
    assert HEADING not in existing, "Append once; retain all historical sections."
    inventory.write_text(existing.rstrip() + "\n\n" + "\n".join(lines), encoding="utf-8")
    print("INVENTORIED", len(history), "confirmed versions", "15 KD smoke cases", flush=True)


if __name__ == "__main__":
    main()
