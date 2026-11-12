"""Link measured and pending research treatments to their dedicated EDA reports."""
from __future__ import annotations

import os
from pathlib import Path

from scripts import recovery_custom_eda_source as eda

ENTITY = "ilhamrafiqin-brawijaya-university"
PROJECT = "yolom26-kd-indotraffic"
SUITE = "20261107T020459Z_45575a7f"


def main():
    if not os.environ.get("WANDB_API_KEY"):
        from dotenv import dotenv_values
        os.environ["WANDB_API_KEY"] = dotenv_values(Path(__file__).resolve().parents[1] / ".env", interpolate=False)["WANDB_API_KEY"]
    import wandb

    wr = eda.ceda_workspace_sdk()
    api = wandb.Api(timeout=60)
    groups = [f"yolo26s-lamp-{name}-{SUITE}" for name in ("ft", "kd")]
    runs = list(api.runs(f"{ENTITY}/{PROJECT}", filters={"group": {"$in": groups}}))
    chosen = {}
    for run in runs:
        value = dict(run.summary)
        if value.get("mode", run.config.get("mode")) != "full_val":
            continue
        seed = value.get("seed", run.config.get("seed"))
        target = value.get("target_speedup", run.config.get("target_speedup"))
        method = value.get("method", run.config.get("method", "finetune"))
        key = method, seed, target
        if key not in chosen or run.created_at > chosen[key][0].created_at:
            chosen[key] = run, value
    blocks = [wr.H1("EDA Recovery YOLO26s · Poltrada V3"),
        wr.P("Buka halaman per perlakuan untuk melihat baseline asal (biru), raw pruned (merah), dan hasil recovery (hijau). "
             "Indeks ini memisahkan hasil penelitian dari smoke CPU. NOT_RUN berarti hasil final belum tersedia; kosong bukan nilai nol."),
        wr.P("Setiap halaman menyediakan diagram per metrik, tabel per kelas/delta/gain, serta galeri channel/LAMP, PR/F1, "
             "confusion matrix, kurva training, dan trade-off. Angka latency hanya dibandingkan pada perangkat/protokol yang sama.")]
    for method in ("finetune", "native", "crosskd", "mgd"):
        slots = [(42, 1.1), (42, 1.2), (42, 1.7), (0, 1.2), (2024, 1.2), (7, 1.2), (3407, 1.2)] if method == "finetune" else [(s, 1.2) for s in (42, 0, 2024, 7, 3407)]
        finished = sum(chosen.get((method, seed, target), (None, {}))[1].get("status") == "PASS" for seed, target in slots)
        blocks += [wr.H2(f"{eda.CEDA_METHODS[method]} · hasil {finished}/{len(slots)} perlakuan")]
        lines = ["| Seed | Target | Status final | EDA / run |", "|---:|---|---|---|"]
        for seed, target in slots:
            run, value = chosen.get((method, seed, target), (None, {}))
            url = value.get("custom_eda_report_url")
            status = value.get("status", "NOT_RUN")
            diagnostic = " · TARGET_NOT_REACHED" if value.get("target_status") == "TARGET_NOT_REACHED" else ""
            link = f"[Perbandingan tiga kondisi]({url})" if url else (f"[Run {run.state}]({run.url})" if run else "NOT_RUN")
            lines.append(f"| {seed} | su{target:.2f}x | {status}{diagnostic} | {link} |")
        blocks.append(wr.MarkdownBlock("\n".join(lines)))
    path = Path("outputs/pruning/recovery_custom_eda_20261108/index_report.json")
    prior = eda.ceda_read(path, {})
    report = wr.Report.from_url(prior["url"]) if prior.get("url") else wr.Report(entity=ENTITY, project=PROJECT)
    report.title = "EDA Recovery YOLO26s - Poltrada V3"
    report.description = "Indeks perbandingan baseline, raw pruned dan tiap metode recovery, per seed/target."
    report.blocks = blocks
    report.save()
    url = report.url.replace("\\", "/")
    loaded = wr.Report.from_url(url)
    assert loaded.id == report.id and loaded.title == report.title
    eda.ceda_write(path, {"status": "CONFIRMED", "url": url, "id": report.id, "blocks": len(loaded.blocks),
                          "scope": "Published full_val runs and explicit pending treatments; smoke excluded."})
    print("EDA_INDEX", url, flush=True)


if __name__ == "__main__":
    main()
