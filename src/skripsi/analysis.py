"""Cross-architecture summary after all research seeds have been evaluated."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
from scipy.stats import friedmanchisquare

from .config import EXPECTED_V2_FINGERPRINT, SEEDS

ARCHITECTURES = ("baseline", "bifpn", "asff", "afpn")
DISPLAY_NAMES = {
    "baseline": "FPN-PAN",
    "bifpn": "BiFPN",
    "asff": "ASFF",
    "afpn": "AFPN",
}


def summarize_experiments(output_root: str | Path) -> dict:
    """Collect completed evaluations and run Friedman only for a full 4×3 matrix."""
    root = Path(output_root).expanduser().resolve()
    reports = root / "reports"
    reports.mkdir(parents=True, exist_ok=True)
    rows = []
    missing = []
    for architecture in ARCHITECTURES:
        for seed in SEEDS:
            run = root / "runs" / f"yolo26n-{architecture}-train-seed-{seed}"
            overall_path = run / "evaluation" / "overall_metrics.json"
            efficiency_path = run / "evaluation" / "efficiency.json"
            if not overall_path.is_file() or not efficiency_path.is_file():
                missing.append({"architecture": architecture, "seed": seed})
                continue
            overall = json.loads(overall_path.read_text(encoding="utf-8"))
            efficiency = json.loads(efficiency_path.read_text(encoding="utf-8"))
            if overall.get("dataset_fingerprint") != EXPECTED_V2_FINGERPRINT:
                raise RuntimeError(f"Fingerprint dataset berbeda pada {run}")
            if (
                overall.get("architecture") != architecture
                or int(overall.get("seed", -1)) != seed
            ):
                raise RuntimeError(f"Identitas eksperimen tidak cocok pada {run}")
            rows.append(
                {
                    "Arsitektur": DISPLAY_NAMES[architecture],
                    "Seed": seed,
                    "mAP50-95": float(overall["mAP50_95"]),
                    "mAP50": float(overall["mAP50"]),
                    "Precision": float(overall["precision"]),
                    "Recall": float(overall["recall"]),
                    "F1-score": float(overall["f1"]),
                    "GPU": efficiency.get("gpu"),
                    "Precision komputasi": efficiency.get("precision"),
                    "Resolusi": efficiency.get("imgsz"),
                    "Dataset fingerprint": overall["dataset_fingerprint"],
                }
            )
    comparison = pd.DataFrame(rows)
    comparison.to_csv(reports / "model_comparison.csv", index=False)
    if missing:
        status = {
            "status": "not_run",
            "reason": "Data 3 seed x 4 arsitektur belum lengkap",
            "missing": missing,
        }
        (reports / "friedman_status.json").write_text(
            json.dumps(status, indent=2), encoding="utf-8"
        )
        return status
    consistency = {
        column: int(comparison[column].dropna().nunique())
        for column in ("GPU", "Precision komputasi", "Resolusi", "Dataset fingerprint")
    }
    if any(count > 1 for count in consistency.values()):
        raise RuntimeError(f"Lingkungan eksperimen tidak konsisten: {consistency}")
    pivot = comparison.pivot(index="Seed", columns="Arsitektur", values="mAP50-95").loc[
        list(SEEDS), list(DISPLAY_NAMES.values())
    ]
    statistic, p_value = friedmanchisquare(*(pivot[column] for column in pivot.columns))
    pivot.reset_index().to_csv(reports / "tabel_3_9_data_friedman.csv", index=False)
    result = {
        "Statistik Friedman": float(statistic),
        "p-value": float(p_value),
        "alpha": 0.05,
        "Keputusan": "Tolak H0" if p_value < 0.05 else "Belum menolak H0",
    }
    pd.DataFrame([result]).to_csv(reports / "hasil_friedman.csv", index=False)
    status = {"status": "completed", "result": result, "consistency": consistency}
    (reports / "friedman_status.json").write_text(
        json.dumps(status, indent=2), encoding="utf-8"
    )
    return status
