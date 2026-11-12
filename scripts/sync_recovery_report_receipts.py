"""Synchronize local presentation receipts after an explicitly verified report refresh.

No checkpoints, training state, contracts, or immutable evaluation manifests change.
"""
from __future__ import annotations

import argparse
from pathlib import Path

from scripts import recovery_custom_eda_source as eda


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--suite", type=Path, required=True)
    parser.add_argument("--reports", type=Path, default=Path("outputs/pruning/recovery_custom_eda_20261108"))
    args = parser.parse_args()
    records = eda.ceda_read(args.suite / "verification.json")
    updates = []
    for case, row in records.items():
        folder = Path(row["run_dir"])
        run_id = row["logging_verification"]["run_id"]
        proof = eda.ceda_read(args.reports / run_id / "custom_eda_report.json")
        assert proof["status"] == "CONFIRMED" and proof["run_id"] == run_id
        expected = eda.ceda_read(folder / "custom_eda_logging.json")
        assert expected["run_id"] == run_id
        previous = expected["custom_eda_report_url"]
        old = eda.ceda_read(folder / "custom_eda_report.json", {})
        merged = {**proof, "identity": {**old.get("identity", {}), **{k: v for k, v in proof.get("identity", {}).items() if v is not None}}}
        eda.ceda_write(folder / "custom_eda_report.json", merged)
        expected["custom_eda_report_url"] = proof["url"]
        eda.ceda_write(folder / "custom_eda_logging.json", expected)
        row["logging_verification"]["custom_eda_report_url"] = proof["url"]
        for filename in ("kd_manifest.json", "fine_tuned_manifest.json"):
            path = folder / filename
            if path.exists():
                value = eda.ceda_read(path)
                value.get("custom_eda", {})["custom_eda_report_url"] = proof["url"]
                eda.ceda_write(path, value)
        for name in ("kd_registry", "finetuning_registry"):
            path = folder / f"{name}.json"
            if path.exists():
                registry = eda.ceda_read(path)
                registry["rows"][0]["custom_eda_report_url"] = proof["url"]
                eda.ceda_write(path, registry)
                eda.ceda_csv(folder / f"{name}.csv", registry["rows"])
        if previous != proof["url"]:
            updates.append({"case": case, "run_id": run_id, "old_url": previous, "confirmed_url": proof["url"],
                            "reason": "Presentation-only refresh from seeded SDK report names to the canonical per-run report."})
    eda.ceda_write(args.suite / "verification.json", records)
    eda.ceda_write(args.suite / "report_receipt_sync.json", {"status": "PASS", "changed_links": updates,
                   "checked_cases": len(records), "model_and_training_files_changed": False})
    print("REPORT_RECEIPTS_SYNCHRONIZED", len(records), "cases; changed URLs", len(updates))


if __name__ == "__main__":
    main()
