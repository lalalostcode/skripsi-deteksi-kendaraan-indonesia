"""Collect actual recovery/EDA evidence; never promote intended runs to measured results."""
from __future__ import annotations

import argparse
import ast
import hashlib
import json
import re
from datetime import datetime, timedelta, timezone
from pathlib import Path

import nbformat

from scripts.generate_standard_finetuning_notebooks import DESTINATIONS

ROOT = Path(__file__).resolve().parents[1]
METHODS = ("native", "crosskd", "mgd")


def read(path, default=None):
    path = Path(path)
    return json.loads(path.read_text(encoding="utf-8")) if path.is_file() else default


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def collect(suite):
    rows = read(suite / "verification.json", {})
    for method in METHODS:
        for key, value in read(suite / method / "verification.json", {}).items():
            # Snapshot-restored evidence retains the same actual epoch and run.
            if not rows.get(key, {}).get("snapshot_restored_run_dir"):
                rows[key] = value
    online = read(suite / "online_verification.json", {})
    for key, value in rows.items():
        verified = online.get(key, {})
        if verified.get("status") == "PASS":
            for name in ("custom_eda_report_url", "custom_eda_artifact_ref", "custom_eda_status"):
                if name in verified:
                    value["logging_verification"][name] = verified[name]
    (suite / "verification.json").write_text(json.dumps(rows, indent=2), encoding="utf-8")
    books = {}
    for platform in ("kaggle", "filkom"):
        for (method, variant), filename in DESTINATIONS.items():
            path = ROOT / "notebooks" / platform / "recovery" / filename
            book = nbformat.read(path, as_version=4)
            nbformat.validate(book)
            implementation = None
            for i, cell in enumerate(book.cells):
                if cell.cell_type != "code":
                    continue
                compile(cell.source, f"{path}:{i}", "exec")
                for node in ast.parse(cell.source).body:
                    if isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id == "IMPLEMENTATION_SHA256" for t in node.targets):
                        implementation = ast.literal_eval(node.value)
            books[f"{platform}/{method}/{variant}"] = {"path": path.relative_to(ROOT).as_posix(), "sha256": sha(path),
                "implementation_sha256": implementation, "custom_eda_sha256": book.metadata.research_workflow.custom_eda_sha256,
                "schema": "PASS", "compile": "PASS"}
    before = read(ROOT / "outputs/recovery_custom_eda_before_20261108.json")
    after = {p.relative_to(ROOT).as_posix(): sha(p) for p in (ROOT / "notebooks").rglob("*.ipynb")}
    changed = [name for name, digest in before.items() if after[name] != digest]
    assert set(changed) == {row["path"] for row in books.values()}
    cases = []
    for key, value in rows.items():
        folder = Path(value["run_dir"])
        manifest = read(folder / "kd_manifest.json", {})
        stages = {name: {**{k: stage.get(k) for k in ("parameters", "gflops", "conv_output_channels", "checkpoint_sha256")},
                         **{k: stage.get("validation", {}).get(k) for k in ("map50_95", "map50", "precision", "recall")},
                         "latency_status": stage.get("latency", {}).get("status", "NOT_RUN")}
                  for name, stage in manifest.get("stages", {}).items()}
        cases.append({"case": key, **value, "stages": stages,
                      "publications": read(folder / "artifact_publications.json", {}),
                      "custom_eda_publication": read(folder / "custom_eda_publication.json", {}),
                      "custom_eda_report": read(folder / "custom_eda_report.json", {})})
    return rows, cases, books, before, after, changed


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--suite", type=Path, default=ROOT / "outputs/pruning/selected_kd_online_delivery_20261108")
    parser.add_argument("--strict", action="store_true")
    args = parser.parse_args()
    args.suite.mkdir(parents=True, exist_ok=True)
    rows, cases, books, before, after, changed = collect(args.suite)
    resumes = {m: read(ROOT / f"outputs/pruning/selected_kd_resume_delivery_20261107/{m}/resume_verification.json", {"status": "NOT_RUN"}) for m in METHODS}
    snapshots = {m: read(args.suite / f"snapshot_{m}.json", {"status": "NOT_RUN"}) for m in METHODS}
    api = read(args.suite / "online_verification.json", {})
    group = read(args.suite / "group_verification.json", {"status": "NOT_RUN"})
    backfill = read(ROOT / "outputs/pruning/recovery_custom_eda_20261108/verification.json", {})
    tests_path = ROOT / "outputs/recovery_custom_eda_delivery_tests_20261108_retry.log"
    if not tests_path.exists():
        tests_path = ROOT / "outputs/recovery_custom_eda_delivery_tests_20261108.log"
    tests = tests_path.read_text(encoding="utf-8") if tests_path.is_file() else ""
    counts = re.findall(r"(\d+) passed", tests)
    complete = (len(rows) == 15 and all(row["status"] == "PASS" for row in rows.values())
                and len(api) == 15 and all(row["status"] == "PASS" for row in api.values())
                and all(row["status"] == "PASS" for row in resumes.values())
                and all(row["status"] == "PASS" for row in snapshots.values()) and group["status"] == "PASS"
                and len(backfill) == 4 and all(row["status"] == "PASS" and row["original_epoch_count"] == row["new_epoch_count"] for row in backfill.values())
                and counts and not re.search(r"\d+ failed", tests))
    report = {"date_jakarta": datetime.now(timezone(timedelta(hours=7))).isoformat(), "status": "PASS" if complete else "PARTIAL",
        "scope": "Notebook implementation, real-checkpoint synthetic-CPU smoke, W&B EDA/publication verification",
        "notebooks": books, "changed_notebooks": sorted(changed), "unchanged_notebooks": len(before) - len(changed),
        "notebook_hashes_before": before, "notebook_hashes_after": after, "tests_passed": int(counts[-1]) if counts else None,
        "test_log": str(tests_path.relative_to(ROOT)),
        "cache_cleanup": read(ROOT / "outputs/recovery_cache_cleanup_20261108.json"),
        "smoke": {"expected_cases": 15, "completed_cases": sum(r["status"] == "PASS" for r in rows.values()),
                  "epochs_per_case": 1, "device": "local CPU", "imgsz": 64, "batch": 2, "precision": "FP32",
                  "train_images": 6, "validation_images": 6, "dataset": "synthetic fixture"},
        "cases": cases, "epoch_resume": resumes, "wandb_snapshot_resume": snapshots,
        "online_api_verification": api, "group_verification": group,
        "report_receipt_sync": read(args.suite / "report_receipt_sync.json"),
        "finished_finetuning_custom_eda": backfill,
        "finished_user_run_audit": read(ROOT / "reports/pruning/recovery/finished_recovery_eda_audit_20261108.json"),
        "reports_only_no_training_session": read(ROOT / "outputs/pruning/recovery_custom_eda_20261108/live_reports.json", {}),
        "raw_restoration": read(ROOT / "reports/pruning/lamp_yolo26s_raw_restored_20261107.json"),
        "finetuning_raw_bindings": read(ROOT / "reports/pruning/lamp_yolo26s_ft_raw_bindings_20261108.json"),
        "finetuning_late_seed_smoke": read(ROOT / "outputs/pruning/finetune_binding_smoke_20261108/verification.json"),
        "yolo26n_cpu_smoke": read(ROOT / "outputs/pruning/recovery_n_custom_eda_cpu_20261108/verification.json"),
        "kaggle_variant_cpu": read(ROOT / "outputs/pruning/selected_kd_kaggle_custom_eda_cpu_20261108/verification.json", {}),
        "new_finetune_notebook_online_smoke": read(ROOT / "outputs/pruning/finetune_custom_eda_online_20261108/verification.json", {}),
        "limitations": ["The agent did not execute 100 epochs, full Poltrada V3 validation, GPU AMP or CUDA latency.",
            "Four finished user FT runs were read and given presentation-only EDA; their 100-epoch counts and checkpoints were retained.",
            "Active user KD runs received reports and a summary link only, without attaching another training session.",
            "YOLO26n legacy fixture lacks pruning inventories: new three-condition EDA is PASS, legacy diagnostics remain PARTIAL/NOT_RUN.",
            "W&B report browser inspection was rejected by automatic review at a CoreWeave redirect; API round-trip and local PNG inspection were used.",
            "One regression test and the first MGD snapshot restore exhausted local disk space; official W&B LRU cache cleanup preserved all experiment outputs, and the interrupted checks were retried.",
            "Earlier failed diagnostic outputs and historical artifact versions are preserved."]}
    target = ROOT / "reports/pruning/recovery/recovery_custom_eda_verification_20261108.json"
    temp = target.with_suffix(".tmp.json")
    temp.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temp.replace(target)
    print(json.dumps({"status": report["status"], "smoke_pass": report["smoke"]["completed_cases"], "api_verified": len(api),
                      "tests": report["tests_passed"], "unchanged_notebooks": report["unchanged_notebooks"]}))
    if args.strict:
        assert complete, "Measured evidence is not complete."


if __name__ == "__main__":
    main()
