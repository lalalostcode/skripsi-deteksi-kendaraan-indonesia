"""Summarize measured retention/cleanup evidence; missing stages stay explicitly incomplete."""
import ast
import hashlib
import re
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

from scripts.cleanup_smoke_artifacts import ROOT, read, save


def main():
    evidence = ROOT / "reports/pruning/recovery/smoke_cleanup_20261108"
    local = read(evidence / "local_deleted.json", {})
    deleted = {row["path"]: row for row in local.get("removed", [])}
    automatic = []
    roots = ("recovery_storage_smoke_20261108", "recovery_storage_offline_smoke_20261108",
             "selected_finetune_storage_resume_20261108", "selected_kd_storage_resume_20261108")
    for name in roots:
        for path in (ROOT / "outputs/pruning" / name).rglob("smoke_local_cleanup.json"):
            receipt = read(path)
            automatic.append({"receipt": str(path), "status": receipt["status"], "bytes_removed": receipt["bytes_removed"]})
            for row in receipt["removed"]:
                absolute = str((path.parent / row["relative_path"]).resolve())
                deleted[absolute] = {**row, "path": absolute}
    remaining = [name for name in deleted if Path(name).is_file()]
    cases = {}
    for name in roots[:2]:
        for method, row in read(ROOT / "outputs/pruning" / name / "verification.json", {}).items():
            cases[method] = row
    resumes = {"finetune": read(ROOT / "outputs/pruning/selected_finetune_storage_resume_20261108/resume_verification.json", {}),
               "mgd": read(ROOT / "outputs/pruning/selected_kd_storage_resume_20261108/mgd/resume_verification.json", {})}
    before = read(ROOT / "outputs/recovery_retention_contracts_before_20261108.json")
    books = []
    for name, expected in before.items():
        book = read(ROOT / name)
        headings = [line for cell in book["cells"] if cell["cell_type"] == "markdown"
                    for line in "".join(cell["source"]).splitlines() if line.startswith("#")]
        implementation = None
        for cell in book["cells"]:
            if cell["cell_type"] != "code":
                continue
            source = "".join(cell["source"])
            compile(source, name, "exec")
            for node in ast.parse(source).body:
                if isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id == "IMPLEMENTATION_SHA256" for t in node.targets):
                    implementation = ast.literal_eval(node.value)
        books.append({"path": name, "sha256": hashlib.sha256((ROOT / name).read_bytes()).hexdigest(),
                      "status": "PASS" if headings == expected["headings"] and implementation == expected["implementation"] else "FAIL",
                      "implementation_sha256": implementation, "storage_sha256": book["metadata"]["research_workflow"]["storage_sha256"]})
    remote = read(evidence / "remote_verification.json", {"status": "NOT_RUN"})
    inputs_local = read(evidence / "protected_local_inputs_after.json", {})
    inputs_remote = read(evidence / "protected_remote_inputs.json", {})
    test_log = ROOT / "outputs/recovery_retention_tests_20261108_final.log"
    test_text = test_log.read_text(encoding="utf-8", errors="replace") if test_log.is_file() else ""
    measured_tests = re.search(r"(\d+) passed", test_text)
    guard_log = ROOT / "outputs/recovery_cleanup_guard_tests_20261108.log"
    guard_text = guard_log.read_text(encoding="utf-8", errors="replace") if guard_log.is_file() else ""
    measured_guard = re.search(r"(\d+) passed, (\d+) skipped", guard_text)
    tests_pass = bool(measured_tests and measured_guard and not re.search(r"\d+ failed", test_text + guard_text))
    passed = (len(cases) == 4 and all(row["status"] == "PASS" for row in cases.values())
              and all(cases[m].get("smoke_artifact_cleanup", {}).get("status") == "PASS" for m in ("finetune", "mgd"))
              and all(row.get("status") == "PASS" for row in resumes.values())
              and len(books) == 10 and all(row["status"] == "PASS" for row in books)
              and not remaining and all(row["status"] == "PASS" for row in automatic)
              and inputs_local.get("status") == inputs_remote.get("status") == remote["status"] == "PASS" and tests_pass)
    report = {"status": "PASS" if passed else "PARTIAL", "at_jakarta": datetime.now(ZoneInfo("Asia/Jakarta")).isoformat(),
        "scope": "recovery best/latest I/O; canonical inputs and full research artifacts preserved",
        "local_deleted_binary_count": len(deleted), "local_deleted_bytes": sum(row["bytes"] for row in deleted.values()),
        "deleted_local_paths_still_present": remaining, "local_cleanup_receipts": automatic,
        "local_old_cleanup": {k: local.get(k) for k in ("status", "bytes_removed", "completed_at", "errors", "locked")},
        "remote_cleanup": remote, "input_integrity_local": inputs_local, "input_access_remote": inputs_remote,
        "notebooks": books, "unchanged_other_notebooks": read(evidence / "changed_notebooks.json"),
        "smoke": cases, "resume": resumes,
        "regression_tests": {"status": "PASS" if tests_pass else "NOT_RUN_OR_FAIL",
                             "passed": int(measured_tests[1]) if measured_tests else None,
                             "log": "outputs/recovery_retention_tests_20261108_final.log",
                             "additional_cleanup_guard_passed": int(measured_guard[1]) if measured_guard else None,
                             "symlink_guard_skipped": int(measured_guard[2]) if measured_guard else None,
                             "guard_log": "outputs/recovery_cleanup_guard_tests_20261108.log"},
        "limits": ["CPU FP32 64px, batch/nbs 2, six synthetic images per train/val; not Poltrada accuracy",
                   "Full validation, CUDA latency and 100-epoch GPU testing of this I/O policy: NOT_RUN",
                   "System-managed W&B history cannot be deleted through Artifact.delete; details in remote receipt",
                   "W&B deletion is soft delete; physical storage reclamation/usage timing was not measured",
                   "Untagged legacy test fixtures and all full research versions were excluded"]}
    save(ROOT / "reports/pruning/recovery/recovery_storage_verification_20261108.json", report)
    print("STORAGE_VERIFICATION", report["status"], len(cases), "methods", len(books), "notebooks",
          f"{report['local_deleted_bytes'] / 2**30:.2f} GiB deleted")


if __name__ == "__main__":
    main()
