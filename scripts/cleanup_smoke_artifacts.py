"""Inventory and remove explicitly identified W&B smoke artifacts and local smoke weights.

Raw research inputs and original measurements are protected. An inventory is
always saved before deletion; local JSON/CSV/log/figure evidence remains available.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

ROOT = Path(__file__).resolve().parents[1]
PROJECT = "ilhamrafiqin-brawijaya-university/yolom26-kd-indotraffic"
SMOKE = re.compile(r"(?:^|[-_])smoke(?:[-_]|$)", re.I)
LOCAL_PREFIXES = ("selected_kd_", "selected_finetune_", "source_matched_cpu", "recovery_kd_", "finetune_binding_smoke_",
                  "finetune_custom_eda_", "recovery_n_custom_eda_", "standard_finetuning_cpu", "pruned_fine_tuned_cpu")


def read(path, default=None):
    return json.loads(Path(path).read_text(encoding="utf-8")) if Path(path).is_file() else default


def save(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(2**20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def protected_collection(name):
    name = name.rsplit("/", 1)[-1].split(":")[0]
    # A smoke run's confusion-matrix Table may contain the word baseline;
    # it is a diagnostic Table, not an input model collection.
    if name.startswith("run-"):
        return False
    return bool(re.match(r"^yolo26s-lamp-su\d+\.\d+x-seed\d+-raw$", name) or
                (("baseline" in name.lower() or "teacher" in name.lower()) and not SMOKE.search(name)))


def local_smoke_roots():
    parent = (ROOT / "outputs/pruning").resolve()
    result = []
    for folder in parent.iterdir():
        cpu_pruning = folder.name.startswith("lamp_") and any(word in folder.name for word in ("cpu", "replay"))
        if not (folder.is_dir() and (folder.name.startswith(LOCAL_PREFIXES) or cpu_pruning) and folder.resolve().parent == parent):
            continue
        manifests = [p for name in ("kd_manifest.json", "fine_tuned_manifest.json", "recovery_manifest.json", "suite_manifest.json")
                     for p in folder.rglob(name)]
        modes = {read(p, {}).get("mode") for p in manifests}
        if any(mode not in {None, "smoke"} for mode in modes):
            continue
        if cpu_pruning and modes != {"smoke"}:
            continue
        if not manifests and not any(word in folder.name for word in ("cpu", "preflight", "resume", "smoke", "kaggle_verified")):
            continue
        result.append(folder)
    # Never infer smoke solely from a test directory name. Older PKD/MGD
    # fixtures without explicit smoke manifests are left outside this scope.
    old_recovery = (ROOT / "outputs/recovery").resolve()
    if old_recovery.is_dir():
        for folder in old_recovery.iterdir():
            if not folder.is_dir() or not folder.name.startswith("lamp_"):
                continue
            modes = {read(p, {}).get("mode") for p in folder.rglob("suite_manifest.json")}
            if modes == {"smoke"}:
                result.append(folder)
    return result


def local_smoke_ids():
    result = set()
    for folder in local_smoke_roots():
        for identity in folder.rglob("wandb_identity.json"):
            value = read(identity, {})
            manifest = read(identity.parent / "kd_manifest.json", read(identity.parent / "fine_tuned_manifest.json", {}))
            contract = read(identity.parent / "resume_contract.json", {}).get("contract", {})
            if manifest.get("mode", contract.get("mode")) == "smoke" and value.get("id"):
                result.add(value["id"])
    return result


def inspect_remote(api, evidence):
    known = local_smoke_ids()
    smoke_runs, active = {}, set()
    # Querying full records also protects against a changed/unsupported regex filter.
    for run in api.runs(PROJECT, per_page=100):
        if SMOKE.search(run.name or "") or SMOKE.search(run.group or "") or run.config.get("mode") == "smoke":
            known.add(run.id)
            smoke_runs[run.id] = {"id": run.id, "name": run.name, "state": run.state}
            if run.state in {"running", "pending", "preempting"}:
                active.add(run.id)
    collections, candidates, protected = [], [], []
    for kind in api.artifact_types(PROJECT):
        for collection in kind.collections(per_page=100):
            name = collection.name
            inferred = next((rid for rid in known if name == rid or name.startswith((f"run-{rid}-", rid + "-"))), None)
            if not (SMOKE.search(name) or inferred):
                continue
            if protected_collection(name):
                protected.append({"type": kind.name, "name": name, "reason": "protected input collection"})
                continue
            versions = list(collection.artifacts(per_page=100))
            entry = {"type": kind.name, "name": name, "id": collection.id, "active_versions": len(versions),
                     "smoke_run_id": inferred, "safe_to_delete": inferred not in active}
            collections.append(entry)
            for artifact in versions:
                record = {"id": artifact.id, "reference": artifact.qualified_name, "type": kind.name,
                          "collection": name, "state": artifact.state, "size": artifact.size,
                          "aliases": list(artifact.aliases), "safe_to_delete": entry["safe_to_delete"]}
                users = artifact.used_by()
                non_smoke = []
                for user in users:
                    if user.id not in known:
                        if SMOKE.search(user.name or "") or user.config.get("mode") == "smoke" or SMOKE.search(user.group or ""):
                            known.add(user.id)
                        else:
                            non_smoke.append({"id": user.id, "name": user.name})
                if non_smoke:
                    record.update(safe_to_delete=False, protected_consumers=non_smoke)
                    entry["safe_to_delete"] = False
                candidates.append(record)
            save(evidence, {"status": "INVENTORY_RUNNING", "project": PROJECT, "smoke_runs": smoke_runs,
                 "known_smoke_run_ids": sorted(known), "active_smoke_run_ids": sorted(active),
                 "collections": collections, "versions": candidates, "protected": protected})
    value = {"status": "PLANNED", "project": PROJECT, "smoke_runs": smoke_runs, "known_smoke_run_ids": sorted(known),
             "active_smoke_run_ids": sorted(active), "collections": collections, "versions": candidates, "protected": protected}
    save(evidence, value)
    return value


def delete_remote(api, plan, receipt):
    done = read(receipt, {"status": "RUNNING", "deleted_versions": [], "deleted_collections": [], "errors": []})
    for item in plan["versions"]:
        if not item["safe_to_delete"] or item["id"] in {row["id"] for row in done["deleted_versions"]}:
            continue
        assert item["reference"].startswith(PROJECT + "/") and not protected_collection(item["collection"])
        try:
            artifact = api.artifact(item["reference"])
            assert artifact.id == item["id"]
            artifact.delete(delete_aliases=True)
            done["deleted_versions"].append(item)
        except Exception as error:
            done["errors"].append({"reference": item["reference"], "error": str(error)})
        save(receipt, done)
    for item in plan["collections"]:
        if not item["safe_to_delete"] or item["id"] in {row["id"] for row in done["deleted_collections"]}:
            continue
        assert not protected_collection(item["name"])
        try:
            collection = api.artifact_collection(type_name=item["type"], name=PROJECT + "/" + item["name"])
            assert collection.id == item["id"]
            # Do not bulk-delete a collection containing any unexpected/new version.
            remaining = list(collection.artifacts(per_page=100))
            if remaining:
                raise ValueError("Collection still contains active versions; inspect before deletion")
            collection.delete()
            done["deleted_collections"].append(item)
        except Exception as error:
            done["errors"].append({"collection": item["name"], "error": str(error)})
        save(receipt, done)
    done.update(status="PASS" if not done["errors"] else "PARTIAL",
                logical_deleted_bytes=sum(row["size"] or 0 for row in done["deleted_versions"]),
                completed_at=datetime.now(ZoneInfo("Asia/Jakarta")).isoformat())
    save(receipt, done)
    return done


def cleanup_local(receipt, *, execute):
    from filelock import FileLock, Timeout
    roots = local_smoke_roots()
    records = read(receipt, {"status": "RUNNING", "removed": [], "locked": [], "errors": []})
    already = {row["path"] for row in records["removed"]}
    for folder in roots:
        # Only generated model binaries; keep metrics, logs, inventories and figures.
        files = [p for p in folder.rglob("*") if p.is_file() and p.suffix.lower() in {".pt", ".pth", ".ckpt", ".safetensors", ".onnx", ".engine"}]
        for path in files:
            resolved = path.resolve()
            assert resolved.is_relative_to(folder.resolve()) and resolved.is_relative_to(ROOT.resolve())
            if str(path) in already:
                continue
            lock_path = next((parent / ".run.lock" for parent in path.parents if (parent / ".run.lock").is_file() and parent.is_relative_to(folder)), None)
            lock = FileLock(str(lock_path), timeout=0) if lock_path else None
            try:
                if lock:
                    lock.acquire()
                item = {"path": str(path), "bytes": path.stat().st_size, "sha256": sha(path), "smoke_root": str(folder)}
                if execute:
                    path.unlink()
                records["removed"].append(item)
                save(receipt, records)
            except Timeout:
                records["locked"].append(str(path))
            except Exception as error:
                records["errors"].append({"path": str(path), "error": str(error)})
            finally:
                if lock:
                    lock.release()
    records.update(status="PASS" if not records["errors"] and not records["locked"] else "PARTIAL", executed=execute,
                   bytes_removed=sum(row["bytes"] for row in records["removed"]),
                   completed_at=datetime.now(ZoneInfo("Asia/Jakarta")).isoformat())
    save(receipt, records)
    return records


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--remote", action="store_true")
    parser.add_argument("--local", action="store_true")
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--output", type=Path, default=ROOT / "reports/pruning/recovery/smoke_cleanup_20261108")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    if args.local:
        result = cleanup_local(args.output / ("local_deleted.json" if args.execute else "local_plan.json"), execute=args.execute)
        print("LOCAL", result["status"], len(result["removed"]), result["bytes_removed"], flush=True)
    if args.remote:
        if not os.environ.get("WANDB_API_KEY"):
            from dotenv import dotenv_values
            os.environ["WANDB_API_KEY"] = dotenv_values(ROOT / ".env", interpolate=False)["WANDB_API_KEY"]
        import wandb
        api = wandb.Api(timeout=60)
        plan_path = args.output / "remote_plan.json"
        plan = read(plan_path)
        if not plan or plan.get("status") != "PLANNED":
            plan = inspect_remote(api, plan_path)
        # Earlier inventories conservatively excluded baseline-labelled Tables.
        # Only restore those tied to already confirmed smoke run IDs.
        for item in list(plan.get("protected", [])):
            if item["type"] != "run_table" or not any(item["name"].startswith(f"run-{rid}-") for rid in plan["known_smoke_run_ids"]):
                continue
            collection = api.artifact_collection(type_name=item["type"], name=PROJECT + "/" + item["name"])
            versions = list(collection.artifacts(per_page=100))
            if versions:
                continue
            plan["collections"].append({"type": item["type"], "name": item["name"], "id": collection.id,
                                        "active_versions": 0, "safe_to_delete": True})
            plan["protected"].remove(item)
        save(plan_path, plan)
        print("REMOTE_PLAN", len(plan["collections"]), len(plan["versions"]), sum(v["size"] or 0 for v in plan["versions"]), flush=True)
        if args.execute:
            result = delete_remote(api, plan, args.output / "remote_deleted.json")
            print("REMOTE_DELETED", result["status"], len(result["deleted_collections"]), len(result["deleted_versions"]), flush=True)


if __name__ == "__main__":
    main()
