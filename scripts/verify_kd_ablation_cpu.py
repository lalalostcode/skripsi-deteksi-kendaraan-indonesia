"""Offline six-candidate smoke and exact epoch-resume check on pinned real checkpoints."""
# ruff: noqa: S102 -- Execute trusted repository notebook declarations for compatibility verification.
from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
from functools import lru_cache
from pathlib import Path

from scripts.generate_kd_ablation_notebooks import ROOT, build_notebook


@lru_cache(maxsize=2)
def verification_book(method):
    # Freeze each source for the whole interrupted/uninterrupted comparison.
    return build_notebook(method)


def load_ablation(method, value, output, fixture, teacher, packages, *, epochs=1, resume=None):
    scope = {"__name__": "__main__", "display": lambda *a: None}
    for i, cell in enumerate(verification_book(method)["cells"]):
        tags = set(cell.get("metadata", {}).get("tags", []))
        if cell["cell_type"] != "code" or tags & {"run", "display", "group-display"}:
            continue
        exec(compile("".join(cell["source"]), f"ablation-{method}:{i}", "exec"), scope)
        if "config" in tags:
            scope.update(MODE="smoke", DEVICE="cpu", IMGSZ=64, BATCH=2, WORKERS=0, INSTALL_DEPENDENCIES=False,
                         WANDB_TRACKING=False, RUN_LATENCY_BENCHMARK=False, SMOKE_EPOCHS=epochs,
                         SMOKE_DATASET_ROOT=fixture, ABLATION_ROOT=output, OUTPUT_DIR=output / method,
                         LOCAL_RAW_PACKAGES=packages, TEACHER_ARTIFACT_REF=None, TEACHER_LOCAL_PATH=teacher,
                         RESUME_RUN_DIR=resume, EDA_DPI=50, EDA_MAX_LAYERS_PER_PAGE=64)
            if method == "crosskd":
                scope.update(KD_SCALE=value, KD_CLS_WEIGHT=value, KD_BOX_WEIGHT=value)
            else:
                scope["MGD_ALPHA"] = value
    scope["torch"].set_num_threads(2)
    return scope


def prepare(scope):
    folder = scope["RUN_DIR"]
    scope["ft_restore_snapshot"](folder)
    manifest, files = scope["ft_source"](folder)
    raw = scope["fine_tuned_load_raw"](files["pruned_fp32.pt"], manifest)
    scope["ft_verify_raw"](raw, manifest)
    data, record, train, yaml = scope["ft_dataset"](manifest, folder / "data")
    contract = scope["ft_contract"](folder, record)
    return raw, files["pruned_fp32.pt"], manifest, data, record, train, yaml, folder, None, contract


def equal(left, right):
    import torch
    if isinstance(left, torch.Tensor):
        return torch.equal(left, right)
    if isinstance(left, dict):
        return left.keys() == right.keys() and all(equal(v, right[k]) for k, v in left.items())
    if isinstance(left, (list, tuple)):
        return len(left) == len(right) and all(equal(x, y) for x, y in zip(left, right, strict=True))
    return left == right


def resume_check(method, root, fixture, teacher, packages):
    value = 1.0 if method == "crosskd" else 2e-5
    def scope(resume=None):
        s = load_ablation(method, value, root, fixture, teacher, packages, epochs=2, resume=resume)
        original = s["fine_tuned_train_args"]
        s["fine_tuned_train_args"] = lambda *a, **k: {**original(*a, **k), "nbs": 4, "warmup_epochs": 0., "close_mosaic": 0}
        return s
    whole = scope()
    expected = whole["ft_fit"](*prepare(whole))
    partial = scope()
    cls = partial["AblationKDTrainer"]
    original_save = cls.save_model
    def interrupt(self):
        value = original_save(self)
        if self.epoch == 0:
            raise KeyboardInterrupt("Injected ablation interruption")
        return value
    cls.save_model = interrupt
    try:
        partial["ft_fit"](*prepare(partial))
    except KeyboardInterrupt:
        pass
    else:
        raise AssertionError("Missing injected interruption")
    restored = scope(partial["RUN_DIR"])
    actual = restored["ft_fit"](*prepare(restored))
    checks = {key: whole["recovery_state_sha256"](expected[key]) == restored["recovery_state_sha256"](actual[key])
              for key in ("live_model", "ema")}
    checks.update({key: equal(expected[key], actual[key]) for key in
                   ("optimizer", "scheduler", "scaler", "gradients", "criterion", "ema_updates", "optimizer_steps", "method_state")})
    assert all(checks.values()), checks
    return {"status": "PASS", "checks": checks, "run_dir": str(restored["RUN_DIR"])}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--resume-only", action="store_true")
    parser.add_argument("--skip-resume", action="store_true")
    parser.add_argument("--finalize-only", action="store_true", help="Recheck saved evidence and protected notebook bytes without training")
    args = parser.parse_args()
    args.output = args.output.resolve()
    args.output.mkdir(parents=True, exist_ok=True)
    os.environ.update(OMP_NUM_THREADS="2", MKL_NUM_THREADS="2", MPLBACKEND="Agg", YOLO_AUTOINSTALL="false", WANDB_MODE="disabled")
    cache = json.loads((ROOT / "outputs/pruning/raw_per_seed_verified/verified_inputs.json").read_text())
    raw = next(v for v in cache.values() if v["seed"] == 42 and v["target"] == 1.2)
    packages = {"yolo26s-lamp-su1.20x-seed42-raw:v1": raw["folder"]}
    fixture = ROOT / "outputs/pruning/lamp_multiseed_cpu_20261106/fixture"
    teacher = ROOT / "outputs/pruning/teacher_yolo26m_v50_verified/best.pt"
    proof = {"status": "RUNNING", "scope": "CPU FP32 synthetic fixture, not Poltrada accuracy", "candidates": {}, "resume": {}}
    if args.finalize_only:
        proof = json.loads((args.output / "verification.json").read_text())
        assert proof["candidates"] or proof["resume"]
        assert all(v["status"] == "PASS" for group in (proof["candidates"], proof["resume"]) for v in group.values())
    def save():
        (args.output / "verification.json").write_text(json.dumps(proof, indent=2), encoding="utf-8")
    if not args.resume_only and not args.finalize_only:
        for method, grid in (("crosskd", [.5, 1., 2.]), ("mgd", [1e-5, 2e-5, 4e-5])):
            for value in grid:
                scope = load_ablation(method, value, args.output / "candidates", fixture, teacher, packages)
                result = scope["ablation_execute"](scope["RUN_DIR"])
                row = {k: result.get(k) for k in ("status", "reason", "epochs", "best_epoch", "raw_sha256", "initial_state_sha256",
                       "trained_state_sha256", "raw_unchanged", "reload_status", "validation_status", "eda_status")}
                row.update(run_dir=str(scope["RUN_DIR"]), smoke=scope["ft_read"](scope["RUN_DIR"] / "ablation_smoke.json"))
                proof["candidates"][f"{method}/{value}"] = row
                save()
                assert result["status"] == "PASS", result.get("errors")
                assert row["raw_unchanged"] and row["smoke"]["teacher_unchanged"]
                print("ABLATION_SMOKE_PASS", method, value, flush=True)
    if not args.skip_resume and not args.finalize_only:
        for method in ("crosskd", "mgd"):
            proof["resume"][method] = resume_check(method, args.output / "resume" / method, fixture, teacher, packages)
            save()
            print("ABLATION_RESUME_PASS", method, flush=True)
    tracked = subprocess.check_output(["git", "ls-files", "*.ipynb"], cwd=ROOT, text=True).splitlines()
    unchanged = True
    notebook_hashes = {}
    for path in tracked:
        content = (ROOT / path).read_bytes()
        # This Windows checkout contains both original LF files and CRLF checkouts.
        head = subprocess.check_output(["git", "show", "HEAD:" + path], cwd=ROOT)
        filtered = subprocess.check_output(["git", "cat-file", "--filters", "HEAD:" + path], cwd=ROOT)
        unchanged = unchanged and content in (head, filtered)
        notebook_hashes[path] = hashlib.sha256(content).hexdigest()
    assert unchanged, "Existing notebook changed"
    proof.update(status="PASS", existing_notebooks_unchanged=unchanged, existing_notebook_count=len(tracked),
                 existing_notebook_sha256=notebook_hashes,
                 gpu_training="NOT_RUN", full_validation_poltrada="NOT_RUN", wandb_online="NOT_RUN", test_split="NOT_RUN")
    save()


if __name__ == "__main__":
    main()
