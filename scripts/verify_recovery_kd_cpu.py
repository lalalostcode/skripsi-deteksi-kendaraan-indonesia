"""CPU smoke of recovery KD on real raw checkpoints with a locked YOLO26m teacher."""
# ruff: noqa: S102, BLE001 -- Execute trusted notebook declarations and isolate each method/variant fixture.
from __future__ import annotations

import argparse
import ast
import json
import os
from pathlib import Path
from unittest.mock import patch

from recovery_kd_method_dependencies import method_dependencies
from verify_lamp_gflops_cpu import fixture_dataset

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "scripts"
METHODS = ("native", "crosskd", "mgd")
TEACHER_V50_SHA = "d749ef0baa0d6c3ae94e83ded1d9999d83c804b76ad5a841fc2ebec953af1428"


def scope_for(method):
    path = ROOT / "notebooks/kaggle/pruning/05_lamp_yolo26n_gflops_no_recovery.ipynb"
    notebook = json.loads(path.read_text(encoding="utf-8"))
    scope = {"display": lambda value: None, "__name__": f"recovery_{method}_cpu"}
    for index, cell in enumerate(notebook["cells"]):
        if set(cell["metadata"].get("tags", [])) & {"config", "imports", "helpers"}:
            exec(compile("".join(cell["source"]), f"pruning#{index}", "exec"), scope)
    for filename in ("lamp_recovery_training_source.py", "standard_finetuning_training_source.py"):
        exec(compile((SOURCE / filename).read_text(encoding="utf-8"), filename, "exec"), scope)
    for source in method_dependencies(ROOT, method):
        exec(compile(source, f"{method}-dependencies", "exec"), scope)
    file = SOURCE / "recovery_kd_methods_source.py"
    exec(compile(file.read_text(encoding="utf-8"), str(file), "exec"), scope)
    file = SOURCE / "lamp_recovery_runtime_source.py"
    settings = next(node for node in ast.parse(file.read_text(encoding="utf-8")).body
                    if isinstance(node, ast.FunctionDef) and node.name == "recovery_settings")
    exec(compile(ast.Module(body=[settings], type_ignores=[]), str(file), "exec"), scope)
    return scope


def run_one(args, method, variant, raw_path, yaml_path, *, check_resume):
    import torch

    env = scope_for(method)
    env.update(RECOVERY_METHOD=method, TEACHER_ARTIFACT_REF=None, TEACHER_LOCAL_PATH=args.teacher,
               TEACHER_SHA256=args.teacher_sha256, TEACHER_CHECKPOINT_FILENAME="best.pt",
               RECOVERY_EPOCHS=3 if check_resume else 1, RECOVERY_OPTIMIZER="AdamW", RECOVERY_LR0=0.0001,
               RECOVERY_WEIGHT_DECAY=0.0005, RECOVERY_WARMUP_EPOCHS=0, RECOVERY_CLOSE_MOSAIC=0,
               RECOVERY_PATIENCE=0, RECOVERY_BATCH=2, RECOVERY_NBS=8 if check_resume else 2,
               RECOVERY_WORKERS=0, RECOVERY_AMP=False, TARGET_DEVICE=torch.device("cpu"), IMGSZ=64)
    settings = env["recovery_settings"]()
    settings.update(plots=False, mosaic=0.0)
    target = args.output_dir / f"{method}-{variant}"
    raw_sha = env["sha256_file"](raw_path)
    raw = torch.load(raw_path, map_location="cpu", weights_only=False)["model"]
    signature = env["recovery_structure_signature"](raw)
    trainer, kwargs, contract = env["recovery_prepare_method"](target)
    contract_sha = env["workflow_digest"](contract)

    def fit(folder, callback=None):
        return env["recovery_fit"](raw, raw_path, raw_sha, yaml_path, raw.names, folder, settings, contract_sha,
                                   val_batch=2, committed_callback=callback, trainer_class=trainer, trainer_kwargs=kwargs)

    state, _ = fit(target / "uninterrupted")
    assert state["epoch"] == settings["epochs"] and state["optimizer_steps"] > 0
    assert env["sha256_file"](raw_path) == raw_sha
    assert all(env["recovery_structure_signature"](state[name]) == signature for name in ("live_model", "ema"))
    assert all(type(state[name]) is env["_RecoveryDetectionModel"] for name in ("live_model", "ema"))
    kd_key = "mgd_weighted_loss" if method == "mgd" else "dis_loss"
    assert state["epoch_rows"][-1]["losses"][kd_key] > 0
    assert all(not key.startswith(("val/kd_", "val/mgd_", "val/dis_loss"))
               for key in state["epoch_rows"][-1]["metrics"])
    result = {"status": "PASS", "method": method, "variant": variant, "epochs": state["epoch"],
              "optimizer_steps": state["optimizer_steps"], "raw_sha256": raw_sha,
              "teacher_sha256": args.teacher_sha256, "teacher_state_sha256": state["method_state"]["teacher_state_sha256"],
              "final_live_sha256": env["recovery_state_sha256"](state["live_model"]), "kd_loss_key": kd_key,
              "kd_loss": state["epoch_rows"][-1]["losses"][kd_key], "resume": "NOT_RUN"}
    if check_resume:
        class SimulatedInterruption(Exception):
            pass

        def interrupt(index, row):
            if index["epoch"] == 1:
                raise SimulatedInterruption

        try:
            fit(target / "interrupted", interrupt)
        except SimulatedInterruption:
            pass
        else:
            raise AssertionError("Expected interruption after first epoch commit.")
        snapshot, _ = env["recovery_committed_state"](target / "interrupted", contract_sha, raw_sha)
        assert snapshot["epoch"] == 1 and snapshot["gradients"] and snapshot["method_state"]["projector_gradients"]
        resumed, _ = fit(target / "interrupted")
        assert resumed["epoch"] == state["epoch"] and resumed["optimizer_steps"] == state["optimizer_steps"]
        for name in ("live_model", "ema"):
            assert env["recovery_state_sha256"](state[name]) == env["recovery_state_sha256"](resumed[name]), name
        for name, value in state["method_state"]["projector"].items():
            assert torch.equal(value, resumed["method_state"]["projector"][name]), name
        result.update(resume="PASS", first_epoch_accumulation=snapshot["accumulate"],
                      first_epoch_pending_projector_gradients=len(snapshot["method_state"]["projector_gradients"]))
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-n", type=Path)
    parser.add_argument("--raw-s", type=Path, required=True)
    teacher = parser.add_mutually_exclusive_group(required=True)
    teacher.add_argument("--teacher", type=Path, help="Local checkpoint of the verified YOLO26m teacher.")
    teacher.add_argument("--synthetic-teacher-m", action="store_true", help="Use randomly initialized M for plumbing only.")
    parser.add_argument("--teacher-sha256", default=TEACHER_V50_SHA, help="Expected SHA-256 of the local teacher.")
    parser.add_argument("--method", nargs="+", choices=METHODS, default=METHODS)
    parser.add_argument("--model", nargs="+", choices=("n", "s"), default=("s",))
    parser.add_argument("--check-resume", action="store_true", help="Run exact resume comparison for S MGD.")
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    args.output_dir = args.output_dir.resolve()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    os.environ.update({"WANDB_MODE": "disabled", "MPLBACKEND": "Agg", "YOLO_AUTOINSTALL": "false",
                       "OMP_NUM_THREADS": "2", "MKL_NUM_THREADS": "2"})
    from hashlib import file_digest

    import requests
    import torch
    import yaml
    from ultralytics.nn.tasks import DetectionModel

    torch.set_num_threads(2)
    dataset = args.output_dir / "fixture"
    yaml_path = fixture_dataset(dataset)
    yaml_path.write_text(yaml.safe_dump({"path": str(dataset), **yaml.safe_load(yaml_path.read_text(encoding="utf-8"))}),
                         encoding="utf-8")
    if "n" in args.model and args.raw_n is None:
        parser.error("--raw-n is required when testing variant n.")
    if args.synthetic_teacher_m:
        torch.manual_seed(42)
        raw = torch.load(args.raw_s.resolve(strict=True), map_location="cpu", weights_only=False)["model"]
        teacher = DetectionModel("yolo26m.yaml", nc=len(raw.names), verbose=False).float()
        teacher.names = raw.names
        # Calibrate only BN buffers on synthetic inputs so the random teacher's
        # raw maps do not collapse to constant head biases at 64 pixels.
        batch_norm = [module for module in teacher.modules() if isinstance(module, torch.nn.BatchNorm2d)]
        momentum = [module.momentum for module in batch_norm]
        for module in batch_norm:
            module.momentum = 1.0
        with torch.no_grad():
            teacher.train()(torch.rand(6, 3, 64, 64))
        for module, value in zip(batch_norm, momentum, strict=True):
            module.momentum = value
        teacher.eval()
        args.teacher = args.output_dir / "synthetic_teacher_yolo26m.pt"
        torch.save({"model": teacher}, args.teacher)
        with args.teacher.open("rb") as stream:
            args.teacher_sha256 = file_digest(stream, "sha256").hexdigest()

    def forbidden(*args, **kwargs):
        raise AssertionError("Offline recovery smoke attempted HTTP.")

    rows = []
    with patch.object(requests.Session, "request", forbidden):
        for method in args.method:
            for variant in args.model:
                try:
                    raw = getattr(args, f"raw_{variant}").resolve(strict=True)
                    result = run_one(args, method, variant, raw, yaml_path,
                                     check_resume=args.check_resume and method == "mgd" and variant == "s")
                except Exception as error:
                    import traceback
                    traceback.print_exc()
                    result = {"status": "FAIL", "method": method, "variant": variant,
                              "error": f"{type(error).__name__}: {error}"}
                rows.append(result)
                print(method, variant, result["status"], flush=True)
                (args.output_dir / "report.json").write_text(json.dumps({"scope": "CPU FP32 64px synthetic train/val",
                    "teacher_origin": "SYNTHETIC_RANDOM_YOLO26M_BN_CALIBRATED" if args.synthetic_teacher_m else "LOCAL_CHECKPOINT",
                    "real_teacher_v50": "TESTED" if args.teacher_sha256 == TEACHER_V50_SHA else "NOT_RUN",
                    "cases": rows, "kaggle": "NOT_RUN", "filkom": "NOT_RUN", "wandb_online": "NOT_RUN"}, indent=2),
                    encoding="utf-8")
    if any(row["status"] != "PASS" for row in rows):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
