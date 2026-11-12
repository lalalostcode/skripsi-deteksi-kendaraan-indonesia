"""Smoke the exact notebook on CPU with real local weights and explicit test adapters.

No Kaggle job, Google/W&B login, or online artifact upload is performed.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import types
import uuid
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
# Keep verification settings writes inside the workspace, including during imports.
_config_directory = ROOT / "tmp/mgd_smoke_20261031/ultralytics-config"
_config_directory.mkdir(parents=True, exist_ok=True)
os.environ["YOLO_CONFIG_DIR"] = str(_config_directory)

import pytest
import torch
from PIL import Image
from ultralytics import YOLO
from ultralytics.nn.tasks import DetectionModel

sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tests"))
# Explicit, offline-only notebook fixture adapters.
import test_mgd_notebook as harness
from test_serial_multiseed_notebooks import FakeWandb


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--teacher", type=Path, required=True)
    parser.add_argument("--student", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()
    teacher_path, student_path = args.teacher.resolve(), args.student.resolve()
    assert teacher_path.is_file() and student_path.is_file()
    notebook_before = harness.TARGET.read_bytes()
    notebook_sha = hashlib.sha256(notebook_before).hexdigest()
    workdir = ROOT / "outputs/mgd_cpu_smoke" / uuid.uuid4().hex
    workdir.mkdir(parents=True)
    previous_threads = torch.get_num_threads()
    torch.set_num_threads(2)
    fixture = harness.environment.__wrapped__(workdir)
    env = next(fixture)
    default_teacher_sha = env["TEACHER_SHA256"]
    notebook_defaults = {name: env[name] for name in ("EPOCHS", "IMGSZ", "BATCH", "AMP", "SEEDS_TO_RUN", "RUN_FULL_TRAINING", "RUN_SMOKE_TEST")}
    result = {"status": "FAIL CPU SMOKE", "notebook_sha256": notebook_sha,
              "gpu_status": "NOT TESTED", "online_wandb": "NOT TESTED - local adapter", "workdir": str(workdir)}
    try:
        with pytest.MonkeyPatch.context() as patch:
            wandb, _events = harness.configure_synthetic_cpu(env, workdir, patch)
            env.update(TEACHER_CHECKPOINT=str(teacher_path), TEACHER_SHA256=default_teacher_sha,
                       MODEL_WEIGHTS=str(student_path), SEEDS_TO_RUN=[42],
                       EPOCHS=100, RUN_SMOKE_TEST=True, RUN_FULL_TRAINING=False)
            # Include an actual short final batch: five training images / batch two.
            dataset = Path(env["DATA"]).parent
            Image.new("RGB", (64, 64), (110, 70, 30)).save(dataset / "train/images/4.png")
            (dataset / "train/labels/4.txt").write_text("0 .5 .5 .25 .25\n")
            teacher_audits = []

            def local_teacher():
                assert env["file_sha256"](teacher_path) == default_teacher_sha
                model = YOLO(str(teacher_path)).model
                assert model.yaml["scale"] == "s" and model.model[-1].nc == 6
                assert env["class_names"](model.names) == harness.NAMES
                reference = DetectionModel("yolo26s.yaml", nc=6, verbose=False)
                assert {k: tuple(v.shape) for k, v in model.state_dict().items()} == {
                    k: tuple(v.shape) for k, v in reference.state_dict().items()}
                provenance = {"version": "v62", "requested_artifact": env["TEACHER_ARTIFACT"],
                              "best_pt_sha256": default_teacher_sha,
                              "local_tensor_schema_and_class_order_verified": True,
                              "remote_artifact_retrieval": "NOT TESTED in this smoke run"}
                teacher_audits.append(provenance)
                return str(teacher_path), provenance
            env["download_teacher"] = local_teacher
            upload_records = []

            def local_init(**kwargs):
                assert kwargs["name"] == "mgd_T26s_S26n_seed42_smoke"
                assert kwargs["job_type"] == "diagnostic" and kwargs["resume"] == "never"
                run = FakeWandb.init(wandb, **kwargs)
                run.use_artifact = lambda reference, type: types.SimpleNamespace(version="v62")
                original_log = run.log_artifact

                def local_log(artifact, aliases):
                    handle = original_log(artifact, aliases)
                    def wait():
                        files = {name: {"path": path, "bytes": Path(path).stat().st_size,
                                        "sha256": env["file_sha256"](Path(path))}
                                 for name, path in artifact.files.items()}
                        handle.wait()
                        assert all(Path(info["path"]).is_file() for info in files.values())
                        upload_records.append({"kind": artifact.type, "files": files, "local_adapter_only": True})
                    return types.SimpleNamespace(wait=wait)
                run.log_artifact = local_log
                return run
            wandb.init = local_init
            contexts, batches = [], []
            original_callbacks = env["make_callbacks"]

            def callbacks(context):
                contexts.append(context)
                callbacks = original_callbacks(context)
                original_start, original_end = callbacks["on_train_start"], callbacks["on_train_batch_end"]
                def start(trainer):
                    original_start(trainer)
                    context["initial_student_digest"] = env["state_digest"](trainer.model.student_model)
                def end(trainer):
                    original_end(trainer)
                    batches.append({"actual_batch": trainer.model.last_actual_batch,
                                    **{k: float(v) for k, v in trainer.loss_items.items()}})
                    context["final_student_digest"] = env["state_digest"](trainer.model.student_model)
                callbacks.update(on_train_start=start, on_train_batch_end=end)
                return callbacks
            env["make_callbacks"] = callbacks
            reports = []
            original_one = env["run_one_seed"]
            def one(seed, queue_dir, *, diagnostic=False):
                assert diagnostic is True  # Prevent accidental full 100-epoch execution.
                report = original_one(seed, queue_dir, diagnostic=diagnostic)
                reports.append(report)
                return report
            env["run_one_seed"] = one
            env["run_seed_queue"]()
            assert len(reports) == len(contexts) == len(teacher_audits) == 1
            smoke, context = reports[0], contexts[0]
            assert smoke["status"] == "PASS" and smoke["requested_epochs"] == smoke["completed_epochs"] == 1
            assert [r["actual_batch"] for r in batches] == [2, 2, 1]
            assert smoke["optimizer_steps"] >= 1 and smoke["teacher_parameters_and_bn_unchanged"]
            assert context["initial_student_digest"] != context["final_student_digest"]
            assert wandb.run is None and all(entry["waited"] for entry in wandb.runs[0].uploads)
            best = Path(smoke["student_checkpoint"])
            child = '''import sys,torch
from ultralytics import YOLO
torch.set_num_threads(2)
m=YOLO(sys.argv[1]).model.float().eval()
assert m.model[-1].nc==6 and not hasattr(m,'teacher_model') and not hasattr(m,'projector')
with torch.no_grad(): y=m(torch.zeros(1,3,64,64))
assert torch.isfinite(y[0]).all()
state=torch.load(sys.argv[2],map_location='cpu',weights_only=False)
assert state['projector'] and state['optimizer']['state'] and state['resume_status']=='RESUME_NOT_VERIFIED'
print('PASS_PORTABLE_STUDENT_AND_MGD_STATE')'''
            process = subprocess.run([sys.executable, "-I", "-c", child, str(best), str(best.parent / "training_state_last.pt")],
                                     capture_output=True, text=True, timeout=120, cwd=workdir, check=False,
                                     env={**os.environ, "OMP_NUM_THREADS": "2", "YOLO_CONFIG_DIR": str(workdir / "child-config")})
            assert process.returncode == 0, process.stdout + process.stderr
            assert "PASS_PORTABLE_STUDENT_AND_MGD_STATE" in process.stdout
            result.update(status="PASS CPU SMOKE - NOT GPU", seed=42, epochs=1,
                          torch=torch.__version__, cuda_available=torch.cuda.is_available(),
                          teacher_sha256=default_teacher_sha, teacher_audit=teacher_audits[0],
                          student_initial_sha256=env["file_sha256"](student_path),
                          student_source="https://github.com/ultralytics/assets/releases/download/v8.4.0/yolo26n.pt",
                          data="SYNTHETIC: 5 train / 4 val images, not Poltrada V3",
                          test_overrides={"device": "cpu", "imgsz": 64, "batch": 2, "workers": 0,
                                          "amp": False, "nbs": 2, "plots": False},
                          pipeline_result=smoke, batches=batches, gradient_norms=context["gradient"],
                          student_parameters_changed=True, portable_checkpoint_and_state=True,
                          local_artifact_checks=upload_records, notebook_defaults=notebook_defaults,
                          notebook_defaults_unchanged=True)
    except BaseException as error:
        result.update(error_type=type(error).__name__, error=str(error))
        raise
    finally:
        fixture.close()
        torch.set_num_threads(previous_threads)
        assert harness.TARGET.read_bytes() == notebook_before
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(json.dumps(result, indent=2), encoding="utf-8")
        print(json.dumps({"status": result["status"], "report": str(args.report.resolve())}), flush=True)


if __name__ == "__main__":
    main()
