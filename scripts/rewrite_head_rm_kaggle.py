"""Section the tested HEAD/RM objectives into clean, standalone Kaggle notebooks."""
from __future__ import annotations

import ast
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
FOLDER = ROOT / "notebooks/yolo26-kd-indotraffic/notebooks"

ENVIRONMENT = '''import importlib.metadata as metadata
import subprocess
import sys

for package, version in (("ultralytics", "8.4.155"), ("wandb", "0.28.1")):
    try:
        installed = metadata.version(package)
    except metadata.PackageNotFoundError:
        installed = None
    if installed != version:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", f"{package}=={version}"])

import copy
import hashlib
import json
import math
import os
import shutil
import tempfile
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path

import torch
from torch import nn
from ultralytics import YOLO, settings
from ultralytics.data.utils import check_det_dataset
from ultralytics.models.yolo.detect import DetectionTrainer
from ultralytics.nn.tasks import DetectionModel
from ultralytics.utils.torch_utils import ModelEMA, unwrap_model
import wandb

if metadata.version("ultralytics") != "8.4.155":
    raise RuntimeError("Restart kernel setelah instalasi Ultralytics.")
'''

SETUP = '''if not torch.cuda.is_available():
    raise RuntimeError("Aktifkan GPU di Kaggle: Settings > Accelerator.")
settings.update({"wandb": False})  # Logging diatur oleh callback notebook.
if not os.environ.get("WANDB_API_KEY"):
    from kaggle_secrets import UserSecretsClient
    os.environ["WANDB_API_KEY"] = UserSecretsClient().get_secret("WANDB_API_KEY")
wandb.login(key=os.environ["WANDB_API_KEY"])
if wandb.run is not None:
    raise RuntimeError("Selesaikan run W&B yang aktif sebelum menjalankan notebook ini.")

SESSION_DIR = Path(PROJECT) / f"{EXPERIMENT_ID}_{datetime.now():%Y%m%d_%H%M%S}_{uuid.uuid4().hex[:6]}"
SESSION_DIR.mkdir(parents=True, exist_ok=False)
print("GPU:", torch.cuda.get_device_name(DEVICE))
print("Output:", SESSION_DIR)
'''

DATASET = '''def class_names(names):
    return [str(names[k]) for k in sorted(names, key=lambda k: int(k))] if isinstance(names, dict) else list(names)

def file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()

data_info = check_det_dataset(DATA, autodownload=False)
if EVAL_SPLIT != "val" or not data_info.get("train") or not data_info.get("val"):
    raise ValueError("Gunakan split train dan val; test tidak dipakai untuk pemilihan checkpoint.")
DATASET_NAMES = class_names(data_info["names"])
DATASET_INFO = {"data": DATA, "names": DATASET_NAMES, "yaml_sha256": file_sha256(DATA)}
print("Kelas:", DATASET_NAMES)
'''

TEACHER = '''artifact = wandb.Api().artifact(TEACHER_ARTIFACT, type="model")
producer = artifact.logged_by()
if artifact.version != "v62" or producer is None or list(producer.path) != [WANDB_ENTITY, WANDB_PROJECT, "q5kp9k9d"]:
    raise ValueError("Artifact bukan teacher baseline YOLO26s yang ditetapkan.")
TEACHER_PATH = Path(artifact.get_path("best.pt").download(root=str(SESSION_DIR / "teacher")))
if file_sha256(TEACHER_PATH) != TEACHER_SHA256:
    raise ValueError("SHA-256 teacher berbeda; tidak ada fallback ke pretrained teacher.")
teacher = YOLO(str(TEACHER_PATH)).model
reference = DetectionModel("yolo26s.yaml", nc=data_info["nc"], verbose=False)
if teacher.yaml.get("scale") != "s" or class_names(teacher.names) != DATASET_NAMES:
    raise ValueError("Arsitektur atau urutan kelas teacher berbeda dari dataset.")
if {k: tuple(v.shape) for k, v in teacher.state_dict().items()} != {k: tuple(v.shape) for k, v in reference.state_dict().items()}:
    raise ValueError("Schema checkpoint bukan YOLO26s yang sesuai.")
TEACHER_INFO = {"artifact": TEACHER_ARTIFACT, "sha256": TEACHER_SHA256,
                "names": DATASET_NAMES, "artifact_metadata": dict(artifact.metadata)}
(SESSION_DIR / "teacher_provenance.json").write_text(json.dumps(TEACHER_INFO, indent=2))
print("Teacher:", artifact.name, "| epoch metadata:", artifact.metadata.get("epoch"))
del teacher, reference
'''

TRAINER = '''def teacher_digest(model):
    digest = hashlib.sha256()
    for name, value in model.state_dict().items():
        digest.update(name.encode())
        digest.update(value.detach().cpu().contiguous().reshape(-1).view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()

def gradient_norm(parameters):
    values = [p.grad.detach().float().norm() for p in parameters if p.grad is not None]
    result = float(torch.stack(values).norm()) if values else 0.0
    if not math.isfinite(result) or result <= 0:
        raise RuntimeError("Gradien harus finite dan nonzero.")
    return result

class StudentEMA(ModelEMA):
    def __init__(self, model):
        super().__init__(unwrap_model(model).student_model)
    def update(self, model):
        super().update(unwrap_model(model).student_model)
    def update_attr(self, model, include=(), exclude=()):
        super().update_attr(unwrap_model(model).student_model, include, exclude)

class ResearchTrainer(DetectionTrainer):
    def set_model_attributes(self):
        super().set_model_attributes()
        if self.args.resume or self.args.compile or self.world_size > 1:
            raise ValueError("Gunakan satu GPU, compile=False, resume=False; RESUME_NOT_VERIFIED.")
        teacher = YOLO(str(TEACHER_PATH)).model.to(self.device).eval().requires_grad_(False)
        self.model = KDModel(self.model, teacher, **METHOD_KWARGS).to(self.device)

    def build_optimizer(self, model, *args, **kwargs):
        model.teacher_model.eval().requires_grad_(False)
        optimizer = super().build_optimizer(model, *args, **kwargs)
        for group in optimizer.param_groups:
            group["params"] = [p for p in group["params"] if p.requires_grad]
        actual = [id(p) for group in optimizer.param_groups for p in group["params"]]
        expected = {id(p) for p in model.parameters() if p.requires_grad}
        if set(actual) != expected or len(actual) != len(set(actual)):
            raise RuntimeError("Parameter optimizer tidak sesuai.")
        return optimizer

    def _setup_train(self):
        super()._setup_train()
        self.ema = None
        self.ema = StudentEMA(self.model)

    def _build_train_pipeline(self):
        if self.batch_size != BATCH:
            raise RuntimeError("Batch berubah setelah OOM; hentikan dan tinjau konfigurasi.")
        super()._build_train_pipeline()

    def preprocess_batch(self, batch):
        batch = super().preprocess_batch(batch)
        self.actual_shape = tuple(batch["img"].shape)
        return batch

    def _handle_nan_recovery(self, epoch):
        if not torch.isfinite(self.loss).all() or not math.isfinite(float(self.fitness)):
            raise RuntimeError("Loss/fitness non-finite; training dihentikan.")
        return False

    def save_model(self):
        # Hanya DetectionModel biasa yang masuk best.pt/last.pt.
        student = copy.deepcopy(self.ema.ema).half().to(memory_format=torch.contiguous_format)
        student.criterion = None
        if not all(torch.isfinite(v).all() for v in student.state_dict().values() if v.is_floating_point()):
            raise RuntimeError("Checkpoint student mengandung NaN/Inf.")
        checkpoint = {"model": student, "ema": None, "optimizer": None, "epoch": int(self.epoch),
                      "best_fitness": float(self.best_fitness), "updates": self.ema.updates,
                      "train_args": vars(self.args), "train_metrics": {**self.metrics, "fitness": self.fitness},
                      "train_results": self.read_results_csv(), "version": metadata.version("ultralytics"),
                      "method": METHOD, "teacher_provenance": TEACHER_INFO, "resume_status": "INFERENCE_ONLY"}
        self.wdir.mkdir(parents=True, exist_ok=True)
        torch.save(checkpoint, self.last)
        if self.fitness == self.best_fitness:
            torch.save(checkpoint, self.best)
        return True
'''

LOGGING = '''def cpu_state(state):
    return {k: v.detach().cpu().clone() for k, v in state.items()}

def make_callbacks(run, diagnostic):
    context = {"steps": 0, "optimizer_steps": 0, "best_epoch": None, "sums": {}, "batches": 0}

    def start(trainer):
        context["teacher_before"] = teacher_digest(trainer.model.teacher_model)
        context["started"] = time.perf_counter()
        if trainer.device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(trainer.device)
        def before_step(optimizer, *_):
            # Fused AdamW memanggil hook juga saat GradScaler melewati update.
            found_inf = getattr(optimizer, "found_inf", None)
            if found_inf is not None and float(found_inf) != 0:
                context["amp_skipped_steps"] = context.get("amp_skipped_steps", 0) + 1
                return
            context["student_grad"] = gradient_norm(trainer.model.student_model.parameters())
            if hasattr(trainer.model, "auxiliary_parameters"):
                context["aux_grad"] = gradient_norm(trainer.model.auxiliary_parameters())
            context["optimizer_steps"] += 1
        context["gradient_hook"] = trainer.optimizer.register_step_pre_hook(before_step)

    def batch_start(trainer):
        trainer._oom_retries = 3  # Tidak mengizinkan native OOM batch-halving.
        if trainer.batch_size != BATCH or int(trainer.args.imgsz) != IMGSZ or bool(trainer.amp) != AMP:
            raise RuntimeError("Batch, resolusi, atau AMP efektif berbeda dari konfigurasi.")

    def batch_end(trainer):
        context["steps"] += 1
        context["batches"] += 1
        for key, value in trainer.model.last_metrics.items():
            if not math.isfinite(float(value)):
                raise RuntimeError(f"Metrik non-finite: {key}")
            context["sums"][key] = context["sums"].get(key, 0) + float(value)
        teacher = trainer.model.teacher_model
        if teacher.training or any(p.grad is not None for p in teacher.parameters()):
            raise RuntimeError("Teacher keluar dari eval atau menerima gradien.")

    def save(trainer):
        epoch = int(trainer.epoch) + 1
        if trainer.fitness == trainer.best_fitness:
            context["best_epoch"] = epoch
        state = {"epoch": epoch, "student": cpu_state(trainer.model.student_model.state_dict()),
                 "auxiliary": cpu_state({k: v for k, v in trainer.model.state_dict().items()
                                         if not k.startswith(("student_model.", "teacher_model."))}),
                 "optimizer": trainer.optimizer.state_dict(), "scaler": trainer.scaler.state_dict(),
                 "resume_status": "RESUME_NOT_VERIFIED"}
        torch.save(state, trainer.wdir / "training_state_last.pt")
        if trainer.fitness == trainer.best_fitness:
            shutil.copy2(trainer.wdir / "training_state_last.pt", trainer.wdir / "training_state_best.pt")
        # Snapshot tetap utuh sampai upload selesai, lalu file sementara dibersihkan.
        with tempfile.TemporaryDirectory(prefix=f"epoch-{epoch:03d}-", dir=str(trainer.save_dir)) as temporary:
            snapshot = Path(temporary)
            item = wandb.Artifact(f"{EXPERIMENT_ID}-{run.id}-checkpoints", type="model",
                    metadata={"epoch": epoch, "best_epoch": context["best_epoch"], "method": METHOD,
                              "diagnostic": diagnostic, "teacher_sha256": TEACHER_SHA256,
                              "resume_status": "RESUME_NOT_VERIFIED"})
            for filename in ("last.pt", "best.pt", "training_state_last.pt", "training_state_best.pt"):
                shutil.copy2(trainer.wdir / filename, snapshot / filename)
                item.add_file(str(snapshot / filename), name=filename)
            run.log_artifact(item, aliases=["latest", f"epoch-{epoch:03d}"] +
                             (["best"] if trainer.fitness == trainer.best_fitness else [])).wait()
        values = {f"train/{k}": v / context["batches"] for k, v in context["sums"].items()}
        values.update(trainer.metrics)
        values.update(trainer.lr)
        values.update(epoch=epoch, best_epoch=context["best_epoch"],
                      elapsed_seconds=time.perf_counter() - context["started"],
                      amp_skipped_steps=context.get("amp_skipped_steps", 0),
                      student_gradient_norm=context.get("student_grad", 0),
                      auxiliary_gradient_norm=context.get("aux_grad", 0))
        if trainer.device.type == "cuda":
            values.update(peak_allocated_gib=torch.cuda.max_memory_allocated() / 2**30,
                          peak_reserved_gib=torch.cuda.max_memory_reserved() / 2**30)
        run.log(values, step=epoch)
        context["sums"], context["batches"] = {}, 0

    return {"on_train_start": start, "on_train_batch_start": batch_start,
            "on_train_batch_end": batch_end, "on_model_save": save}, context
'''

TRAINING = '''def experiment_config():
    return {"method": METHOD, "model": MODEL_WEIGHTS, "data": DATA, "epochs": EPOCHS,
            "imgsz": IMGSZ, "batch": BATCH, "seed": SEED, "optimizer": OPTIMIZER, "lr0": LR,
            "weight_decay": WEIGHT_DECAY, "warmup_epochs": WARMUP_EPOCHS,
            "close_mosaic": CLOSE_MOSAIC, "amp": AMP, "workers": WORKERS,
            "method_kwargs": METHOD_KWARGS, "teacher_sha256": TEACHER_SHA256}

def run_training(diagnostic=False):
    if not 1 <= EPOCHS <= 100:
        raise ValueError("Gunakan 1 sampai 100 epoch.")
    config = experiment_config()
    name = ("diagnostic" if diagnostic else "training") + "_" + uuid.uuid4().hex[:6]
    run = wandb.init(entity=WANDB_ENTITY, project=WANDB_PROJECT, group=WANDB_GROUP,
                     name=f"{EXPERIMENT_ID}_seed{SEED}_{name}", id=uuid.uuid4().hex[:8], resume="never",
                     tags=[METHOD, "teacher-s", "student-n", f"seed{SEED}", "poltrada-v3", name.split("_")[0]],
                     config={**config, "dataset": DATASET_INFO, "teacher": TEACHER_INFO}, dir=str(SESSION_DIR))
    callbacks, context = make_callbacks(run, diagnostic)
    trainer = None
    try:
        trainer = ResearchTrainer(overrides={"model": MODEL_WEIGHTS, "data": DATA, "epochs": 1 if diagnostic else EPOCHS,
            "imgsz": IMGSZ, "batch": BATCH, "seed": SEED, "device": DEVICE, "workers": WORKERS,
            "optimizer": OPTIMIZER, "lr0": LR, "weight_decay": WEIGHT_DECAY,
            "warmup_epochs": WARMUP_EPOCHS, "close_mosaic": CLOSE_MOSAIC, "amp": AMP,
            "patience": 100, "split": EVAL_SPLIT, "project": str(SESSION_DIR), "name": name,
            "plots": not diagnostic, "save": True, "exist_ok": False, "resume": False,
            "compile": False, "distill_model": None})
        for event, callback in callbacks.items():
            trainer.add_callback(event, callback)
        trainer.train()
        if context["optimizer_steps"] < (2 if diagnostic else 1):
            raise RuntimeError("Belum ada cukup optimizer step nyata yang berhasil.")
        if teacher_digest(trainer.model.teacher_model) != context["teacher_before"]:
            raise RuntimeError("Parameter atau BatchNorm teacher berubah.")
        if not all(math.isfinite(float(v)) for v in trainer.metrics.values()):
            raise RuntimeError("Metrik validasi non-finite.")
        student = YOLO(str(trainer.best)).model.eval()
        if student.yaml.get("scale") != "n" or hasattr(student, "teacher_model"):
            raise RuntimeError("Checkpoint inferensi bukan student-only.")
        result = {"status": "PASS", "config": config, "epochs_this_run": 1 if diagnostic else EPOCHS,
                  "training_batches": context["steps"], "best_pt": str(trainer.best),
                  "best_epoch": context["best_epoch"], "metrics": trainer.metrics,
                  "optimizer_steps": context["optimizer_steps"],
                  "seconds": time.perf_counter() - context["started"], "resume": "RESUME_NOT_VERIFIED"}
        (Path(trainer.save_dir) / "summary.json").write_text(json.dumps(result, indent=2))
        run.summary.update(result)
        run.finish(exit_code=0)
        return result
    except BaseException:
        run.finish(exit_code=1)
        raise
    finally:
        if context.get("gradient_hook"):
            context["gradient_hook"].remove()
'''


def cell(kind, identifier, source):
    item = {"cell_type": kind, "id": identifier, "metadata": {}, "source": source.strip().splitlines(keepends=True)}
    if kind == "code":
        item.update(execution_count=None, outputs=[])
    return item


def rewrite(method, number):
    path = FOLDER / f"{number}_{method}_yolo26n_teacher_yolo26s.ipynb"
    notebook = json.loads(path.read_bytes())
    class_name = "HEADModel" if method == "head" else "RMModel"
    existing = {c["id"]: "".join(c["source"]) for c in notebook["cells"]}
    if method + "-algorithm" in existing:
        algorithm = existing[method + "-algorithm"]
    else:
        algorithm = (existing[method + "-helpers"] + "\n\n" +
                     existing[method + "-model"].split("\nKDModel =")[0] + "\n\n" +
                     existing[method + "-numerical"])
    tree = ast.parse(algorithm)
    model = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == class_name)
    lines = algorithm.splitlines(keepends=True)
    helpers = "".join(lines[:model.lineno - 1])
    model_source = "".join(lines[model.lineno - 1:model.end_lineno])
    oracle = "".join(lines[model.end_lineno:])
    verification = "print(" + ("numerical_verification" if method == "head" else "rm_numerical_verification") + "())"
    if not oracle.rstrip().endswith(verification):
        oracle += "\n\n" + verification + "\n"
    kwargs = ({"akd_weight": 1.0, "ckd_weight": 1.0, "assistant_det_weight": 1.0} if method == "head"
              else {"rm_weight": 4.0, "rm_cls_weight": 1.0, "rm_iou_weight": 1.0,
                    "candidate_topk": 10, "candidate_iou_threshold": 0.0})
    config = f'''EXPERIMENT_ID = "{number}_{method}_yolo26n_teacher_yolo26s"
METHOD = "{method}"
MODEL_WEIGHTS = "yolo26n.pt"
DATA = "/kaggle/input/datasets/ilhamrafiqin/poltrada-v3-yolo26-clean/data.yaml"
EPOCHS = 100
IMGSZ = 640
BATCH = 16
SEED = 42
DEVICE = 0
WORKERS = 2
OPTIMIZER = "AdamW"
LR = 0.001
WEIGHT_DECAY = 0.0005
WARMUP_EPOCHS = 3.0
CLOSE_MOSAIC = 10
AMP = True
EVAL_SPLIT = "val"
PROJECT = "/kaggle/working/results"
RUN_FULL_TRAINING = False
METHOD_KWARGS = {kwargs!r}
WANDB_ENTITY = "ilhamrafiqin-brawijaya-university"
WANDB_PROJECT = "yolom26-kd-indotraffic"
WANDB_GROUP = "T26s_S26n"
TEACHER_ARTIFACT = "ilhamrafiqin-brawijaya-university/yolom26-kd-indotraffic/02_baseline_yolo26s-q5kp9k9d-checkpoints:v62"
TEACHER_SHA256 = "5bc3ae61f8da1c9f03916922de10146805c27c64a11e46edd766a00d0f6f18fd"
'''
    title = "HEAD Adapted for YOLO26" if method == "head" else "Rank Mimicking (RM-only)"
    theory = ("Assistant adalah salinan Detect teacher yang trainable dan menerima neck student melalui adapter. "
              "AKD = detection loss assistant + mean MSE fitur intermediate assistant terhadap teacher. "
              "CKD = mean MSE fitur klasifikasi student terhadap teacher, melalui projector. "
              "Loss total = detection student + AKD + CKD. Ini adaptasi satu keluarga YOLO26, bukan reproduksi heterogeneous HEAD asli. "
              "[Paper HEAD](https://www.ecva.net/papers/eccv_2022/papers_ECCV/papers/136690310.pdf) / "
              "[kode penulis](https://github.com/LutingWang/HEAD)." if method == "head" else
              "RM memindahkan ranking kandidat per objek: softmax pada sumbu kandidat, lalu KL teacher ke student. "
              "Dua indikatornya adalah skor kelas GT dan IoU box dengan GT; mean antar-instance, bobot RM 4. "
              "Kandidat adalah cell P3/P4/P5 sebelum NMS, dipilih teacher top-10 IoU di dalam GT dengan ownership tunggal. "
              "Skor student tetap terhubung ke autograd. Pemilihan kandidat adalah adaptasi untuk YOLO26; tidak ada PFI. "
              "[Paper RM](https://ojs.aaai.org/index.php/AAAI/article/download/20018/19777).")
    parts = [
        ("markdown", "intro", f"# {title}: YOLO26s → YOLO26n\n\nPoltrada V3 · seed 42 · Kaggle GPU.\n\nTambahkan dataset, aktifkan GPU/Internet, dan secret `WANDB_API_KEY`. Run All dengan default menjalankan smoke test **1 epoch** pada 640/batch 16, termasuk validasi dan checkpoint. Ubah `RUN_FULL_TRAINING=True` untuk training setelah smoke test lulus.\n\n{theory}"),
        ("markdown", "config-md", "## 1. Konfigurasi\nUbah parameter penelitian di cell ini."),
        ("code", "config", config),
        ("markdown", "environment-md", "## 2. Persiapan Kaggle\nPyTorch GPU bawaan Kaggle tetap digunakan."),
        ("code", "environment", ENVIRONMENT), ("code", "setup", SETUP),
        ("markdown", "dataset-md", "## 3. Dataset dan teacher\nSplit dan anotasi tetap. Teacher dikunci ke `best.pt` artifact v62; metadata epoch-nya 63, bukan 62."),
        ("code", "dataset", DATASET), ("code", "teacher", TEACHER),
        ("markdown", "loss-md", "## 4. Loss dan model\nDetection loss resmi YOLO26 tetap digunakan pada kedua cabang. KD ditambahkan sekali dengan faktor batch. Modul tambahan HEAD hanya untuk training."),
        ("code", "helpers", helpers), ("code", "model", model_source + f"\n\nKDModel = {class_name}\n"),
        ("markdown", "trainer-md", "## 5. Trainer dan checkpoint\nEMA, validasi, dan `best.pt` memakai student biasa. State training tambahan disimpan terpisah; resume belum diverifikasi."),
        ("code", "trainer", TRAINER),
        ("markdown", "logging-md", "## 6. Logging W&B\nLoss per komponen, mAP, precision/recall, LR, gradien, VRAM, dan durasi dicatat. Snapshot setiap epoch baru dibersihkan setelah upload selesai."),
        ("code", "logging", LOGGING), ("code", "training", TRAINING),
        ("markdown", "diagnostic-md", "## 7. Smoke test Kaggle — 1 epoch\nSatu epoch pada seluruh split train, lalu validasi dan load ulang student. Jika OOM/AMP/loss gagal, training berhenti tanpa menurunkan batch atau resolusi otomatis."),
        ("code", "numerical", oracle),
        ("code", "diagnostic", "DIAGNOSTIC_RESULT = run_training(diagnostic=True)\nprint(json.dumps(DIAGNOSTIC_RESULT, indent=2))"),
        ("markdown", "full-md", "## 8. Training dan hasil\nSatu metode, satu seed, maksimal 100 epoch. Untuk uji satu epoch saja, set `EPOCHS=1`."),
        ("code", "full", '''TRAINING_RESULT = None
if RUN_FULL_TRAINING:
    if DIAGNOSTIC_RESULT["status"] != "PASS" or DIAGNOSTIC_RESULT["config"] != experiment_config():
        raise RuntimeError("Jalankan ulang smoke test setelah mengubah konfigurasi.")
    TRAINING_RESULT = run_training(diagnostic=False)
else:
    print("Smoke test selesai. Full training belum dijalankan.")'''),
        ("code", "results", '''result = TRAINING_RESULT or DIAGNOSTIC_RESULT
print("Checkpoint student:", result["best_pt"])
print("Best epoch:", result["best_epoch"])
print("Validation:", result["metrics"])
print("Durasi (detik):", round(result["seconds"], 2))'''),
    ]
    notebook["cells"] = [cell(kind, f"{method}-{identifier}", source) for kind, identifier, source in parts]
    notebook["metadata"].pop("head_rm_sources", None)
    notebook["metadata"]["head_rm_kaggle"] = {"method": method, "runtime": "sectioned-kaggle"}
    path.write_text(json.dumps(notebook, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    print(path.name, sum(len("".join(c["source"])) for c in notebook["cells"] if c["cell_type"] == "code"))


if __name__ == "__main__":
    for method, number in (("head", "13"), ("rm", "14")):
        rewrite(method, number)
