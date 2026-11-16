"""Build two standalone FILKOM ablations without writing any existing notebook."""
from __future__ import annotations

import ast
import hashlib
import json
import re
import runpy
from copy import deepcopy
from pprint import pformat

from scripts import generate_selected_kd_notebooks as selected

ROOT = selected.base.ROOT
NAMES = {
    "crosskd": "06_yolo26s_crosskd_kdscale_ablation.ipynb",
    "mgd": "07_yolo26s_mgd_alpha_ablation.ipynb",
}


def put(cell, source):
    if cell["cell_type"] == "code":
        ast.parse(source)
        cell.update(execution_count=None, outputs=[])
    cell["source"] = source.strip().splitlines(keepends=True)
    cell["id"] = hashlib.sha256(source.encode()).hexdigest()[:12]


def tagged(book, tag):
    return next(c for c in book["cells"] if tag in c.get("metadata", {}).get("tags", []))


def build_notebook(method):
    if method not in NAMES:
        raise ValueError(method)
    book = selected.build_notebook("filkom", method)
    mgd = selected.build_notebook("filkom", "mgd")
    config = "".join(tagged(mgd, "config")["source"])
    config = config.replace("RECOVERY_METHOD = 'mgd'", f"RECOVERY_METHOD = {method!r}")
    config = re.sub(r"SOURCE_NOTEBOOK = .*", f"SOURCE_NOTEBOOK = 'notebooks/filkom/recovery/{selected.base.METHOD_NOTEBOOKS[method][0]}'", config)
    config = re.sub(r"STUDENT_INPUTS = \[.*?\n\]", 'STUDENT_INPUTS = ["yolo26s-lamp-su1.20x-seed42-raw:v1"]', config, flags=re.DOTALL)
    config = re.sub(r"OUTPUT_DIR = Path\(os.environ.get\(\"SKRIPSI_OUTPUT_DIR\".*", 'ABLATION_ROOT = Path(os.environ.get("SKRIPSI_ABLATION_OUTPUT_DIR", WORKSPACE_ROOT / "outputs/filkom/recovery/ablation")).expanduser()', config)
    config = config.replace("OUTPUT_DIR = OUTPUT_DIR / RECOVERY_METHOD", "OUTPUT_DIR = ABLATION_ROOT / RECOVERY_METHOD")
    config = config.replace("AUTO_DELETE_SMOKE_ARTIFACTS = True", "AUTO_DELETE_SMOKE_ARTIFACTS = False")
    config = config.replace("AUTO_DELETE_LOCAL_SMOKE_WEIGHTS = True", "AUTO_DELETE_LOCAL_SMOKE_WEIGHTS = False")
    config = re.sub(r"MGD_ALPHA = .*\nMGD_MASK_RATIO = .*", "", config)
    knob = ('KD_SCALE_VALUES = [0.5, 1.0, 2.0]\nKD_SCALE = 1.0\n'
            'KD_CLS_WEIGHT = KD_SCALE\nKD_BOX_WEIGHT = KD_SCALE\nKD_CLS_GAMMA = 1.0' if method == "crosskd" else
            'MGD_ALPHA_VALUES = [1e-5, 2e-5, 4e-5]\nMGD_ALPHA = 2e-5\nMGD_MASK_RATIO = 0.65')
    put(tagged(book, "config"), config + "\n" + knob + '\nREUSE_EXISTING_DEFAULT = True\n')
    pin = {k: v for k, v in selected.selected_pins().items() if v["seed"] == 42}
    selection = "".join(tagged(book, "selection")["source"])
    selection = selection[:selection.index("STUDENT_PINS =")] + "STUDENT_PINS = " + pformat(pin, width=110) + "\n" + selection[selection.index("if len(STUDENT_INPUTS)"):]
    selection += '''
if SEED != 42 or EVAL_SPLIT != "val":
    raise ValueError("Ablation hanya seed 42 dan validation.")
if RECOVERY_METHOD == "crosskd":
    if KD_SCALE not in KD_SCALE_VALUES or KD_SCALE_VALUES != [0.5, 1.0, 2.0]:
        raise ValueError("Pilih satu KD_SCALE dari grid terkunci.")
    if (KD_CLS_WEIGHT, KD_BOX_WEIGHT, KD_CLS_GAMMA) != (KD_SCALE, KD_SCALE, 1.0):
        raise ValueError("Bobot CrossKD harus mengikuti KD_SCALE, gamma=1.")
    ABLATION_VALUE, ABLATION_PARAMETER = float(KD_SCALE), "kd_scale"
    PARAMETER_TOKEN = "scale" + f"{KD_SCALE:.1f}".replace(".", "p")
else:
    if MGD_ALPHA not in MGD_ALPHA_VALUES or MGD_ALPHA_VALUES != [1e-5, 2e-5, 4e-5] or MGD_MASK_RATIO != 0.65:
        raise ValueError("Pilih satu MGD_ALPHA dari grid terkunci; mask=0.65.")
    ABLATION_VALUE, ABLATION_PARAMETER = float(MGD_ALPHA), "mgd_alpha"
    PARAMETER_TOKEN = "alpha" + f"{MGD_ALPHA:.0e}".replace("e-0", "e-")
OUTPUT_DIR = OUTPUT_DIR / PARAMETER_TOKEN
'''
    put(tagged(book, "selection"), selection)
    put(tagged(book, "protocol"), "".join(tagged(mgd, "protocol")["source"]))
    bootstrap = "".join(tagged(mgd, "bootstrap")["source"])
    bootstrap = bootstrap.replace("MGD YOLO26s memerlukan", "Ablation KD YOLO26s memerlukan")
    start, end = bootstrap.index("if RESUME_RUN_DIR:"), bootstrap.index("\nPINS =")
    bootstrap = bootstrap[:start] + '''EXECUTION_ID = f"{datetime.now(UTC):%Y%m%dT%H%M%SZ}-{uuid.uuid4().hex[:8]}"
if RESUME_RUN_DIR:
    RUN_DIR = Path(RESUME_RUN_DIR).expanduser().resolve(strict=True)
    saved = json.loads((RUN_DIR / "ablation_identity.json").read_text())
    if (saved["method"], saved["value"], saved["mode"]) != (RECOVERY_METHOD, ABLATION_VALUE, MODE):
        raise ValueError("Resume berasal dari kandidat atau mode lain.")
    EXECUTION_ID = saved["execution_id"]
else:
    RUN_DIR = OUTPUT_DIR.expanduser().resolve() / EXECUTION_ID
    RUN_DIR.mkdir(parents=True, exist_ok=False)
''' + bootstrap[end:]
    put(tagged(book, "bootstrap"), bootstrap)
    runtime = "".join(tagged(book, "runtime")["source"])
    old_implementation = re.findall(r"^IMPLEMENTATION_SHA256 = ['\"]([0-9a-f]+)['\"]", runtime, re.MULTILINE)[-1]
    runtime = runtime.replace("selected-raw-kd-v1", "filkom-kd-ablation-v1")
    runtime = runtime.replace('type="model" if label != "results" else "results"',
                              'type="model" if label in {"checkpoints", "resume"} else "results"')
    dependencies = runpy.run_path(str(ROOT / "scripts/recovery_kd_method_dependencies.py"))["method_dependencies"]
    method_sha = hashlib.sha256("\n".join(dependencies(ROOT, method, source_matched=True)).encode()).hexdigest()
    if method == "crosskd":
        for name in ("filkom_mgd_labels_source.py", "filkom_mgd_dataset_source.py"):
            runtime += "\n\n" + (ROOT / "scripts" / name).read_text(encoding="utf-8")
    # The overlay changes orchestration only; the embedded method definitions stay identical.
    runtime += f"\nMETHOD_SOURCE_SHA256 = {method_sha!r}\nHISTORICAL_IMPLEMENTATIONS = [{old_implementation!r}]\n"
    runtime += "\n\n" + (ROOT / "scripts/kd_ablation_source.py").read_text(encoding="utf-8")
    put(tagged(book, "runtime"), runtime)
    for c in book["cells"]:
        if c["cell_type"] == "code":
            put(c, "".join(c["source"]).replace("from datetime import UTC, datetime", "from datetime import datetime, timezone\nUTC = timezone.utc")
                .replace("from datetime import UTC as _RECOVERY_UTC", "from datetime import timezone as _RecoveryTimezone\n_RECOVERY_UTC = _RecoveryTimezone.utc"))
    digest = hashlib.sha256("\n".join("".join(c["source"]) for c in book["cells"] if c["cell_type"] == "code"
                            and not set(c["metadata"]["tags"]) & {"run", "display", "group-display"}).encode()).hexdigest()
    put(tagged(book, "runtime"), "".join(tagged(book, "runtime")["source"]) + f"\nIMPLEMENTATION_SHA256 = {digest!r}\n")
    put(tagged(book, "run"), '''print(f"{RECOVERY_METHOD} | seed=42 | {ABLATION_PARAMETER}={ABLATION_VALUE} | {MODE} | {RUN_DIR}", flush=True)
RESULT = ablation_execute(RUN_DIR)
if RESULT.get("status") != "PASS":
    raise RuntimeError(f"{RESULT.get('status')}: {RESULT.get('reason')} | {RUN_DIR / 'kd_manifest.json'}")
''')
    put(tagged(book, "group-display"), '''# Dapat dijalankan ulang setelah kandidat lain selesai, tanpa mengulang training.
if LOAD_GROUP_RESULTS:
    ablation_report(sync=WANDB_TRACKING)
else:
    print("Aktifkan LOAD_GROUP_RESULTS untuk tabel tiga kandidat dan kurva sensitivitas.")
''')
    markdown = [c for c in book["cells"] if c["cell_type"] == "markdown"]
    texts = [
        f"# Ablation YOLO26s · {method.upper()} · FILKOM\n\nPertanyaan: bagaimana bobot KD memengaruhi validation mAP50–95 pada kondisi awal tetap? Satu sesi menjalankan satu kandidat. Hasil belum diukur sampai notebook dieksekusi di server.",
        "## 01 — Pilih satu kandidat\n\nUbah hanya KD_SCALE atau MGD_ALPHA pada grid yang tersedia. Seed terkunci 42; student raw 1.20× :v1 dan teacher M v50 beserta hash sama untuk semua kandidat. Restart Kernel sebelum mengganti kandidat. Default 100 epoch, batch 16, 640 px, AdamW 0.001 mengikuti MGD FILKOM.",
        "## 02 — Bootstrap server dan input\n\nPython >=3.11; Torch/torchvision CUDA kernel dipertahankan. W&B key: environment lalu REPO_ROOT/.env. Workspace default ~/skripsi-kd; dataset datasets/roboflow-poltrada-v3. Override output khusus SKRIPSI_ABLATION_OUTPUT_DIR. Fingerprint lokal dicatat dan dikunci antar kandidat, tanpa wajib sama dengan snapshot pruning. Adapter label campuran tidak mengubah sumber. Test split tidak dipakai.",
        "## 03 — Smoke, training dan resume\n\nModel sementara memeriksa loss, gradien dan optimizer step sebelum training; raw asli dimuat independen. CrossKD hanya menskalakan kedua loss secara proporsional; MGD hanya mengubah alpha. Best mengikuti EMA/fitness native MGD. Snapshot lengkap setiap epoch dan upload tiap lima epoch. Gunakan RESUME_RUN_DIR atau RESUME_SNAPSHOT_REF versi tetap untuk kandidat yang sama. Default historis diaudit sebelum dipakai kembali; seed 0 ditolak. Kegagalan bukan bukti performa buruk.",
        "## 04 — Evaluasi final dan EDA\n\nBest native FP32 dievaluasi ulang pada full validation; struktur/parameter/GFLOPs harus tetap sama dengan raw. Baseline, raw dan recovered memakai evaluator serta label policy sama. Kurva Ultralytics training dipisahkan dari final_validation/*. Loss rekap berasal dari epoch best, mentah dan berbobot. Durasi adalah jumlah epoch_seconds, termasuk validation training; memory adalah peak proses training, bukan kebutuhan inference. Smoke memakai namespace terpisah dan tidak ikut seleksi penelitian.",
        "## 05 — Rekap tiga kandidat\n\nAktifkan LOAD_GROUP_RESULTS untuk memuat registry lokal dan hasil W&B terverifikasi yang sebanding, menggambar sensitivitas dan kurva training. Nilai kosong tetap NOT_RUN. Pemilihan menunggu tiga hasil lengkap; seri dilaporkan. Pemenang di ujung rentang tidak membuktikan optimum global. Satu seed tidak mengukur signifikansi atau variasi antar seed.",
    ]
    for c, s in zip(markdown, texts, strict=True):
        put(c, s)
    book["metadata"]["research_workflow"].update(protocol="filkom-kd-ablation-v1", method=method, seed=42)
    return deepcopy(book)


def main():
    for method, name in NAMES.items():
        path = ROOT / "notebooks/filkom/recovery" / name
        path.write_text(json.dumps(build_notebook(method), ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
        print(path)


if __name__ == "__main__":
    main()
