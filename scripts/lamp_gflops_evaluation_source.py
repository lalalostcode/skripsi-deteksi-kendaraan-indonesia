"""Dataset and student-selection declarations embedded in notebook 05."""
# ruff: noqa: F821 -- Notebook runtime supplies the V2 evaluator and helpers.

import hashlib
import json
import logging
import math
from contextlib import contextmanager
from pathlib import Path
from unittest.mock import patch

POLTRADA_V3_ID = "ilhamrafiqin/poltrada-v3-yolo26-clean"
POLTRADA_V3_MOUNT = Path("/kaggle/input/datasets") / POLTRADA_V3_ID


def _gflops_split_images(root, entries):
    """Resolve directories and image lists using dataset-relative identities."""
    from ultralytics.data.utils import IMG_FORMATS

    if isinstance(entries, str):
        entries = [entries]
    if not isinstance(entries, list) or not entries:
        raise ValueError("Split Poltrada harus berupa direktori atau daftar gambar lokal.")
    images = []
    for entry in entries:
        if not isinstance(entry, str) or not entry.strip():
            raise ValueError("Path split kosong atau tidak valid.")
        path = (root / entry).resolve()
        if (not path.exists() or not path.is_relative_to(root)) and entry.startswith("../"):
            # Roboflow exports commonly use ../valid/images from data.yaml.
            path = (root / entry.removeprefix("../")).resolve()
        if not path.is_relative_to(root):
            raise ValueError(f"Path split berada di luar DATASET_ROOT: {path}")
        if path.is_dir():
            files = (item for item in path.rglob("*") if item.is_file())
        elif path.is_file() and path.suffix.lower() == ".txt":
            files = (path.parent / line.strip() for line in path.read_text(encoding="utf-8-sig").splitlines()
                     if line.strip())
        else:
            raise FileNotFoundError(f"Direktori atau daftar gambar split tidak tersedia: {path}")
        for image in files:
            if image.suffix.lower().removeprefix(".") not in IMG_FORMATS:
                continue
            image = image.resolve()
            if not image.is_file() or not image.is_relative_to(root):
                raise ValueError(f"Gambar tidak tersedia atau berada di luar DATASET_ROOT: {image}")
            images.append(image)
    if not images:
        raise ValueError("Split Poltrada tidak berisi gambar yang dapat dievaluasi.")
    if len(set(images)) != len(images):
        raise ValueError("Split Poltrada memuat path gambar duplikat.")
    return sorted(images, key=lambda image: image.relative_to(root).as_posix())


def _gflops_fingerprint(root, images, split, names):
    """Hash every selected image and label; missing and empty labels stay distinct."""
    from ultralytics.data.utils import img2label_paths

    records = []
    for image, label in zip(images, img2label_paths([str(image) for image in images]), strict=True):
        label = Path(label).resolve()
        if not label.is_relative_to(root):
            raise ValueError(f"Label berada di luar DATASET_ROOT: {label}")
        records.append({"image": image.relative_to(root).as_posix(), "image_sha256": sha256_file(image),
                        "label": label.relative_to(root).as_posix(),
                        "label_sha256": sha256_file(label) if label.is_file() else None})
    # Encoding and sorted path order make the digest independent of the mount point.
    payload = json.dumps({"split": split, "class_names": names, "records": records},
                         ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(payload).hexdigest(), records


def gflops_prepare_dataset(mode, data_yaml, dataset_root, suite_dir, *, seed,
                           expected_fingerprint=None, dataset_id=None):
    """Freeze the input set before evaluating any model in the suite."""
    import yaml

    if mode not in {"smoke", "full_val", "test"}:
        raise ValueError("Mode evaluasi harus smoke, full_val, atau test.")
    if expected_fingerprint is not None:
        if (not isinstance(expected_fingerprint, str) or len(expected_fingerprint) != 64
                or any(char not in "0123456789abcdefABCDEF" for char in expected_fingerprint)):
            raise ValueError("Fingerprint full val harus berupa SHA-256 heksadesimal.")
        expected_fingerprint = expected_fingerprint.lower()
    source = Path(data_yaml).expanduser().resolve()
    if not source.is_file():
        return None
    root = Path(dataset_root).expanduser().resolve() if dataset_root is not None else source.parent
    if not root.is_dir():
        raise FileNotFoundError(f"DATASET_ROOT tidak tersedia: {root}")
    raw = yaml.safe_load(source.read_text(encoding="utf-8-sig"))
    if not isinstance(raw, dict) or not isinstance(raw.get("names"), (dict, list)) or not raw["names"]:
        raise ValueError("DATA_YAML harus memuat names Poltrada yang berurutan.")
    class_names = ordered_names(raw["names"])
    if raw.get("nc", len(class_names)) != len(class_names):
        raise ValueError("nc di DATA_YAML berbeda dari jumlah names.")

    split = "test" if mode == "test" else "val"
    val_images = _gflops_split_images(root, raw.get("val"))
    full_val_fingerprint = None
    if mode != "smoke" or expected_fingerprint is not None:
        full_val_fingerprint, val_records = _gflops_fingerprint(root, val_images, "val", class_names)
        if expected_fingerprint is not None and full_val_fingerprint != expected_fingerprint:
            raise ValueError("Fingerprint full val tidak cocok dengan konfigurasi yang dikunci.")

    official_mount = root == POLTRADA_V3_MOUNT.resolve() and source.is_relative_to(root)
    verified_copy = dataset_id == POLTRADA_V3_ID and expected_fingerprint is not None
    provenance_status = "VERIFIED" if official_mount or verified_copy else "UNVERIFIED"
    provenance_basis = ("KAGGLE_MOUNT" if official_mount else
                        "EXPECTED_FULL_VAL_FINGERPRINT" if verified_copy else "USER_PATH_ONLY")
    if mode == "test":
        if not raw.get("test") or not raw.get("train"):
            return None
        images = _gflops_split_images(root, raw["test"])
        fingerprint, records = _gflops_fingerprint(root, images, split, class_names)
        test_hashes = {row["image_sha256"] for row in records}
        if any(row["image_sha256"] in test_hashes for row in val_records):
            raise ValueError("Split test berbagi konten gambar dengan full val.")
        train_images = _gflops_split_images(root, raw["train"])
        if any(sha256_file(image) in test_hashes for image in train_images):
            raise ValueError("Split test berbagi konten gambar dengan train.")
    elif mode == "smoke":
        ranked = sorted(val_images, key=lambda image: (
            hashlib.sha256(f"{seed}:{image.relative_to(root).as_posix()}".encode()).hexdigest(),
            image.relative_to(root).as_posix()))
        images = sorted(ranked[:128], key=lambda image: image.relative_to(root).as_posix())
        fingerprint, records = _gflops_fingerprint(root, images, split, class_names)
    else:
        images = val_images
        fingerprint, records = full_val_fingerprint, val_records
    if any(image.with_suffix(".npy").is_file() for image in images):
        raise ValueError("Cache gambar .npy berada di samping citra evaluasi. Gunakan salinan dataset tanpa cache.")

    folder = Path(suite_dir).resolve()
    folder.mkdir(parents=True, exist_ok=True)
    paths = folder / f"evaluation_{mode}_images.txt"
    paths.write_text("".join(f"{image.as_posix()}\n" for image in images), encoding="utf-8")
    # This descriptor permits no download or implicit split fallback. An unused train
    # key is required by Ultralytics even though this notebook only calls val().
    descriptor = {"path": str(root), "train": str(paths), "val": str(paths),
                  "names": class_names, "nc": len(class_names)}
    if split == "test":
        descriptor["test"] = str(paths)
    evaluation_yaml = folder / f"evaluation_{mode}_data.yaml"
    evaluation_yaml.write_text(yaml.safe_dump(descriptor, sort_keys=False, allow_unicode=True), encoding="utf-8")
    manifest = {"mode": mode, "split": split, "seed": seed, "image_count": len(images),
                "full_val_image_count": len(val_images), "fingerprint": fingerprint,
                "full_val_fingerprint": full_val_fingerprint, "provenance_status": provenance_status,
                "provenance_basis": provenance_basis,
                "dataset_id": POLTRADA_V3_ID if official_mount else dataset_id,
                "source_yaml": str(source), "source_yaml_sha256": sha256_file(source),
                "evaluation_yaml_sha256": sha256_file(evaluation_yaml), "class_names": class_names,
                "test_content_disjoint_from_train_val": True if mode == "test" else None}
    manifest_path = folder / f"evaluation_{mode}_manifest.json"
    write_json(manifest_path, {**manifest, "files": records})
    return {**manifest, "manifest_path": str(manifest_path), "yaml": str(evaluation_yaml),
            "info": descriptor, "val_paths": [str(paths)], "image_files": [str(image) for image in images]}


def gflops_validation_arguments(dataset, output_dir, label):
    """Preserve notebook 03 settings and explicitly select the prepared split."""
    arguments = validation_arguments(dataset["yaml"], output_dir, label)
    arguments.update({"split": dataset["split"], "fraction": 1.0})
    return arguments


@contextmanager
def gflops_disable_label_cache():
    """Force label parsing from the fingerprinted files without changing the input dataset."""
    from ultralytics.data import dataset as sdk_dataset

    def set_in_memory_version(_prefix, _path, records, version):
        # get_labels() expects the writer's in-memory version field.
        records["version"] = version

    with (patch.object(sdk_dataset, "load_dataset_cache_file", side_effect=FileNotFoundError),
          patch.object(sdk_dataset, "save_dataset_cache_file", side_effect=set_in_memory_version)):
        yield


@contextmanager
def gflops_require_complete_nms():
    """Fail validation if the SDK stops processing a batch because of its NMS time limit."""
    from ultralytics.utils import LOGGER

    timeouts = []

    class TimeoutHandler(logging.Handler):
        def emit(self, record):
            message = record.getMessage()
            if "NMS time limit" in message and "exceeded" in message:
                timeouts.append(message)

    handler = TimeoutHandler()
    LOGGER.addHandler(handler)
    try:
        yield
    finally:
        LOGGER.removeHandler(handler)
    if timeouts:
        raise RuntimeError(f"Validasi parsial karena NMS timeout: {timeouts[0]}")


def gflops_validation_metadata(validator, arguments, dataset):
    """Verify the actual SDK backend and loaded images before exporting mAP."""
    if validator is None or getattr(validator, "args", None) is None:
        raise ValueError("Validator SDK aktual tidak terekam.")
    fields = ("split", "imgsz", "batch", "workers", "device", "conf", "iou", "max_det",
              "quantize", "nms", "augment", "plots", "fraction", "rect", "data")
    effective = {field: getattr(validator.args, field, None) for field in fields}
    effective["device"] = str(validator.device)
    effective["data"] = str(effective["data"])
    effective["end2end"] = getattr(validator, "end2end", None)
    if (effective["split"] != dataset["split"] or effective["end2end"] is not False
            or effective["nms"] is not None or effective["quantize"] is not None):
        raise ValueError("Validator SDK tidak memakai split, FP32, atau mode one-to-many dengan NMS yang ditetapkan.")
    loaded = getattr(getattr(validator.dataloader, "dataset", None), "im_files", ())
    if sorted(str(Path(path).resolve()) for path in loaded) != sorted(dataset["image_files"]):
        raise ValueError("Daftar gambar validator berbeda dari manifest evaluasi.")
    effective["dataloader_batch"] = validator.dataloader.batch_size
    if effective["batch"] != arguments["batch"] or effective["dataloader_batch"] != min(arguments["batch"], len(loaded)):
        raise ValueError("Batch validator aktual berbeda dari protokol (atau batas ukuran subset).")
    return {"split": dataset["split"], "evaluation_mode": dataset["mode"],
            "dataset_fingerprint": dataset["fingerprint"],
            "dataset_provenance_status": dataset["provenance_status"],
            "validation_image_count": len(loaded), "validation_arguments_requested": arguments,
            "validation_arguments_effective": effective}


def _gflops_complete_student_row(row):
    if (row.get("evaluation_mode") != "full_val" or row.get("dataset_provenance_status") != "VERIFIED"
            or any(row.get(key) != "PASS" for key in ("structural_status", "reload_status", "validation_status"))
            or row.get("latency_status") not in {"PASS", "MEASURED"}):
        return False
    fields = ("target_speedup", "gflops_speedup", "map50_95", "baseline_map50_95", "gflops",
              "parameters", "latency_median_ms")
    if any(not isinstance(row.get(key), (float, int)) or isinstance(row[key], bool)
           or not math.isfinite(row[key]) for key in fields):
        return False
    return (row.get("model_variant") in {"n", "s"} and row["target_speedup"] in {1.02, 1.05, 1.10, 1.20, 1.70}
            and row.get("target_status", "REACHED") == "REACHED"
            and row.get("baseline_validation_parity_status", "PASS") == "PASS"
            and row["gflops_speedup"] >= row["target_speedup"]
            and 0 <= row["map50_95"] <= 1 and 0 <= row["baseline_map50_95"] <= 1
            and row["baseline_map50_95"] - row["map50_95"] <= 0.02 + 1e-12
            and all(row[key] > 0 for key in ("gflops", "parameters", "latency_median_ms")))


def gflops_select_student(rows, mode):
    """Select one comparable full-validation student from both variants."""
    if mode != "full_val":
        return {"status": "NOT_RUN", "reason": "Student dipilih hanya dari full_val.", "selected": None}
    eligible = [row for row in rows if _gflops_complete_student_row(row)]
    if not eligible:
        return {"status": "NO_QUALIFYING_STUDENT", "selected": None}
    fingerprints = {row.get("dataset_fingerprint") for row in eligible}
    if len(fingerprints) != 1 or not next(iter(fingerprints)):
        raise ValueError("Kandidat student berasal dari fingerprint full val berbeda atau tidak tercatat.")
    best = min(eligible, key=lambda row: (-row["target_speedup"], -row["map50_95"], row["latency_median_ms"],
                                          row["parameters"], row["gflops"], row["model_variant"],
                                          row.get("checkpoint_sha256") or ""))
    return {"status": "SELECTED", "selected": dict(best)}


def _gflops_verified_checkpoint(path, expected):
    checkpoint = Path(path).resolve()
    if not checkpoint.is_file() or sha256_file(checkpoint) != expected:
        raise ValueError(f"Checkpoint berbeda dari SHA-256 yang dikunci: {checkpoint}")
    return checkpoint


def gflops_select_candidate_files(paths):
    """Combine independent N/S full-val exports under the same measurement contract."""
    files = [Path(path).resolve() for path in paths]
    if len(files) != len(set(files)):
        raise ValueError("File kandidat yang sama tidak boleh digabung dua kali.")
    variants, fingerprints, contracts, rows = set(), set(), [], []
    hashes = {str(path): sha256_file(path) for path in files}
    for path in files:
        record = json.loads(path.read_text(encoding="utf-8"))
        if (record.get("format_version") not in {1, 2} or record.get("mode") != "full_val"
                or record.get("status") not in {"PASS", "PARTIAL"}
                or record.get("evaluation_status") not in {"PASS", "PARTIAL"}
                or record.get("dataset_provenance_status") != "VERIFIED"):
            raise ValueError("Gabungan student memerlukan ekspor full_val VERIFIED yang selesai PASS.")
        members = set(record.get("model_variants", []))
        if not members or not members <= {"n", "s"} or variants.intersection(members):
            raise ValueError("Setiap varian harus mempunyai tepat satu sumber full_val.")
        variants.update(members)
        fingerprints.add(record.get("dataset_fingerprint"))
        contract = record.get("comparison_contract")
        if not isinstance(contract, dict) or not contract:
            raise ValueError("Kontrak evaluasi dan perangkat latency tidak tercatat.")
        contracts.append(contract)
        for source in record.get("rows", []):
            if source.get("model_variant") not in members:
                raise ValueError("Varian baris kandidat berbeda dari deklarasi ekspor.")
            row = dict(source)
            for key in ("checkpoint", "baseline_checkpoint"):
                relative = row.get(key + "_relative_path")
                if relative:
                    relocated = (path.parent / relative).resolve()
                    if not relocated.is_relative_to(path.parent):
                        raise ValueError("Path checkpoint kandidat keluar dari folder ekspor.")
                    row[key + "_path"] = str(relocated)
            row["selection_source_sha256"] = hashes
            rows.append(row)
    if len(fingerprints) > 1 or (fingerprints and not next(iter(fingerprints))):
        raise ValueError("Fingerprint full validation N/S berbeda atau kosong.")
    if contracts and any(contract != contracts[0] for contract in contracts[1:]):
        raise ValueError("Protokol, implementasi, atau perangkat latency N/S berbeda.")
    if variants != {"n", "s"}:
        return {"status": "NOT_RUN", "selected": None,
                "reason": "Pemilihan akhir menunggu hasil full_val dari N dan S.",
                "available_variants": sorted(variants)}
    result = gflops_select_student(rows, "full_val")
    if result.get("selected"):
        selected = result["selected"]
        _gflops_verified_checkpoint(selected["checkpoint_path"], selected["checkpoint_sha256"])
        _gflops_verified_checkpoint(selected["baseline_checkpoint_path"], selected["baseline_checkpoint_sha256"])
    return {**result, "source_files": hashes, "model_variants": sorted(variants)}


def gflops_lock_student(selection, path):
    """Write an immutable-by-convention full-val lock, verified again before test."""
    if selection.get("status") != "SELECTED":
        return None
    student = selection["selected"]
    if not _gflops_complete_student_row(student) or not student.get("dataset_fingerprint"):
        raise ValueError("Lock student memerlukan hasil full val yang lolos seleksi.")
    _gflops_verified_checkpoint(student["checkpoint_path"], student["checkpoint_sha256"])
    _gflops_verified_checkpoint(student["baseline_checkpoint_path"], student["baseline_checkpoint_sha256"])
    lock = Path(path)
    record = {"format_version": 1, "status": "LOCKED", "selected": student}
    if lock.exists():
        if json.loads(lock.read_text(encoding="utf-8")) != record:
            raise FileExistsError(f"Lock student sudah berisi pilihan berbeda: {lock}")
    else:
        # Exclusive creation prevents another run replacing the frozen selection.
        with lock.open("x", encoding="utf-8") as stream:
            json.dump(record, stream, ensure_ascii=False, indent=2, allow_nan=False)
    return record


def gflops_run_locked_test(lock_path, dataset, output_dir, *, checkpoint_path=None, baseline_checkpoint_path=None):
    """Evaluate the fixed baseline/student pair only on an independent test split."""
    if dataset is None:
        return {"status": "NOT_RUN", "reason": "Split test/train independen belum tersedia."}
    if (dataset.get("mode") != "test" or dataset.get("split") != "test"
            or dataset.get("provenance_status") != "VERIFIED"
            or dataset.get("test_content_disjoint_from_train_val") is not True):
        raise ValueError("Test memerlukan split independen yang provenance-nya VERIFIED.")
    record = json.loads(Path(lock_path).read_text(encoding="utf-8"))
    student = record.get("selected", {})
    if (record.get("status") != "LOCKED" or record.get("format_version") != 1
            or not _gflops_complete_student_row(student)
            or student["dataset_fingerprint"] != dataset.get("full_val_fingerprint")):
        raise ValueError("Lock student tidak cocok dengan full val Poltrada yang telah diverifikasi.")
    baseline = _gflops_verified_checkpoint(baseline_checkpoint_path or student["baseline_checkpoint_path"],
                                           student["baseline_checkpoint_sha256"])
    pruned = _gflops_verified_checkpoint(checkpoint_path or student["checkpoint_path"], student["checkpoint_sha256"])
    before = evaluate_checkpoint(baseline, dataset, output_dir, "baseline_test")
    after = evaluate_checkpoint(pruned, dataset, output_dir, "student_test")
    return {"status": "PASS", "baseline": before, "student": after,
            "delta_map50_95_pp": 100 * (after["map50_95"] - before["map50_95"]),
            "test_fingerprint": dataset["fingerprint"], "checkpoint_sha256": student["checkpoint_sha256"]}
