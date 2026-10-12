"""Roboflow v2 working copy, YOLO label normalization, and provenance audit."""

from __future__ import annotations

import hashlib
import json
import math
import os
import shutil
from dataclasses import dataclass
from pathlib import Path

import pandas as pd
import yaml
from PIL import Image

from .config import CLASS_NAMES, EXPECTED_V2, EXPECTED_V2_FINGERPRINT, ExperimentConfig

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
NON_BLOCKING_STATUS = {"ok", "empty_label"}
BOUNDARY_EPS = 1e-5


@dataclass
class PreparedDataset:
    root: Path
    runtime_yaml: Path
    split_images: dict[str, Path]
    audit: pd.DataFrame
    issues: pd.DataFrame
    fingerprint: str


def get_secret(name: str) -> str | None:
    """Environment locally, Kaggle Secrets in Kaggle; never log the value."""
    if not Path("/kaggle/working").is_dir():
        from dotenv import load_dotenv

        load_dotenv(Path(__file__).resolve().parents[2] / ".env", override=False)
    value = os.getenv(name)
    if value:
        return value
    if Path("/kaggle/working").is_dir():
        try:
            from kaggle_secrets import UserSecretsClient

            return UserSecretsClient().get_secret(name)
        except Exception:
            pass
    return None


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _source_yaml(root: Path) -> Path:
    candidates = sorted(root.rglob("data.yaml"))
    if not candidates:
        raise FileNotFoundError(f"data.yaml tidak ditemukan di {root}")
    if len(candidates) > 1:
        raise RuntimeError(f"Lebih dari satu data.yaml di {root}: {candidates}")
    return candidates[0]


def _working_copy(config: ExperimentConfig) -> Path:
    """Never normalize labels in the user's source dataset."""
    destination = config.output_root / "dataset"
    marker = destination / ".modular_source.json"
    source = config.dataset_dir
    identity = {
        "workspace": config.workspace,
        "project": config.project,
        "version": config.dataset_version,
        "local_source": str(source) if source else None,
    }
    if destination.exists():
        if (
            not marker.exists()
            or json.loads(marker.read_text(encoding="utf-8")) != identity
        ):
            raise RuntimeError(
                f"Dataset kerja sudah ada dengan sumber berbeda: {destination}"
            )
        _source_yaml(destination)
        return destination

    destination.parent.mkdir(parents=True, exist_ok=True)
    if source is not None:
        if not source.is_dir():
            raise FileNotFoundError(source)
        if destination == source or destination.is_relative_to(source):
            raise ValueError("output_root tidak boleh berada di dalam dataset sumber")
        shutil.copytree(source, destination)
    else:
        key = get_secret("ROBOFLOW_API_KEY")
        if not key:
            raise RuntimeError(
                "Dataset v2 belum tersedia; set ROBOFLOW_API_KEY atau dataset_dir"
            )
        from roboflow import Roboflow

        downloaded = (
            Roboflow(api_key=key)
            .workspace(config.workspace)
            .project(config.project)
            .version(config.dataset_version)
            .download("yolov8", location=str(destination), overwrite=False)
        )
        if not destination.exists():
            raise FileNotFoundError(
                f"Roboflow mengembalikan {downloaded.location}; dataset kerja tidak ada"
            )
    _source_yaml(destination)
    marker.write_text(json.dumps(identity, indent=2), encoding="utf-8")
    return destination


def _resolve_split(
    source_yaml: Path, raw: dict, key: str, aliases: tuple[str, ...]
) -> Path:
    candidates: list[Path] = []
    value = raw.get(key)
    if value:
        path = Path(str(value))
        candidates.append(
            path if path.is_absolute() else (source_yaml.parent / path).resolve()
        )
    candidates.extend(
        (source_yaml.parent / alias / "images").resolve() for alias in aliases
    )
    for candidate in candidates:
        if candidate.is_dir():
            return candidate
    raise FileNotFoundError(f"Split {key} tidak ditemukan; kandidat: {candidates}")


def _polygon_to_bbox(parts: list[str]) -> str:
    coords = [float(value) for value in parts[1:]]
    if len(coords) < 6 or len(coords) % 2:
        raise ValueError("Polygon memerlukan minimal tiga pasangan koordinat")
    if not all(math.isfinite(value) and 0 <= value <= 1 for value in coords):
        raise ValueError("Koordinat polygon di luar [0,1]")
    xs, ys = coords[::2], coords[1::2]
    x_min, x_max, y_min, y_max = min(xs), max(xs), min(ys), max(ys)
    if x_max <= x_min or y_max <= y_min:
        raise ValueError("Polygon menghasilkan bounding box kosong")
    return f"{parts[0]} {(x_min + x_max) / 2:.10f} {(y_min + y_max) / 2:.10f} {x_max - x_min:.10f} {y_max - y_min:.10f}"


def normalize_labels(split_images: dict[str, Path], reports_dir: Path) -> pd.DataFrame:
    """Preserve five-column labels; convert polygon lines exactly as the source notebooks."""
    rows: list[dict] = []
    for split, image_dir in split_images.items():
        label_dir = image_dir.parent / "labels"
        if not label_dir.exists():
            continue
        for label in sorted(label_dir.rglob("*.txt")):
            source = label.read_text(encoding="utf-8")
            converted: list[str] = []
            changed = False
            for number, raw in enumerate(source.splitlines(), 1):
                line = raw.strip()
                if not line:
                    continue
                parts = line.split()
                if len(parts) == 5:
                    converted.append(line)
                elif len(parts) >= 7 and len(parts) % 2 == 1:
                    result = _polygon_to_bbox(parts)
                    converted.append(result)
                    changed = True
                    rows.append(
                        {
                            "split": split,
                            "label": str(label),
                            "line": number,
                            "original_columns": len(parts),
                            "action": "polygon_to_bbox",
                            "original": line,
                            "converted": result,
                        }
                    )
                else:
                    raise ValueError(f"Format label tidak dikenali: {label}:{number}")
            if changed:
                backup = (
                    reports_dir
                    / "label_backup_before_polygon_conversion"
                    / split
                    / label.relative_to(label_dir)
                )
                backup.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(label, backup)
                label.write_text("\n".join(converted) + "\n", encoding="utf-8")
    columns = (
        "split",
        "label",
        "line",
        "original_columns",
        "action",
        "original",
        "converted",
    )
    report = pd.DataFrame(rows, columns=columns)
    reports_dir.mkdir(parents=True, exist_ok=True)
    report_path = reports_dir / "polygon_conversion.csv"
    if report_path.exists():
        previous = pd.read_csv(report_path)
        report = pd.concat((previous, report), ignore_index=True).drop_duplicates(
            ["split", "label", "line"], keep="last"
        )
    report.to_csv(report_path, index=False)
    return report


def validate_label_file(
    path: Path, num_classes: int = len(CLASS_NAMES)
) -> tuple[str, int, dict | None]:
    if not path.exists():
        return "missing_label", 0, {"issue": "missing_label"}
    lines = [
        line.strip()
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if not lines:
        return "empty_label", 0, None
    for number, line in enumerate(lines, 1):
        parts = line.split()
        if len(parts) != 5:
            return (
                "invalid_label",
                number - 1,
                {"issue": "column_count", "line": number, "raw": line},
            )
        try:
            cls, xc, yc, width, height = map(float, parts)
        except ValueError:
            return (
                "invalid_label",
                number - 1,
                {"issue": "non_numeric", "line": number, "raw": line},
            )
        if not all(map(math.isfinite, (cls, xc, yc, width, height))):
            return (
                "invalid_label",
                number - 1,
                {"issue": "non_finite", "line": number, "raw": line},
            )
        if not cls.is_integer() or not 0 <= cls < num_classes:
            return (
                "invalid_label",
                number - 1,
                {"issue": "invalid_class", "line": number, "raw": line},
            )
        if not (0 <= xc <= 1 and 0 <= yc <= 1 and 0 < width <= 1 and 0 < height <= 1):
            return (
                "invalid_label",
                number - 1,
                {"issue": "invalid_coordinate", "line": number, "raw": line},
            )
        overflow = max(
            0.0,
            -(xc - width / 2),
            -(yc - height / 2),
            xc + width / 2 - 1,
            yc + height / 2 - 1,
        )
        if overflow > BOUNDARY_EPS:
            return (
                "invalid_label",
                number - 1,
                {
                    "issue": "out_of_bounds",
                    "line": number,
                    "overflow": overflow,
                    "raw": line,
                },
            )
    return "ok", len(lines), None


def audit_dataset(
    split_images: dict[str, Path], reports_dir: Path, *, strict_v2: bool = True
) -> tuple[pd.DataFrame, pd.DataFrame, str]:
    """Audit every image/label and stop on blocking issues or cross-split leaks."""
    rows: list[dict] = []
    issues: list[dict] = []
    image_hashes: dict[str, set[str]] = {}
    for split, image_dir in split_images.items():
        label_dir = image_dir.parent / "labels"
        images = sorted(
            path
            for path in image_dir.rglob("*")
            if path.is_file() and path.suffix.lower() in IMAGE_EXTS
        )
        for image in images:
            label = label_dir / image.relative_to(image_dir).with_suffix(".txt")
            row = {
                "split": split,
                "image": str(image),
                "label": str(label),
                "objects": 0,
                "status": "ok",
                "width": None,
                "height": None,
            }
            try:
                with Image.open(image) as opened:
                    opened.verify()
                with Image.open(image) as opened:
                    row["width"], row["height"] = opened.size
            except Exception as exc:
                row["status"] = f"corrupt_image:{type(exc).__name__}"
            if row["status"] == "ok":
                status, count, issue = validate_label_file(label)
                row["status"], row["objects"] = status, count
                if issue:
                    issues.append(
                        {
                            "split": split,
                            "image": str(image),
                            "label": str(label),
                            **issue,
                        }
                    )
            row["sha256"] = sha256_file(image)
            row["label_sha256"] = sha256_file(label) if label.exists() else None
            image_hashes.setdefault(row["sha256"], set()).add(split)
            rows.append(row)
    audit = pd.DataFrame(rows)
    issue_frame = pd.DataFrame(issues)
    reports_dir.mkdir(parents=True, exist_ok=True)
    audit.to_csv(reports_dir / "dataset_audit.csv", index=False)
    if not issue_frame.empty:
        issue_frame.to_csv(reports_dir / "dataset_audit_issues.csv", index=False)
    fingerprint_payload = "\n".join(
        sorted(
            f"{row['split']}|{Path(row['image']).name}|{row['sha256']}|{row['label_sha256']}"
            for row in rows
        )
    )
    fingerprint = hashlib.sha256(fingerprint_payload.encode("utf-8")).hexdigest()
    blocking = (
        audit[~audit["status"].isin(NON_BLOCKING_STATUS)] if not audit.empty else audit
    )
    leaks = [digest for digest, splits in image_hashes.items() if len(splits) > 1]
    if len(blocking) or leaks:
        raise RuntimeError(
            f"Audit gagal: {len(blocking)} baris blocking, {len(leaks)} duplikasi lintas split; lihat {reports_dir}"
        )
    if strict_v2:
        for split, expected in EXPECTED_V2.items():
            part = audit[audit["split"] == split]
            actual = {
                "images": len(part),
                "annotations": int(part["objects"].sum()),
                "empty_labels": int((part["status"] == "empty_label").sum()),
            }
            if actual != expected:
                raise RuntimeError(f"Provenance v2 {split}: {actual} != {expected}")
        if fingerprint != EXPECTED_V2_FINGERPRINT:
            raise RuntimeError(
                f"Fingerprint dataset v2 berbeda: {fingerprint} != {EXPECTED_V2_FINGERPRINT}"
            )
    return audit, issue_frame, fingerprint


def prepare_dataset(
    config: ExperimentConfig, *, strict_v2: bool = True
) -> PreparedDataset:
    root = _working_copy(config)
    source_yaml = _source_yaml(root)
    raw = yaml.safe_load(source_yaml.read_text(encoding="utf-8"))
    names = raw.get("names", [])
    if isinstance(names, dict):
        names = [names[key] for key in sorted(names, key=lambda key: int(key))]
    if tuple(names) != CLASS_NAMES or int(raw.get("nc", len(names))) != len(
        CLASS_NAMES
    ):
        raise RuntimeError(f"Urutan kelas tidak sesuai: {names}")
    roboflow_meta = raw.get("roboflow") or {}
    if (
        roboflow_meta
        and int(roboflow_meta.get("version", config.dataset_version))
        != config.dataset_version
    ):
        raise RuntimeError("Versi Roboflow pada data.yaml tidak sesuai")
    splits = {
        "train": _resolve_split(source_yaml, raw, "train", ("train",)),
        "val": _resolve_split(source_yaml, raw, "val", ("valid", "val")),
        "test": _resolve_split(source_yaml, raw, "test", ("test",)),
    }
    for path in splits.values():
        if not path.is_relative_to(root):
            raise RuntimeError(f"Split berada di luar salinan kerja: {path}")
    normalize_labels(splits, config.reports_dir)
    audit, issues, fingerprint = audit_dataset(
        splits, config.reports_dir, strict_v2=strict_v2
    )
    runtime_yaml = config.output_root / "data_runtime.yaml"
    runtime_yaml.write_text(
        yaml.safe_dump(
            {
                "path": str(source_yaml.parent),
                "train": str(splits["train"]),
                "val": str(splits["val"]),
                "test": str(splits["test"]),
                "nc": len(CLASS_NAMES),
                "names": list(CLASS_NAMES),
            },
            sort_keys=False,
            allow_unicode=True,
        ),
        encoding="utf-8",
    )
    return PreparedDataset(root, runtime_yaml, splits, audit, issues, fingerprint)
