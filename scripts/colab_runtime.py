"""Portable helpers embedded into each generated Colab notebook (no repo imports)."""

import hashlib
import os
import shutil
import stat
import zipfile
from pathlib import Path, PurePosixPath

import yaml


def colab_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def colab_api_key(name="WANDB_API_KEY"):
    """Prefer an existing environment value; otherwise request a Colab Secret."""
    value = os.environ.get(name, "").strip()
    if not value:
        try:
            from google.colab import userdata

            value = userdata.get(name)
        except Exception:  # noqa: BLE001 -- Normalize UI failures without exposing secret-bearing payloads.
            raise RuntimeError(
                f"Add {name} in Colab Secrets (key icon), enable notebook access, "
                "then rerun this cell. Never paste the key into notebook code."
            ) from None
    if not isinstance(value, str) or not value.strip():
        raise RuntimeError(f"Colab Secret {name} is empty; update it and rerun this cell.")
    os.environ[name] = value.strip()
    return value.strip()


def colab_require_gpu(torch_module):
    if not torch_module.cuda.is_available():
        raise RuntimeError("Select Runtime > Change runtime type > GPU in Colab, then Run All.")


def colab_extract_zip(archive, destination):
    """Reject unsafe/duplicate members before extracting; ZIP CRCs validate reads."""
    destination = Path(destination).resolve()
    with zipfile.ZipFile(archive) as source:
        members = source.infolist()
        targets = []
        seen = set()
        for member in members:
            name = member.filename.replace("\\", "/")
            relative = PurePosixPath(name)
            if member.is_dir() and str(relative) == ".":
                continue  # Common `zip -r archive.zip .` root-directory entry.
            if (relative.is_absolute() or not relative.parts or ".." in relative.parts
                    or any(":" in part for part in relative.parts)
                    or stat.S_ISLNK(member.external_attr >> 16)):
                raise ValueError(f"Unsafe ZIP member: {member.filename!r}")
            target = destination.joinpath(*relative.parts).resolve()
            if not target.is_relative_to(destination) or target in seen:
                raise ValueError(f"Unsafe or duplicate ZIP member: {member.filename!r}")
            seen.add(target)
            targets.append((member, target))
        needed = sum(member.file_size for member in members)
        if needed > shutil.disk_usage(destination.parent).free:
            raise OSError("Insufficient Colab local disk for the extracted dataset.")
        destination.mkdir(parents=True, exist_ok=False)
        for member, target in targets:
            if member.is_dir():
                target.mkdir(parents=True, exist_ok=True)
            else:
                target.parent.mkdir(parents=True, exist_ok=True)
                with source.open(member) as incoming, target.open("wb") as outgoing:
                    shutil.copyfileobj(incoming, outgoing)
        return {"files": sum(not member.is_dir() for member in members), "bytes": needed}


def colab_resolve_reference(value, root, extracted, declared_root, source_declaration):
    """Relocate the declared root, keeping directory names and split membership."""
    value = str(value).replace("\\", "/")
    relative = PurePosixPath(value)
    if relative.is_absolute():
        prefixes = [str(declared_root or ""), str(PurePosixPath(source_declaration).parent)]
        for prefix in prefixes:
            prefix_path = PurePosixPath(prefix)
            if prefix and prefix_path.is_absolute() and relative.is_relative_to(prefix_path):
                relative = relative.relative_to(prefix_path)
                break
        else:
            raise ValueError(f"Cannot relocate absolute dataset path: {value}")
    candidate = root.joinpath(*relative.parts).resolve()
    # Same leading ../ compatibility used by the pinned Ultralytics loader.
    if not candidate.exists() and value.startswith("../"):
        candidate = (root / value[3:]).resolve()
    if not candidate.is_relative_to(extracted) or not candidate.exists():
        raise FileNotFoundError(f"Dataset path is missing or outside the extracted ZIP: {value}")
    return candidate


def prepare_colab_dataset(zip_path, local_directory, yaml_relative, source_declaration):
    """Copy a Drive ZIP locally and write a separate, fully resolved runtime YAML."""
    zip_path, local = Path(zip_path), Path(local_directory).resolve()
    if not zip_path.is_file():
        raise FileNotFoundError(f"Dataset ZIP not found: {zip_path}. Edit DATASET_ZIP in Colab configuration.")
    local.mkdir(parents=True, exist_ok=True)
    runtime_yaml = local / "runtime/data.yaml"
    # A completed setup is immutable within this run; do not overwrite/reuse partial extraction.
    if (local / "source").exists() or runtime_yaml.exists():
        raise RuntimeError("Dataset setup already started for this run. Rerun the Colab configuration cell to create a new run directory.")
    archive = local / "dataset.zip"
    if zip_path.stat().st_size > shutil.disk_usage(local).free:
        raise OSError("Insufficient Colab local disk to copy DATASET_ZIP.")
    shutil.copyfile(zip_path, archive)
    archive_sha = colab_sha256(archive)
    if colab_sha256(zip_path) != archive_sha:
        raise ValueError("Dataset ZIP changed or was corrupted during the copy from Drive.")
    extracted = local / "source"
    inventory = colab_extract_zip(archive, extracted)
    if yaml_relative:
        original_yaml = (extracted / yaml_relative).resolve()
        if not original_yaml.is_relative_to(extracted) or not original_yaml.is_file():
            raise FileNotFoundError("DATASET_YAML_RELATIVE must name a YAML file inside the ZIP.")
    else:
        candidates = sorted(p for p in extracted.rglob("*") if p.suffix.lower() in {".yaml", ".yml"})
        if len(candidates) != 1:
            choices = [str(p.relative_to(extracted)) for p in candidates]
            raise ValueError(f"Expected one dataset YAML, found {choices}. Set DATASET_YAML_RELATIVE explicitly.")
        original_yaml = candidates[0]
    class UniqueDatasetLoader(yaml.SafeLoader):
        pass

    def unique_mapping(loader, node):
        mapping = {}
        for key_node, value_node in node.value:
            key = loader.construct_object(key_node)
            if key in mapping:
                raise ValueError(f"Duplicate dataset YAML key: {key!r}; fix the source ZIP explicitly.")
            mapping[key] = loader.construct_object(value_node)
        return mapping

    UniqueDatasetLoader.add_constructor(yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG, unique_mapping)
    raw = yaml.load(original_yaml.read_text(encoding="utf-8-sig"), Loader=UniqueDatasetLoader)
    if not isinstance(raw, dict) or not raw.get("train") or not raw.get("val"):
        raise ValueError("Dataset YAML must declare train, val and class names.")
    names = raw.get("names")
    if isinstance(names, dict):
        keys = sorted(names, key=lambda key: int(key))
        if [int(key) for key in keys] != list(range(len(keys))):
            raise ValueError("Class IDs must be consecutive starting at zero.")
        ordered_names = [str(names[key]) for key in keys]
    elif isinstance(names, list):
        ordered_names = [str(name) for name in names]
    else:
        raise TypeError("Dataset YAML names must be a list or a class-ID mapping.")
    if not ordered_names or int(raw.get("nc", len(ordered_names))) != len(ordered_names):
        raise ValueError("Dataset nc and class names disagree.")
    declared_root = raw.get("path")
    root = original_yaml.parent
    if declared_root and not PurePosixPath(str(declared_root).replace("\\", "/")).is_absolute():
        root = (root / declared_root).resolve()
    if not root.is_relative_to(extracted) or not root.is_dir():
        raise ValueError("Dataset YAML root must resolve inside the extracted ZIP.")
    relocated = dict(raw)
    relocated["path"] = str(root)
    runtime_yaml.parent.mkdir(parents=True, exist_ok=True)
    split_counts = {}
    # Match Ultralytics 8.4.155 IMG_FORMATS, including its less common formats.
    image_suffixes = {".avif", ".bmp", ".dng", ".heic", ".heif", ".jp2", ".jpeg", ".jpg", ".mpo", ".png", ".tif", ".tiff", ".webp"}
    for split in ("train", "val", "test"):
        if not raw.get(split):
            continue
        values = raw[split] if isinstance(raw[split], list) else [raw[split]]
        outputs, count = [], 0
        for index, value in enumerate(values):
            if not isinstance(value, str):
                raise TypeError(f"Invalid {split} path: {value!r}")
            path = colab_resolve_reference(value, root, extracted, declared_root, source_declaration)
            if path.is_dir():
                count += sum(p.is_file() and p.suffix.lower() in image_suffixes for p in path.rglob("*"))
                outputs.append(str(path))
            elif path.suffix.lower() == ".txt":
                entries = []
                for line in path.read_text(encoding="utf-8-sig").splitlines():
                    if not line.strip():
                        continue
                    image_root = path.parent if line.strip().startswith("./") else root
                    image = colab_resolve_reference(line.strip(), image_root, extracted, declared_root, source_declaration)
                    if not image.is_file() or image.suffix.lower() not in image_suffixes:
                        raise ValueError(f"Split list references a non-image: {line}")
                    entries.append(str(image))
                derived = runtime_yaml.parent / f"{split}-{index}.txt"
                derived.write_text("\n".join(entries) + "\n", encoding="utf-8")
                outputs.append(str(derived))
                count += len(entries)
            else:
                raise ValueError(f"Split must reference an image directory or TXT list: {value}")
        if count == 0:
            raise ValueError(f"No images found in {split}; check the ZIP and YAML paths.")
        relocated[split] = outputs if isinstance(raw[split], list) else outputs[0]
        split_counts[split] = count
    runtime_yaml.write_text(yaml.safe_dump(relocated, sort_keys=False, allow_unicode=True), encoding="utf-8")
    provenance = {
        "runtime": "google-colab", "drive_zip": str(zip_path), "zip_sha256": archive_sha,
        "extracted_inventory": inventory, "original_yaml": str(original_yaml),
        "original_yaml_sha256": colab_sha256(original_yaml), "runtime_yaml": str(runtime_yaml),
        "runtime_yaml_sha256": colab_sha256(runtime_yaml), "source_dataset_declaration": source_declaration,
        "class_names": ordered_names, "split_image_counts": split_counts,
        "relocation_only": True,
        "identity_note": "ZIP provenance is recorded; equality with the original Kaggle image/label bytes is not independently established.",
    }
    return str(runtime_yaml), provenance
