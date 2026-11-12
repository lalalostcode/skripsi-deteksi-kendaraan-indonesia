"""Read mixed YOLO box/polygon labels for detection without editing source files."""
# ruff: noqa: F821 -- Embedded after the notebook's SDK and common helper declarations.
from ultralytics.data import dataset as _FilkomLabelDataset

FILKOM_MGD_LABEL_POLICY = "mixed-box-polygon-to-detection-v1"
_filkom_label_verifier = _FilkomLabelDataset.verify_image_label
_filkom_label_cache_context = gflops_disable_label_cache


def filkom_mgd_prepare_label_adapter(root, images, folder):
    from ultralytics.data.utils import img2label_paths
    from ultralytics.utils.ops import segments2boxes

    root, folder = Path(root).resolve(), Path(folder)
    mapping, records = {}, []
    for label_name in img2label_paths([str(image) for image in images]):
        label = Path(label_name).resolve()
        if not label.is_file() or str(label) in mapping:
            continue
        rows = [line.split() for line in label.read_text(encoding="utf-8").splitlines() if line.strip()]
        if not (any(len(row) == 5 for row in rows) and any(len(row) > 6 for row in rows)):
            continue
        converted, polygon_count = [], 0
        for index, row in enumerate(rows, 1):
            if len(row) == 5:
                converted.append(" ".join(row))
            elif len(row) >= 7 and (len(row) - 1) % 2 == 0:
                segment = np.asarray(row[1:], dtype=np.float32).reshape(-1, 2)
                box = segments2boxes([segment])[0]
                converted.append(row[0] + " " + " ".join(format(float(value), ".9g") for value in box))
                polygon_count += 1
            else:
                raise ValueError(f"Format label campuran tidak valid: {label}, baris {index}.")
        relative = label.relative_to(root)
        prepared = folder / "label_adapter" / "converted" / relative
        prepared.parent.mkdir(parents=True, exist_ok=True)
        text = "\n".join(converted) + "\n"
        if not prepared.is_file() or prepared.read_text(encoding="utf-8") != text:
            prepared.write_text(text, encoding="utf-8", newline="\n")
        row = {"source_label": relative.as_posix(), "source_sha256": sha256_file(label),
               "prepared_label": prepared.relative_to(folder).as_posix(), "prepared_sha256": sha256_file(prepared),
               "row_count": len(rows), "polygon_rows_converted": polygon_count}
        records.append(row)
        mapping[str(label)] = {**row, "path": prepared}
    globals()["FILKOM_MGD_LABEL_MAP"] = mapping
    audit = {"policy": FILKOM_MGD_LABEL_POLICY, "source_files_modified": False,
             "converted_file_count": len(records), "files": records}
    write_json(folder / "label_adapter" / "manifest.json", audit)
    print(f"Label adapter FILKOM | mixed files={len(records)} | source unchanged", flush=True)
    return audit


def filkom_mgd_verify_image_label(args):
    prepared = globals().get("FILKOM_MGD_LABEL_MAP", {}).get(str(Path(args[1]).resolve()))
    if prepared:
        if sha256_file(args[1]) != prepared["source_sha256"] or sha256_file(prepared["path"]) != prepared["prepared_sha256"]:
            raise ValueError(f"Label berubah setelah preflight: {args[1]}")
        args = (args[0], str(prepared["path"]), *args[2:])
    return _filkom_label_verifier(args)


@contextmanager
def gflops_disable_label_cache():
    # Both training and all three evaluation stages use the same label reader.
    with _filkom_label_cache_context(), patch.object(_FilkomLabelDataset, "verify_image_label", filkom_mgd_verify_image_label):
        yield


def fine_tuned_check_images(loader, expected):
    observed = [Path(path).resolve() for path in loader.dataset.im_files]
    wanted = [Path(path).resolve() for path in expected]
    if len(observed) != len(wanted) or set(observed) != set(wanted):
        missing = sorted(str(path) for path in set(wanted) - set(observed))
        unexpected = sorted(str(path) for path in set(observed) - set(wanted))
        audit = {"prefix": getattr(loader.dataset, "prefix", ""), "expected_count": len(wanted),
                 "observed_count": len(observed), "missing": missing, "unexpected": unexpected,
                 "duplicate_count": len(observed) - len(set(observed))}
        path = Path(RUN_DIR) / "dataloader_mismatch.json"
        write_json(path, audit)
        raise ValueError(f"Citra aktual dataloader berbeda: expected={len(wanted)}, observed={len(observed)}, "
                         f"missing={len(missing)}, unexpected={len(unexpected)}. Detail: {path}")
