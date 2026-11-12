"""Record the local FILKOM dataset without requiring the raw pruning dataset hash."""
# ruff: noqa: F821 -- Embedded only in the FILKOM MGD notebook.
_filkom_mgd_comparison_settings = kd_comparison_settings


def ft_dataset(manifest, folder):
    original = manifest["dataset"]
    diagnostic_fixture = MODE == "smoke" and SMOKE_DATASET_ROOT is not None
    root = Path(SMOKE_DATASET_ROOT) if diagnostic_fixture else DATASET_ROOT
    source_yaml = root / "data.yaml" if diagnostic_fixture else DATA_YAML
    if MODE == "full_val" and (original["provenance_status"] != "VERIFIED" or manifest["validation_status"] != "PASS"):
        raise ValueError("Full validation membutuhkan raw dengan dataset terverifikasi.")
    dataset = gflops_prepare_dataset(MODE, source_yaml, root, folder, seed=manifest["seed"],
                                    expected_fingerprint=None, dataset_id=DATASET_ID)
    if dataset is None:
        raise FileNotFoundError("Atur DATASET_ROOT dan DATA_YAML.")
    record, train, _ = workflow_dataset_manifest(dataset, source_yaml, root, folder)
    if record["class_names"] != original["class_names"]:
        raise ValueError("Urutan kelas dataset FILKOM berbeda dari raw.")
    record.update(
        diagnostic_fixture=diagnostic_fixture,
        fingerprint_policy="record-only-filkom-mgd-v1",
        raw_dataset_reference={"dataset_id": original.get("dataset_id"),
                               "train_fingerprint": original["train_fingerprint"],
                               "val_fingerprint": original["full_val_fingerprint"]},
        matches_raw_dataset=(record["train_fingerprint"] == original["train_fingerprint"]
                             and record["val_fingerprint"] == original["full_val_fingerprint"]),
    )
    record["comparability_to_raw_training"] = "SAME_DATASET" if record["matches_raw_dataset"] else "DIFFERENT_DATASET"
    write_json(Path(folder) / "dataset_manifest.json", record)
    # Bind cross-run summaries to the actual train/val contents, including when
    # the local dataset happens to match the original pruning dataset.
    globals()["FILKOM_MGD_DATASET"] = record
    print(f"Dataset FILKOM | fingerprint policy=record-only | matches_raw={record['matches_raw_dataset']} "
          f"| val={record['val_fingerprint']}", flush=True)
    data_yaml = folder / "train_val_data.yaml"
    train_list = folder / "train_images.txt"
    train_list.write_text("".join(f"{p.as_posix()}\n" for p in train), encoding="utf-8")
    data_yaml.write_text(yaml.safe_dump({**dataset["info"], "train": str(train_list.resolve())}, sort_keys=False), encoding="utf-8")
    return dataset, record, train, data_yaml


def kd_comparison_settings():
    settings = _filkom_mgd_comparison_settings()
    record = globals().get("FILKOM_MGD_DATASET", {})
    return {**settings, "dataset_policy": "record-only-filkom-mgd-v1",
            "dataset_id": DATASET_ID, "train_fingerprint": record.get("train_fingerprint"),
            "val_fingerprint": record.get("val_fingerprint")}
