"""Connect the presentation layer to the selected raw FT/KD runtime."""
# ruff: noqa: F821 -- Notebook declarations provide the existing runtime.
_ceda_ft_eda = ft_eda
_ceda_ft_log_results = ft_log_results
_ceda_ft_registry = ft_registry
_ceda_ft_verify_logging = ft_verify_logging


def ft_eda(result, folder, epochs, files):
    report = _ceda_ft_eda(result, folder, epochs, files)
    ceda_build(result, folder)
    return report


def ft_log_results(run, result, folder):
    keys = _ceda_ft_log_results(run, result, folder)
    ceda_log(run, result, folder)
    return keys


def ft_registry(result, folder):
    row = _ceda_ft_registry(result, folder)
    row.update({key: value for key, value in result.get("custom_eda", {}).items() if key.startswith("custom_eda_")})
    filename = "kd_registry" if RECOVERY_METHOD != "finetune" else "finetuning_registry"
    path = folder / f"{filename}.json"
    value = ceda_read(path)
    value["rows"][0].update(row)
    ceda_write(path, value)
    write_csv(folder / f"{filename}.csv", [row])
    return row


def ft_verify_logging(result, folder):
    proof = _ceda_ft_verify_logging(result, folder)
    expected = ceda_read(folder / "custom_eda_logging.json")
    if not expected:
        raise ValueError("EDA custom belum dicatat di W&B.")
    remote = wandb.Api(timeout=60).run(f"{WANDB_ENTITY}/{WANDB_PROJECT}/{proof['run_id']}")
    try:
        for key in expected["image_keys"] + expected["table_keys"]:
            value = remote.summary.get(key)
            if not value or not value.get("_type"):
                raise ValueError(f"Panel EDA custom belum ditemukan di W&B: {key}")
        receipt = ceda_read(folder / "custom_eda_receipt.json")
        for key, name in receipt["images"].items():
            if remote.summary[key].get("sha256") != ceda_hash(folder / "custom_eda" / name):
                raise ValueError(f"SHA-256 gambar W&B berbeda dari EDA lokal: {key}")
        for key in expected["table_keys"]:
            name = key.rsplit("/", 1)[-1]
            if remote.summary[key].get("nrows") != len(ceda_read(folder / "custom_eda" / f"{name}.json")):
                raise ValueError(f"Jumlah baris Table W&B berbeda: {key}")
        if remote.summary.get("custom_eda_report_url") != expected["custom_eda_report_url"]:
            raise ValueError("Tautan halaman EDA custom belum tercatat di summary W&B.")
    except Exception:
        remote.summary.update({"status": "PARTIAL", "logging_verified": False, "custom_eda_status": "FAIL"})
        raise
    proof.update(custom_eda_status="PASS", custom_eda_report_url=expected["custom_eda_report_url"],
                 custom_eda_artifact_ref=expected["custom_eda_artifact_ref"])
    return proof
