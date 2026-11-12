"""Connect the same charts and report to each YOLO26n recovery job."""
# ruff: noqa: F821 -- Notebook declarations provide the existing runtime.
_ceda_suite_build_eda = recovery_build_eda
_ceda_suite_log_comparison = recovery_log_comparison


def recovery_build_eda(result, folder, epoch_rows, files):
    report = _ceda_suite_build_eda(result, folder, epoch_rows, files)
    ceda_build(result, folder)
    return report


def recovery_log_comparison(run, result, folder):
    _ceda_suite_log_comparison(run, result, folder)
    # Include the full diagnostic gallery, which the legacy logger only stored in its artifact.
    import wandb
    keys = []
    for path in sorted((_CedaPath(folder) / "eda").rglob("*.png")):
        key = "visuals/eda/" + path.relative_to(_CedaPath(folder) / "eda").with_suffix("").as_posix()
        run.log({key: wandb.Image(str(path))})
        keys.append(key)
    ceda_write(_CedaPath(folder) / "wandb_media.json", {"keys": keys, "image_count": len(keys)})
    ceda_log(run, result, folder)
