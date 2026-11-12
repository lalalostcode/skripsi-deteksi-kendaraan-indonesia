"""Prepare W&B credentials before FILKOM MGD downloads any artifacts."""
# ruff: noqa: F821 -- Notebook configuration provides os, Path, and WANDB_TRACKING.


def filkom_mgd_load_wandb_key():
    if os.environ.get("WANDB_API_KEY"):
        return "environment"
    import shlex

    repo = Path(globals().get("REPO_ROOT", Path.home() / "skripsi-kd/repo")).expanduser()
    env_file = repo / ".env"
    if not env_file.is_file():
        return None
    for line in env_file.read_text(encoding="utf-8-sig").splitlines():
        assignment = line.strip()
        if assignment.startswith("export "):
            assignment = assignment[7:].lstrip()
        name, separator, raw_value = assignment.partition("=")
        if not separator or name.strip() != "WANDB_API_KEY":
            continue
        try:
            values = shlex.split(raw_value, comments=True, posix=True)
        except ValueError:
            raise RuntimeError(f"Format WANDB_API_KEY tidak valid di {env_file}; periksa tanda kutip.") from None
        if not values or values == [""]:
            continue
        if len(values) != 1:
            raise RuntimeError(f"Format WANDB_API_KEY tidak valid di {env_file}; nilai harus satu token.")
        os.environ["WANDB_API_KEY"] = values[0]
        return str(env_file)
    return None


FILKOM_MGD_AUTH_SOURCE = filkom_mgd_load_wandb_key()
if WANDB_TRACKING and FILKOM_MGD_AUTH_SOURCE is None:
    raise RuntimeError(
        "WANDB_API_KEY belum tersedia. Set pada environment kernel atau file .env di REPO_ROOT "
        "(default ~/skripsi-kd/repo/.env) sebelum menjalankan training."
    )
print("W&B credential:", "ready" if FILKOM_MGD_AUTH_SOURCE else "not configured (tracking disabled)", flush=True)
