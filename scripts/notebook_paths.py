"""Explicit notebook locations for repository tooling, never notebook runtime imports."""
from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK_ROOT = ROOT / "notebooks"

GFLOPS_PRUNING = ("05_lamp_yolo26n_gflops_no_recovery.ipynb", "06_lamp_yolo26s_gflops_no_recovery.ipynb")
MULTISEED_PRUNING = "06_lamp_yolo26s_multiseed_wandb.ipynb"
MULTISEED_TEACHER = "02_teacher_yolo26m_multiseed.ipynb"
PRUNED_FINE_TUNED = "01_yolo26s_pruned_fine_tuned.ipynb"
RAW_PRUNED_RECOVERY = (
    "01_yolo26n_standard_finetuning.ipynb", "02_yolo26s_standard_finetuning.ipynb",
    "03_yolo26ns_native_kd.ipynb", "04_yolo26ns_crosskd.ipynb", "05_yolo26ns_mgd_kd.ipynb",
)

# Teacher roles are taken from experiment configuration, not the student filename.
NOTEBOOK_CATEGORIES = {
    "kaggle": {
        "eda": ("00_eda_poltrada_v3.ipynb", "00_results_eda-poltrada.ipynb"),
        "teacher": ("01_teacher_yolo26m.ipynb", MULTISEED_TEACHER),
        "baseline": ("02_baseline_yolo26s.ipynb", "03_baseline_yolo26n.ipynb", "03_baseline_yolo26n_scratch.ipynb"),
        "kd/teacher_m": (
            "04_native_yolo26s.ipynb", "05_native_yolo26n.ipynb", "06_crosskd_yolo26s.ipynb",
            "07_crosskd_yolo26n.ipynb", "08_csakd_yolo26s.ipynb", "09_csakd_yolo26n.ipynb",
        ),
        "kd/teacher_s": (
            "10_native_yolo26n_teacher_yolo26s.ipynb", "11_crosskd_yolo26n_teacher_yolo26s.ipynb",
            "12_csakd_yolo26n_teacher_yolo26s.ipynb", "12_pkd_yolo26n.ipynb",
            "13_head_yolo26n_teacher_yolo26s.ipynb", "14_rm_yolo26n_teacher_yolo26s.ipynb",
            "15_mgd_yolo26n_teacher_yolo26s.ipynb",
        ),
        "pruning": ("01_lamp_yolo26n_no_recovery.ipynb", "02_lamp_yolo26s_no_recovery.ipynb",
                    "03_lamp_yolo26n_extended_no_recovery.ipynb", "04_lamp_yolo26ns_finetune_recovery.ipynb",
                    *GFLOPS_PRUNING, MULTISEED_PRUNING),
        "recovery": RAW_PRUNED_RECOVERY,
        "baseline/fine-tunned": (PRUNED_FINE_TUNED,),
    },
    "filkom": {"teacher": (MULTISEED_TEACHER,), "pruning": GFLOPS_PRUNING, "recovery": RAW_PRUNED_RECOVERY,
               "fine-tunned": (PRUNED_FINE_TUNED,)},
    "colab": {
        "eda": ("00_eda_poltrada_v3.ipynb",),
        "teacher": ("01_teacher_yolo26m.ipynb",),
        "baseline": ("02_baseline_yolo26s.ipynb", "03_baseline_yolo26n_pretrained.ipynb", "04_baseline_yolo26n_scratch.ipynb"),
        "kd/teacher_m": (
            "05_native_yolo26m_to_yolo26s.ipynb", "06_native_yolo26m_to_yolo26n.ipynb",
            "07_crosskd_yolo26m_to_yolo26s.ipynb", "08_crosskd_yolo26m_to_yolo26n.ipynb",
            "09_csakd_yolo26m_to_yolo26s.ipynb", "10_csakd_yolo26m_to_yolo26n.ipynb",
        ),
        "kd/teacher_s": (
            "11_native_yolo26s_to_yolo26n.ipynb", "12_crosskd_yolo26s_to_yolo26n.ipynb",
            "13_csakd_yolo26s_to_yolo26n.ipynb", "14_pearson_pkd_yolo26s_to_yolo26n.ipynb",
            "15_head_yolo26s_to_yolo26n.ipynb", "16_rank_mimicking_yolo26s_to_yolo26n.ipynb",
        ),
    },
    "amd": {"baseline": ("03_baseline_yolo26n_amd_roboflow.ipynb",)},
    "local": {"eda": ("01_eda_dataset.ipynb",), "analysis": ("03_baseline_seed42_failure_analysis.ipynb",)},
}

NOTEBOOK_LOCATIONS = {
    platform: {filename: Path(platform) / category / filename
               for category, filenames in categories.items() for filename in filenames}
    for platform, categories in NOTEBOOK_CATEGORIES.items()
}


def notebook_path(platform: str, filename: str) -> Path:
    """Resolve a registered filename; reject unknown or ambiguous location guesses."""
    try:
        return NOTEBOOK_ROOT / NOTEBOOK_LOCATIONS[platform][filename]
    except KeyError as error:
        raise ValueError(f"Unknown notebook: platform={platform!r}, filename={filename!r}") from error


def notebook_paths(platform: str) -> list[Path]:
    """Enumerate the complete explicit collection for tests and generators."""
    if platform not in NOTEBOOK_LOCATIONS:
        raise ValueError(f"Unknown notebook platform: {platform!r}")
    return sorted(NOTEBOOK_ROOT / relative for relative in NOTEBOOK_LOCATIONS[platform].values())
