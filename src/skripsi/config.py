"""The shared, versioned experiment protocol."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Literal

Architecture = Literal["baseline", "bifpn", "asff", "afpn"]

CLASS_NAMES = (
    "Bus",
    "Mobil Penumpang",
    "Pejalan Kaki",
    "Sepeda Motor",
    "Truck",
    "Unmotorized",
)
SEEDS = (42, 123, 2026)
EXPECTED_V2 = {
    "train": {"images": 6775, "annotations": 125108, "empty_labels": 16},
    "val": {"images": 1790, "annotations": 32076, "empty_labels": 2},
    "test": {"images": 913, "annotations": 16158, "empty_labels": 9},
}
EXPECTED_V2_FINGERPRINT = (
    "970d28b5cd507f8dcd054edee478783fa1bf45eb9bac10263896d98f4a96f7cd"
)


@dataclass(frozen=True)
class ExperimentConfig:
    """One architecture and seed, with identical data/training/evaluation settings."""

    architecture: Architecture = "baseline"
    seed: int = 42
    dataset_dir: Path | None = None
    output_root: Path = field(
        default_factory=lambda: (
            Path("/kaggle/working/yolo26n_research")
            if Path("/kaggle/working").is_dir()
            else Path("./outputs/research")
        )
    )
    model_weights: str = "yolo26n.pt"
    workspace: str = "ilham-rafiqin"
    project: str = "poltrada-original-raw-2"
    dataset_version: int = 2
    imgsz: int = 640
    epochs: int = 100
    patience: int = 25
    train_batch: int = 16
    eval_batch: int = 1
    optimizer: str = "AdamW"
    lr0: float = 0.001
    weight_decay: float = 0.0005
    warmup_epochs: float = 3.0
    amp: bool = True
    workers: int = 4
    close_mosaic: int = 10
    flipud: float = 0.0
    mixup: float = 0.0
    cutmix: float = 0.0
    save_period: int = 1
    artifact_period: int = 5
    prediction_conf: float = 0.001
    prediction_iou: float = 0.7
    benchmark_warmup: int = 50
    benchmark_runs: int = 200
    wandb_project: str = "skripsi-yolo26n-indotraffic"
    wandb_entity: str | None = None
    git_commit: str | None = None

    def __post_init__(self) -> None:
        if self.architecture not in ("baseline", "bifpn", "asff", "afpn"):
            raise ValueError(f"Arsitektur tidak dikenal: {self.architecture}")
        if self.seed not in SEEDS:
            raise ValueError(f"Seed harus salah satu dari {SEEDS}")
        if self.dataset_version != 2:
            raise ValueError("Seluruh eksperimen modular memakai dataset versi 2")
        if self.imgsz != 640:
            raise ValueError("Protokol AP skala memakai kanvas 640x640")
        if self.train_batch < 1 or self.eval_batch < 1:
            raise ValueError("Ukuran batch harus positif")
        object.__setattr__(
            self, "output_root", Path(self.output_root).expanduser().resolve()
        )
        if self.dataset_dir is not None:
            object.__setattr__(
                self, "dataset_dir", Path(self.dataset_dir).expanduser().resolve()
            )

    @property
    def run_name(self) -> str:
        return f"yolo26n-{self.architecture}-train-seed-{self.seed}"

    @property
    def run_dir(self) -> Path:
        return self.output_root / "runs" / self.run_name

    @property
    def reports_dir(self) -> Path:
        return self.output_root / "reports"

    def as_dict(self) -> dict:
        return {
            key: str(value) if isinstance(value, Path) else value
            for key, value in asdict(self).items()
        }
