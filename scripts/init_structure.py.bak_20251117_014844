from __future__ import annotations

from pathlib import Path


def get_root() -> Path:
    """Landmarking root = folder with 1_ANNOTATOR.bat."""
    # scripts/ -> parent = tools/2_Landmarking_v1.0
    return Path(__file__).resolve().parent.parent


def ensure_dirs(root: Path) -> None:
    """
    Create all service directories if they do not exist.
    According to ТЗ_1.0: status/, models/current/, models/history/,
    config/, logs/, datasets/.
    """
    subdirs = [
        "status",
        "models/current",
        "models/history",
        "config",
        "logs",
        "datasets",
    ]
    for sub in subdirs:
        (root / sub).mkdir(parents=True, exist_ok=True)


def ensure_config(root: Path) -> None:
    """
    Create default config/hrnet_config.yaml if missing.
    Values follow ТЗ_1.0 (simple HRNet W32 config).
    """
    cfg_dir = root / "config"
    cfg_dir.mkdir(parents=True, exist_ok=True)
    cfg_path = cfg_dir / "hrnet_config.yaml"
    if cfg_path.exists():
        return

    default_yaml = """# HRNet training config for GM Landmarking
model_type: "hrnet_w32"
input_size: 256  # 0 = do not resize (see resize_mode)
resize_mode: "resize"  # "resize" or "original"
keep_aspect_ratio: true
batch_size: 8
learning_rate: 0.0005
max_epochs: 100
train_val_split: 0.9
flip_augmentation: true
rotation_augmentation_deg: 15
scale_augmentation: 0.3
weight_decay: 0.0001
"""
    cfg_path.write_text(default_yaml, encoding="utf-8")


def ensure_status_header(root: Path) -> None:
    """
    Create empty status/localities_status.csv with header
    if file does not exist yet.
    """
    status_dir = root / "status"
    status_dir.mkdir(parents=True, exist_ok=True)
    status_csv = status_dir / "localities_status.csv"
    if status_csv.exists():
        return

    header = (
        "locality,status,auto_quality,"
        "last_model_run,last_update,n_images,n_labeled\n"
    )
    status_csv.write_text(header, encoding="utf-8")


def main() -> int:
    root = get_root()
    ensure_dirs(root)
    ensure_config(root)
    ensure_status_header(root)
    # No verbose output here: this script is called on every start.
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
