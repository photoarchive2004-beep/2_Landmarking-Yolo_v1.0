param()

$ErrorActionPreference = "Stop"

$root = Get-Location
Write-Host "=== step01_yolo_init_structure.ps1 ==="
Write-Host "Root: $root"

# 1) Патчим 1_ANNOTATOR.bat: привязываем к 2_Landmarking-Yolo_v1.0
$annotPath = Join-Path $root "1_ANNOTATOR.bat"
if (!(Test-Path $annotPath)) {
    throw "1_ANNOTATOR.bat not found: $annotPath"
}

$stamp = Get-Date -Format "yyyyMMdd_HHmmss"

$backupAnnot = Join-Path $root ("1_ANNOTATOR.bat.bak_" + $stamp)
Copy-Item $annotPath $backupAnnot -Force
Write-Host "Backup of 1_ANNOTATOR.bat -> $backupAnnot"

$annotText = Get-Content $annotPath -Raw -Encoding UTF8
$annotText = $annotText -replace "2_Landmarking_v1.0","2_Landmarking-Yolo_v1.0"
Set-Content -Path $annotPath -Value $annotText -Encoding UTF8
Write-Host "Patched TOOL_DIR in 1_ANNOTATOR.bat to 2_Landmarking-Yolo_v1.0"

# 2) Переписываем scripts\init_structure.py под YOLO
$initPath = Join-Path $root "scripts\init_structure.py"
if (!(Test-Path $initPath)) {
    throw "scripts\init_structure.py not found: $initPath"
}

$backupInit = Join-Path $root ("scripts\init_structure.py.bak_" + $stamp)
Copy-Item $initPath $backupInit -Force
Write-Host "Backup of init_structure.py -> $backupInit"

# ВНИМАНИЕ: это уже содержимое самого init_structure.py
$initCode = @"
from __future__ import annotations

from pathlib import Path


def get_root() -> Path:
    """
    Landmarking YOLO root (LM_ROOT) = folder that contains 1_ANNOTATOR.bat
    and 2_TRAIN-INFER_YOLO.bat (see ТЗ-YOLO).
    scripts/ -> parent = 2_Landmarking-Yolo_v1.0
    """
    return Path(__file__).resolve().parent.parent


def ensure_dirs(root: Path) -> None:
    """
    Create all service directories if they do not exist.

    LM_ROOT/
      scripts/
      status/
      models/
        current/
        history/
      config/
      logs/
      datasets/
      cfg/
    """
    subdirs = [
        "scripts",
        "status",
        "models/current",
        "models/history",
        "config",
        "logs",
        "datasets",
        "cfg",
    ]
    for sub in subdirs:
        (root / sub).mkdir(parents=True, exist_ok=True)


def read_lm_number(root: Path) -> int:
    """
    Read number of landmarks from LM_number.txt if available.
    If missing or invalid, return 0 (user can edit config later).
    """
    lm_file = root / "LM_number.txt"
    if not lm_file.exists():
        return 0

    text = lm_file.read_text(encoding="utf-8").strip()
    try:
        return int(text)
    except ValueError:
        return 0


def ensure_yolo_config(root: Path) -> None:
    """
    Create default config/yolo_config.yaml if missing.

    Structure follows ТЗ-YOLO:
      model / resize / train / augment / infer
    """
    cfg_dir = root / "config"
    cfg_dir.mkdir(parents=True, exist_ok=True)
    cfg_path = cfg_dir / "yolo_config.yaml"
    if cfg_path.exists():
        return

    n_kpts = read_lm_number(root)

    default_yaml = f"""# YOLO pose config for GM Landmarking (auto-created)
model:
  pretrained_weights: "yolov8m-pose.pt"
  num_keypoints: {n_kpts}
  class_names: ["class"]

resize:
  enabled: true
  long_side: 1280
  stride_multiple: 32

train:
  train_val_split: 0.9
  max_epochs: 100
  batch_size: 8
  learning_rate: 0.0005
  weight_decay: 0.0001

augment:
  rotation_deg: 15
  rotation_prob: 0.5
  scale_min: 0.8
  scale_max: 1.25
  scale_prob: 0.5
  brightness: 0.1
  contrast: 0.1
  color_prob: 0.5
  horizontal_flip: false

infer:
  conf_threshold: 0.25
  iou_threshold: 0.5
"""
    cfg_path.write_text(default_yaml, encoding="utf-8")


def ensure_status_header(root: Path) -> None:
    """
    Create empty status/localities_status.csv with header
    if file does not exist yet (format from ТЗ-YOLO).
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
    ensure_yolo_config(root)
    ensure_status_header(root)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
"@

Set-Content -Path $initPath -Value $initCode -Encoding UTF8
Write-Host "Rewritten scripts\init_structure.py for YOLO config."

# 3) Синхронизация с GitHub (по ТЗ)
$syncScript = Join-Path $root "sync_repo_yolo_v3.ps1"
if (Test-Path $syncScript) {
    Write-Host "Running sync_repo_yolo_v3.ps1 ..."
    powershell -NoProfile -ExecutionPolicy Bypass -File $syncScript
}
else {
    Write-Host "[WARN] sync_repo_yolo_v3.ps1 not found, skipping auto git sync."
}

Write-Host "=== step01_yolo_init_structure.ps1 done ==="
