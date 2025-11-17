from __future__ import annotations

import argparse
import csv
import json
import os
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

STATUS_FILE = "status/localities_status.csv"
QUALITY_FILE = "models/current/quality.json"
YOLO_CONFIG = "config/yolo_config.yaml"


@dataclass
class LocalityStatus:
    name: str
    status: str = ""
    auto_quality: str = ""
    last_model_run: str = ""
    last_update: str = ""
    n_images: int = 0
    n_labeled: int = 0


def get_root(arg_root: Optional[str]) -> Path:
    if arg_root:
        return Path(arg_root).resolve()
    # scripts/ -> parent = 2_Landmarking-Yolo_v1.0
    return Path(__file__).resolve().parent.parent


def load_status(root: Path) -> List[LocalityStatus]:
    status_path = root / STATUS_FILE
    rows: List[LocalityStatus] = []
    if not status_path.exists():
        return rows

    with status_path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                rows.append(
                    LocalityStatus(
                        name=row.get("locality", "").strip(),
                        status=row.get("status", "").strip(),
                        auto_quality=row.get("auto_quality", "").strip(),
                        last_model_run=row.get("last_model_run", "").strip(),
                        last_update=row.get("last_update", "").strip(),
                        n_images=int(row.get("n_images", "0") or 0),
                        n_labeled=int(row.get("n_labeled", "0") or 0),
                    )
                )
            except Exception:
                continue
    return rows


def format_locality_row(idx: int, loc: LocalityStatus) -> str:
    if loc.n_images > 0:
        pct = int(round(100.0 * loc.n_labeled / loc.n_images))
        count_str = f"[{loc.n_labeled}/{loc.n_images}] {pct:3d}%"
    else:
        count_str = "[0/0]   0%"

    status = loc.status or "(no status)"
    aq = f" {loc.auto_quality}" if loc.auto_quality else ""
    return f"[{idx}] {loc.name:<30s} {count_str}  {status}{aq}"


def print_localities_block(localities: List[LocalityStatus]) -> None:
    print()
    print("Localities (from status/localities_status.csv):")
    print()
    if not localities:
        print("  (no localities found)")
        print()
        return

    for i, loc in enumerate(localities, start=1):
        print(" ", format_locality_row(i, loc))
    print()
    print("[0] Back to main menu")
    print()


def pick_locality(localities: List[LocalityStatus]) -> Optional[LocalityStatus]:
    if not localities:
        print("No localities found.")
        return None
    print_localities_block(localities)
    choice = input("Select locality (0 = cancel): ").strip()
    if choice == "0":
        return None
    try:
        idx = int(choice)
    except ValueError:
        print("Invalid selection.")
        return None
    if idx < 1 or idx > len(localities):
        print("Invalid selection.")
        return None
    return localities[idx - 1]


def load_yolo_config(root: Path) -> Dict[str, Any]:
    cfg_path = root / YOLO_CONFIG
    if not cfg_path.exists():
        return {}
    try:
        import yaml  # type: ignore

        with cfg_path.open("r", encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}
        if not isinstance(data, dict):
            return {}
        return data
    except Exception as exc:
        print(f"[WARN] Cannot read YOLO config: {exc}")
        return {}


def show_yolo_config(root: Path) -> None:
    cfg = load_yolo_config(root)
    if not cfg:
        print("No YOLO config found (config/yolo_config.yaml).")
        return

    model = cfg.get("model", {})
    resize = cfg.get("resize", {})
    train = cfg.get("train", {})
    augment = cfg.get("augment", {})
    infer = cfg.get("infer", {})

    print("YOLO config (config/yolo_config.yaml)")
    print()
    print("model:")
    print(f"  pretrained_weights: {model.get('pretrained_weights', '?')}")
    print(f"  num_keypoints:      {model.get('num_keypoints', '?')}")
    print(f"  class_names:        {model.get('class_names', [])}")
    print()
    print("resize:")
    print(f"  enabled:            {resize.get('enabled', True)}")
    print(f"  long_side:          {resize.get('long_side', 1280)}")
    print(f"  stride_multiple:    {resize.get('stride_multiple', 32)}")
    print()
    print("train:")
    print(f"  train_val_split:    {train.get('train_val_split', 0.9)}")
    print(f"  max_epochs:         {train.get('max_epochs', 100)}")
    print(f"  batch_size:         {train.get('batch_size', 8)}")
    print(f"  learning_rate:      {train.get('learning_rate', 0.0005)}")
    print(f"  weight_decay:       {train.get('weight_decay', 0.0001)}")
    print()
    print("augment:")
    for key in (
        "rotation_deg",
        "rotation_prob",
        "scale_min",
        "scale_max",
        "scale_prob",
        "brightness",
        "contrast",
        "color_prob",
        "horizontal_flip",
    ):
        if key in augment:
            print(f"  {key}: {augment[key]}")
    print()
    print("infer:")
    print(f"  conf_threshold:     {infer.get('conf_threshold', 0.25)}")
    print(f"  iou_threshold:      {infer.get('iou_threshold', 0.5)}")
    print()
    print("To change these values:")
    print('  1) Open file "config/yolo_config.yaml" with a text editor (for example Notepad).')
    print("  2) Change only the numbers or true/false values.")
    print("  3) Save the file.")
    print("New training runs will automatically use the new settings.")
    print("Do not change the parameter names, only their values.")
    print()


def show_model_info(root: Path) -> None:
    q_path = root / QUALITY_FILE
    if not q_path.exists():
        print("No trained YOLO model found. Please run action 1) Train / finetune model first.")
        return
    try:
        with q_path.open("r", encoding="utf-8") as f:
            data = json.load(f)
    except Exception as exc:
        print(f"[ERR] Cannot read {q_path}: {exc}")
        return

    run_id = data.get("run_id", "?")
    model_type = data.get("model_type", "yolo_pose")
    num_kpts = data.get("num_keypoints", data.get("n_keypoints", "?"))
    n_train = int(data.get("n_train_images", 0) or 0)
    n_val = int(data.get("n_val_images", 0) or 0)
    train_share = float(data.get("train_share", 0.0) or 0.0)
    val_share = float(data.get("val_share", 0.0) or 0.0)
    pck_percent = float(data.get("pck_r_percent", 0.0) or 0.0)
    resize_enabled = bool(data.get("resize_enabled", True))
    resize_long_side = data.get("resize_long_side", 1280)

    print("Current model:")
    print()
    print(f"Run id:       {run_id}")
    print(f"Model type:   {model_type}, {num_kpts} keypoints")
    print()
    train_pct = int(round(train_share * 100)) if (n_train + n_val) > 0 else 0
    val_pct = int(round(val_share * 100)) if (n_train + n_val) > 0 else 0
    print(f"Train images: {n_train} ({train_pct} %)")
    print(f"Val images:   {n_val} ({val_pct} %)")
    print()
    print(f"Validation PCK@R: {int(round(pck_percent))} %")
    print()
    print(f"Resize enabled:   {resize_enabled}")
    print(f"Resize long side: {resize_long_side}")
    print()


def run_train_yolo(root: Path, base_localities: Path, localities: List[LocalityStatus]) -> None:
    """
    Action 1) Train / finetune YOLO model on MANUAL localities.

    According to ТЗ-YOLO:
      - use only MANUAL localities from status/localities_status.csv
      - build YOLO-pose dataset in datasets/yolo/<run_id>/
      - call scripts/train_yolo.py which trains model and computes PCK@R
    """
    manual_localities = [loc for loc in localities if loc.status.upper() == "MANUAL"]
    if not manual_localities:
        print("No MANUAL localities. Nothing to train.")
        return

    script = root / "scripts" / "train_yolo.py"
    if not script.is_file():
        print(f"[ERR] YOLO training script not found: {script}")
        return

    # Python executable (prefer same env as this script)
    py = os.environ.get("PYTHON_EXE") or sys.executable or "python"

    env = os.environ.copy()
    env["LM_ROOT"] = str(root)
    env["BASE_LOCALITIES"] = str(base_localities)

    cmd = [str(py), str(script), "--root", str(root), "--base", str(base_localities)]

    print()
    print(f"[INFO] Starting YOLO training via {script.name} ...")
    print()

    try:
        result = subprocess.run(cmd, env=env)
    except Exception as exc:
        print(f"[ERR] Cannot start train_yolo.py: {exc}")
        return

    if result.returncode != 0:
        print(f"[ERR] YOLO training script finished with code {result.returncode}.")
    else:
        print("[INFO] YOLO training finished successfully.")
    print()


def run_autolabel_yolo(root: Path, base_localities: Path, localities: List[LocalityStatus]) -> None:
    """
    Placeholder for YOLO autolabel.
    """
    print()
    print("[INFO] YOLO autolabel (action 2) is not implemented yet in this step.")
    print("       Autolabeling will be available after training code is added.")
    print()


def run_open_in_annotator(root: Path, base_localities: Path, localities: List[LocalityStatus]) -> None:
    loc = pick_locality(localities)
    if loc is None:
        return

    # Немедленно обновляем строку в status/localities_status.csv
    status_path = root / STATUS_FILE
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    rows: List[Dict[str, Any]] = []
    fieldnames: List[str] = []

    if status_path.exists():
        with status_path.open("r", encoding="utf-8", newline="") as f:
            reader = csv.DictReader(f)
            fieldnames = reader.fieldnames or []
            for row in reader:
                if row.get("locality", "").strip() == loc.name:
                    row["status"] = "MANUAL"
                    row["auto_quality"] = ""
                    row["last_update"] = now
                rows.append(row)

        if fieldnames:
            with status_path.open("w", encoding="utf-8", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
                for row in rows:
                    writer.writerow(row)

    # Запускаем 1_ANNOTATOR.bat
    bat = root / "1_ANNOTATOR.bat"
    if not bat.exists():
        print(f"[ERR] Annotator launcher not found: {bat}")
        return

    try:
        os.system(f'"{bat}"')
    except Exception as exc:
        print(f"[ERR] Cannot start annotator: {exc}")
        return

    # После закрытия аннотатора пересобираем статусы
    try:
        import rebuild_localities_status  # type: ignore

        rebuild_localities_status.main()
    except Exception as exc:
        print(f"[WARN] rebuild_localities_status failed after annotator: {exc}")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", dest="root", default=None)
    parser.add_argument("--base-localities", dest="base_localities", default=None)
    args = parser.parse_args()

    root = get_root(args.root)
    if args.base_localities:
        base_localities = Path(args.base_localities)
    else:
        cfg_dir = root / "cfg"
        last_base = cfg_dir / "last_base.txt"
        if last_base.exists():
            base_localities = Path(last_base.read_text(encoding="utf-8").strip())
        else:
            base_localities = root

    print("=== GM Landmarking: YOLO Trainer (v1.0) ===")
    print()
    print("Base folder:")
    print(f"  {base_localities}")
    print()

    while True:
        print("Actions:")
        print("  1) Train / finetune YOLO model on MANUAL localities")
        print("  2) Autolabel locality with current YOLO model")
        print("  3) Open locality in annotator")
        print("  4) Show current model info")
        print("  5) Show / edit YOLO config")
        print()
        print("  0) Quit")
        print()

        choice = input("Select action: ").strip()
        print()

        if choice == "0":
            break

        localities = load_status(root)
        print_localities_block(localities)

        if choice == "1":
            run_train_yolo(root, base_localities, localities)
        elif choice == "2":
            run_autolabel_yolo(root, base_localities, localities)
        elif choice == "3":
            run_open_in_annotator(root, base_localities, localities)
        elif choice == "4":
            show_model_info(root)
        elif choice == "5":
            show_yolo_config(root)
        else:
            print("Unknown action.")

        print()

    return 0


if __name__ == "__main__":
    try:
        code = main()
    except Exception as exc:
        print(f"[ERR] Trainer crashed: {exc}")
        code = 1
    raise SystemExit(code)

