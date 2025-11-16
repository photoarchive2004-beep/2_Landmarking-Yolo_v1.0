from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

try:
    import yaml  # type: ignore
except ImportError:  # pragma: no cover
    yaml = None


STATUS_FILE = "status/localities_status.csv"

CONFIG_DEFAULTS: Dict[str, Any] = {
    "model_type": "hrnet_w32",
    "input_size": 256,
    "resize_mode": "resize",
    "keep_aspect_ratio": True,
    "batch_size": 8,
    "learning_rate": 0.0005,
    "max_epochs": 100,
    "train_val_split": 0.9,
    "flip_augmentation": True,
    "rotation_augmentation_deg": 15,
    "scale_augmentation": 0.3,
    "weight_decay": 0.0001,
}


@dataclass
class Locality:
    locality: str
    status: str
    auto_quality: str
    last_model_run: str
    last_update: str
    n_images: int
    n_labeled: int


def get_landmark_root(arg_root: Optional[str]) -> Path:
    """Return landmark module root (tools/2_Landmarking_v1.0)."""
    # Всегда берём папку на два уровня выше scripts/, как в ТЗ_1.0.
    # Аргумент --root игнорируем, чтобы не уезжать в D:\GM и т.п.
    return Path(__file__).resolve().parent.parent


def read_last_base(landmark_root: Path) -> Optional[Path]:
    cfg_path = landmark_root / "cfg" / "last_base.txt"
    if not cfg_path.exists():
        return None
    text = cfg_path.read_text(encoding="utf-8").strip()
    if not text:
        return None
    return Path(text)


def load_localities(landmark_root: Path) -> List[Locality]:
    status_path = landmark_root / STATUS_FILE
    if not status_path.exists():
        return []

    rows: List[Locality] = []
    with status_path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                n_images = int(row.get("n_images", "") or 0)
            except ValueError:
                n_images = 0
            try:
                n_labeled = int(row.get("n_labeled", "") or 0)
            except ValueError:
                n_labeled = 0

            rows.append(
                Locality(
                    locality=row.get("locality", ""),
                    status=row.get("status", ""),
                    auto_quality=row.get("auto_quality", ""),
                    last_model_run=row.get("last_model_run", ""),
                    last_update=row.get("last_update", ""),
                    n_images=n_images,
                    n_labeled=n_labeled,
                )
            )
    return rows


def save_localities(landmark_root: Path, localities: List[Locality]) -> None:
    status_path = landmark_root / STATUS_FILE
    status_path.parent.mkdir(parents=True, exist_ok=True)

    with status_path.open("w", encoding="utf-8", newline="") as f:
        fieldnames = [
            "locality",
            "status",
            "auto_quality",
            "last_model_run",
            "last_update",
            "n_images",
            "n_labeled",
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for loc in localities:
            writer.writerow(
                {
                    "locality": loc.locality,
                    "status": loc.status,
                    "auto_quality": loc.auto_quality,
                    "last_model_run": loc.last_model_run,
                    "last_update": loc.last_update,
                    "n_images": loc.n_images,
                    "n_labeled": loc.n_labeled,
                }
            )


def format_status(loc: Locality) -> str:
    if loc.status == "MANUAL":
        return "MANUAL"
    if loc.status == "AUTO":
        if loc.auto_quality:
            return f"AUTO {loc.auto_quality}"
        return "AUTO"
    return ""


def calc_percent(loc: Locality) -> int:
    if loc.n_images <= 0:
        return 0
    return int(round(100.0 * loc.n_labeled / max(1, loc.n_images)))


def print_localities_block(localities: List[Locality]) -> None:
    print()
    print("Localities (photos/<locality>/png):")
    print()

    if not localities:
        print(" (no localities found)")
        return

    # Аккуратные столбцы имени, статуса и счётчиков
    max_name = max((len(loc.locality) for loc in localities), default=0)
    name_width = max(max_name + 2, 24)  # минимум 24 символа
    count_width = 16

    for idx, loc in enumerate(localities, start=1):
        percent = calc_percent(loc)
        status_text = format_status(loc)

        prefix = f"[{idx:2d}] "
        name_part = f"{loc.locality}".ljust(name_width)
        name_part = prefix + name_part

        count_part = f"[{loc.n_labeled}/{loc.n_images}] {percent:3d}%".ljust(count_width)
        status_col = status_text if status_text else ""

        line = f"{name_part} {status_col:10s} {count_part}"
        print(line)


def read_quality(landmark_root: Path) -> Dict[str, Any]:
    quality_path = landmark_root / "models" / "current" / "quality.json"
    if not quality_path.exists():
        return {}

    try:
        data: Any = json.loads(quality_path.read_text(encoding="utf-8"))
    except Exception:
        return {}

    if not isinstance(data, dict):
        return {}
    return data


def run_training(landmark_root: Path) -> None:
    localities = load_localities(landmark_root)
    manual_localities = [loc for loc in localities if loc.status == "MANUAL"]

    if not manual_localities:
        print()
        print("No MANUAL localities found.")
        print("Nothing to train on.")
        return

    train_script = landmark_root / "scripts" / "train_hrnet.py"
    if not train_script.exists():
        print()
        print("[ERR] train_hrnet.py not found.")
        print("Cannot start training.")
        return

    print()
    print("=== HRNet training started (action 1: MANUAL localities) ===")
    print()

    cmd = [sys.executable, str(train_script)]
    try:
        result = subprocess.run(cmd, check=False)
    except Exception as exc:  # noqa: BLE001
        print(f"[ERR] Training failed: {exc}")
        return

    if result.returncode != 0:
        print(f"[WARN] train_hrnet.py exited with code {result.returncode}.")

    print("=== HRNet training finished ===")


def choose_autolabel_locality(localities: List[Locality]) -> Optional[Locality]:
    # Только локальности без статуса или с AUTO
    candidates: List[Locality] = [
        loc for loc in localities if loc.status in ("", "AUTO")
    ]

    print()
    print("Localities available for autolabel:")
    print()

    if not candidates:
        print(" (no localities available – only MANUAL localities found)")
        return None

    max_name = max((len(loc.locality) for loc in candidates), default=0)
    name_width = max(max_name + 2, 24)

    for idx, loc in enumerate(candidates, start=1):
        status_text = format_status(loc)
        status_block = status_text if status_text else ""
        print(
            f"[{idx}] {loc.locality.ljust(name_width)} "
            f"{status_block:8s} ({loc.n_images} imgs, {loc.n_labeled} csv)"
        )

    print()

    try:
        choice = input("Select locality number (or 0 to cancel): ").strip()
    except EOFError:
        return None

    if not choice:
        return None
    if not choice.isdigit():
        print("Please enter a number.")
        return None

    idx = int(choice)
    if idx <= 0:
        return None
    if idx > len(candidates):
        print("No such locality.")
        return None

    return candidates[idx - 1]


def run_autolabel(landmark_root: Path, base_localities: Optional[Path]) -> None:
    localities = load_localities(landmark_root)
    if not localities:
        print()
        print("No localities found in status/localities_status.csv.")
        return

    # Проверяем, что есть обученная модель
    model_path = landmark_root / "models" / "current" / "hrnet_best.pth"
    if not model_path.exists():
        print()
        print("No trained model found (models/current/hrnet_best.pth is missing).")
        print("Please run action 1 (Train) first.")
        return

    loc = choose_autolabel_locality(localities)
    if not loc:
        return

    if base_localities is None:
        base_localities = read_last_base(landmark_root)

    if base_localities is None:
        print()
        print("Base localities folder is not set (cfg/last_base.txt is empty).")
        return

    png_dir = base_localities / loc.locality / "png"
    if not png_dir.exists():
        print()
        print(f"Folder not found: {png_dir}")
        return

    png_list = sorted(png_dir.glob("*.png"))
    if not png_list:
        print()
        print(f"No *.png images found in {png_dir}")
        return

    infer_script = landmark_root / "scripts" / "infer_hrnet.py"
    if not infer_script.exists():
        print()
        print("[ERR] infer_hrnet.py not found.")
        print("Cannot run autolabel.")
        return

    print()
    print(f'=== Autolabel started for locality "{loc.locality}" ===')
    print()

    cmd = [
        sys.executable,
        str(infer_script),
        "--landmark-root",
        str(landmark_root),
        "--base",
        str(base_localities),
        "--locality",
        loc.locality,
    ]

    try:
        result = subprocess.run(cmd, check=False)
    except Exception as exc:  # noqa: BLE001
        print(f"[ERR] Autolabel failed: {exc}")
        return

    if result.returncode != 0:
        print(f"[ERR] Autolabel script returned code {result.returncode}.")
        return

    # Пересчитываем n_images / n_labeled для выбранной локальности
    png_list = sorted(png_dir.glob("*.png"))
    n_images = len(png_list)
    n_labeled = 0
    for img_path in png_list:
        csv_path = img_path.with_suffix(".csv")
        if csv_path.exists():
            n_labeled += 1

    loc.n_images = n_images
    loc.n_labeled = n_labeled

    # Читаем quality.json
    quality = read_quality(landmark_root)
    auto_quality = ""
    run_id = ""

    if quality:
        try:
            pck_percent = int(round(float(quality.get("pck_r_percent", 0))))
            auto_quality = str(pck_percent)
        except Exception:
            auto_quality = ""
        run_id = str(quality.get("run_id", "") or "")

    loc.status = "AUTO"
    loc.auto_quality = auto_quality
    loc.last_model_run = run_id
    loc.last_update = datetime.now().isoformat(timespec="seconds")

    save_localities(landmark_root, localities)

    if auto_quality:
        print(f'Autolabel done for locality "{loc.locality}".')
        print(f"Status: AUTO {auto_quality}")
    else:
        print(f'Autolabel done for locality "{loc.locality}".')
        print("Status: AUTO")

    print()
    input("Press Enter to exit...")


def choose_auto_locality(localities: List[Locality]) -> Optional[Locality]:
    auto_localities = [loc for loc in localities if loc.status == "AUTO"]

    print()
    print("AUTO localities:")
    print()

    if not auto_localities:
        print(" (no AUTO localities found)")
        return None

    max_name = max((len(loc.locality) for loc in auto_localities), default=0)
    name_width = max(max_name + 2, 24)

    for idx, loc in enumerate(auto_localities, start=1):
        status_text = format_status(loc)
        print(
            f"[{idx}] {loc.locality.ljust(name_width)} "
            f"{status_text:8s} ({loc.n_images} imgs, {loc.n_labeled} csv)"
        )

    print()

    try:
        choice = input("Select locality to review (or 0 to cancel): ").strip()
    except EOFError:
        return None

    if not choice:
        return None
    if not choice.isdigit():
        print("Please enter a number.")
        return None

    idx = int(choice)
    if idx <= 0:
        return None
    if idx > len(auto_localities):
        print("No such locality.")
        return None

    return auto_localities[idx - 1]


def run_review_auto(landmark_root: Path) -> None:
    localities = load_localities(landmark_root)
    if not localities:
        print()
        print("No localities found in status/localities_status.csv.")
        print()
        input("Press Enter to exit...")
        return

    loc = choose_auto_locality(localities)
    if not loc:
        print()
        input("Press Enter to exit...")
        return

    annotator_bat = landmark_root / "1_ANNOTATOR.bat"
    if not annotator_bat.exists():
        print()
        print("[ERR] 1_ANNOTATOR.bat not found in module root.")
        print()
        input("Press Enter to exit...")
        return

    flag_dir = landmark_root / "status"
    flag_dir.mkdir(parents=True, exist_ok=True)
    flag_path = flag_dir / f"review_done_{loc.locality}.flag"

    # Старый флаг на всякий случай удаляем
    try:
        if flag_path.exists():
            flag_path.unlink()
    except Exception:
        pass

    print()
    print(f'Launching annotator for locality "{loc.locality}" in REVIEW_AUTO mode...')
    print("Close annotator when review is finished.")
    print()

    env = os.environ.copy()
    env["GM_LOCALITY"] = loc.locality
    env["GM_MODE"] = "REVIEW_AUTO"

    try:
        subprocess.run(
            ["cmd", "/c", str(annotator_bat)],
            check=False,
            env=env,
        )
    except Exception as exc:  # noqa: BLE001
        print(f"[ERR] Failed to launch annotator: {exc}")
        print()
        input("Press Enter to exit...")
        return

    # После выхода аннотатора проверяем флаг
    if flag_path.exists():
        # Переводим локальность в MANUAL
        loc.status = "MANUAL"
        loc.auto_quality = ""
        loc.last_update = datetime.now().isoformat(timespec="seconds")

        save_localities(landmark_root, localities)

        try:
            flag_path.unlink()
        except Exception:
            pass

        print()
        print(f'Locality "{loc.locality}" marked as MANUAL (after review).')
    else:
        print()
        print("Review was not finished (no flag file found).")
        print(f'Locality "{loc.locality}" remains {format_status(loc) or "(no status)"}.')

    print()
    input("Press Enter to exit...")


def show_model_info(landmark_root: Path) -> None:
    print()
    quality = read_quality(landmark_root)
    if not quality:
        print("Model is not trained yet (models/current/quality.json not found).")
        print()
        input("Press Enter to exit...")
        return

    print("Current model info:")
    print()

    run_id = quality.get("run_id", "")
    print(f"Run id: {run_id}")
    print("Model: HRNet-W32 (18 keypoints)")

    n_train = quality.get("n_train_images", 0)
    n_val = quality.get("n_val_images", 0)
    train_share = quality.get("train_share", 0)
    val_share = quality.get("val_share", 0)

    try:
        train_percent = int(round(float(train_share) * 100))
    except Exception:
        train_percent = 0

    try:
        val_percent = int(round(float(val_share) * 100))
    except Exception:
        val_percent = 0

    print()
    print(f"Train images: {n_train} ({train_percent}%)")
    print(f"Val images:   {n_val}  ({val_percent}%)")

    pck = quality.get("pck_r_percent", None)
    if pck is not None:
        try:
            pck_int = int(round(float(pck)))
        except Exception:
            pck_int = 0
        print()
        print(f"PCK@R (validation): {pck_int} %")

    used_manual = quality.get("n_manual_localities", None)
    if used_manual is not None:
        try:
            used_manual_int = int(used_manual)
        except Exception:
            used_manual_int = None
        if used_manual_int is not None:
            print()
            print(f"Used MANUAL localities: {used_manual_int}")

    print()
    input("Press Enter to exit...")


def load_or_create_config(landmark_root: Path) -> Dict[str, Any]:
    cfg_dir = landmark_root / "config"
    cfg_dir.mkdir(parents=True, exist_ok=True)
    cfg_path = cfg_dir / "hrnet_config.yaml"

    cfg: Dict[str, Any] = dict(CONFIG_DEFAULTS)

    if yaml is None:
        # Нет PyYAML – просто гарантируем существование файла с простым форматом
        if not cfg_path.exists():
            with cfg_path.open("w", encoding="utf-8") as f:
                for key, value in CONFIG_DEFAULTS.items():
                    f.write(f"{key}: {value}\n")
        return cfg
def show_model_settings(landmark_root):
    """
    Print HRNet/MMPose training settings from config/hrnet_config.yaml
    in a simple, user-friendly way (including crop margins).
    """
    from pathlib import Path

    root = Path(landmark_root)
    cfg_path = root / "config" / "hrnet_config.yaml"

    print()
    print("=== Model settings (config/hrnet_config.yaml) ===")

    # Load config safely: if PyYAML is missing or file is broken,
    # fall back to built-in defaults from CONFIG_DEFAULTS.
    if yaml is None:
        print("PyYAML is not installed in this environment.")
        print("Using default values from CONFIG_DEFAULTS.")
        cfg = dict(CONFIG_DEFAULTS)
    else:
        try:
            if cfg_path.exists():
                with cfg_path.open("r", encoding="utf-8") as f:
                    cfg = yaml.safe_load(f) or {}
            else:
                print("Config file 'config/hrnet_config.yaml' not found.")
                print("It will be created automatically when you run the trainer.")
                cfg = dict(CONFIG_DEFAULTS)
        except Exception as exc:
            print("Error reading config/hrnet_config.yaml:")
            print(f"  {exc}")
            print("Using default values from CONFIG_DEFAULTS.")
            cfg = dict(CONFIG_DEFAULTS)

    def show_one(name, default, description_lines):
        value = cfg.get(name, default)
        print(f"{name} = {value}")
        for line in description_lines:
            print(f"  - {line}")
        print()

    # === Basic HRNet training params ===
    show_one("model_type", "hrnet_w32", [
        "HRNet backbone type. 'hrnet_w32' is a good default."
    ])

    show_one("input_size", 1280, [
        "Target size for the LONG side of the image in pixels.",
        "The image is resized keeping aspect ratio.",
    ])

    show_one("resize_mode", "resize", [
        "'resize': rescale images so that the long side = input_size.",
        "'original': keep original resolution (only safe changes)."
    ])

    show_one("keep_aspect_ratio", True, [
        "If True: image proportions are always preserved.",
        "This MUST stay True for geometric morphometrics."
    ])

    show_one("batch_size", 2, [
        "How many images are processed in one training step.",
        "Bigger batch uses more GPU memory."
    ])

    show_one("learning_rate", 0.0005, [
        "How fast the model learns."
    ])

    show_one("max_epochs", 150, [
        "Maximum number of passes through the training data."
    ])

    show_one("train_val_split", 0.9, [
        "Part of data used for training. 0.9 = 90% train, 10% validation."
    ])

    show_one("flip_augmentation", True, [
        "If True: random horizontal flips during training."
    ])

    show_one("rotation_augmentation_deg", 15, [
        "Maximum random rotation angle in degrees during training."
    ])

    show_one("scale_augmentation", 0.3, [
        "Random scaling of images during training.",
        "0.3 means up to ±30% size change."
    ])

    show_one("weight_decay", 0.0001, [
        "Regularization to reduce overfitting."
    ])

    show_one("crop_margin_x_percent", 0.15, [
        "Extra space LEFT and RIGHT of the landmarks bounding box.",
        "Value is a fraction of bbox width for EACH side.",
        "Example: 0.15 -> +15% bbox width on the left and +15% on the right."
    ])

    show_one("crop_margin_y_percent", 0.5, [
        "Extra space ABOVE and BELOW the landmarks bounding box.",
        "Value is a fraction of bbox height for EACH side.",
        "Example: 0.5 -> +50% bbox height above and +50% below."
    ])

    print("To change these values:")
    print('  1) Open file "config/hrnet_config.yaml" in a text editor (for example Notepad).')
    print("  2) Change numbers or true/false values.")
    print("  3) Save the file.")
    print("New training runs will automatically use the new settings.")
    print("Do not change parameter names, only their values.")
    print()
    input("Press Enter to exit...")

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", dest="root", default=None)
    parser.add_argument("--base", dest="base", default=None)
    parser.add_argument("--base-localities", dest="base_localities", default=None)
    args = parser.parse_args()

    landmark_root = get_landmark_root(args.root)

    base_localities: Optional[Path] = None
    base_arg = args.base_localities or args.base
    if base_arg:
        base_localities = Path(base_arg)

    localities = load_localities(landmark_root)

    print("=== GM Landmarking: HRNet Trainer (v1.0) ===")
    print()
    print("1) Train / Finetune model on MANUAL localities")
    print("2) Autolabel locality with current model")
    print("3) Review AUTO locality in annotator (set MANUAL by button)")
    print("4) Info about current model / metrics")
    print("5) Model settings")
    print()
    print("0) Quit")
    print()

    choice = input("Select action: ").strip()

    # После выбора действия показываем список локальностей (для информации)
    print_localities_block(localities)

    if choice == "1":
        run_training(landmark_root)
        print()
        input("Press Enter to exit...")
    elif choice == "2":
        run_autolabel(landmark_root, base_localities)
    elif choice == "3":
        run_review_auto(landmark_root)
    elif choice == "4":
        show_model_info(landmark_root)
    elif choice == "5":
        show_model_settings(landmark_root)
    else:
        # 0 или что-то другое – просто выходим
        return


if __name__ == "__main__":
    main()

