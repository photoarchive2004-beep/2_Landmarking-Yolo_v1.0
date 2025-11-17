from __future__ import annotations

import argparse
import csv
import json
import os
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List

try:
    import torch  # type: ignore
except Exception:  # pragma: no cover
    torch = None  # type: ignore

try:
    from ultralytics import YOLO  # type: ignore
except Exception:  # pragma: no cover
    YOLO = None  # type: ignore

try:
    from PIL import Image  # type: ignore
except Exception:  # pragma: no cover
    Image = None  # type: ignore

try:
    import yaml  # type: ignore
except Exception:  # pragma: no cover
    yaml = None  # type: ignore


STATUS_FILE = "status/localities_status.csv"
QUALITY_FILE = "models/current/quality.json"
YOLO_CONFIG = "config/yolo_config.yaml"


def _log(msg: str, fh) -> None:
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    line = f"[{ts}] {msg}"
    print(line)
    if fh is not None:
        fh.write(line + "\n")
        fh.flush()


def _get_root(arg_root: str | None) -> Path:
    if arg_root:
        return Path(arg_root).resolve()
    return Path(__file__).resolve().parent.parent


def _read_lm_number(root: Path) -> int:
    """Читаем LM_number.txt; если нет/ошибка — возвращаем 0."""
    lm_path = root / "LM_number.txt"
    if not lm_path.is_file():
        return 0
    try:
        text = lm_path.read_text(encoding="utf-8").strip()
        return int(text)
    except Exception:
        return 0


def _load_yolo_config(root: Path) -> Dict[str, Any]:
    cfg_path = root / YOLO_CONFIG
    if yaml is None:
        raise RuntimeError("PyYAML is not installed, cannot read yolo_config.yaml")
    if not cfg_path.is_file():
        raise RuntimeError(f"YOLO config not found: {cfg_path}")
    with cfg_path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    if not isinstance(data, dict):
        raise RuntimeError("Invalid structure in yolo_config.yaml")
    return data


def _load_quality(root: Path) -> Dict[str, Any]:
    q_path = root / QUALITY_FILE
    if not q_path.is_file():
        return {}
    try:
        with q_path.open("r", encoding="utf-8") as f:
            data = json.load(f)
        return data if isinstance(data, dict) else {}
    except Exception:
        return {}


def _list_images(locality_dir: Path) -> List[Path]:
    exts = (".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp")
    files: List[Path] = []
    if not locality_dir.is_dir():
        return files
    for p in sorted(locality_dir.iterdir()):
        if p.suffix.lower() in exts:
            files.append(p)
    return files


def infer_locality(root: Path, base_localities: Path, locality_name: str) -> int:
    logs_dir = root / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)
    log_file_path = logs_dir / "infer_yolo_last.log"
    log_file = log_file_path.open("w", encoding="utf-8")

    try:
        _log(f"YOLO autolabel start for locality: {locality_name}", log_file)
        _log(f"LM_ROOT={root}", log_file)
        _log(f"BASE_LOCALITIES={base_localities}", log_file)

        if YOLO is None:
            _log("Ultralytics YOLO is not installed.", log_file)
            return 1
        if torch is None:
            _log("PyTorch is not installed.", log_file)
            return 1
        if Image is None:
            _log("Pillow (PIL) is not installed.", log_file)
            return 1

        # Загружаем конфиг YOLO
        cfg = _load_yolo_config(root)
        model_cfg = cfg.get("model", {}) or {}
        resize_cfg = cfg.get("resize", {}) or {}
        infer_cfg = cfg.get("infer", {}) or {}

        # Число ключевых точек: сначала LM_number.txt, затем num_keypoints из конфига
        n_kpts_from_lm = _read_lm_number(root)
        try:
            n_kpts_from_cfg = int(model_cfg.get("num_keypoints") or 0)
        except Exception:
            n_kpts_from_cfg = 0
        num_keypoints = n_kpts_from_lm or n_kpts_from_cfg
        if num_keypoints <= 0:
            _log(
                "Number of keypoints is not defined. "
                "Please set LM_number.txt and/or model.num_keypoints in config/yolo_config.yaml.",
                log_file,
            )
            return 1

        # Размер для инференса: long_side из секции resize
        try:
            img_long_side = int(resize_cfg.get("long_side", 1280) or 1280)
        except Exception:
            img_long_side = 1280
        if img_long_side < 640:
            img_long_side = 640

        quality = _load_quality(root)
        run_id = str(quality.get("run_id", ""))
        pck_percent = quality.get("pck_r_percent", None)
        if isinstance(pck_percent, (int, float)):
            auto_quality = int(round(float(pck_percent)))
        else:
            auto_quality = ""

        model_path = root / "models" / "current" / "yolo_best.pt"
        if not model_path.is_file():
            _log(f"Model file not found: {model_path}", log_file)
            return 1

        device = "cuda" if (torch is not None and torch.cuda.is_available()) else "cpu"
        _log(f"Loading YOLO model from {model_path}", log_file)
        _log(f"Using device: {device}", log_file)
        _log(f"Inference long_side (imgsz) = {img_long_side}", log_file)

        model = YOLO(str(model_path))

        conf_thres = float(infer_cfg.get("conf_threshold", 0.25))
        iou_thres = float(infer_cfg.get("iou_threshold", 0.5))
        _log(f"Inference parameters: conf={conf_thres}, iou={iou_thres}", log_file)

        locality_dir = base_localities / locality_name / "png"
        if not locality_dir.is_dir():
            _log(f"Locality images folder not found: {locality_dir}", log_file)
            return 1

        image_paths = _list_images(locality_dir)
        if not image_paths:
            _log("No images found for this locality.", log_file)
            return 1

        _log(f"Found {len(image_paths)} images for locality.", log_file)

        n_labeled = 0
        for img_path in image_paths:
            _log(f"Processing image: {img_path.name}", log_file)
            try:
                results = model(
                    str(img_path),
                    device=device,
                    conf=conf_thres,
                    iou=iou_thres,
                    imgsz=img_long_side,
                    verbose=False,
                )
            except Exception as exc:
                _log(f"[WARN] Inference failed for {img_path.name}: {exc}", log_file)
                continue

            if not results:
                _log(f"[WARN] No results returned for {img_path.name}", log_file)
                continue

            r = results[0]
            if r.keypoints is None or r.keypoints.xy is None:
                _log(f"[WARN] No keypoints predicted for {img_path.name}", log_file)
                continue

            k_xy = r.keypoints.xy
            if k_xy is None or len(k_xy) == 0:
                _log(f"[WARN] Empty keypoints tensor for {img_path.name}", log_file)
                continue

            pts = k_xy[0]
            if pts.shape[0] != num_keypoints:
                _log(
                    f"[WARN] Predicted {pts.shape[0]} keypoints, "
                    f"but expected {num_keypoints}. Will truncate/pad with -1.",
                    log_file,
                )

            coords: List[float] = []
            for i in range(num_keypoints):
                if i < pts.shape[0]:
                    x = float(pts[i, 0])
                    y = float(pts[i, 1])
                    if x != x or y != y:  # NaN check
                        x, y = -1.0, -1.0
                else:
                    x, y = -1.0, -1.0
                coords.extend([x, y])

            out_csv = img_path.with_suffix(".csv")
            row = ",".join(f"{v:.3f}" if v != -1.0 else "-1" for v in coords)
            out_csv.write_text(row + "\n", encoding="utf-8")
            n_labeled += 1
            _log(f"Wrote landmarks CSV: {out_csv.name}", log_file)

        n_images = len(image_paths)
        _log(f"Labeled {n_labeled}/{n_images} images for locality '{locality_name}'.", log_file)

        status_path = root / STATUS_FILE
        if status_path.exists():
            rows: List[Dict[str, Any]] = []
            with status_path.open("r", encoding="utf-8", newline="") as f:
                reader = csv.DictReader(f)
                fieldnames = reader.fieldnames or []
                for row in reader:
                    if row.get("locality", "").strip() == locality_name:
                        row["status"] = "AUTO"
                        row["auto_quality"] = str(auto_quality) if auto_quality != "" else ""
                        row["last_model_run"] = run_id
                        row["last_update"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                        row["n_images"] = str(n_images)
                        row["n_labeled"] = str(n_labeled)
                    rows.append(row)

            if fieldnames:
                with status_path.open("w", encoding="utf-8", newline="") as f:
                    writer = csv.DictWriter(f, fieldnames=fieldnames)
                    writer.writeheader()
                    for row in rows:
                        writer.writerow(row)
            _log("Updated status/localities_status.csv for this locality.", log_file)
        else:
            _log("status/localities_status.csv not found, cannot update status.", log_file)

        _log("YOLO autolabel finished.", log_file)
        return 0

    finally:
        try:
            log_file.close()
        except Exception:
            pass


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", dest="root", default=None)
    parser.add_argument("--base", dest="base", default=None)
    parser.add_argument("--locality", dest="locality", required=True)
    args = parser.parse_args()

    root = _get_root(args.root)
    if args.base:
        base_localities = Path(args.base)
    else:
        cfg_dir = root / "cfg"
        last_base = cfg_dir / "last_base.txt"
        if last_base.exists():
            base_localities = Path(last_base.read_text(encoding="utf-8").strip())
        else:
            base_localities = root

    return infer_locality(root, base_localities, args.locality)


if __name__ == "__main__":
    exit_code = main()
    raise SystemExit(exit_code)
