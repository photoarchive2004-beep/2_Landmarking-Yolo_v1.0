from __future__ import annotations

import argparse
import csv
import json
import os
import random
import shutil
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import math

import numpy as np
from PIL import Image

try:
    import yaml  # type: ignore
except Exception:  # pragma: no cover
    yaml = None

try:
    import torch  # type: ignore
except Exception:  # pragma: no cover
    torch = None

try:
    from ultralytics import YOLO  # type: ignore
except Exception:  # pragma: no cover
    YOLO = None


STATUS_FILE = "status/localities_status.csv"
YOLO_CONFIG_FILE = "config/yolo_config.yaml"
LOGS_DIR = "logs"
TRAIN_LOG_NAME = "train_yolo_last.log"


# -----------------------------
# Логирование
# -----------------------------
def _open_log(root: Path) -> Tuple[Any, Path]:
    logs_dir = root / LOGS_DIR
    logs_dir.mkdir(parents=True, exist_ok=True)
    log_path = logs_dir / TRAIN_LOG_NAME
    f = log_path.open("w", encoding="utf-8")
    return f, log_path


def _log(msg: str, log_file: Any) -> None:
    ts = time.strftime("%Y-%m-%d %H:%M:%S")
    line = f"[{ts}] {msg}"
    print(line)
    try:
        log_file.write(line + "\n")
        log_file.flush()
    except Exception:
        pass


# -----------------------------
# Конфиг YOLO
# -----------------------------
@dataclass
class YoloResizeConfig:
    enabled: bool = True
    long_side: int = 1280
    stride_multiple: int = 32


@dataclass
class YoloTrainConfig:
    train_val_split: float = 0.9
    max_epochs: int = 100
    batch_size: int = 8
    learning_rate: float = 5e-4
    weight_decay: float = 1e-4
    early_stop_patience: int = 10


@dataclass
class YoloModelConfig:
    pretrained_weights: str = "yolov8m-pose.pt"
    num_keypoints: int = 0
    class_names: List[str] = None  # type: ignore[assignment]


@dataclass
class YoloConfig:
    model: YoloModelConfig
    resize: YoloResizeConfig
    train: YoloTrainConfig
    augment: Dict[str, Any]
    infer: Dict[str, Any]


def _read_lm_number(root: Path) -> int:
    lm_path = root / "LM_number.txt"
    if not lm_path.is_file():
        return 0
    try:
        text = lm_path.read_text(encoding="utf-8").strip()
        return int(text)
    except Exception:
        return 0


def _load_yolo_config(root: Path, log_file: Any) -> YoloConfig:
    cfg_path = root / YOLO_CONFIG_FILE
    if yaml is None:
        raise RuntimeError("PyYAML is not installed, cannot read yolo_config.yaml")

    if not cfg_path.is_file():
        raise RuntimeError(f"YOLO config not found: {cfg_path}")

    with cfg_path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}

    if not isinstance(data, dict):
        raise RuntimeError("Invalid structure in yolo_config.yaml")

    model_data = data.get("model", {}) or {}
    resize_data = data.get("resize", {}) or {}
    train_data = data.get("train", {}) or {}
    augment_data = data.get("augment", {}) or {}
    infer_data = data.get("infer", {}) or {}

    # Число ключевых точек: сначала LM_number.txt, затем model.num_keypoints
    n_kpts_from_file = int(model_data.get("num_keypoints") or 0)
    n_kpts_from_lm = _read_lm_number(root)
    num_keypoints = n_kpts_from_lm or n_kpts_from_file

    if num_keypoints <= 0:
        raise RuntimeError(
            "Number of keypoints is not defined. "
            "Please set LM_number.txt and/or model.num_keypoints in config/yolo_config.yaml"
        )

    model_cfg = YoloModelConfig(
        pretrained_weights=str(model_data.get("pretrained_weights", "yolov8m-pose.pt")),
        num_keypoints=num_keypoints,
        class_names=list(model_data.get("class_names", ["class"])),
    )

    resize_cfg = YoloResizeConfig(
        enabled=bool(resize_data.get("enabled", True)),
        long_side=int(resize_data.get("long_side", 1280)),
        stride_multiple=int(resize_data.get("stride_multiple", 32)),
    )

    train_cfg = YoloTrainConfig(
        train_val_split=float(train_data.get("train_val_split", 0.9)),
        max_epochs=int(train_data.get("max_epochs", 100)),
        batch_size=int(train_data.get("batch_size", 8)),
        learning_rate=float(train_data.get("learning_rate", 5e-4)),
        weight_decay=float(train_data.get("weight_decay", 1e-4)),
        early_stop_patience=int(train_data.get("early_stop_patience", 10)),
    )

    _log(
        f"YOLO config: num_keypoints={model_cfg.num_keypoints}, "
        f"pretrained_weights={model_cfg.pretrained_weights}, "
        f"resize.enabled={resize_cfg.enabled}, resize.long_side={resize_cfg.long_side}, "
        f"train.max_epochs={train_cfg.max_epochs}, train.batch_size={train_cfg.batch_size}, "
        f"train.early_stop_patience={train_cfg.early_stop_patience}",
        log_file,
    )

    return YoloConfig(
        model=model_cfg,
        resize=resize_cfg,
        train=train_cfg,
        augment=augment_data,
        infer=infer_data,
    )

# -----------------------------
# Чтение статусов и выбор MANUAL
# -----------------------------
@dataclass
class Sample:
    img_path: Path
    csv_path: Path
    locality: str


def _load_manual_samples(
    root: Path,
    base: Path,
    log_file: Any,
) -> Tuple[List[Sample], int]:
    status_path = root / STATUS_FILE
    if not status_path.is_file():
        raise RuntimeError(f"Status file not found: {status_path}")

    manual_localities: List[str] = []
    with status_path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            loc = (row.get("locality") or "").strip()
            status = (row.get("status") or "").strip().upper()
            if status == "MANUAL" and loc:
                manual_localities.append(loc)

    manual_localities = sorted(set(manual_localities))

    if not manual_localities:
        _log("No MANUAL localities in localities_status.csv. Nothing to train.", log_file)
        return [], 0

    _log(f"MANUAL localities: {', '.join(manual_localities)}", log_file)

    samples: List[Sample] = []
    for loc in manual_localities:
        png_dir = base / loc / "png"
        if not png_dir.is_dir():
            _log(f"[WARN] png folder not found for locality: {png_dir}", log_file)
            continue

        img_files: List[Path] = []
        for ext in ("*.png", "*.jpg", "*.jpeg", "*.tif", "*.tiff"):
            img_files.extend(sorted(png_dir.glob(ext)))

        n_valid = 0
        for img_path in img_files:
            csv_path = img_path.with_suffix(".csv")
            if not csv_path.is_file():
                continue
            samples.append(Sample(img_path=img_path, csv_path=csv_path, locality=loc))
            n_valid += 1

        _log(f"Locality {loc}: {n_valid} labeled images", log_file)

    _log(f"Total labeled images across MANUAL localities: {len(samples)}", log_file)
    return samples, len(manual_localities)


# -----------------------------
# Подготовка датасета YOLO
# -----------------------------
def _read_keypoints_from_csv(csv_path: Path, num_kpts: int) -> Optional[np.ndarray]:
    with csv_path.open("r", encoding="utf-8") as f:
        reader = csv.reader(f)
        row = next(reader, None)
        if row is None:
            return None
        try:
            vals = [float(x) for x in row]
        except Exception:
            return None

    if len(vals) != 2 * num_kpts:
        return None

    pts = np.array(vals, dtype=np.float32).reshape(num_kpts, 2)
    return pts


def _resize_and_pad_image(
    img: Image.Image,
    resize_cfg: YoloResizeConfig,
) -> Tuple[Image.Image, float, int, int]:
    """
    Возвращает (новое_изображение, scale, new_w, new_h).
    Если ресайз выключен — scale=1.0, размеры исходные.
    Паддинг делаем вправо и вниз.
    """
    w, h = img.size
    if not resize_cfg.enabled:
        return img, 1.0, w, h

    long_side = max(w, h)
    if long_side <= 0:
        return img, 1.0, w, h

    scale = float(resize_cfg.long_side) / float(long_side)
    new_w = int(round(w * scale))
    new_h = int(round(h * scale))

    # Паддинг до кратности stride_multiple
    stride = max(1, int(resize_cfg.stride_multiple))
    pad_w = int(math.ceil(new_w / stride) * stride)
    pad_h = int(math.ceil(new_h / stride) * stride)

    img_resized = img.resize((new_w, new_h), Image.BILINEAR)
    canvas = Image.new("RGB", (pad_w, pad_h))
    canvas.paste(img_resized, (0, 0))

    return canvas, scale, pad_w, pad_h


def _create_yolo_dataset(
    root: Path,
    samples: List[Sample],
    cfg: YoloConfig,
    log_file: Any,
) -> Tuple[Path, str, int, int]:
    """
    Строит датасет YOLO-pose в datasets/yolo/<run_id>/...
    Возвращает (dataset_root, run_id, n_train, n_val).
    """
    if not samples:
        raise RuntimeError("Empty sample list")

    run_id = time.strftime("%Y%m%d_%H%M%S")
    ds_root = root / "datasets" / "yolo" / run_id

    img_train = ds_root / "images" / "train"
    img_val = ds_root / "images" / "val"
    lbl_train = ds_root / "labels" / "train"
    lbl_val = ds_root / "labels" / "val"

    for p in (img_train, img_val, lbl_train, lbl_val):
        p.mkdir(parents=True, exist_ok=True)

    rng = random.Random(42)
    indices = list(range(len(samples)))
    rng.shuffle(indices)

    split = cfg.train.train_val_split
    split = max(0.0, min(1.0, split))
    n_total = len(indices)
    n_train = int(round(n_total * split))
    n_train = max(1, min(n_total - 1, n_train)) if n_total > 1 else n_total
    train_idx = set(indices[:n_train])

    num_kpts = cfg.model.num_keypoints

    def process_one(idx: int) -> Optional[str]:
        s = samples[idx]
        try:
            img = Image.open(s.img_path).convert("RGB")
        except Exception:
            _log(f"[WARN] Cannot open image: {s.img_path}", log_file)
            return None

        kps = _read_keypoints_from_csv(s.csv_path, num_kpts)
        if kps is None:
            _log(f"[WARN] Invalid CSV for image: {s.csv_path}", log_file)
            return None

        img_proc, scale, W2, H2 = _resize_and_pad_image(img, cfg.resize)

        # Пересчёт keypoints в нормированные координаты YOLO
        # bbox = весь кадр
        x_center = 0.5
        y_center = 0.5
        width = 1.0
        height = 1.0

        kpt_triplets: List[float] = []
        for i in range(num_kpts):
            x, y = float(kps[i, 0]), float(kps[i, 1])
            if x <= 0.0 and y <= 0.0:
                # невидимая точка
                kpt_triplets.extend([0.0, 0.0, 0.0])
            else:
                x_scaled = x * scale
                y_scaled = y * scale
                x_norm = x_scaled / float(max(1, W2))
                y_norm = y_scaled / float(max(1, H2))
                kpt_triplets.extend([x_norm, y_norm, 2.0])

        # Сохраняем изображение и метку
        subset = "train" if idx in train_idx else "val"
        if subset == "train":
            img_dir = img_train
            lbl_dir = lbl_train
        else:
            img_dir = img_val
            lbl_dir = lbl_val

        stem = s.img_path.stem
        out_img_path = img_dir / (stem + s.img_path.suffix)
        out_lbl_path = lbl_dir / (stem + ".txt")

        try:
            img_proc.save(out_img_path)
        except Exception:
            _log(f"[WARN] Cannot save processed image: {out_img_path}", log_file)
            return None

        values: List[float] = [0.0, x_center, y_center, width, height]
        values.extend(kpt_triplets)
        line = " ".join(f"{v:.6f}" if isinstance(v, float) else str(v) for v in values)

        with out_lbl_path.open("w", encoding="utf-8") as f:
            f.write(line + "\n")

        return subset

    n_train_final = 0
    n_val_final = 0
    for idx in indices:
        subset = process_one(idx)
        if subset == "train":
            n_train_final += 1
        elif subset == "val":
            n_val_final += 1

    if n_train_final == 0 or n_val_final == 0:
        raise RuntimeError(
            f"Invalid train/val split after filtering: train={n_train_final}, val={n_val_final}"
        )

    _log(
        f"YOLO dataset built at {ds_root} (train={n_train_final}, val={n_val_final})",
        log_file,
    )

    # dataset.yaml
    ds_yaml = ds_root / "dataset.yaml"
    names = cfg.model.class_names or ["class"]
    data_yaml = {
        "path": str(ds_root),
        "train": "images/train",
        "val": "images/val",
        "names": {0: str(names[0])},
        "kpt_shape": [cfg.model.num_keypoints, 3],
    }
    if yaml is not None:
        with ds_yaml.open("w", encoding="utf-8") as f:
            yaml.safe_dump(data_yaml, f, allow_unicode=True)
    else:
        with ds_yaml.open("w", encoding="utf-8") as f:
            json.dump(data_yaml, f, indent=2, ensure_ascii=False)

    return ds_root, run_id, n_train_final, n_val_final


# -----------------------------
# PCK@R
# -----------------------------
def compute_pck_at_r(
    pred: "torch.Tensor",  # (N, K, 2)
    gt: "torch.Tensor",  # (N, K, 2)
    R: float,
) -> float:
    """
    PCK@R: доля точек, попавших в радиус R от разметки.
    Точки с gt <= (0,0) игнорируем.
    """
    mask = (gt[..., 0] > 0) | (gt[..., 1] > 0)
    if mask.sum().item() == 0:
        return 0.0
    dists = torch.norm(pred - gt, dim=2)
    correct = (dists <= R) & mask
    return float(correct.sum().item()) / float(mask.sum().item())


def _evaluate_pck(
    model_path: Path,
    val_samples: List[Sample],
    cfg: YoloConfig,
    log_file: Any,
) -> float:
    """
    Запускает инференс YOLO-позы на валидационных изображениях и считает PCK@R.
    Для упрощения считаем R = 5% от long_side (как в HRNet через input_size).
    """
    if torch is None:
        _log("[WARN] PyTorch is not installed, cannot compute PCK@R.", log_file)
        return 0.0
    if YOLO is None:
        _log("[WARN] Ultralytics YOLO is not installed, cannot compute PCK@R.", log_file)
        return 0.0

    device = "cuda" if torch.cuda.is_available() else "cpu"
    _log(f"Loading YOLO model for evaluation on device: {device}", log_file)

    model = YOLO(str(model_path))

    num_kpts = cfg.model.num_keypoints
    all_pred: List[torch.Tensor] = []
    all_gt: List[torch.Tensor] = []

    for s in val_samples:
        try:
            img = Image.open(s.img_path).convert("RGB")
        except Exception:
            _log(f"[WARN] Cannot open image during eval: {s.img_path}", log_file)
            continue

        w, h = img.size
        kps = _read_keypoints_from_csv(s.csv_path, num_kpts)
        if kps is None:
            _log(f"[WARN] Invalid CSV during eval: {s.csv_path}", log_file)
            continue

        # GT в пикселях исходного изображения
        gt_arr = np.zeros((num_kpts, 2), dtype=np.float32)
        for i in range(num_kpts):
            x, y = float(kps[i, 0]), float(kps[i, 1])
            if x <= 0.0 and y <= 0.0:
                gt_arr[i, 0] = 0.0
                gt_arr[i, 1] = 0.0
            else:
                gt_arr[i, 0] = x
                gt_arr[i, 1] = y

        try:
            results = model.predict(
                source=str(s.img_path),
                verbose=False,
                imgsz=cfg.resize.long_side if cfg.resize.enabled else max(w, h),
                device=device,
            )
        except Exception as exc:
            _log(f"[WARN] YOLO predict failed for {s.img_path}: {exc}", log_file)
            continue

        if not results:
            _log(f"[WARN] Empty YOLO result for {s.img_path}", log_file)
            continue

        r = results[0]
        if getattr(r, "keypoints", None) is None:
            _log(f"[WARN] No keypoints in YOLO result for {s.img_path}", log_file)
            continue

        try:
            # Нормированные координаты (0..1), берём первую детекцию
            kpn = r.keypoints.xyn[0].cpu().numpy()  # type: ignore[attr-defined]
        except Exception:
            _log(f"[WARN] Cannot extract keypoints.xyn for {s.img_path}", log_file)
            continue

        if kpn.shape[0] != num_kpts:
            _log(
                f"[WARN] Keypoints count mismatch for {s.img_path}: "
                f"pred={kpn.shape[0]}, expected={num_kpts}",
                log_file,
            )
            continue

        pred_arr = np.zeros((num_kpts, 2), dtype=np.float32)
        for i in range(num_kpts):
            x_n, y_n = float(kpn[i, 0]), float(kpn[i, 1])
            pred_arr[i, 0] = x_n * float(w)
            pred_arr[i, 1] = y_n * float(h)

        gt_t = torch.from_numpy(gt_arr).float().unsqueeze(0)
        pred_t = torch.from_numpy(pred_arr).float().unsqueeze(0)

        all_gt.append(gt_t)
        all_pred.append(pred_t)

    if not all_gt or not all_pred:
        _log("[WARN] No valid samples for PCK@R evaluation.", log_file)
        return 0.0

    gt_cat = torch.cat(all_gt, dim=0)
    pred_cat = torch.cat(all_pred, dim=0)

    # R как 5% от long_side (аналогично HRNet input_size)
    if cfg.resize.enabled and cfg.resize.long_side > 0:
        R = float(max(1, int(round(min(cfg.resize.long_side, cfg.resize.long_side) * 0.05))))
    else:
        R = 10.0

    pck = compute_pck_at_r(pred_cat, gt_cat, R)
    _log(f"Validation PCK@R = {pck:.4f}", log_file)
    return float(pck)


# -----------------------------
# Основная функция обучения
# -----------------------------
def train_yolo(root: Path, base: Path) -> int:
    log_file, log_path = _open_log(root)
    _log(f"LM_ROOT = {root}", log_file)
    _log(f"Base localities = {base}", log_file)

    try:
        cfg = _load_yolo_config(root, log_file)
        samples, n_manual_localities = _load_manual_samples(root, base, log_file)

        if not samples:
            _log("No labeled images in MANUAL localities. Abort training.", log_file)
            log_file.close()
            return 0

        ds_root, run_id, n_train, n_val = _create_yolo_dataset(root, samples, cfg, log_file)

        if YOLO is None:
            raise RuntimeError("Ultralytics YOLO is not installed.")

        if torch is None:
            raise RuntimeError("PyTorch is not installed.")

        # Обучение YOLO-позы
        history_root = root / "models" / "history"
        history_root.mkdir(parents=True, exist_ok=True)
        history_run_dir = history_root / run_id

        device = "cuda" if torch.cuda.is_available() else "cpu"

        # Если есть уже обученная модель в models/current/yolo_best.pt,
        # продолжаем обучение от неё, иначе используем базовые веса из конфига.
        current_best = root / "models" / "current" / "yolo_best.pt"
        if current_best.is_file():
            pretrained_path = current_best
            _log(f"Found existing model in models/current: {pretrained_path}", log_file)
        else:
            pretrained_path = Path(cfg.model.pretrained_weights)
            if not pretrained_path.is_absolute():
                pretrained_path = root / pretrained_path
            _log(f"Using pretrained weights from config: {pretrained_path}", log_file)

        _log(f"Using device: {device}", log_file)

        model = YOLO(str(pretrained_path))

        train_args: Dict[str, Any] = {
            "data": str(ds_root / "dataset.yaml"),
            "epochs": int(cfg.train.max_epochs),
            "imgsz": int(cfg.resize.long_side if cfg.resize.enabled else max(640, cfg.resize.long_side)),
            "batch": int(cfg.train.batch_size),
            "lr0": float(cfg.train.learning_rate),
            "weight_decay": float(cfg.train.weight_decay),
            "project": str(history_root),
            "name": run_id,
            "exist_ok": True,
            "verbose": True,
            "device": device,
        }

        if getattr(cfg.train, "early_stop_patience", 0) > 0:
            train_args["patience"] = int(cfg.train.early_stop_patience)

        _log(f"Starting YOLO training with args: {train_args}", log_file)
        try:
            model.train(**train_args)
        except Exception as exc:
            # Если попытались тренировать на CUDA и что-то пошло не так — пробуем ещё раз на CPU.
            if device == "cuda":
                _log(f"[WARN] YOLO training on CUDA failed: {exc}. Falling back to CPU.", log_file)
                device = "cpu"
                train_args["device"] = device
                _log(f"Restarting YOLO training on device: {device}", log_file)
                try:
                    model.train(**train_args)
                except Exception as exc2:
                    _log(f"[ERR] YOLO training failed on CPU: {exc2}", log_file)
                    log_file.close()
                    return 1
            else:
                _log(f"[ERR] YOLO training failed: {exc}", log_file)
                log_file.close()
                return 1

        # Ищем лучший чекпоинт
        weights_dir = history_run_dir / "weights"
        best_src = weights_dir / "best.pt"
        if not best_src.is_file():
            # fallback: просто последний
            last_src = weights_dir / "last.pt"
            if last_src.is_file():
                best_src = last_src
            else:
                _log(f"[ERR] best.pt not found in {weights_dir}", log_file)
                log_file.close()
                return 1

        yolo_best_hist = history_run_dir / "yolo_best.pt"
        yolo_best_hist.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(best_src, yolo_best_hist)

        models_current = root / "models" / "current"
        models_current.mkdir(parents=True, exist_ok=True)
        yolo_best_curr = models_current / "yolo_best.pt"
        shutil.copy2(best_src, yolo_best_curr)

        # Считаем PCK@R по валидации
        # Отбираем те сэмплы, которые попали во val (по структуре датасета)
        val_samples: List[Sample] = []
        for s in samples:
            stem = s.img_path.stem
            # Если есть label в labels/val -> это валидация
            lbl_path = ds_root / "labels" / "val" / f"{stem}.txt"
            if lbl_path.is_file():
                val_samples.append(s)

        pck = _evaluate_pck(yolo_best_curr, val_samples, cfg, log_file)
        pck_percent = float(round(pck * 100.0))

        train_share = float(n_train) / float(max(1, n_train + n_val))
        val_share = float(n_val) / float(max(1, n_train + n_val))

        # metrics.json
        metrics: Dict[str, Any] = {
            "run_id": run_id,
            "model_type": "yolo_pose",
            "num_keypoints": cfg.model.num_keypoints,
            "n_manual_localities": n_manual_localities,
            "n_train_images": n_train,
            "n_val_images": n_val,
            "train_share": train_share,
            "val_share": val_share,
            "pck_r": float(pck),
            "pck_r_percent": float(pck_percent),
            "resize_enabled": bool(cfg.resize.enabled),
            "resize_long_side": int(cfg.resize.long_side),
        }
        metrics_path = history_run_dir / "metrics.json"
        with metrics_path.open("w", encoding="utf-8") as f:
            json.dump(metrics, f, indent=2, ensure_ascii=False)

        # train_config.yaml
        train_cfg_path = history_run_dir / "train_config.yaml"
        cfg_dict = {
            "model": {
                "pretrained_weights": cfg.model.pretrained_weights,
                "num_keypoints": cfg.model.num_keypoints,
                "class_names": cfg.model.class_names,
            },
            "resize": {
                "enabled": cfg.resize.enabled,
                "long_side": cfg.resize.long_side,
                "stride_multiple": cfg.resize.stride_multiple,
            },
            "train": {
                "train_val_split": cfg.train.train_val_split,
                "max_epochs": cfg.train.max_epochs,
                "batch_size": cfg.train.batch_size,
                "learning_rate": cfg.train.learning_rate,
                "weight_decay": cfg.train.weight_decay,
            },
            "augment": cfg.augment,
            "infer": cfg.infer,
        }
        try:
            if yaml is not None:
                with train_cfg_path.open("w", encoding="utf-8") as f:
                    yaml.safe_dump(cfg_dict, f, allow_unicode=True)
            else:
                with train_cfg_path.open("w", encoding="utf-8") as f:
                    json.dump(cfg_dict, f, indent=2, ensure_ascii=False)
        except Exception:
            pass

        # train_log.txt
        history_log = history_run_dir / "train_log.txt"
        try:
            shutil.copy2(log_path, history_log)
        except Exception:
            pass

        # quality.json в models/current
        quality: Dict[str, Any] = {
            "run_id": run_id,
            "model_type": "yolo_pose",
            "num_keypoints": cfg.model.num_keypoints,
            "n_manual_localities": n_manual_localities,
            "n_train_images": n_train,
            "n_val_images": n_val,
            "train_share": train_share,
            "val_share": val_share,
            "pck_r_percent": float(pck_percent),
            "resize_enabled": bool(cfg.resize.enabled),
            "resize_long_side": int(cfg.resize.long_side),
        }
        quality_path = models_current / "quality.json"
        with quality_path.open("w", encoding="utf-8") as f:
            json.dump(quality, f, indent=2, ensure_ascii=False)

        # Краткий отчёт в консоль
        print()
        print(f"Used MANUAL localities: {n_manual_localities}")
        train_pct = int(round(train_share * 100.0))
        val_pct = int(round(val_share * 100.0))
        print(f"Train images: {n_train} ({train_pct} %)")
        print(f"Val images:   {n_val} ({val_pct} %)")
        print()
        print(f"PCK@R (validation): {int(round(pck_percent))} %")
        print()
        print("Model saved as: models/current/yolo_best.pt")
        print(f"Run id: {run_id}")

        log_file.close()
        return 0

    except Exception as exc:
        _log(f"[ERR] Unexpected error in train_yolo: {exc}", log_file)
        log_file.close()
        return 1


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description="Train / finetune YOLO pose model on MANUAL localities (GM Landmarking)."
    )
    parser.add_argument(
        "--root",
        type=str,
        help="LM_ROOT (folder with 1_ANNOTATOR.bat, 2_TRAIN-INFER_YOLO.bat)",
    )
    parser.add_argument(
        "--base",
        type=str,
        help="Base localities folder (<base_localities>)",
    )
    args = parser.parse_args(argv)

    root_env = os.environ.get("LM_ROOT")
    base_env = os.environ.get("BASE_LOCALITIES")

    root_str = args.root or root_env
    base_str = args.base or base_env

    if not root_str or not base_str:
        print("[ERR] Both --root and --base (or env LM_ROOT/BASE_LOCALITIES) must be provided.")
        return 1

    root = Path(root_str).resolve()
    base = Path(base_str).resolve()
    return train_yolo(root, base)


if __name__ == "__main__":
    raise SystemExit(main())


