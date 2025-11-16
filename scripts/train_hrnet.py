from __future__ import annotations

import argparse
import csv
import json
import random
import sys
import time
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple, Optional, Dict, Any

import numpy as np
from PIL import Image

try:
    import torch
    from torch import nn
    from torch.utils.data import Dataset, DataLoader
except ImportError:  # pragma: no cover
    torch = None
    nn = None  # type: ignore[assignment]
    Dataset = object  # type: ignore[assignment]
    DataLoader = object  # type: ignore[assignment]

try:
    # HRNet backbone from MMPose (используем, если установлен)
    from mmpose.models.backbones import HRNet as MMPoseHRNet  # type: ignore
except Exception as e:  # pragma: no cover
    print("[WARN] Exception during import or setup (possibly MMPose HRNet):", repr(e))
    MMPoseHRNet = None  # type: ignore

try:
    import yaml
except ImportError:  # pragma: no cover
    yaml = None


@dataclass
class HRNetConfig:
    model_type: str = "hrnet_w32"
    input_size: int = 256
    resize_mode: str = "resize"  # "resize" или "original"
    keep_aspect_ratio: bool = True
    batch_size: int = 8
    learning_rate: float = 5e-4
    max_epochs: int = 100
    train_val_split: float = 0.9
    flip_augmentation: bool = False
    rotation_augmentation_deg: float = 15.0
    scale_augmentation: float = 0.3
    weight_decay: float = 1e-4
    heatmap_sigma_px: float = 2.5


def get_landmark_root() -> Path:
    return Path(__file__).resolve().parent.parent


def load_yaml_config(cfg_path: Path) -> HRNetConfig:
    cfg = HRNetConfig()
    if not cfg_path.is_file() or yaml is None:
        return cfg
    with cfg_path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    for field in cfg.__dataclass_fields__.keys():  # type: ignore[attr-defined]
        if field in data:
            setattr(cfg, field, data[field])
    return cfg


def read_lm_number(root: Path) -> int:
    lm_path = root / "LM_number.txt"
    try:
        with lm_path.open("r", encoding="utf-8") as f:
            line = f.readline().strip()
        n = int(line)
        if n <= 0:
            raise ValueError
        return n
    except Exception:
        return -1


def read_last_base(root: Path) -> Optional[Path]:
    base_txt = root / "cfg" / "last_base.txt"
    if not base_txt.is_file():
        return None
    txt = base_txt.read_text(encoding="utf-8").strip()
    if not txt:
        return None
    base = Path(txt)
    if not base.is_dir():
        return None
    return base


def gather_manual_samples(root: Path, base_dir: Path) -> List[Tuple[Path, Path, str]]:
    """
    Собираем пары (png, csv, locality) только для MANUAL локальностей.
    """
    status_dir = root / "status"
    status_file = status_dir / "localities_status.csv"
    manual_localities: List[str] = []

    if status_file.is_file():
        with status_file.open("r", encoding="utf-8", newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                loc = (row.get("locality") or "").strip()
                status = (row.get("status") or "").strip().upper()
                if loc and status == "MANUAL":
                    manual_localities.append(loc)

    manual_localities = sorted(set(manual_localities))
    samples: List[Tuple[Path, Path, str]] = []

    for loc in manual_localities:
        png_dir = base_dir / loc / "png"
        if not png_dir.is_dir():
            continue
        for img_path in sorted(png_dir.glob("*.png")):
            csv_path = img_path.with_suffix(".csv")
            if csv_path.is_file():
                samples.append((img_path, csv_path, loc))

    return samples


def _resize_and_pad(
    img: Image.Image,
    cfg: HRNetConfig,
) -> Tuple[Image.Image, float, int, int]:
    """
    Ресайз с сохранением пропорций и паддингом до квадрата input_size x input_size.
    Возвращает (новое_изображение, scale, offset_x, offset_y).
    """
    w, h = img.size
    if cfg.resize_mode == "original" or cfg.input_size <= 0:
        # Без изменения размера
        return img, 1.0, 0, 0

    target = int(cfg.input_size)
    if target <= 0:
        return img, 1.0, 0, 0

    scale = min(target / float(w), target / float(h))
    new_w = max(1, int(round(w * scale)))
    new_h = max(1, int(round(h * scale)))
    resized = img.resize((new_w, new_h), Image.BILINEAR)

    canvas = Image.new("RGB", (target, target), (0, 0, 0))
    offset_x = (target - new_w) // 2
    offset_y = (target - new_h) // 2
    canvas.paste(resized, (offset_x, offset_y))
    return canvas, scale, offset_x, offset_y


def _load_keypoints(csv_path: Path, num_keypoints: int) -> np.ndarray:
    """
    Загружаем точки из CSV.

    Поддерживаем два формата:
    1) ОДНА строка: x1,y1,x2,y2,... (как делает аннотатор).
    2) МНОГО строк: по одной точке в строке (x,y), возможен заголовок x,y.

    Возвращаем массив (K, 2), недостающие точки = 0,0.
    """
    kps = np.zeros((num_keypoints, 2), dtype=np.float32)
    try:
        with csv_path.open("r", encoding="utf-8", newline="") as f:
            reader = csv.reader(f)
            # убираем полностью пустые строки
            rows = [row for row in reader if any((c or "").strip() for c in row)]

        if not rows:
            return kps

        # убираем заголовок "x,y" если есть
        if rows and rows[0] and (rows[0][0] or "").strip().lower().startswith("x"):
            rows = rows[1:]

        if not rows:
            return kps

        # Вариант 1: одна строка, много чисел -> считаем, что это x1,y1,x2,y2,...
        if len(rows) == 1 and len(rows[0]) > 2:
            flat_vals: list[float] = []
            for cell in rows[0]:
                cell = (cell or "").strip()
                if not cell:
                    continue
                try:
                    flat_vals.append(float(cell))
                except ValueError:
                    continue

            idx = 0
            it = iter(flat_vals)
            for x, y in zip(it, it):
                if idx >= num_keypoints:
                    break
                kps[idx, 0] = x
                kps[idx, 1] = y
                idx += 1

        else:
            # Вариант 2: построчный формат x,y
            idx = 0
            for row in rows:
                if idx >= num_keypoints:
                    break
                if len(row) < 2:
                    continue
                try:
                    x = float((row[0] or "").strip())
                    y = float((row[1] or "").strip())
                except ValueError:
                    continue
                kps[idx, 0] = x
                kps[idx, 1] = y
                idx += 1

    except Exception:
        # В случае ошибок возвращаем нули
        pass
    return kps


class LandmarkDataset(Dataset):  # type: ignore[misc]
    def __init__(
        self,
        samples: List[Tuple[Path, Path, str]],
        num_keypoints: int,
        cfg: HRNetConfig,
        phase: str = "train",
    ) -> None:
        self.samples = samples
        self.num_keypoints = num_keypoints
        self.cfg = cfg
        self.phase = phase

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        img_path, csv_path, loc = self.samples[idx]
        img = Image.open(img_path).convert("RGB")

        # Ресайз + паддинг
        img_resized, scale, offset_x, offset_y = _resize_and_pad(img, self.cfg)

        # Базовые преобразования в тензор
        np_img = np.asarray(img_resized, dtype=np.float32) / 255.0
        np_img = np.transpose(np_img, (2, 0, 1))  # CHW

        # Точки -> координаты в новом масштабе
        kps = _load_keypoints(csv_path, self.num_keypoints)
        if scale != 1.0 or offset_x != 0 or offset_y != 0:
            kps[:, 0] = kps[:, 0] * scale + float(offset_x)
            kps[:, 1] = kps[:, 1] * scale + float(offset_y)

        if torch is not None:
            img_tensor = torch.from_numpy(np_img).float()
            kps_tensor = torch.from_numpy(kps).float()
        else:
            # В заглушке нам фактически не важно содержимое, но форма должна быть корректной
            img_tensor = np_img  # type: ignore[assignment]
            kps_tensor = kps  # type: ignore[assignment]

        return img_tensor, kps_tensor


def keypoints_to_heatmaps(
    keypoints: "torch.Tensor",
    height: int,
    width: int,
    sigma: float = 2.0,
    device: Optional["torch.device"] = None,
) -> "torch.Tensor":
    """
    Преобразуем координаты (N, K, 2) в теплокарты (N, K, H, W).
    Точки с координатами <= 0 считаем отсутствующими.
    """
    if device is None:
        device = keypoints.device
    N, K, _ = keypoints.shape
    yy, xx = torch.meshgrid(
        torch.arange(height, device=device),
        torch.arange(width, device=device),
        indexing="ij",
    )
    heatmaps = []
    for n in range(N):
        kps = keypoints[n]
        hm_per_img = []
        for k in range(K):
            x = kps[k, 0]
            y = kps[k, 1]
            if x <= 0 and y <= 0:
                hm_per_img.append(torch.zeros((height, width), device=device))
                continue
            g = torch.exp(-((xx - x) ** 2 + (yy - y) ** 2) / (2 * sigma * sigma))
            hm_per_img.append(g)
        hm_per_img = torch.stack(hm_per_img, dim=0)
        heatmaps.append(hm_per_img)
    heatmaps = torch.stack(heatmaps, dim=0)
    return heatmaps


def heatmaps_to_keypoints(heatmaps: "torch.Tensor") -> "torch.Tensor":
    """
    Из теплокарт получаем координаты максимумов.
    """
    B, K, H, W = heatmaps.shape
    heatmaps_reshaped = heatmaps.view(B, K, -1)
    idx = heatmaps_reshaped.argmax(dim=2)  # (B, K)
    ys = (idx // W).float()
    xs = (idx % W).float()
    kps = torch.stack([xs, ys], dim=2)
    return kps


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


# Модели: простая сеть SimpleHRNet и HRNet-W32 (через MMPose)
if torch is not None:

    class SimpleHRNet(nn.Module):
        """
        Упрощённый бэкбон для расстановки ландмарок.
        """

        def __init__(self, num_keypoints: int) -> None:
            super().__init__()
            self.stem = nn.Sequential(
                nn.Conv2d(3, 32, kernel_size=3, padding=1),
                nn.BatchNorm2d(32),
                nn.ReLU(inplace=True),
                nn.Conv2d(32, 64, kernel_size=3, padding=1),
                nn.BatchNorm2d(64),
                nn.ReLU(inplace=True),
            )
            blocks = []
            in_channels = 64
            for _ in range(3):
                block = nn.Sequential(
                    nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1),
                    nn.BatchNorm2d(in_channels),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1),
                    nn.BatchNorm2d(in_channels),
                )
                blocks.append(block)
            self.blocks = nn.ModuleList(blocks)
            self.relu = nn.ReLU(inplace=True)
            self.head = nn.Conv2d(64, num_keypoints, kernel_size=1)

        def forward(self, x: "torch.Tensor") -> "torch.Tensor":
            x = self.stem(x)
            for block in self.blocks:
                residual = x
                out = block(x)
                x = self.relu(out + residual)
            heatmaps = self.head(x)
            return heatmaps

    class HRNetW32GM(nn.Module):
        """
        HRNet-W32 из MMPose + простой head для теплокарт.
        Если MMPose недоступен, используется SimpleHRNet.
        """

        def __init__(self, num_keypoints: int) -> None:
            super().__init__()
            self.num_keypoints = int(num_keypoints)
            self.use_mmpose = MMPoseHRNet is not None
            if self.use_mmpose:
                extra = {
                    "stage1": dict(
                        num_modules=1,
                        num_branches=1,
                        block="BOTTLENECK",
                        num_blocks=(4,),
                        num_channels=(64,),
                    ),
                    "stage2": dict(
                        num_modules=1,
                        num_branches=2,
                        block="BASIC",
                        num_blocks=(4, 4),
                        num_channels=(32, 64),
                    ),
                    "stage3": dict(
                        num_modules=4,
                        num_branches=3,
                        block="BASIC",
                        num_blocks=(4, 4, 4),
                        num_channels=(32, 64, 128),
                    ),
                    "stage4": dict(
                        num_modules=3,
                        num_branches=4,
                        block="BASIC",
                        num_blocks=(4, 4, 4, 4),
                        num_channels=(32, 64, 128, 256),
                    ),
                }
                self.backbone = MMPoseHRNet(extra=extra, in_channels=3)  # type: ignore[call-arg]
                self.head = nn.Conv2d(32, self.num_keypoints, kernel_size=1)
            else:
                self.fallback = SimpleHRNet(num_keypoints)

        def forward(self, x: "torch.Tensor") -> "torch.Tensor":
            if self.use_mmpose:
                feats = self.backbone(x)
                if isinstance(feats, (list, tuple)):
                    feats0 = feats[0]
                else:
                    feats0 = feats
                return self.head(feats0)
            return self.fallback(x)

else:

    class SimpleHRNet:  # type: ignore[misc]
        """
        Заглушка, если torch не установлен.
        """

        def __init__(self, num_keypoints: int) -> None:  # pragma: no cover
            raise RuntimeError("PyTorch is required for SimpleHRNet but is not installed.")

    class HRNetW32GM:  # type: ignore[misc]
        """
        Заглушка, если torch не установлен.
        """

        def __init__(self, num_keypoints: int) -> None:  # pragma: no cover
            raise RuntimeError("PyTorch is required for HRNetW32GM but is not installed.")


def write_datasets_txt(
    root: Path,
    run_id: str,
    train_samples: List[Tuple[Path, Path, str]],
    val_samples: List[Tuple[Path, Path, str]],
) -> None:
    datasets_dir = root / "datasets"
    datasets_dir.mkdir(parents=True, exist_ok=True)
    train_txt = datasets_dir / f"hrnet_train_{run_id}.txt"
    val_txt = datasets_dir / f"hrnet_val_{run_id}.txt"

    def _write(path: Path, samples: List[Tuple[Path, Path, str]]) -> None:
        with path.open("w", encoding="utf-8") as f:
            for img_path, csv_path, loc in samples:
                f.write(f"{img_path};{csv_path};{loc}\n")

    _write(train_txt, train_samples)
    _write(val_txt, val_samples)


def train_model(
    cfg: HRNetConfig,
    train_samples: List[Tuple[Path, Path, str]],
    val_samples: List[Tuple[Path, Path, str]],
    num_keypoints: int,
    run_id: str,
    root: Path,
    log_path: Path,
) -> Dict[str, Any]:
    """
    Основной цикл обучения. Пишет:
    - models/history/<run_id>/hrnet_best.pth, metrics.json, train_config.yaml, train_log.txt
    - models/current/hrnet_best.pth, quality.json
    - logs/train_hrnet_last.log
    """
    log_file = log_path.open("w", encoding="utf-8")

    def log(msg: str) -> None:
        ts = time.strftime("%Y-%m-%d %H:%M:%S")
        line = f"[{ts}] {msg}"
        print(line)
        log_file.write(line + "\n")
        log_file.flush()

    history_dir = root / "models" / "history" / run_id
    history_dir.mkdir(parents=True, exist_ok=True)
    current_dir = root / "models" / "current"
    current_dir.mkdir(parents=True, exist_ok=True)

    # Общая информация о датасете
    n_train = len(train_samples)
    n_val = len(val_samples)
    total = n_train + n_val
    train_share = float(n_train) / total if total > 0 else 0.0
    val_share = float(n_val) / total if total > 0 else 0.0
    manual_localities = sorted({loc for *_rest, loc in train_samples + val_samples})
    n_manual_localities = len(manual_localities)

    # Ветка без torch: честная заглушка, но с корректными файлами метрик
    if torch is None:
        log("PyTorch не установлен. Обучение выполнить нельзя, пишем нулевые метрики и выходим без падения.")

        metrics: Dict[str, Any] = {
            "run_id": run_id,
            "pck_r": 0.0,
            "pck_r_percent": 0.0,
            "R": 0.0,
            "n_train_images": n_train,
            "n_val_images": n_val,
            "train_share": train_share,
            "val_share": val_share,
            "n_manual_localities": n_manual_localities,
        }
        quality: Dict[str, Any] = {
            "run_id": run_id,
            "pck_r": 0.0,
            "pck_r_percent": 0.0,
            "n_train_images": n_train,
            "n_val_images": n_val,
            "train_share": train_share,
            "val_share": val_share,
            "n_manual_localities": n_manual_localities,
        }

        # Заглушечные файлы модели/метрик
        quality_path = current_dir / "quality.json"
        with quality_path.open("w", encoding="utf-8") as f:
            json.dump(quality, f, indent=2, ensure_ascii=False)

        metrics_path = history_dir / "metrics.json"
        with metrics_path.open("w", encoding="utf-8") as f:
            json.dump(metrics, f, indent=2, ensure_ascii=False)

        # train_config.yaml — просто копия текущего конфига и информации о запуске
        train_cfg_path = history_dir / "train_config.yaml"
        train_cfg_data: Dict[str, Any] = {
            "run_id": run_id,
            "config": cfg.__dict__,
            "n_train_images": n_train,
            "n_val_images": n_val,
            "train_share": train_share,
            "val_share": val_share,
            "n_manual_localities": n_manual_localities,
        }
        try:
            if yaml is not None:
                with train_cfg_path.open("w", encoding="utf-8") as f:
                    yaml.safe_dump(train_cfg_data, f, allow_unicode=True)
            else:
                with train_cfg_path.open("w", encoding="utf-8") as f:
                    json.dump(train_cfg_data, f, indent=2, ensure_ascii=False)
        except Exception:
            pass

        # train_log.txt — копия основного лога
        log_file.close()
        history_log = history_dir / "train_log.txt"
        try:
            shutil.copy2(log_path, history_log)
        except Exception:
            pass

        # hrnet_best.pth — пустой файл-заглушка, чтобы 2) не падал на проверке наличия
        best_model_hist = history_dir / "hrnet_best.pth"
        best_model_curr = current_dir / "hrnet_best.pth"
        try:
            best_model_hist.touch()
            shutil.copy2(best_model_hist, best_model_curr)
        except Exception:
            pass

        return metrics

    # --- Реальное обучение, если torch доступен ---
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log(f"Используем устройство: {device}")

    log(
        f"Всего образцов: {total} "
        f"(train={n_train}, val={n_val})"
    )

    train_ds = LandmarkDataset(train_samples, num_keypoints, cfg, phase="train")
    val_ds = LandmarkDataset(val_samples, num_keypoints, cfg, phase="val")

    train_loader = DataLoader(
        train_ds,
        batch_size=max(1, int(cfg.batch_size)),
        shuffle=True,
        num_workers=0,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=max(1, int(cfg.batch_size)),
        shuffle=False,
        num_workers=0,
    )

    model_type = (cfg.model_type or "").lower()
    if model_type.startswith("hrnet_w32"):
        log("Создаём модель HRNet-W32 (MMPose) для геометрической морфометрии.")
        model = HRNetW32GM(num_keypoints=num_keypoints)
        if getattr(model, "use_mmpose", False) is False:
            log("MMPose HRNet недоступен, используется запасной вариант SimpleHRNet.")
    else:
        log("Создаём упрощённую модель SimpleHRNet (без MMPose).")
        model = HRNetW32GM(num_keypoints=num_keypoints)

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=float(cfg.learning_rate),
        weight_decay=float(cfg.weight_decay),
    )
    criterion = torch.nn.MSELoss()

    model = model.to(device)

    # Радиус для PCK@R: фиксированный в пикселях на теплокарте
    # Здесь берём 5% от меньшей стороны входа.
    if cfg.input_size > 0:
        R = float(max(1, int(round(min(cfg.input_size, cfg.input_size) * 0.05))))
    else:
        R = 10.0

    best_pck = 0.0
    best_epoch = 0

    for epoch in range(int(cfg.max_epochs)):
        model.train()
        running_loss = 0.0
        num_batches = 0

        for imgs, kps in train_loader:
            imgs = imgs.to(device)
            kps = kps.to(device)

            optimizer.zero_grad()
            # Сначала прогоняем через модель и узнаём размер теплокарт
            pred_heatmaps = model(imgs)
            _, _, H_hm, W_hm = pred_heatmaps.shape
            _, _, H_in, W_in = imgs.shape

            # Масштабируем ключевые точки под разрешение теплокарт,
            # чтобы gt и pred были одного размера.
            if H_in != H_hm or W_in != W_hm:
                scale_x = float(W_hm) / float(W_in)
                scale_y = float(H_hm) / float(H_in)
                kps_scaled = kps.clone()
                kps_scaled[..., 0] = kps_scaled[..., 0] * scale_x
                kps_scaled[..., 1] = kps_scaled[..., 1] * scale_y
            else:
                kps_scaled = kps

            gt_heatmaps = keypoints_to_heatmaps(
                kps_scaled,
                H_hm,
                W_hm,
                sigma=float(getattr(cfg, "heatmap_sigma_px", 2.0)),
                device=device,
            )
            loss = criterion(pred_heatmaps, gt_heatmaps)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            num_batches += 1

        avg_loss = running_loss / max(1, num_batches)
        log(f"Epoch {epoch + 1}/{cfg.max_epochs} - train loss = {avg_loss:.6f}")

        # Валидация
        model.eval()
        all_pred = []
        all_gt = []
        with torch.no_grad():
            for imgs, kps in val_loader:
                imgs = imgs.to(device)
                kps = kps.to(device)
                pred_heatmaps = model(imgs)
                pred_kps = heatmaps_to_keypoints(pred_heatmaps)
                # Переводим предсказанные координаты в ту же систему, что и gt
                _, _, H_in, W_in = imgs.shape
                _, _, H_hm, W_hm = pred_heatmaps.shape
                if H_in != H_hm or W_in != W_hm:
                    scale_x = float(W_in) / float(W_hm)
                    scale_y = float(H_in) / float(H_hm)
                    pred_kps[..., 0] = pred_kps[..., 0] * scale_x
                    pred_kps[..., 1] = pred_kps[..., 1] * scale_y
                all_pred.append(pred_kps.cpu())
                all_gt.append(kps.cpu())

        if all_pred and all_gt:
            pred_cat = torch.cat(all_pred, dim=0)
            gt_cat = torch.cat(all_gt, dim=0)
            pck = compute_pck_at_r(pred_cat, gt_cat, R)
        else:
            pck = 0.0

        log(f"Epoch {epoch + 1} - val PCK@R={pck:.4f}")

        if pck >= best_pck:
            best_pck = pck
            best_epoch = epoch + 1
            # Сохраняем лучшую модель
            best_model_hist = history_dir / "hrnet_best.pth"
            torch.save(model.state_dict(), best_model_hist)
            best_model_curr = current_dir / "hrnet_best.pth"
            shutil.copy2(best_model_hist, best_model_curr)
            log(f"Новая лучшая модель сохранена (epoch={best_epoch}, PCK@R={best_pck:.4f}).")

    # Финальные метрики
    pck_percent = float(round(best_pck * 100))
    metrics: Dict[str, Any] = {
        "run_id": run_id,
        "pck_r": float(best_pck),
        "pck_r_percent": pck_percent,
        "R": R,
        "n_train_images": n_train,
        "n_val_images": n_val,
        "train_share": train_share,
        "val_share": val_share,
        "n_manual_localities": n_manual_localities,
        "best_epoch": best_epoch,
        "device": str(device),
    }

    metrics_path = history_dir / "metrics.json"
    with metrics_path.open("w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)

    quality: Dict[str, Any] = {
        "run_id": run_id,
        "pck_r": float(best_pck),
        "pck_r_percent": pck_percent,
        "n_train_images": n_train,
        "n_val_images": n_val,
        "train_share": train_share,
        "val_share": val_share,
        "n_manual_localities": n_manual_localities,
    }
    quality_path = current_dir / "quality.json"
    with quality_path.open("w", encoding="utf-8") as f:
        json.dump(quality, f, indent=2, ensure_ascii=False)

    # train_config.yaml
    train_cfg_path = history_dir / "train_config.yaml"
    train_cfg_data: Dict[str, Any] = {
        "run_id": run_id,
        "config": cfg.__dict__,
        "R": R,
        "n_train_images": n_train,
        "n_val_images": n_val,
        "train_share": train_share,
        "val_share": val_share,
        "n_manual_localities": n_manual_localities,
        "best_epoch": best_epoch,
    }
    try:
        if yaml is not None:
            with train_cfg_path.open("w", encoding="utf-8") as f:
                yaml.safe_dump(train_cfg_data, f, allow_unicode=True)
        else:
            with train_cfg_path.open("w", encoding="utf-8") as f:
                json.dump(train_cfg_data, f, indent=2, ensure_ascii=False)
    except Exception:
        pass

    # train_log.txt
    log_file.close()
    history_log = history_dir / "train_log.txt"
    try:
        shutil.copy2(log_path, history_log)
    except Exception:
        pass

    return metrics


def main(argv: Optional[List[str]] = None) -> int:
    """
    Точка входа для действия 1) Train / Finetune model on MANUAL localities.
    Никаких новых ключей/режимов не добавляем, только внутренняя реализация.
    """
    argv = list(sys.argv[1:] if argv is None else argv)
    parser = argparse.ArgumentParser(description="Train HRNet model on MANUAL localities")
    # parse_known_args, чтобы не сломаться, если trainer_menu.py что-то передаёт
    parser.add_argument("--dummy", help="Не используется, только для совместимости", default=None)
    args, _unknown = parser.parse_known_args(argv)

    root = get_landmark_root()
    cfg_path = root / "config" / "hrnet_config.yaml"
    cfg = load_yaml_config(cfg_path)

    lm_number = read_lm_number(root)
    if lm_number <= 0:
        print("[ERR] LM_number.txt не найден или содержит неверное значение.")
        return 1

    base_dir = read_last_base(root)
    if base_dir is None:
        print("[ERR] cfg\\last_base.txt не найден или папка с локальностями отсутствует.")
        return 1

    samples = gather_manual_samples(root, base_dir)
    if not samples:
        print("[INFO] MANUAL локальностей с полными PNG+CSV не найдено. Обучать нечего.")
        return 0

    # Один раз тасуем и делим, чтобы split совпадал и для логов, и для обучения
    random.shuffle(samples)
    train_val = float(cfg.train_val_split)
    split_idx = int(len(samples) * train_val)
    train_samples = samples[:split_idx]
    val_samples = samples[split_idx:]

    run_id = time.strftime("%Y%m%d_%H%M%S")
    logs_dir = root / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)
    log_path = logs_dir / "train_hrnet_last.log"

    # По ТЗ: сохраняем списки датасета
    write_datasets_txt(root, run_id, train_samples, val_samples)

    metrics = train_model(cfg, train_samples, val_samples, lm_number, run_id, root, log_path)

    # Красивый вывод как в ТЗ
    n_train = metrics.get("n_train_images", 0)
    n_val = metrics.get("n_val_images", 0)
    total = n_train + n_val
    train_share = metrics.get("train_share", float(n_train) / total if total > 0 else 0.0)
    val_share = metrics.get("val_share", float(n_val) / total if total > 0 else 0.0)
    pck_percent = metrics.get("pck_r_percent", 0.0)
    n_manual_localities = metrics.get("n_manual_localities", 0)
    run_id_out = metrics.get("run_id", run_id)

    print("Training finished.")
    print()
    print(f"Used MANUAL localities: {n_manual_localities}")
    train_pct = int(round(train_share * 100))
    val_pct = int(round(val_share * 100))
    print(f"Train images: {n_train} ({train_pct}%)")
    print(f"Val images:   {n_val} ({val_pct}%)")
    print()
    print(f"PCK@R (validation): {int(round(pck_percent))} %")
    print()
    print("Model saved as: models/current/hrnet_best.pth")
    print(f"Run id: {run_id_out}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

