from __future__ import annotations

import argparse
import csv
import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple, Optional

import numpy as np
from PIL import Image

try:
    import torch
    from torch import nn
except ImportError:  # pragma: no cover
    torch = None
    nn = None  # type: ignore[assignment]

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
    resize_mode: str = "resize"
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


# Модель должна совпадать с train_hrnet.py
if torch is not None:

    class SimpleHRNet(nn.Module):
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
        def __init__(self, num_keypoints: int) -> None:  # pragma: no cover
            raise RuntimeError("PyTorch is required for SimpleHRNet but is not installed.")

    class HRNetW32GM:  # type: ignore[misc]
        def __init__(self, num_keypoints: int) -> None:  # pragma: no cover
            raise RuntimeError("PyTorch is required for HRNetW32GM but is not installed.")


def build_model_for_infer(cfg: HRNetConfig, num_keypoints: int) -> "nn.Module":
    model_type = (cfg.model_type or "").lower()
    if model_type.startswith("hrnet_w32"):
        return HRNetW32GM(num_keypoints=num_keypoints)
    return SimpleHRNet(num_keypoints=num_keypoints)


def _write_dummy_csv(csv_path: Path, num_keypoints: int) -> None:
    """
    Записываем CSV в формате одной строки: x1,y1,x2,y2,... .
    Используется в заглушках, если нет модели или torch.
    """
    with csv_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        row = []
        for _ in range(num_keypoints):
            row.extend([0.0, 0.0])
        writer.writerow(row)


def infer_for_locality(
    root: Path,
    base_dir: Path,
    locality: str,
    num_keypoints: int,
    cfg: HRNetConfig,
) -> None:
    """
    Автолейблинг одной локальности.
    """
    png_dir = base_dir / locality / "png"
    if not png_dir.is_dir():
        print(f"[ERR] Папка с изображениями не найдена: {png_dir}")
        return

    models_current = root / "models" / "current"
    model_path = models_current / "hrnet_best.pth"

    # Ветка без torch или без модели: честная заглушка
    if torch is None or not model_path.is_file():
        if torch is None:
            print("[WARN] PyTorch не установлен. Создаём CSV с нулевыми координатами (заглушка).")
        else:
            print("[WARN] Файл модели models/current/hrnet_best.pth не найден. Создаём CSV с нулевыми координатами (заглушка).")

        for img_path in sorted(png_dir.glob("*.png")):
            csv_path = img_path.with_suffix(".csv")
            _write_dummy_csv(csv_path, num_keypoints)
        return

    # Реальный инференс
    assert torch is not None
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Загружаем модель для инференса: {model_path} (device={device})")

    model = build_model_for_infer(cfg, num_keypoints)
    state = torch.load(model_path, map_location=device)
    if isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]
    model.load_state_dict(state)
    model.to(device)
    model.eval()

    for img_path in sorted(png_dir.glob("*.png")):
        img = Image.open(img_path).convert("RGB")
        img_resized, scale, offset_x, offset_y = _resize_and_pad(img, cfg)

        np_img = np.asarray(img_resized, dtype=np.float32) / 255.0
        np_img = np.transpose(np_img, (2, 0, 1))  # CHW
        tensor = torch.from_numpy(np_img).unsqueeze(0).to(device)

        with torch.no_grad():
            heatmaps = model(tensor)
            kps_resized = heatmaps_to_keypoints(heatmaps)[0].cpu().numpy()  # (K, 2)

        # Переводим координаты обратно в систему исходного изображения
        if scale > 0:
            kps_orig = np.zeros_like(kps_resized, dtype=np.float32)
            kps_orig[:, 0] = (kps_resized[:, 0] - float(offset_x)) / scale
            kps_orig[:, 1] = (kps_resized[:, 1] - float(offset_y)) / scale
        else:
            kps_orig = kps_resized

        # Запись CSV в формате одной строки: x1,y1,x2,y2,...
        csv_path = img_path.with_suffix(".csv")
        with csv_path.open("w", encoding="utf-8", newline="") as f:
            writer = csv.writer(f)
            row = []
            for i in range(num_keypoints):
                x = float(kps_orig[i, 0]) if i < len(kps_orig) else 0.0
                y = float(kps_orig[i, 1]) if i < len(kps_orig) else 0.0
                row.extend([x, y])
            writer.writerow(row)


def main(argv: Optional[List[str]] = None) -> int:
    """
    Точка входа для действия 2) Autolabel locality with current model.
    Внешний интерфейс не меняем, стараемся быть максимально совместимыми:
    - база берётся из cfg/last_base.txt, если не передана явно;
    - локальность берётся из аргумента --locality или переменной окружения GM_LOCALITY.
    """
    argv = list(sys.argv[1:] if argv is None else argv)
    parser = argparse.ArgumentParser(description="Autolabel locality with current HRNet model")
    parser.add_argument("--base", help="Базовая папка локальностей", default=None)
    parser.add_argument("--locality", help="Имя локальности", default=None)
    args, _unknown = parser.parse_known_args(argv)

    root = get_landmark_root()
    cfg_path = root / "config" / "hrnet_config.yaml"
    cfg = load_yaml_config(cfg_path)

    lm_number = read_lm_number(root)
    if lm_number <= 0:
        print("[ERR] LM_number.txt не найден или содержит неверное значение.")
        return 1

    base_dir: Optional[Path]
    if args.base:
        base_dir = Path(args.base)
    else:
        base_dir = read_last_base(root)

    if base_dir is None or not base_dir.is_dir():
        print("[ERR] Базовая папка локальностей не определена или не существует.")
        return 1

    locality = args.locality or os.environ.get("GM_LOCALITY", "").strip()
    if not locality:
        print("[ERR] Локальность не указана (нет --locality и GM_LOCALITY).")
        return 1

    print(f"[INFO] Autolabel locality: {locality}")
    print(f"[INFO] Base dir: {base_dir}")

    infer_for_locality(root, base_dir, locality, lm_number, cfg)

    print("[INFO] Autolabel finished for locality:", locality)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())





