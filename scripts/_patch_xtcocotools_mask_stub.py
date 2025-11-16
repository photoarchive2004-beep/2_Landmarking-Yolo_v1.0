from pathlib import Path
import traceback

print("[INFO] Import xtcocotools ...")
try:
    import xtcocotools  # type: ignore
except Exception as e:
    print("[ERR] Cannot import xtcocotools:", repr(e))
    traceback.print_exc()
    raise SystemExit(1)

mask_path = Path(xtcocotools.__file__).resolve().parent / "mask.py"
backup_path = mask_path.with_name("mask_orig_gm_backup.py")

print("[INFO] xtcocotools module file:", xtcocotools.__file__)
print("[INFO] mask.py path:", mask_path)

if not mask_path.is_file():
    print("[ERR] mask.py not found, nothing to patch.")
    raise SystemExit(1)

# Делаем бэкап оригинального mask.py один раз
if not backup_path.exists():
    print("[INFO] Saving original mask.py to", backup_path)
    backup_path.write_text(mask_path.read_text(encoding="utf-8"), encoding="utf-8")
else:
    print("[INFO] Backup already exists:", backup_path)

stub = """\"\"\"[GM PATCH] Simplified xtcocotools.mask for GM HRNet training.

В этой заглушке мы полностью отключаем использование скомпилированного _mask,
который конфликтует с текущей версией NumPy. Модуль нужен только для того,
чтобы успешно импортировался MMPose/COCO, сами операции с масками нам по ТЗ не нужны.
\"\"\"

import numpy as np  # noqa: F401


def _disabled(name: str):
    raise NotImplementedError(
        f\"xtcocotools.mask.{name} is disabled in GM stub (COCO masks are not used in this project)\"
    )


def encode(*args, **kwargs):
    return _disabled("encode")


def decode(*args, **kwargs):
    return _disabled("decode")


def area(*args, **kwargs):
    return _disabled("area")


def toBbox(*args, **kwargs):
    return _disabled("toBbox")


def iou(*args, **kwargs):
    return _disabled("iou")


def merge(*args, **kwargs):
    return _disabled("merge")


def frPyObjects(*args, **kwargs):
    return _disabled("frPyObjects")
"""

print("[INFO] Writing GM stub to mask.py ...")
mask_path.write_text(stub, encoding="utf-8")
print("[INFO] Stub xtcocotools.mask written successfully.")

# --- Проверяем импорт HRNet из MMPose ---
print("\\n[INFO] Now testing MMPose HRNet import ...")
import numpy
print("numpy version:", numpy.__version__)
try:
    from mmpose.models.backbones import HRNet as MMPoseHRNet  # type: ignore
    print("[OK] MMPoseHRNet imported successfully:", MMPoseHRNet)
except Exception as e:
    print("[ERR] Still cannot import MMPoseHRNet:", repr(e))
    traceback.print_exc()
    raise SystemExit(1)
