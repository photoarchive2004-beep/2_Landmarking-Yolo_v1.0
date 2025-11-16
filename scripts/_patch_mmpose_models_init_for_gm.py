from pathlib import Path
import sys

print("[INFO] Importing mmpose to locate models/__init__.py ...")
try:
    import mmpose  # type: ignore
except Exception as e:
    print("[ERR] Cannot import mmpose:", repr(e))
    sys.exit(1)

models_dir = Path(mmpose.__file__).resolve().parent / "models"
init_path = models_dir / "__init__.py"
print("[INFO] mmpose models/__init__.py:", init_path)

if not init_path.is_file():
    print("[ERR] __init__.py not found:", init_path)
    sys.exit(1)

text = init_path.read_text(encoding="utf-8")

needle = "from .distillers import *"
if needle not in text:
    print("[WARN] Line 'from .distillers import *' not found, nothing to patch.")
else:
    print("[INFO] Patching models/__init__.py: disabling distillers import to avoid xtcocotools.")
    patched = text.replace(
        needle,
        "# [GM] disabled distillers for HRNet-GM project (no need for DWPose/COCO)\n# "
        + needle
    )
    init_path.write_text(patched, encoding="utf-8")
    print("[INFO] Patch applied.")

print("[INFO] Testing HRNet import from MMPose ...")
try:
    from mmpose.models.backbones import HRNet as MMPoseHRNet  # type: ignore
    print("[OK] MMPoseHRNet imported successfully:", MMPoseHRNet)
    sys.exit(0)
except Exception as e:
    import traceback
    print("[ERR] Still cannot import MMPoseHRNet:", repr(e))
    traceback.print_exc()
    sys.exit(1)
