from pathlib import Path
import traceback

print("[INFO] Importing mmpose to locate heads/__init__.py ...")
try:
    import mmpose  # type: ignore
except Exception as e:
    print("[ERR] Cannot import mmpose:", repr(e))
    traceback.print_exc()
    raise SystemExit(1)

base = Path(mmpose.__file__).resolve().parent
heads_init = base / "models" / "heads" / "__init__.py"
print("[INFO] mmpose heads __init__:", heads_init)

if not heads_init.is_file():
    print("[ERR] heads __init__.py not found, cannot patch.")
    raise SystemExit(1)

text = heads_init.read_text(encoding="utf-8")

needle = "from .hybrid_heads import DEKRHead, RTMOHead, VisPredictHead"
if needle not in text:
    print("[WARN] Target import line not found in heads/__init__.py, nothing to patch.")
else:
    replacement = """try:
    from .hybrid_heads import DEKRHead, RTMOHead, VisPredictHead
except Exception as e:  # pragma: no cover
    print("[WARN] GM: hybrid_heads (RTMOHead) disabled because mmdet is missing:", repr(e))
    DEKRHead = RTMOHead = VisPredictHead = None
"""
    text = text.replace(needle, replacement)
    heads_init.write_text(text, encoding="utf-8")
    print("[INFO] Patch applied to heads/__init__.py")

print("[INFO] Testing HRNet import after patch ...")
try:
    from mmpose.models.backbones import HRNet as MMPoseHRNet  # type: ignore
    print("[OK] MMPoseHRNet imported:", MMPoseHRNet)
except Exception as e:
    print("[ERR] Still cannot import MMPoseHRNet:", repr(e))
    traceback.print_exc()
    raise SystemExit(1)
