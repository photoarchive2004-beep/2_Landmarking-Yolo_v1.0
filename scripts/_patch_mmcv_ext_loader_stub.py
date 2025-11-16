from pathlib import Path
import traceback

print("[INFO] Importing mmcv to locate utils/ext_loader.py ...")
try:
    import mmcv  # type: ignore
except Exception as e:
    print("[ERR] Cannot import mmcv:", repr(e))
    traceback.print_exc()
    raise SystemExit(1)

base = Path(mmcv.__file__).resolve().parent
ext_loader_path = base / "utils" / "ext_loader.py"
print("[INFO] mmcv ext_loader.py:", ext_loader_path)

if not ext_loader_path.is_file():
    print("[ERR] ext_loader.py not found, cannot patch.")
    raise SystemExit(1)

text = ext_loader_path.read_text(encoding="utf-8")

if "[GM PATCH] stub load_ext" in text:
    print("[INFO] GM stub for load_ext already present, skip patch.")
else:
    stub = """
# [GM PATCH] stub load_ext to allow missing mmcv._ext for GM HRNet-only use.
def load_ext(name, funcs):
    \\"\\"
    Simplified loader for mmcv C-extensions.

    If mmcv._ext is available, use it.
    If not, return a dummy module so that imports do not fail.
    Heavy ops that rely on these extensions are not used in GM HRNet pipeline.
    \\"\\"
    import importlib
    import types
    try:
        ext = importlib.import_module('mmcv._ext')
        return ext
    except Exception as e:  # pragma: no cover
        print("[WARN] GM: mmcv._ext not available, returning dummy ext for", name, ":", repr(e))
        dummy = types.SimpleNamespace()
        # Optionally attach dummy functions for requested ops
        try:
            for fn_name in (funcs or []):
                def _make_dummy(n):
                    def _fn(*args, **kwargs):
                        raise NotImplementedError(f"mmcv._ext op '{n}' is disabled in GM stub")
                    return _fn
                setattr(dummy, fn_name, _make_dummy(fn_name))
        except Exception:
            pass
        return dummy
"""
    print("[INFO] Appending GM stub load_ext to ext_loader.py ...")
    text = text.rstrip() + "\\n" + stub + "\\n"
    ext_loader_path.write_text(text, encoding="utf-8")
    print("[INFO] GM stub appended to ext_loader.py")

# --- Проверяем импорт HRNet после патча ---
print("\\n[INFO] Testing MMPose HRNet import with patched mmcv.ext_loader ...")
try:
    from mmpose.models.backbones import HRNet as MMPoseHRNet  # type: ignore
    print("[OK] MMPoseHRNet imported:", MMPoseHRNet)
    raise SystemExit(0)
except Exception as e:
    print("[ERR] Still cannot import MMPoseHRNet:", repr(e))
    traceback.print_exc()
    raise SystemExit(1)
