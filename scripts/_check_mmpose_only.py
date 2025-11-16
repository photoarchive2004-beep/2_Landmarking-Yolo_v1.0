import importlib, sys

def check(name):
    try:
        m = importlib.import_module(name)
        ver = getattr(m, "__version__", "?")
        print(f"{name}: OK (version {ver})")
    except Exception as e:
        print(f"{name}: FAIL ({e!r})")

print("Python:", sys.version)
for pkg in ("mmengine", "mmcv", "mmpose"):
    check(pkg)
