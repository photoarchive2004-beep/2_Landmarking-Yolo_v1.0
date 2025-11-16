import importlib

def show(name):
    try:
        m = importlib.import_module(name)
        ver = getattr(m, "__version__", "?")
        print(f"{name}: {ver}")
    except Exception as e:
        print(f"{name}: FAIL ({e!r})")

for pkg in ("numpy", "torch", "mmengine", "mmcv", "mmpose"):
    show(pkg)
