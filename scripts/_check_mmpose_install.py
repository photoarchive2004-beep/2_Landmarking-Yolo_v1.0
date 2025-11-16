import sys
print("Python:", sys.version)
try:
    import mmengine
    import mmcv
    import mmpose
    print("mmengine:", mmengine.__version__)
    print("mmcv:", mmcv.__version__)
    print("mmpose:", mmpose.__version__)
    print("OK: all core MMPose deps imported.")
except Exception as e:
    print("IMPORT ERROR:", repr(e))
