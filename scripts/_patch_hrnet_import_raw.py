from pathlib import Path

root = Path(__file__).resolve().parent
scripts_dir = root / "scripts"

def patch_file(path: Path) -> None:
    print(f"[INFO] Patching {path} ...")
    if not path.is_file():
        print(f"[WARN] File not found, skip: {path}")
        return
    text = path.read_text(encoding="utf-8")

    needle = "from mmpose.models.backbones import HRNet as MMPoseHRNet"
    if needle not in text:
        print(f"[WARN] Import line not found in {path.name}, skip.")
        return

    # Ищем блок try/except вокруг импорта
    idx_import = text.index(needle)
    # ищем начало блока по ближайшему 'try:' выше
    idx_try = text.rfind("try:", 0, idx_import)
    if idx_try == -1:
        print(f"[WARN] 'try:' before HRNet import not found in {path.name}, skip.")
        return

    # ищем конец блока по строке с MMPoseHRNet = None
    end_marker = "MMPoseHRNet = None  # type: ignore"
    idx_end_marker = text.find(end_marker, idx_import)
    if idx_end_marker == -1:
        print(f"[WARN] end marker not found in {path.name}, skip.")
        return
    idx_block_end = idx_end_marker + len(end_marker)

    old_block = text[idx_try:idx_block_end]

    new_block = '''
def _load_mmpose_hrnet() -> object:
    """Load HRNet backbone from MMPose without importing datasets/evaluation."""
    from pathlib import Path
    import importlib.util
    import mmpose  # type: ignore

    base = Path(mmpose.__file__).resolve().parent
    hrnet_py = base / "models" / "backbones" / "hrnet.py"
    if not hrnet_py.is_file():
        raise FileNotFoundError(f"MMPose HRNet file not found: {hrnet_py}")
    spec = importlib.util.spec_from_file_location("gm_mmpose_hrnet", hrnet_py)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot create module spec for {hrnet_py}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)  # type: ignore[arg-type]
    if not hasattr(module, "HRNet"):
        raise AttributeError("HRNet class not found in MMPose hrnet.py")
    return module.HRNet

try:
    MMPoseHRNet = _load_mmpose_hrnet()
except Exception as e:  # pragma: no cover
    print("[WARN] Exception during import or setup (possibly MMPose HRNet):", repr(e))
    MMPoseHRNet = None  # type: ignore
'''

    new_text = text[:idx_try] + new_block + text[idx_block_end:]
    path.write_text(new_text, encoding="utf-8")
    print(f"[INFO] Patched {path.name} successfully.")

for name in ("train_hrnet.py", "infer_hrnet.py"):
    patch_file(scripts_dir / name)
