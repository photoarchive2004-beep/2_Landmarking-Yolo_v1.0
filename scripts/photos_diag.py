import os, glob, sys
HERE = os.path.dirname(__file__)
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
PHOTOS = os.path.join(ROOT, "photos")
print(f"[DIAG] ROOT   = {ROOT}")
print(f"[DIAG] PHOTOS = {PHOTOS}")
if not os.path.isdir(PHOTOS):
    print("[ERR] photos dir not found")
    sys.exit(1)

dirs = sorted([d for d in os.listdir(PHOTOS) if os.path.isdir(os.path.join(PHOTOS, d))])
if not dirs:
    print("[DIAG] photos is empty")
    sys.exit(0)

for d in dirs:
    loc = os.path.join(PHOTOS, d)
    png_dir = os.path.join(loc, "png")
    has_png_dir = os.path.isdir(png_dir)
    cnt_png_in_dir = len(glob.glob(os.path.join(png_dir, "*.png"))) if has_png_dir else 0
    cnt_png_flat   = len(glob.glob(os.path.join(loc, "*.png")))
    print(f"{d}: png_subdir={has_png_dir}, png_in_subdir={cnt_png_in_dir}, png_in_root={cnt_png_flat}")