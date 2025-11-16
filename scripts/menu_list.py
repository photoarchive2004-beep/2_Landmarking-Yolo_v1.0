import os, sys

HERE = os.path.dirname(__file__)
def default_root():
    # scripts -> 2_Landmarking_v1.0 -> tools -> GM
    return os.path.abspath(os.path.join(HERE, "..", "..", ".."))

def parse_args(argv):
    args = {"mode":"print", "pick":None, "root":None}
    it = iter(range(len(argv)))
    i = 0
    while i < len(argv):
        a = argv[i]
        if a == "--print": args["mode"] = "print"
        elif a == "--pick":
            args["mode"] = "pick"
            if i+1 < len(argv):
                try: args["pick"] = int(argv[i+1]); i += 1
                except: pass
        elif a == "--root":
            if i+1 < len(argv):
                args["root"] = argv[i+1]; i += 1
        i += 1
    return args

def read_npoints(LM_FILE):
    try:
        with open(LM_FILE, "r", encoding="utf-8-sig") as f:
            s = f.readline().strip()
        n = int(s)
        return n if n > 1 else None
    except:
        return None

def parse_wide_nums(line):
    vals = []
    for t in [x.strip() for x in line.split(",") if x.strip()]:
        try: vals.append(float(t))
        except: pass
    return vals

def image_is_done(png_path, N):
    base = os.path.splitext(os.path.basename(png_path))[0]
    csv_path = os.path.join(os.path.dirname(png_path), base + ".csv")
    if not os.path.exists(csv_path): 
        return False
    try:
        with open(csv_path, "r", encoding="utf-8-sig") as f:
            rows = [r.strip() for r in f if r.strip()]
        if not rows: return False
        for line in rows:
            nums = parse_wide_nums(line)
            if N and len(nums) == 2 * N:
                return True
        return False
    except:
        return False

def has_any_scale(png_dir):
    try:
        for fn in os.listdir(png_dir):
            if fn.lower().endswith(".scale.csv"):
                return True
    except: pass
    return False

def collect_localities(PHOTOS):
    if not os.path.isdir(PHOTOS):
        return []
    names = []
    for d in sorted(os.listdir(PHOTOS)):
        png_dir = os.path.join(PHOTOS, d, "png")
        if os.path.isdir(png_dir): 
            names.append(d)
    return names

def main():
    a = parse_args(sys.argv[1:])
    ROOT = os.path.abspath(a["root"]) if a["root"] else default_root()
    TOOL = os.path.join(ROOT, "tools", "2_Landmarking_v1.0")
    CFG = os.path.abspath(os.path.join(HERE, "..", "cfg", "last_base.txt"))
    _base = None
    if os.path.isfile(CFG):
        try:
            with open(CFG, "r", encoding="utf-8") as f:
                _base = f.read().strip()
        except OSError:
            _base = None
    if _base:
        PHOTOS = _base
    else:
        PHOTOS = os.path.join(ROOT, "photos")
    LM_FILE = os.path.join(TOOL, "LM_number.txt")
    N = read_npoints(LM_FILE)
    locs = collect_localities(PHOTOS)

    if a["mode"] == "print":
        if not locs:
            print("No localities found in 'photos'.")
            return 0
        for i, loc in enumerate(locs, 1):
            png_dir = os.path.join(PHOTOS, loc, "png")
            pngs = [os.path.join(png_dir, f) for f in os.listdir(png_dir) if f.lower().endswith(".png")]
            total = len(pngs)
            done  = sum(image_is_done(p, N) for p in pngs) if (N and total) else 0
            pct   = int(round((done / total * 100), 0)) if total else 0
            line  = f"{i}) {loc} [{done}/{total}] {pct}%"
            if total and done == total and not has_any_scale(png_dir):
                line += "  Set Scale!"
            print(line)
        return 0

    if a["mode"] == "pick":
        if not (a["pick"] and 1 <= a["pick"] <= len(locs)): 
            return 2
        print(locs[a["pick"]-1], end="")
        return 0

    return 0

if __name__ == "__main__":
    sys.exit(main())

