import sys, argparse, shutil, time
from pathlib import Path
from PIL import Image, ImageTk
import tkinter as tk
from tkinter import messagebox, simpledialog

# ---------- helpers ----------
def read_npoints(root):
    lm = Path(root) / "tools/2_Landmarking_v1.0/LM_number.txt"
    try:
        n = int(lm.read_text(encoding="utf-8-sig").splitlines()[0].strip())
        return n if n > 1 else None
    except Exception:
        return None

def list_images(png_dir):
    return sorted(str(p) for p in Path(png_dir).glob("*.png"))

def save_csv_wide(csv_path, pts_list):
    flat = []
    for (x, y) in pts_list:
        flat += [f"{x:.2f}", f"{y:.2f}"]
    Path(csv_path).write_text(",".join(flat), encoding="utf-8")

def load_existing_csv(csv_path, N):
    p = Path(csv_path)
    if not p.exists():
        return []
    try:
        txt = p.read_text(encoding="utf-8-sig")
        for line in [l.strip() for l in txt.splitlines() if l.strip()]:
            parts = [t.strip() for t in line.split(",") if t.strip() != ""]
            vals = list(map(float, parts))
            pts = []
            for i in range(0, min(len(vals), 2 * N), 2):
                pts.append((vals[i], vals[i + 1]))
            return pts
    except Exception:
        pass
    return []

def read_any_csv_points(csv_path):
    p = Path(csv_path)
    if not p.exists():
        return []
    pts = []
    try:
        txt = p.read_text(encoding="utf-8-sig")
        line = next((l.strip() for l in txt.splitlines() if l.strip()), "")
        parts = [t.strip() for t in line.split(",") if t.strip() != ""]
        vals = list(map(float, parts))
        for i in range(0, len(vals) - 1, 2):
            pts.append((vals[i], vals[i + 1]))
    except Exception:
        pass
    return pts

def move_to_bad(img_path):
    p = Path(img_path)
    bad = p.parent / "bad"
    bad.mkdir(exist_ok=True)
    for f in [p, p.with_suffix(".csv")]:
        try:
            if f.exists():
                shutil.move(str(f), str(bad / f.name))
        except Exception:
            pass

# ---------- mini-GPA ----------
def center_scale(pts):
    cx = sum(x for x, _ in pts) / len(pts)
    cy = sum(y for _, y in pts) / len(pts)
    centered = [(x - cx, y - cy) for x, y in pts]
    cs = (sum(x * x + y * y for x, y in centered)) ** 0.5 or 1.0
    return [(x / cs, y / cs) for x, y in centered]

def rotate_to(A, B):
    sxx = sum(ax * bx + ay * by for (ax, ay), (bx, by) in zip(A, B))
    syx = sum(ay * bx - ax * by for (ax, ay), (bx, by) in zip(A, B))
    r = (sxx * sxx + syx * syx) ** 0.5 or 1.0
    c = sxx / r
    s = syx / r
    return [(c * ax - s * ay, s * ax + c * ay) for (ax, ay) in A]

def mean_shape(shapes):
    N = len(shapes[0])
    mx = [0.0] * N
    my = [0.0] * N
    for sh in shapes:
        for i, (x, y) in enumerate(sh):
            mx[i] += x
            my[i] += y
    k = len(shapes)
    return [(mx[i] / k, my[i] / k) for i in range(N)]

def gpa_align(shapes, iters=10):
    aligned = [center_scale(s) for s in shapes]
    m = mean_shape(aligned)
    for _ in range(iters):
        aligned = [rotate_to(center_scale(s), m) for s in shapes]
        new_m = mean_shape(aligned)
        drift = sum((x1 - x2) ** 2 + (y1 - y2) ** 2 for (x1, y1), (x2, y2) in zip(m, new_m))
        if drift < 1e-8:
            break
        m = new_m
    return aligned, m

# ---------- GUI ----------
class AnnotGUI(tk.Tk):
    def __init__(self, root, png_dir, n_points, start_from=None, scale_wizard=False):
        super().__init__()
        self.title("GM Points Annotator - Custom")
        self.configure(background="black")

        self.root_dir = root
        self.png_dir = png_dir
        self.N = n_points or read_npoints(root) or 0
        if self.N <= 1:
            messagebox.showerror("Error", "LM_number.txt invalid or missing (N>1 required).")
            self.destroy()
            sys.exit(2)

        logs_dir = Path(root) / "tools/2_Landmarking_v1.0/logs"
        logs_dir.mkdir(exist_ok=True, parents=True)
        self.log_last = logs_dir / "annot_gui_last.log"
        def _write_log(msg):
            try:
                cur = self.log_last.read_text(encoding="utf-8") if self.log_last.exists() else ""
            except Exception:
                cur = ""
            self.log_last.write_text(cur + time.strftime("[%H:%M:%S] ") + msg + "\n", encoding="utf-8")
        self._log = _write_log

        self.images = list_images(png_dir)
        if not self.images:
            messagebox.showinfo("Empty", "No PNG images found.")
            self.destroy()
            sys.exit(0)

        self.idx = 0
        if start_from:
            sf = start_from.lower().replace(".png", "")
            for i, fp in enumerate(self.images):
                if Path(fp).stem.lower() == sf or sf in Path(fp).stem.lower():
                    self.idx = i
                    break

        # --- top bar ---
        top = tk.Frame(self, bg="black")
        top.pack(fill=tk.X)
        self.banner = tk.Label(top, text="", fg="#00FFFF", bg="black", font=("Segoe UI", 12, "bold"))
        self.banner.pack(side=tk.LEFT, padx=(8, 6), pady=4, fill=tk.X, expand=True)

        # QC tolerance controls
        self.qc_tol = 0.25
        self.qcTolVar = tk.StringVar(value="0.25")
        self.qcTolLabel = tk.Label(top, text="tol=0.25", fg="#FFA500", bg="black", font=("Segoe UI", 10, "bold"))
        self.qcTolLabel.pack(side=tk.RIGHT, padx=(4,4), pady=4)
        self.qcTolBtn = tk.Button(top, text="Set tol", command=self.set_qc_tol)
        self.qcTolBtn.pack(side=tk.RIGHT, padx=(4,4), pady=4)
        self.qcTolEntry = tk.Entry(top, width=5, textvariable=self.qcTolVar, justify="center")
        self.qcTolEntry.pack(side=tk.RIGHT, padx=(4,2), pady=4)

        self.btnQC = tk.Button(top, text="Quick Check (F9)", command=self.toggle_qc)
        self.btnQC.pack(side=tk.RIGHT, padx=8, pady=4)

        self.btnScale = tk.Button(top, text="Save scale (Enter)", command=lambda: self.finish_scale(True))

        # --- canvas + status ---
        self.canvas = tk.Canvas(self, bg="gray10", highlightthickness=0)
        self.canvas.pack(fill=tk.BOTH, expand=True)
        self.status = tk.Label(self, text="", anchor="w", bg="black", fg="white")
        self.status.pack(fill=tk.X)
        self.font_lbl = ("Segoe UI", 11, "bold")

        # state
        self.radius = 6
        self.zoom = 1.0
        self.offset = [0.0, 0.0]
        self.dragging_idx = None
        self.dragging_canvas = False
        self.photo = None
        self.img = None
        self.slots = [None] * self.N

        # modes
        self.scale_mode = bool(scale_wizard)
        self.scale_pts = []
        self.qc_mode = False
        self.qc_list = []
        self.qc_marks = {}
        self.qc_pos = 0

        # bindings
        self.bind("<KeyPress>", self.on_key)
        self.bind("<KeyRelease>", self.on_key_up)
        self.canvas.bind("<Button-1>", self.on_left_click)
        self.canvas.bind("<B1-Motion>", self.on_left_drag)
        self.canvas.bind("<ButtonRelease-1>", self.on_left_release)
        self.canvas.bind("<Button-3>", self.on_right_press)
        self.canvas.bind("<B3-Motion>", self.on_right_drag)
        self.canvas.bind("<ButtonRelease-3>", self.on_right_release)
        self.canvas.bind("<MouseWheel>", self.on_wheel)
        self.bind("<Control-f>", self.search_dialog)
        self.bind("f", self.search_dialog)
        self.bind("<Control-s>", self.ctrl_save)
        self.bind("<F9>", self.toggle_qc)
        self.bind("x", self.hot_move_to_bad)
        self.bind("X", self.hot_move_to_bad)
        self.bind("<Control-d>", self.delete_near_force)
        self.protocol("WM_DELETE_WINDOW", self.on_close)

        # first load
        self._auto_fit_once = True
        self._loaded = False
        self.load_image(self.images[self.idx])

        # auto-scale fallback for first PNG
        first_png = Path(self.images[0]).name
        scale_must = Path(self.images[0]).with_name(first_png + ".scale.csv")
        if not scale_must.exists():
            self.scale_mode = True

        # maximize window (not fullscreen), then fit image to width
        self.after(50, self._maximize_and_fit)
        self.apply_banner()

    # ---------- window fit helpers ----------
    def _maximize_and_fit(self):
        try:
            self.state("zoomed")  # not fullscreen
        except Exception:
            pass
        self.after(150, self._fit_width_on_open)

    def _fit_width_on_open(self):
        if not self._auto_fit_once or not self.img:
            return
        cw = self.canvas.winfo_width()
        ch = self.canvas.winfo_height()
        if cw < 50 or ch < 50:
            self.after(100, self._fit_width_on_open)
            return
        w, h = self.img.size
        self.zoom = max(0.1, min(20.0, cw / w))
        disp_h = h * self.zoom
        self.offset[0] = 0
        self.offset[1] = (ch - disp_h) / 2  # vertical center
        self.redraw("hq")
        self._auto_fit_once = False

    # ---------- basic helpers ----------
    def _count_pts(self):
        return sum(1 for p in self.slots if p is not None)

    def _first_free(self):
        for i, p in enumerate(self.slots):
            if p is None:
                return i
        return None

    def _pointer_on_canvas(self):
        return (
            self.canvas.winfo_pointerx() - self.canvas.winfo_rootx(),
            self.canvas.winfo_pointery() - self.canvas.winfo_rooty(),
        )

    # coords
    def d2i(self, dx, dy):
        return ((dx - self.offset[0]) / self.zoom, (dy - self.offset[1]) / self.zoom)

    def i2d(self, ix, iy):
        return (ix * self.zoom + self.offset[0], iy * self.zoom + self.offset[1])

    # IO
    def load_image(self, fp):
        self.img_path = fp
        self.img = Image.open(fp).convert("RGB")
        if self._auto_fit_once:
            self.zoom = 1.0
            self.offset = [0.0, 0.0]
        self.slots = [None] * self.N
        if not self.scale_mode:
            pts = load_existing_csv(Path(fp).with_suffix(".csv"), self.N)
            for i, (x, y) in enumerate(pts):
                if i < self.N:
                    self.slots[i] = (x, y)
        self.redraw("hq")
        self.update_status()
        self._loaded = True
        self._log(f"[LOAD] {Path(fp).name} pts={self._count_pts()}/{self.N} scale={self.scale_mode}")

    def save_current(self):
        if self.scale_mode:
            if len(self.scale_pts) == 2:
                p = Path(self.img_path)
                scale_csv = p.with_name(p.name + ".scale.csv")
                scale_csv.write_text(
                    f"{self.scale_pts[0][0]:.2f},{self.scale_pts[0][1]:.2f}\n{self.scale_pts[1][0]:.2f},{self.scale_pts[1][1]:.2f}\n",
                    encoding="utf-8",
                )
                self._log(f"[SCALE] saved {scale_csv.name}")
        else:
            csv = Path(self.img_path).with_suffix(".csv")
            pts = [pt for pt in self.slots if pt is not None]
            if pts:
                save_csv_wide(csv, pts)
                self._log(f"[SAVE] {csv.name} pts={len(pts)} OK")
            elif csv.exists():
                csv.unlink()
                self._log(f"[SAVE] {csv.name} removed (0 pts)")

    def auto_save_on_switch(self):
        c = self._count_pts()
        if (not self.scale_mode) and (c > 1) and (c != self.N):
            if not messagebox.askyesno(
                "Mismatch", f"Landmark count mismatch: expected {self.N}, got {c}.\nContinue and save anyway?"
            ):
                self._log("[MIS] user chose NO")
                return False
        self.save_current()
        return True

    # UI
    def apply_banner(self):
        if self.scale_mode:
            self.banner.config(
                text="SCALE MODE — place TWO cyan squares on 10 mm, then Enter / Save button.",
                fg="#00FFFF",
            )
            try:
                self.btnScale.pack(side=tk.RIGHT, padx=8, pady=4)
            except Exception:
                pass
            try:
                self.btnQC.configure(state="disabled")
                self.qcTolEntry.configure(state="disabled")
                self.qcTolBtn.configure(state="disabled")
                self.qcTolLabel.configure(text=f"tol={self.qc_tol:.2f}")
            except Exception:
                pass
        else:
            if self.qc_mode:
                total = len(self.qc_list)
                self.banner.config(text=f"QC MODE — {total} suspicious frames. F9 to exit.", fg="#FF5555")
                try:
                    self.btnQC.configure(text="Exit Quick Check (F9)")
                except Exception:
                    pass
            else:
                self.banner.config(text="", fg="#00FFFF")
                try:
                    self.btnQC.configure(text="Quick Check (F9)")
                except Exception:
                    pass
            try:
                self.btnScale.pack_forget()
            except Exception:
                pass
            try:
                self.btnQC.configure(state="normal")
                self.qcTolEntry.configure(state="normal")
                self.qcTolBtn.configure(state="normal")
                self.qcTolLabel.configure(text=f"tol={self.qc_tol:.2f}")
            except Exception:
                pass
        self.update_status()

    def update_status(self):
        i = self.idx + 1
        T = len(self.images)
        loc = Path(self.png_dir).parent.name
        name = Path(self.img_path).name
        tail = ""
        if self.qc_mode and self.qc_list:
            tail = f" | QC {self.qc_pos + 1}/{len(self.qc_list)}"
        extra = " | SCALE MODE (10 mm)" if self.scale_mode else f" | +/- size {self.radius}px | X/Ctrl+B: move to bad{tail}"
        self.status.config(text=f"Image {i} of {T} | {name} | {loc}{extra}")
        self.title(f"GM Points Annotator - {name} [{i}/{T}]")

    def _draw_label(self, dx, dy, text, color="yellow"):
        for ox, oy in ((-1, 0), (1, 0), (0, -1), (0, 1)):
            self.canvas.create_text(dx + ox, dy + oy, text=text, fill="black", anchor="w", font=self.font_lbl)
        self.canvas.create_text(dx, dy, text=text, fill=color, anchor="w", font=self.font_lbl)

    def redraw(self, quality="fast"):
        self.canvas.delete("all")
        w, h = self.img.size
        res = Image.NEAREST if quality == "fast" else Image.BILINEAR
        disp = self.img.resize((max(1, int(w * self.zoom)), max(1, int(h * self.zoom))), res)
        self.photo = ImageTk.PhotoImage(disp)
        self.canvas.create_image(self.offset[0], self.offset[1], anchor="nw", image=self.photo)

        if self.scale_mode:
            for i, (x, y) in enumerate(self.scale_pts, start=1):
                dx, dy = self.i2d(x, y)
                r = self.radius + 2
                self.canvas.create_rectangle(dx - r, dy - r, dx + r, dy + r, outline="#00FFFF", width=2)
                self._draw_label(dx + r + 6, dy, f"S{i}", "#00FFFF")
            return

        for idx, pt in enumerate(self.slots, start=1):
            if pt is None:
                continue
            x, y = pt
            dx, dy = self.i2d(x, y)
            r = self.radius
            self.canvas.create_oval(dx - r, dy - r, dx + r, dy + r, outline="red", width=2)
            self._draw_label(dx + r + 6, dy, str(idx))

        if self.qc_mode and self.qc_marks:
            marks = self.qc_marks.get(self.idx, [])
            for (k, j) in marks:
                pt = self.slots[k] if 0 <= k < len(self.slots) else None
                if pt:
                    dx, dy = self.i2d(pt[0], pt[1])
                    r = self.radius + 3
                    self.canvas.create_oval(dx - r, dy - r, dx + r, dy + r, outline="#FF00FF", width=3)
                    self._draw_label(dx + r + 8, dy, f"{k + 1}->{j + 1}", "#FF00FF")

    # events
    def on_left_click(self, e):
        if self.scale_mode:
            if len(self.scale_pts) < 2:
                ix, iy = self.d2i(e.x, e.y)
                self.scale_pts.append((ix, iy))
                self.redraw()
                return
            j = self._pick(self.scale_pts, e.x, e.y)
            if j is not None:
                self.dragging_idx = ("scale", j)
                return

        j = self._pick_slots(e.x, e.y)
        if j is not None:
            self.dragging_idx = ("slot", j)
        else:
            free = self._first_free()
            if free is not None:
                ix, iy = self.d2i(e.x, e.y)
                self.slots[free] = (ix, iy)
                self.redraw()

    def on_left_drag(self, e):
        if self.dragging_idx:
            kind, j = self.dragging_idx
            ix, iy = self.d2i(e.x, e.y)
            if kind == "scale":
                self.scale_pts[j] = (ix, iy)
            else:
                self.slots[j] = (ix, iy)
            self.redraw()

    def on_left_release(self, e):
        self.dragging_idx = None

    def on_right_press(self, e):
        self.dragging_canvas = True
        self._last = (e.x, e.y)

    def on_right_drag(self, e):
        if self.dragging_canvas:
            dx = e.x - self._last[0]
            dy = e.y - self._last[1]
            self.offset[0] += dx
            self.offset[1] += dy
            self._last = (e.x, e.y)
            self.redraw("fast")

    def on_right_release(self, e):
        self.dragging_canvas = False

    def on_wheel(self, e):
        if (e.state & 0x0004) == 0:
            return
        factor = 1.1 if e.delta > 0 else (1 / 1.1)
        mx, my = e.x, e.y
        ix, iy = self.d2i(mx, my)
        self.zoom = max(0.1, min(20.0, self.zoom * factor))
        dx, dy = self.i2d(ix, iy)
        self.offset[0] += (mx - dx)
        self.offset[1] += (my - dy)
        self.redraw("fast")
        if hasattr(self, "_zoom_hq_timer") and self._zoom_hq_timer:
            try:
                self.after_cancel(self._zoom_hq_timer)
            except Exception:
                pass
        self._zoom_hq_timer = self.after(70, lambda: self.redraw("hq"))

    def _pick(self, arr, dx, dy, tol=8):
        best = None
        bestd = 1e9
        for j, (x, y) in enumerate(arr):
            px, py = self.i2d(x, y)
            d = (px - dx) ** 2 + (py - dy) ** 2
            if d < bestd and d <= (max(self.radius, tol)) ** 2:
                best, bestd = j, d
        return best

    def _pick_slots(self, dx, dy, tol=8):
        best = None
        bestd = 1e9
        for j, pt in enumerate(self.slots):
            if pt is None:
                continue
            x, y = pt
            px, py = self.i2d(x, y)
            d = (px - dx) ** 2 + (py - dy) ** 2
            if d < bestd and d <= (max(self.radius, tol)) ** 2:
                best, bestd = j, d
        return best

    def finish_scale(self, *_):
        if len(self.scale_pts) == 2:
            self.save_current()
            messagebox.showinfo("Scale saved", "Scale file saved. Starting landmarking.")
            self.scale_mode = False
            self.apply_banner()
            self.redraw("hq")

    def on_key(self, e):
        k = e.keysym.lower()
        if k in ("plus", "equal"):
            self.radius = min(30, self.radius + 1)
            self.redraw()
        elif k == "minus":
            self.radius = max(2, self.radius - 1)
            self.redraw()
        elif k == "return" and self.scale_mode:
            self.finish_scale()
        elif k == "right":
            self.next_image()
        elif k == "left":
            self.prev_image()
        elif k == "delete" and (not self.scale_mode):
            cx, cy = self._pointer_on_canvas()
            j = self._pick_slots(cx, cy)
            if j is not None:
                self.slots[j] = None
                self.redraw()
        elif k == "b" and (e.state & 0x0004) and (not self.scale_mode):
            move_to_bad(self.img_path)
            self.after(10, self.next_image)
        elif k == "h":
            messagebox.showinfo(
                "Help",
                "F9: QC swap-only | tol: fraction (e.g., 0.25)\n"
                "Left/Right: prev/next | LMB: add/drag | RMB drag: pan | Ctrl+wheel: zoom | +/-: size\n"
                "Delete/Ctrl+D: delete | X/Ctrl+B: move to bad | Ctrl+S: save | Enter in Scale: save scale | F: search",
            )

    def on_key_up(self, e):
        pass

    def ctrl_save(self, *_):
        self.save_current()

    def delete_near_force(self, *_):
        cx, cy = self._pointer_on_canvas()
        j = self._pick_slots(cx, cy)
        if j is not None:
            self.slots[j] = None
            self.redraw()

    # ---------- QC core ----------
    def set_qc_tol(self):
        s = (self.qcTolVar.get() or "").strip().replace(",", ".")
        try:
            v = float(s)
        except Exception:
            messagebox.showinfo("QC tol", "Enter a number like 0.25")
            return
        v = max(0.0, min(0.9, v))
        self.qc_tol = v
        try:
            self.qcTolLabel.configure(text=f"tol={self.qc_tol:.2f}")
        except Exception:
            pass
        if self.qc_mode:
            self._qc_refresh_and_locate()

    def run_qc(self):
        # After mini-GPA: LM_k is closer to centroid of LM_j than to its own LM_k by >= tol
        N = self.N
        imgs = self.images
        shapes, idxs = [], []
        for i, fp in enumerate(imgs):
            pts = read_any_csv_points(Path(fp).with_suffix(".csv"))
            if len(pts) == N:
                shapes.append(pts)
                idxs.append(i)
        self.qc_list = []
        self.qc_marks = {}
        if len(shapes) < 2:
            return
        aligned, mean = gpa_align(shapes, iters=10)
        problems = []
        eps = 1e-9
        for idx_img, sh in zip(idxs, aligned):
            reasons = []
            marks = []
            for k, (x, y) in enumerate(sh):
                mx, my = mean[k]
                d_self = ((x - mx) ** 2 + (y - my) ** 2) ** 0.5
                best_d = 1e18
                best_j = None
                for j, (qx, qy) in enumerate(mean):
                    if j == k:
                        continue
                    d = ((x - qx) ** 2 + (y - qy) ** 2) ** 0.5
                    if d < best_d:
                        best_d, best_j = d, j
                # margin rule
                if best_d + eps < d_self * (1.0 - self.qc_tol):
                    reasons.append(f"LM{k + 1} closer to LM{best_j + 1}")
                    marks.append((k, best_j))
            if reasons:
                problems.append((idx_img, "; ".join(reasons)))
                self.qc_marks[idx_img] = marks
        problems.sort(key=lambda x: x[0])
        self.qc_list = problems
        out_path = Path(self.root_dir) / "tools/2_Landmarking_v1.0/logs" / "qc_last.txt"
        out_path.write_text(
            "\n".join([f"{Path(self.images[i]).name}: {r}" for i, r in self.qc_list]),
            encoding="utf-8",
        )

    def _qc_refresh_and_locate(self):
        self.run_qc()
        if not self.qc_list:
            self.qc_mode = False
            self.apply_banner()
            messagebox.showinfo("QC finished", "No suspicious frames left.")
            self.redraw("hq")
            return False
        idxs = [i for i, _ in self.qc_list]
        if self.idx in idxs:
            self.qc_pos = idxs.index(self.idx)
        else:
            pos = 0
            for p, i in enumerate(idxs):
                if i >= self.idx:
                    pos = p
                    break
            self.qc_pos = pos
            self.idx = idxs[self.qc_pos]
            self.load_image(self.images[self.idx])
        self.apply_banner()
        return True

    def toggle_qc(self, *_):
        if self.scale_mode:
            messagebox.showinfo("Scale first", "Finish Scale Wizard first.")
            return
        if not self.qc_mode:
            if not self._qc_refresh_and_locate():
                return
            self.qc_mode = True
            self.apply_banner()
            self.load_image(self.images[self.idx])
        else:
            self.qc_mode = False
            self.apply_banner()
            self.redraw("hq")

    # navigation
    def prev_image(self):
        if not self.auto_save_on_switch():
            return
        if self.qc_mode:
            if not self._qc_refresh_and_locate():
                return
            self.qc_pos = (self.qc_pos - 1) % len(self.qc_list)
            self.idx = self.qc_list[self.qc_pos][0]
            self.load_image(self.images[self.idx])
            return
        if self.idx > 0:
            self.idx -= 1
            self.load_image(self.images[self.idx])

    def next_image(self):
        if not self.auto_save_on_switch():
            return
        if self.qc_mode:
            if not self._qc_refresh_and_locate():
                return
            self.qc_pos = (self.qc_pos + 1) % len(self.qc_list)
            self.idx = self.qc_list[self.qc_pos][0]
            self.load_image(self.images[self.idx])
            return
        if self.idx < len(self.images) - 1:
            self.idx += 1
            self.load_image(self.images[self.idx])

    # search / bad / close
    def search_dialog(self, *_):
        q = simpledialog.askstring("Search", "Enter filename (part, no extension):", parent=self)
        if not q:
            return
        q = q.lower()
        for i, fp in enumerate(self.images):
            s = Path(fp).stem.lower()
            if (q in s) or (s == q):
                if not self.auto_save_on_switch():
                    return
                self.idx = i
                self.load_image(self.images[self.idx])
                return
        messagebox.showinfo("Not found", f"No file matching: {q}")

    def hot_move_to_bad(self, *_):
        if not self.scale_mode:
            move_to_bad(self.img_path)
            self.after(10, self.next_image)

    def on_close(self):
        try:
            self.save_current()
        finally:
            self.destroy()

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--images", required=True)
    ap.add_argument("--n-points", type=int, default=None)
    ap.add_argument("--start-from", default=None)
    ap.add_argument("--scale-wizard", action="store_true")
    a = ap.parse_args()
    N = a.n_points or read_npoints(a.root)
    if not N or N <= 1:
        print("[ERR] invalid N", file=sys.stderr)
        return 2
    app = AnnotGUI(a.root, a.images, N, start_from=a.start_from, scale_wizard=a.scale_wizard)
    app.mainloop()
    return 0

if __name__ == "__main__":
    sys.exit(main())