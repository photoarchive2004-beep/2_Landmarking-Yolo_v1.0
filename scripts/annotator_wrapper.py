from __future__ import annotations

import os
import sys
import subprocess
from pathlib import Path
import argparse


def parse_args() -> argparse.Namespace:
    """
    Simple pass-through for annotator CLI arguments.
    We forward them to annot_gui_custom.py.
    """
    parser = argparse.ArgumentParser(
        description="GM Landmarking: annotator wrapper (adds Review finished flag in REVIEW_AUTO mode)."
    )
    parser.add_argument("--root", required=False, help="Project root (passed through to GUI).")
    parser.add_argument("--images", required=True, help="Path to PNG images for selected locality.")
    parser.add_argument("--first", required=False, help="Optional first image name (if used by GUI).")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    exe = sys.executable

    # LANDMARK_ROOT = tools/2_Landmarking_v1.0
    tool_dir = Path(__file__).resolve().parent.parent
    gui = tool_dir / "annot_gui_custom.py"
    if not gui.exists():
        print(f"[ERR] annotator GUI not found: {gui}")
        sys.exit(1)

    mode = os.environ.get("GM_MODE", "").upper()
    locality = (os.environ.get("GM_LOCALITY") or "").strip()

    # Build command for original GUI
    cmd: list[str] = [exe, str(gui)]
    if args.root:
        cmd += ["--root", args.root]
    cmd += ["--images", args.images]
    if args.first:
        cmd += ["--first", args.first]

    # Normal/manual mode: just run GUI as before
    if mode != "REVIEW_AUTO" or not locality:
        rc = subprocess.call(cmd)
        sys.exit(rc)

    # REVIEW_AUTO mode: start GUI + small window with "Review finished" button.
    print()
    print(f'Starting annotator in REVIEW_AUTO mode for locality "{locality}"...')
    print("When you finish checking auto landmarks, press 'Review finished' in the small window.")
    print()

    try:
        gui_proc = subprocess.Popen(cmd)
    except Exception as exc:
        print("[ERR] Cannot start annotator GUI:")
        print(f"      {exc}")
        sys.exit(1)

    # Try to use Tkinter for a small helper window
    try:
        import tkinter as tk
        from tkinter import ttk
    except Exception:
        print("[WARN] tkinter is not available, falling back to console confirmation.")
        _confirm_via_console(locality)
        rc = gui_proc.wait()
        sys.exit(rc)

    def on_finish() -> None:
        _create_review_flag(locality)
        root_win.destroy()

    def on_close_without() -> None:
        # Just close helper window, do not create flag
        root_win.destroy()

    root_win = tk.Tk()
    root_win.title("GM Landmarking: Review AUTO")

    msg = (
        f'Locality "{locality}"\n\n'
        "1) Check and fix all AUTO landmarks in the main annotator window.\n"
        "2) When you are sure, click \"Review finished\" here.\n\n"
        "If you close this window without the button, MANUAL status will NOT be set."
    )
    label = ttk.Label(root_win, text=msg, justify="left")
    label.pack(padx=12, pady=12)

    btn_finish = ttk.Button(root_win, text="Review finished", command=on_finish)
    btn_finish.pack(pady=(0, 6))

    btn_cancel = ttk.Button(root_win, text="Close without flag", command=on_close_without)
    btn_cancel.pack(pady=(0, 12))

    # Hotkeys: Enter / Ctrl+Enter => Review finished
    root_win.bind("<Return>", lambda event: on_finish())
    root_win.bind("<Control-Return>", lambda event: on_finish())

    root_win.mainloop()

    # After helper window is closed we just wait for GUI to finish
    rc = gui_proc.wait()
    sys.exit(rc)


def _create_review_flag(locality: str) -> None:
    """
    Create status/review_done_<locality>.flag relative to LANDMARK_ROOT
    (tools/2_Landmarking_v1.0), as required in ТЗ_1.0.
    """
    tool_dir = Path(__file__).resolve().parent.parent
    flag_dir = tool_dir / "status"
    try:
        flag_dir.mkdir(parents=True, exist_ok=True)
    except Exception:
        # If directory already exists or cannot be created, we still try to write file.
        pass

    flag_path = flag_dir / f"review_done_{locality}.flag"
    try:
        flag_path.write_text("review finished\n", encoding="utf-8")
        print(f'[INFO] Review flag created: {flag_path}')
    except Exception as exc:
        print(f"[ERR] Cannot create review flag for locality '{locality}': {exc}")


def _confirm_via_console(locality: str) -> None:
    """
    Fallback variant if Tkinter is not available:
    ask in console and create flag only if user clearly confirms.
    """
    ans = input(
        f'Did you finish review for locality "{locality}"? '
        'Type "YES" to create MANUAL flag: '
    ).strip().upper()
    if ans == "YES":
        _create_review_flag(locality)
    else:
        print("[INFO] Review not confirmed, flag was not created.")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        sys.exit(0)
