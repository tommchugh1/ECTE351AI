# Triple_I_APP.py — Flask (streamvideo.py) streaming + optional MediaMTX + Realtime_Filtering

import tkinter as tk
import tkinter.font as tkfont
from tkinter import messagebox, filedialog
from PIL import Image, ImageTk
import os
import webbrowser
import subprocess
import sys
import atexit
from pathlib import Path
from typing import Optional

# ===================== USER DATA =====================

USERS = {
    "Joy Pasala": "7452408",
    "Jonathan Walsh": "pass1",
    "Tom Mchugh": "6413717",
    "Jacob Rhados": "8002812",
    "Jerome Eid": "pass4",
    "Jason Watson": "7678721",
}

# ===================== PATHS / CONSTANTS =====================

BG = "#ffffff"
BTN_COLOR = "#007ACC"
BTN_TEXT = "white"
ENTRY_BG = "white"
REMEMBER_FILE = "remember_me.txt"

# Absolute script paths (update as needed)
SCRIPT_PATHS = {
    "execute":   Path(r"C:\Users\Group8\Desktop\Yolo Demo\ECTE351AI\AI\execute.py"),
    "train":     Path(r"C:\Users\Group8\Desktop\Yolo Demo\ECTE351AI\AI\train.py"),
    "auto":      Path(r"C:\Users\Group8\Desktop\Yolo Demo\ECTE351AI\VideoProcessing\AutomateKdenlive.py"),
    "xml2yolo":  Path(r"C:\Users\Group8\Desktop\Yolo Demo\ECTE351AI\VideoProcessing\KdenliveXMLtoYOLOv8.py"),
    "viz_one":   Path(r"C:\Users\Group8\Desktop\Yolo Demo\ECTE351AI\VideoProcessing\visualiseTXT.py"),
    "viz_batch": Path(r"C:\Users\Group8\Desktop\Yolo Demo\ECTE351AI\VideoProcessing\BatchVisualiseTXTBB.py"),
    # Optional local stream launcher you had before
    "stream":    Path(r"C:\Users\Group8\Desktop\Yolo Demo\ECTE351AI\GUI\Video_Feed.py"),
    # Realtime filtering script
    "realtime":  Path(r"C:\Users\Group8\Desktop\Yolo Demo\ECTE351AI\GUI\Realtime_Filtering.py"),
}

# Pi streaming folder (contains streamvideo.py)
PI_MEDIA_DIR = Path(r"C:\Users\Group8\Desktop\Yolo Demo\ECTE351AI\Pi\mediamtx")
STREAM_SCRIPT = PI_MEDIA_DIR / "streamvideo.py"  # Flask server

# Visual scale
TITLE_FONT     = ("Arial", 28, "bold")
SECTION_FONT   = ("Arial", 24, "bold")
LABEL_FONT     = ("Arial", 14, "bold")
BTN_FONT       = ("Arial", 16, "bold")

TILE_W, TILE_H = 200, 180
TILE_PADX, TILE_PADY = 28, 20

# Dashboard palette
COLORS = {
    "grey":  "#e0e0e0",
    "blue":  "#e8f5ff",
    "peach": "#ffe0b2",
    "green": "#c8e6c9",
    "pink":  "#ffcdd2",
}

# Logo (change if needed)
LOGO_PATH = r"C:\Users\joypa\Downloads\logo_final.jpg"

# Live feed URL (Flask streamvideo.py serves this)
LIVE_FEED_URL = "http://localhost:5000/video_feed"

# Optional RTSP (if you later use MediaMTX)
RTSP_URL = "rtsp://localhost:8554/bolts"

# ===================== ROOT =====================

root = tk.Tk()
root.title("YourQualityCheck")
root.state("zoomed")
root.configure(bg=BG)
root.resizable(False, False)

# Fonts for hover transitions
_icon_normal = tkfont.Font(family="Arial", size=36, weight="normal")
_icon_bold   = tkfont.Font(family="Arial", size=36, weight="bold")

# Keep PhotoImage refs alive
_IMG_REFS: list[ImageTk.PhotoImage] = []
def keep_image_ref(img):
    _IMG_REFS.append(img)

# ===================== HELPERS =====================

def run_script(path: Path, extra_args=None, extra_env=None):
    """Launch a Python script in a separate process."""
    if not isinstance(path, Path):
        path = Path(path)
    if not path.exists():
        messagebox.showerror("Error", f"File not found:\n{path}")
        return
    try:
        args = [sys.executable, str(path)]
        if extra_args:
            args.extend(extra_args)
        env = os.environ.copy()
        if extra_env:
            env.update(extra_env)
        subprocess.Popen(args, cwd=str(path.parent), shell=True, env=env)
    except Exception as e:
        messagebox.showerror("Error", f"Failed to run {path.name}:\n{e}")

def open_live_feed():
    try:
        webbrowser.open(LIVE_FEED_URL)
    except Exception as e:
        messagebox.showerror("Error", f"Failed to open live feed URL:\n{e}")

def hex_shift(hex_color: str, pct: float) -> str:
    h = hex_color.lstrip("#")
    r, g, b = int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)
    clamp = lambda x: max(0, min(255, int(x)))
    r, g, b = clamp(r + pct*255), clamp(g + pct*255), clamp(b + pct*255)
    return f"#{r:02x}{g:02x}{b:02x}"

def clear_root():
    for w in root.winfo_children():
        w.destroy()

# ---------- Flask stream (streamvideo.py) management ----------

_flask_proc: Optional[subprocess.Popen] = None

def start_flask_stream():
    """Start Flask MJPEG stream (Pi/mediamtx/streamvideo.py)."""
    global _flask_proc
    try:
        if not STREAM_SCRIPT.exists():
            messagebox.showerror(
                "Stream",
                f"Could not find streamvideo.py:\n{STREAM_SCRIPT}\n\n"
                "Please check the path."
            )
            return
        if _flask_proc and _flask_proc.poll() is None:
            messagebox.showinfo("Stream", "Flask stream is already running.")
            return
        _flask_proc = subprocess.Popen(
            [sys.executable, str(STREAM_SCRIPT)],
            cwd=str(STREAM_SCRIPT.parent),
            shell=True
        )
        messagebox.showinfo("Stream", "Flask stream started.\nOpening the live feed page…")
        open_live_feed()
    except Exception as e:
        messagebox.showerror("Stream", f"Failed to start Flask stream:\n{e}")

def stop_flask_stream():
    """Stop Flask MJPEG stream if running."""
    global _flask_proc
    try:
        if _flask_proc and _flask_proc.poll() is None:
            _flask_proc.terminate()
            _flask_proc = None
            messagebox.showinfo("Stream", "Flask stream stopped.")
        else:
            messagebox.showinfo("Stream", "Flask stream is not running.")
    except Exception as e:
        messagebox.showerror("Stream", f"Failed to stop Flask stream:\n{e}")

# ---------- Optional MediaMTX helpers (kept for future use) ----------

_MEDIAMTX_HINT_FILE = Path(__file__).with_name("mediamtx_dir.txt")
_mediamtx_proc: Optional[subprocess.Popen] = None

def _load_mediamtx_dir() -> Path:
    try:
        if _MEDIAMTX_HINT_FILE.exists():
            p = Path(_MEDIAMTX_HINT_FILE.read_text(encoding="utf-8").strip().strip('"'))
            if p.exists():
                return p
    except Exception:
        pass
    return PI_MEDIA_DIR

def _save_mediamtx_dir(folder: Path) -> None:
    try:
        _MEDIAMTX_HINT_FILE.write_text(str(folder), encoding="utf-8")
    except Exception:
        pass

def _resolve_mediamtx_exe(folder: Path) -> Optional[Path]:
    names = ["mediamtx.exe", "rtsp-simple-server.exe"] if os.name == "nt" else ["mediamtx"]
    for n in names:
        cand = folder / n
        if cand.exists():
            return cand
    for n in names:
        for p in folder.rglob(n):
            if p.is_file():
                return p
    if os.name == "nt":
        for p in folder.rglob("mediamtx*.exe"):
            if p.is_file():
                return p
    return None

def set_mediamtx_dir():
    d = filedialog.askdirectory(title="Select the folder that contains mediamtx")
    if not d:
        return
    folder = Path(d)
    exe = _resolve_mediamtx_exe(folder)
    if not exe and not (folder / "streamvideo.py").exists():
        messagebox.showerror(
            "MediaMTX",
            "Neither mediamtx executable nor streamvideo.py found in that folder."
        )
        return
    _save_mediamtx_dir(folder)
    messagebox.showinfo("MediaMTX", f"Saved folder:\n{folder}")

def start_mediamtx():
    """Try to start MediaMTX if available; otherwise fall back to Flask stream."""
    global _mediamtx_proc
    try:
        folder = _load_mediamtx_dir()
        exe = _resolve_mediamtx_exe(folder)
        if not exe:
            # No mediamtx → fall back gracefully to Flask stream
            start_flask_stream()
            return
        if _mediamtx_proc and _mediamtx_proc.poll() is None:
            messagebox.showinfo("MediaMTX", "MediaMTX is already running.")
            return
        _mediamtx_proc = subprocess.Popen([str(exe)], cwd=str(exe.parent), shell=True)
        messagebox.showinfo("MediaMTX", f"MediaMTX started from:\n{exe.parent}")
    except Exception as e:
        messagebox.showerror("MediaMTX", f"Failed to start MediaMTX:\n{e}")

def stop_mediamtx():
    global _mediamtx_proc
    try:
        if _mediamtx_proc and _mediamtx_proc.poll() is None:
            _mediamtx_proc.terminate()
            _mediamtx_proc = None
            messagebox.showinfo("MediaMTX", "MediaMTX stopped.")
        else:
            messagebox.showinfo("MediaMTX", "MediaMTX is not running.")
    except Exception as e:
        messagebox.showerror("MediaMTX", f"Failed to stop MediaMTX:\n{e}")

# ---------- Realtime Filtering (still supported) ----------

def run_realtime_filtering():
    path = SCRIPT_PATHS.get("realtime")
    if not path or not path.exists():
        messagebox.showerror("Realtime Filtering", f"Realtime_Filtering.py not found:\n{path}")
        return
    # If you later switch to RTSP via MediaMTX, pass RTSP_URL; for now it can use default camera.
    run_script(path, extra_args=[RTSP_URL], extra_env={"RTSP_URL": RTSP_URL})

def open_rtsp_in_player():
    try:
        if os.name == "nt":
            os.startfile(RTSP_URL)  # type: ignore[attr-defined]
        else:
            webbrowser.open(RTSP_URL)
    except Exception:
        webbrowser.open(RTSP_URL)

def _shutdown():
    try:
        stop_flask_stream()
    except Exception:
        pass
    try:
        stop_mediamtx()
    except Exception:
        pass

atexit.register(_shutdown)

# ===================== TILES (with hover) =====================

def make_tile(parent, title, icon_text, bg_color, command):
    container = tk.Frame(parent, bg="white")
    tile = tk.Frame(
        container, bg=bg_color, width=TILE_W, height=TILE_H,
        highlightthickness=1, highlightbackground="#b0b0b0", relief="flat", bd=2
    )
    tile.pack_propagate(False)
    tile.pack(padx=8, pady=(0, 8))

    icon_lbl = tk.Label(tile, text=icon_text, bg=bg_color, fg="black", font=_icon_normal)
    icon_lbl.pack(expand=True)

    title_lbl = tk.Label(container, text=title, font=LABEL_FONT, bg="white")
    title_lbl.pack()

    hover_bg = hex_shift(bg_color, -0.06)
    def on_enter(_): tile.configure(bg=hover_bg, relief="raised"); icon_lbl.configure(bg=hover_bg, font=_icon_bold)
    def on_leave(_): tile.configure(bg=bg_color, relief="flat");   icon_lbl.configure(bg=bg_color, font=_icon_normal)
    def on_click(_): command()

    for w in (tile, icon_lbl):
        w.bind("<Enter>", on_enter)
        w.bind("<Leave>", on_leave)
        w.bind("<Button-1>", on_click)

    return container

# ===================== SPLASH / LOGIN =====================

def show_splash():
    clear_root()
    splash = tk.Frame(root, bg=BG)
    splash.pack(fill="both", expand=True)

    tk.Label(splash, text="Welcome to YourQualityCheck", font=TITLE_FONT, bg=BG)\
      .pack(pady=(20, 10))

    try:
        img = Image.open(LOGO_PATH).resize((320, 320), Image.Resampling.LANCZOS)
        logo = ImageTk.PhotoImage(img)
        keep_image_ref(logo)
        tk.Label(splash, image=logo, bg=BG).pack(pady=(0, 6))
        tk.Label(splash, text="Sponsored by Triple - I", font=("Arial", 12), bg=BG).pack()
    except Exception:
        tk.Label(splash, text="[Logo not found]", font=("Arial", 18), bg=BG).pack(pady=20)

    root.after(1000, show_login)

def show_login():
    clear_root()
    login = tk.Frame(root, bg=BG)
    login.place(relx=0.5, rely=0.40, anchor="center")  # slightly higher

    try:
        img = Image.open(LOGO_PATH).resize((220, 220), Image.Resampling.LANCZOS)
        login_logo = ImageTk.PhotoImage(img)
        keep_image_ref(login_logo)
        tk.Label(login, image=login_logo, bg=BG).grid(row=0, column=0, columnspan=3, pady=(0, 8))
        tk.Label(login, text="Sponsored by Triple - I", bg=BG, font=("Arial", 12)).grid(row=1, column=0, columnspan=3, pady=(0, 14))
    except Exception:
        tk.Label(login, text="QuAck", font=("Arial", 22, "bold"), bg=BG).grid(row=0, column=0, columnspan=3, pady=(0, 12))

    tk.Label(login, text="User Name:", bg=BG, font=("Arial", 16)).grid(row=2, column=0, padx=12, pady=8, sticky="e")
    user_entry = tk.Entry(login, font=("Arial", 16), width=28, bg=ENTRY_BG)
    user_entry.grid(row=2, column=1, columnspan=2, sticky="w")
    if os.path.exists(REMEMBER_FILE):
        try:
            user_entry.insert(0, Path(REMEMBER_FILE).read_text(encoding="utf-8").strip())
        except Exception:
            pass

    tk.Label(login, text="Password:", bg=BG, font=("Arial", 16)).grid(row=3, column=0, padx=12, pady=8, sticky="e")
    pass_entry = tk.Entry(login, show="*", font=("Arial", 16), width=28, bg=ENTRY_BG)
    pass_entry.grid(row=3, column=1, columnspan=2, sticky="w")

    def toggle_pw():
        if pass_entry.cget("show") == "*":
            pass_entry.config(show=""); show_btn.config(text="Hide Password")
        else:
            pass_entry.config(show="*"); show_btn.config(text="Show Password")

    show_btn = tk.Button(login, text="Show Password", command=toggle_pw,
                         font=("Arial", 12), bg=BTN_COLOR, fg="white")
    show_btn.grid(row=4, column=0, columnspan=3, pady=(6, 4))

    remember_var = tk.BooleanVar(value=os.path.exists(REMEMBER_FILE))
    tk.Checkbutton(login, text="Remember Me", variable=remember_var, bg=BG, font=("Arial", 12))\
      .grid(row=5, column=0, columnspan=3)

    def forgot_pw(_=None):
        uname = user_entry.get().strip()
        if uname in USERS:
            messagebox.showinfo("Password Recovery", f"Password for {uname}: {USERS[uname]}")
        else:
            messagebox.showerror("Error", "Enter a valid username first.")
    lbl_fp = tk.Label(login, text="Forgot Password?", fg="blue", cursor="hand2", bg=BG, font=("Arial", 12))
    lbl_fp.grid(row=6, column=0, columnspan=3, pady=(0, 10))
    lbl_fp.bind("<Button-1>", forgot_pw)

    def do_login(_evt=None):
        uname = user_entry.get().strip()
        pwd = pass_entry.get()
        if uname in USERS and USERS[uname] == pwd:
            if remember_var.get():
                try: Path(REMEMBER_FILE).write_text(uname, encoding="utf-8")
                except Exception: pass
            else:
                try:
                    if os.path.exists(REMEMBER_FILE):
                        Path(REMEMBER_FILE).unlink()
                except Exception:
                    pass
            show_dashboard(uname)
        else:
            messagebox.showerror("Access Denied", "Invalid username or password.")

    tk.Button(login, text="Login", command=do_login, font=("Arial", 18, "bold"),
              width=14, bg=BTN_COLOR, fg="white").grid(row=7, column=0, columnspan=3, pady=(8, 4))
    root.bind("<Return>", do_login)
    user_entry.focus_set()

# ===================== DASHBOARD =====================

def show_dashboard(username):
    clear_root()
    main = tk.Frame(root, bg="white")
    main.pack(fill="both", expand=True)

    tk.Label(main, text="Welcome to YourQualityCheck", font=TITLE_FONT, bg="white").pack(pady=(18, 6))
    try:
        img = Image.open(LOGO_PATH).resize((220, 220), Image.Resampling.LANCZOS)
        logo = ImageTk.PhotoImage(img)
        keep_image_ref(logo)
        tk.Label(main, image=logo, bg="white").pack()
        tk.Label(main, text="Sponsored by Triple - I", bg="white", font=("Arial", 12)).pack(pady=(2, 26))
    except Exception:
        tk.Label(main, text="QuAck", font=("Arial", 22, "bold"), bg="white").pack(pady=(2, 26))

    tiles_frame = tk.Frame(main, bg="white")
    tiles_frame.pack()

    tiles = [
        ("Profile",       "👤", COLORS["grey"],  lambda: render_section("profile", username)),
        ("Inventory",     "🧊", COLORS["blue"],  lambda: render_section("inventory", username)),
        ("Camera Feed",   "📷", COLORS["peach"], lambda: render_section("camera", username)),
        ("Photo Gallery", "🖼", COLORS["green"], lambda: render_section("gallery", username)),
        ("Logout",        "📱", COLORS["pink"],  show_login),
    ]
    for i, (title, icon, color, cmd) in enumerate(tiles):
        make_tile(tiles_frame, title, icon, color, cmd).grid(row=0, column=i, padx=TILE_PADX, pady=TILE_PADY)

# ===================== SECTION RENDERING =====================

def render_section(section, username):
    clear_root()

    sidebar = tk.Frame(root, bg="#bdbdbd", width=200)
    sidebar.pack(side="left", fill="y")
    sidebar.pack_propagate(False)

    def sbtn(text, active, cmd):
        tk.Button(sidebar, text=text, font=("Arial", 14, "bold" if active else "normal"),
                  bg="#eeeeee" if active else "#bdbdbd", relief="flat",
                  anchor="w", padx=14, pady=12, command=cmd).pack(fill="x", pady=2)

    sbtn("← Dashboard", False, lambda: show_dashboard(username))
    sbtn("👤 Profile",   section=="profile",   lambda: render_section("profile", username))
    sbtn("📦 Inventory", section=="inventory", lambda: render_section("inventory", username))
    sbtn("📷 Camera",    section=="camera",    lambda: render_section("camera", username))
    sbtn("🖼 Gallery",   section=="gallery",   lambda: render_section("gallery", username))
    sbtn("🚪 Logout",    False,                show_login)

    content = tk.Frame(root, bg="white")
    content.pack(side="right", expand=True, fill="both")

    tk.Label(content, text={
        "profile": "👤 Profile",
        "inventory": "📦 Model Management",
        "camera": "📷 Camera Feed",
        "gallery": "🖼 Dataset Tools"
    }.get(section, "Section"), font=SECTION_FONT, bg="white").pack(pady=(18, 10))

    tiles_frame = tk.Frame(content, bg="white")
    tiles_frame.pack(pady=10)

    if section == "profile":
        tk.Label(content, text=f"User: {username}", font=("Arial", 16), bg="white").pack(pady=(0, 8))
        make_tile(tiles_frame, "Back to Dashboard", "🏠", COLORS["grey"],
                  lambda: show_dashboard(username)).grid(row=0, column=0, padx=TILE_PADX, pady=TILE_PADY)

    elif section == "inventory":
        tiles = [
            ("Train Model",          "📚", COLORS["blue"],  lambda: run_script(SCRIPT_PATHS["train"])),
            ("Automate Annotations", "⚙",  COLORS["green"], lambda: run_script(SCRIPT_PATHS["auto"])),
            ("Convert XML→YOLO",     "📂", COLORS["peach"], lambda: run_script(SCRIPT_PATHS["xml2yolo"])),
        ]
        for i, (title, icon, color, cmd) in enumerate(tiles):
            make_tile(tiles_frame, title, icon, color, cmd).grid(row=0, column=i, padx=TILE_PADX, pady=TILE_PADY)

    elif section == "camera":
        tiles = [
            ("Set Media Folder",     "📁", COLORS["grey"],  set_mediamtx_dir),
            ("Start Flask Stream",   "▶️", COLORS["green"], start_flask_stream),
            ("Stop Flask Stream",    "⏹", COLORS["pink"],  stop_flask_stream),
            ("Open Live Feed",       "🌐", COLORS["peach"], open_live_feed),
            ("Run Inference",        "🎯", COLORS["blue"],  lambda: run_script(SCRIPT_PATHS["execute"])),
            # Optional RTSP / MediaMTX if you add the binary later:
            ("Start MediaMTX",       "🟢", COLORS["green"], start_mediamtx),
            ("Stop MediaMTX",        "⛔", COLORS["pink"],  stop_mediamtx),
            ("Realtime Filtering",   "🪄", COLORS["blue"],  run_realtime_filtering),
            ("Open RTSP in VLC",     "🎬", COLORS["peach"], open_rtsp_in_player),
        ]
        for i, (title, icon, color, cmd) in enumerate(tiles):
            r, c = divmod(i, 3)  # 3 per row
            make_tile(tiles_frame, title, icon, color, cmd).grid(row=r, column=c, padx=TILE_PADX, pady=TILE_PADY)

    elif section == "gallery":
        tiles = [
            ("Visualise (Single)", "🔍", COLORS["green"], lambda: run_script(SCRIPT_PATHS["viz_one"])),
            ("Batch Visualise",    "🖼", COLORS["pink"],  lambda: run_script(SCRIPT_PATHS["viz_batch"])),
        ]
        for i, (title, icon, color, cmd) in enumerate(tiles):
            make_tile(tiles_frame, title, icon, color, cmd).grid(row=0, column=i, padx=TILE_PADX, pady=TILE_PADY)

# ===================== START =====================

def main():
    show_splash()
    root.mainloop()

if __name__ == "__main__":
    main()
