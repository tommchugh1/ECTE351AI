# Triple_I_APP.py — Single-Stream (Flask) integration, polished UI

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
import socket
import time
from openFeed import open_stream_popup

# ===================== USER DATA =====================

USERS = {
    "Joy Pasala": "7452408",
    "Jonathan Walsh": "pass1",
    "Tom McHugh": "6413717",
    "Jacob Rhados": "8002812",
    "Jerome Eid": "pass4",
    "Jason Watson": "7678721",
}

# ===================== PATHS / CONSTANTS =====================

BG = "#ffffff"
BTN_COLOR = "#007ACC"
ENTRY_BG = "white"
REMEMBER_FILE = "remember_me.txt"

MEDIAMTX_HOST = "0.0.0.0"
RTSP_PORT = 8889
CHECK_INTERVAL_MS = 800

# Relative script paths
working_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

SCRIPT_PATHS = {
    "execute":   Path(os.path.join(working_dir, "AI", "execute.py")),
    "train":     Path(os.path.join(working_dir, "AI", "train.py")),
    "auto":      Path(os.path.join(working_dir, "VideoProcessing", "AutomateKdenlive.py")),
    "xml2yolo":  Path(os.path.join(working_dir, "VideoProcessing", "KdenliveXMLtoYOLOv8.py")),
    "viz_one":   Path(os.path.join(working_dir, "VideoProcessing", "visualiseTXT.py")),
    "viz_batch": Path(os.path.join(working_dir, "VideoProcessing", "BatchVisualiseTXTBB.py")),
    # Optional local preview launcher; if absent we just open the URL
    "stream":    Path(os.path.join(working_dir, "PI", "streamFeed.py")),
    # Realtime filtering script (OpenCV)
    "realtime":  Path(os.path.join(working_dir, "GUI", "Realtime_Filtering.py")),
}

# Folder that should contain streamvideo.py (you can change in-app)
DEFAULT_STREAM_DIR = Path(os.path.join(working_dir, "PI", "mediamtx"))
STREAM_DIR_FILE = Path("stream_folder.txt")  # remembers the chosen folder
STREAM_FILENAME = "testStream.py"           # Streaming 
STREAM_URL_HTTP = "http://10.12.10.242:8889/cam1"

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
logo_path = os.path.join(working_dir, "GUI", "logo_final.jpg")

# ---- MediaMTX headless controller (no UI) ----
class MediaMTXController:
    def __init__(self, host=MEDIAMTX_HOST, port=RTSP_PORT, folder=None):
        self.host = host
        self.port = port
        self.proc = None
        # Where mediamtx + mediamtx.yml live; default to same folder as your PI/mediamtx
        if folder is None:
            self.folder = Path(os.path.join(working_dir, "PI", "mediamtx"))
        else:
            self.folder = Path(folder)
        self.binary = self.folder / ("mediamtx.exe" if sys.platform.startswith("win") else "mediamtx")
        self.config = self.folder / "mediamtx.yml"

    def start(self):
        if self.proc and self.proc.poll() is None:
            return
        if not self.binary.exists():
            messagebox.showerror("MediaMTX", f"Binary not found:\n{self.binary}")
            return
        if not self.config.exists():
            messagebox.showerror("MediaMTX", f"Config not found:\n{self.config}")
            return
        try:
            with open(os.devnull, "wb") as devnull:
                self.proc = subprocess.Popen(
                    [str(self.binary), str(self.config)],
                    cwd=str(self.folder),
                    stdout=devnull,
                    stderr=devnull
                )
        except Exception as e:
            messagebox.showerror("MediaMTX", f"Failed to start:\n{e}")

    def stop(self):
        if self.proc and self.proc.poll() is None:
            try:
                self.proc.terminate()
                deadline = time.time() + 2.5
                while time.time() < deadline and self.proc.poll() is None:
                    time.sleep(0.1)
                if self.proc.poll() is None:
                    self.proc.kill()
            except Exception as e:
                messagebox.showerror("MediaMTX", f"Failed to stop:\n{e}")
        self.proc = None

    def process_alive(self):
        return self.proc is not None and self.proc.poll() is None

    def port_open(self, timeout=0.05):
        if not self.process_alive():
            return False
        try:
            with socket.create_connection((self.host, self.port), timeout=timeout):
                return True
        except OSError:
            return False

    def status(self):
        if self.process_alive():
            return "running" if self.port_open() else "starting"
        return "stopped"


# ===================== ROOT =====================

root = tk.Tk()
MTX = MediaMTXController()
root.title("YourQualityCheck")
#root.state("zoomed")
root.attributes('-fullscreen', True)
root.configure(bg=BG)
root.resizable(False, False)

# Fonts for hover transitions
_icon_normal = tkfont.Font(family="Arial", size=36, weight="normal")
_icon_bold   = tkfont.Font(family="Arial", size=36, weight="bold")

# Keep PhotoImage references alive
_IMG_REFS = []
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
            args.extend(list(extra_args))
        env = os.environ.copy()
        if extra_env:
            env.update(extra_env)
        subprocess.Popen(args, cwd=str(path.parent), env=env)
    except Exception as e:
        messagebox.showerror("Error", f"Failed to run {path.name}:\n{e}")

def open_live_feed():
    try:
        open_stream_popup(root, STREAM_URL_HTTP, title="Camera Feed")
    except Exception as e:
        messagebox.showerror("Error", f"Failed to open live feed URL:\n{e}")

def hex_shift(hex_color: str, pct: float) -> str:
    """Lighten/darken a hex color by pct (-0.4..+0.4)."""
    h = hex_color.lstrip("#")
    r, g, b = int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)
    clamp = lambda x: max(0, min(255, int(x)))
    r, g, b = clamp(r + pct*255), clamp(g + pct*255), clamp(b + pct*255)
    return f"#{r:02x}{g:02x}{b:02x}"

def clear_root():
    for w in root.winfo_children():
        w.destroy()

# ---------- Stream Folder remember/load ----------
def load_stream_dir() -> Path:
    if STREAM_DIR_FILE.exists():
        try:
            p = Path(STREAM_DIR_FILE.read_text(encoding="utf-8").strip())
            if p.exists():
                return p
        except Exception:
            pass
    return DEFAULT_STREAM_DIR

def save_stream_dir(p: Path):
    try:
        STREAM_DIR_FILE.write_text(str(p), encoding="utf-8")
    except Exception:
        pass

STREAM_DIR = load_stream_dir()

def choose_stream_folder():
    global STREAM_DIR
    folder = filedialog.askdirectory(title="Select folder that contains streamvideo.py")
    if not folder:
        return
    p = Path(folder)
    if not (p / STREAM_FILENAME).exists():
        messagebox.showerror("Stream", f"Could not find {STREAM_FILENAME} in:\n{p}")
        return
    STREAM_DIR = p
    save_stream_dir(p)
    messagebox.showinfo("Stream", f"Media folder set to:\n{p}")

# ---------- Flask Stream process mgmt ----------
_stream_proc = None

def _stream_script_path() -> Path:
    return STREAM_DIR / STREAM_FILENAME

def start_stream():
    MTX.start()

def stop_stream():
    MTX.stop()

def _shutdown():
    try:
        stop_stream()
    except Exception:
        pass

atexit.register(_shutdown)

# ---------- Realtime Filtering (consumes MJPEG URL) ----------
def run_realtime_filtering():
    path = SCRIPT_PATHS.get("realtime")
    if not path or not path.exists():
        messagebox.showerror("Realtime Filtering", f"Realtime_Filtering.py not found:\n{path}")
        return
    # Many OpenCV scripts accept the URL as argv[1]; also pass in env
    run_script(path, extra_args=[STREAM_URL_HTTP], extra_env={"STREAM_URL": STREAM_URL_HTTP})

# ===================== TILES (with hover) =====================

def make_tile(parent, title, icon_text, bg_color, command):
    container = tk.Frame(parent, bg="white")
    tile = tk.Frame(container, bg=bg_color, width=TILE_W, height=TILE_H,
                    highlightthickness=1, highlightbackground="#b0b0b0",
                    relief="flat", bd=2)
    tile.pack_propagate(False)
    tile.pack(padx=8, pady=(0, 8))

    icon_lbl = tk.Label(tile, text=icon_text, bg=bg_color, fg="black", font=_icon_normal)
    icon_lbl.pack(expand=True)

    title_lbl = tk.Label(container, text=title, font=LABEL_FONT, bg="white")
    title_lbl.pack()

    hover_bg = hex_shift(bg_color, -0.06)

    def on_enter(_):
        tile.configure(bg=hover_bg, relief="raised")
        icon_lbl.configure(bg=hover_bg, font=_icon_bold)

    def on_leave(_):
        tile.configure(bg=bg_color, relief="flat")
        icon_lbl.configure(bg=bg_color, font=_icon_normal)

    def on_click(_):
        command()

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
        img = Image.open(logo_path).resize((320, 320), Image.Resampling.LANCZOS)
        logo = ImageTk.PhotoImage(img)
        keep_image_ref(logo)
        tk.Label(splash, image=logo, bg=BG).pack(pady=(0, 6))
    except Exception:
        tk.Label(splash, text="[Logo not found]", font=("Arial", 18), bg=BG).pack(pady=20)

    #Temporary disable login
    #root.after(1000, show_login)
    root.after(1000, lambda: show_dashboard("Triple I"))

def show_login():
    clear_root()
    login = tk.Frame(root, bg=BG)
    login.place(relx=0.5, rely=0.40, anchor="center")  # slightly higher

    try:
        img = Image.open(logo_path).resize((220, 220), Image.Resampling.LANCZOS)
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
        img = Image.open(logo_path).resize((220, 220), Image.Resampling.LANCZOS)
        logo = ImageTk.PhotoImage(img); keep_image_ref(logo)
        tk.Label(main, image=logo, bg="white").pack()
    except Exception:
        tk.Label(main, text="QuAck", font=("Arial", 22, "bold"), bg="white").pack(pady=(2, 26))

    tiles_frame = tk.Frame(main, bg="white"); tiles_frame.pack()

    tiles = [
        ("Profile",       "👤", COLORS["grey"],  lambda: render_section("profile", username)),
        ("Inventory",     "📦", COLORS["blue"],  lambda: render_section("inventory", username)),
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
    sidebar.pack(side="left", fill="y"); sidebar.pack_propagate(False)

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

    content = tk.Frame(root, bg="white"); content.pack(side="right", expand=True, fill="both")

    tk.Label(content, text={
        "profile": "👤 Profile",
        "inventory": "📦 Model Management",
        "camera": "📷 Camera Feed",
        "gallery": "🖼 Dataset Tools"
    }.get(section, "Section"), font=SECTION_FONT, bg="white").pack(pady=(18, 10))

    tiles_frame = tk.Frame(content, bg="white"); tiles_frame.pack(pady=10)

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
        # --- Camera Control Tiles ---
        tiles = [
            ("Set Media Folder",  "📁", COLORS["grey"],  choose_stream_folder),
            ("Start Stream",      "🟢",  COLORS["green"], start_stream),
            ("Stop Stream",       "🔴",  COLORS["pink"],  stop_stream),
            ("Open Live Feed",    "🌐", COLORS["peach"], open_live_feed),
            ("Realtime Filtering","🪄",  COLORS["blue"],  run_realtime_filtering),
        ]

        for i, (title, icon, color, cmd) in enumerate(tiles):
            r, c = divmod(i, 3)
            make_tile(tiles_frame, title, icon, color, cmd).grid(row=r, column=c, padx=TILE_PADX, pady=TILE_PADY)

        # --- Streaming Status Tile (6th Tile) ---
        status_tile_frame = tk.Frame(tiles_frame, bg="white")
        tile_bg = COLORS["grey"]
        status_tile = tk.Frame(status_tile_frame, bg=tile_bg, width=TILE_W, height=TILE_H,
                            highlightthickness=1, highlightbackground="#b0b0b0",
                            relief="flat", bd=2)
        status_tile.pack_propagate(False)
        status_tile.pack(padx=8, pady=(0, 8))

        status_label = tk.Label(status_tile, text="Streaming: Stopped",
                                bg=tile_bg, fg="black",
                                font=("Arial", 16, "bold"), wraplength=TILE_W - 20)
        status_label.pack(expand=True)

        title_label = tk.Label(status_tile_frame, text="Status", font=LABEL_FONT, bg="white")
        title_label.pack()

        # Place as 6th tile
        status_tile_frame.grid(row=1, column=2, padx=TILE_PADX, pady=TILE_PADY)

        # --- Periodic Status Updater ---
        def update_status_tile():
            st = MTX.status()

            if st == "running":
                bg = "#2aa745"  # green
                fg = "white"
                text = "Streaming\nActive"
            elif st == "starting":
                bg = "#e0a800"  # amber
                fg = "black"
                text = "Streaming\nStarting…"
            else:
                bg = "#c33"     # red
                fg = "white"
                text = "Streaming\nStopped"

            # Apply colors and text
            status_tile.config(bg=bg)
            status_label.config(bg=bg, fg=fg, text=text)
            title_label.config(bg="white")
            tiles_frame.after(CHECK_INTERVAL_MS, update_status_tile)

        update_status_tile()



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
