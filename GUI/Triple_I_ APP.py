

import tkinter as tk
import tkinter.font as tkfont
from tkinter import messagebox
from PIL import Image, ImageTk
import os
import subprocess
import sys
from pathlib import Path
import socket
import time
import threading
import cv2
from datetime import datetime
from openFeed import open_stream_popup
from Realtime_Filtering import open_filtered_popup
from statusUpdater import (
    StatusMonitor,
    check_status_rtsp_port,
    nudge_monitor_fast
)
# USER DATA 

USERS = {
    "Joy Pasala": "7452408",
    "Jonathan Walsh": "pass1",
    "Tom McHugh": "6413717",
    "Jacob Rhados": "8002812",
    "Jerome Eid": "pass4",
    "Jason Watson": "7678721",
}

# PATHS / CONSTANTS 

BG = "#ffffff"
BTN_COLOR = "#007ACC"
ENTRY_BG = "white"
REMEMBER_FILE = "remember_me.txt"

PI_HOST = "10.27.27.10"
RTSP_TCP_PORT = 8554
CHECK_INTERVAL_MS = 800
STREAM_URL_RTSP = f"rtsp://{PI_HOST}:{RTSP_TCP_PORT}/cam1"

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
    "yellow": '#FFEE8C'
}

# Logo (change if needed)
logo_path = os.path.join(working_dir, "GUI", "logo_final.jpg")

# Sends command to Pi to start feed.
def send_command_to_pi(command: str, pi_ip: str = PI_HOST, port: int = 9001) -> str:
    
    # Sends 'start_feed' or 'stop_feed' to the socket server on the Pi.
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.settimeout(1.5)
            s.connect((pi_ip, port))
            s.sendall(command.encode("utf-8"))
            response = s.recv(1024).decode("utf-8", errors="ignore")
            print(f"[Pi Response] {response}")
            return response
    except Exception as e:
        print(f"[ERROR] Failed to send command to {pi_ip}:{port} -> {e}")
        return "ERROR"

# ROOT 

root = tk.Tk()
root.title("YourQualityCheck")
#root.state("zoomed")
root.bind("<Escape>", lambda e: root.attributes("-fullscreen", False))
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

# HELPERS 

def rtsp_describe_ok(host: str, port: int, path: str = "cam1", timeout: float = 0.5) -> bool:
    """
    Lightweight check: send RTSP DESCRIBE to the specific path.
    Returns True only if the stream/path is actually published (HTTP 200).
    """
    try:
        with socket.create_connection((host, port), timeout=timeout) as s:
            req = (
                f"DESCRIBE rtsp://{host}:{port}/{path} RTSP/1.0\r\n"
                "CSeq: 1\r\n"
                "User-Agent: YQC/1.0\r\n"
                "Accept: application/sdp\r\n\r\n"
            ).encode("ascii")
            s.sendall(req)
            data = s.recv(1024)
            # first line like: b"RTSP/1.0 200 OK\r\n..."
            return b" 200 " in data[:64]
    except OSError:
        return False

def check_status_rtsp_stream(host: str, port: int, path: str = "cam1") -> str:
    return "running" if rtsp_describe_ok(host, port, path) else "stopped"



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
        open_stream_popup(root, STREAM_URL_RTSP, title="Camera Feed")
    except Exception as e:
        messagebox.showerror("Error", f"Failed to open live feed URL:\n{e}")

def on_snapshot_clicked(current_status, _snapshot_busy):
    if _snapshot_busy:
        return
    if current_status["value"] != "running":
        print("[SNAPSHOT] Skipped: stream inactive")
        return

    _snapshot_busy = True

    def worker():
        try:
            cap = cv2.VideoCapture(STREAM_URL_RTSP)  # RTSP URL
            ok, frame = cap.read()
            cap.release()
            if not ok or frame is None:
                raise RuntimeError("Failed to grab frame")

            os.makedirs("snapshots", exist_ok=True)
            filename = f"snapshot_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jpg"
            path = os.path.join("snapshots", filename)
            cv2.imwrite(path, frame)
            print(f"[SNAPSHOT] Saved to {path}")
            # notify on UI thread
            root.after(0, lambda: messagebox.showinfo("Snapshot Captured", f"Image saved to:\n{path}"))
        except Exception as e:
            print(f"[SNAPSHOT] Error: {e}")
            root.after(0, lambda: messagebox.showerror("Capture Error", f"{e}"))
        finally:
            _snapshot_busy = False

    threading.Thread(target=worker, daemon=True).start()

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

# Realtime filtering
def run_realtime_filtering():
    try:    
        open_filtered_popup(root, STREAM_URL_RTSP, title="Filtered Feed")
    except Exception as e:
        messagebox.showerror("Error", f"Failed to open filtered feed:\n{e}")

# TILES (with hover) 

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

# SPLASH / LOGIN 

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

# DASHBOARD 

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

# SECTION RENDERING 

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
        # Camera Control Tiles

        # Button state references
        state = {"starting": False}

        def on_start_clicked():
            send_command_to_pi("start_feed", pi_ip=PI_HOST)
            nudge_monitor_fast(monitor, ms=250, for_seconds=3.0)

        def on_stop_clicked():
            send_command_to_pi("stop_feed", pi_ip=PI_HOST)
            nudge_monitor_fast(monitor, ms=250, for_seconds=2.0)

        current_status = {"value": "stopped"}
        _snapshot_busy = False

        # --- Build Tiles ---
        tiles = [
            ("Start Stream",      "🟢", COLORS["green"], on_start_clicked),
            ("Stop Stream",       "🔴", COLORS["pink"], on_stop_clicked),
            ("Open Live Feed",    "🌐", COLORS["blue"], open_live_feed),
            ("Realtime Filtering","🪄", COLORS["peach"],  run_realtime_filtering),
            ("Capture Snapshot", "📷", COLORS["yellow"], on_snapshot_clicked(current_status, _snapshot_busy)),
        ]
        

        tile_refs = []
        for i, (title, icon, color, cmd) in enumerate(tiles):
            r, c = divmod(i, 3)
            tile = make_tile(tiles_frame, title, icon, color, cmd)
            tile.grid(row=r, column=c, padx=TILE_PADX, pady=TILE_PADY)
            tile_refs.append(tile)

        # Streaming Status Tile (6th tile)
        status_tile_frame = tk.Frame(tiles_frame, bg="white")
        status_tile = tk.Frame(
            status_tile_frame, bg=COLORS["grey"],
            width=TILE_W, height=TILE_H,
            highlightthickness=1, highlightbackground="#b0b0b0",
            relief="flat", bd=2
        )
        status_tile.pack_propagate(False)
        status_tile.pack(padx=8, pady=(0, 8))

        status_label = tk.Label(
            status_tile, text="Streaming\nStopped",
            bg=COLORS["grey"], fg="black",
            font=("Arial", 18, "bold"),  # larger font inside tile
            wraplength=TILE_W - 20, justify="center"
        )
        status_label.pack(expand=True)

        title_label = tk.Label(status_tile_frame, text="Status", font=LABEL_FONT, bg="white")
        title_label.pack()

        # grid as 6th tile
        status_tile_frame.grid(row=1, column=2, padx=TILE_PADX, pady=TILE_PADY)

        # --- Non-blocking periodic status monitor ---
        def paint_status(st: str):
            current_status["value"] = st

            if st == "running":
                bg, fg, text = "#2aa745", "white", "Streaming\nActive"
            elif st == "starting":
                bg, fg, text = "#e0a800", "black", "Streaming\nStarting…"
            else:
                bg, fg, text = "#c33", "white", "Streaming\nStopped"

            status_tile.config(bg=bg)
            status_label.config(bg=bg, fg=fg, text=text)
            title_label.config(bg="white")

        # RTSP Decribe
        check_fn = lambda: check_status_rtsp_stream(PI_HOST, RTSP_TCP_PORT, "cam1")
        monitor = StatusMonitor(root, check_fn, paint_status, interval_ms=800)
        content.bind("<Destroy>", lambda e: monitor.stop())



    elif section == "gallery":
        tiles = [
            ("Visualise (Single)", "🔍", COLORS["green"], lambda: run_script(SCRIPT_PATHS["viz_one"])),
            ("Batch Visualise",    "🖼", COLORS["pink"],  lambda: run_script(SCRIPT_PATHS["viz_batch"])),
        ]
        for i, (title, icon, color, cmd) in enumerate(tiles):
            make_tile(tiles_frame, title, icon, color, cmd).grid(row=0, column=i, padx=TILE_PADX, pady=TILE_PADY)



# START 

def main():
    show_splash()
    root.mainloop()

if __name__ == "__main__":
    main()
