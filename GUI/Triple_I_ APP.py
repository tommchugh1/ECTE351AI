import tkinter as tk
import tkinter.font as tkfont
from tkinter import messagebox
from PIL import Image, ImageTk
import os
import subprocess
import sys
from pathlib import Path
import socket
import threading
import numpy as np
import cv2
import time
from openFeed import open_stream_popup
from Realtime_Filtering import open_filtered_popup
from statusUpdater import (
    StatusMonitor,
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

PI_HOST = "10.12.132.203"
RTSP_TCP_PORT = 8554
CHECK_INTERVAL_MS = 800
STREAM_URL_RTSP = f"rtsp://{PI_HOST}:{RTSP_TCP_PORT}/cam1"

# Relative script paths
working_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

sys.path.insert(1, working_dir)
from Realtime_Filtering import get_filtered_frames
from AI.execute import run_inference_on_generator

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

        # --- Build Tiles ---
        tiles = [
            ("Start Stream",      "🟢", COLORS["green"], on_start_clicked),
            ("Stop Stream",       "🔴", COLORS["pink"], on_stop_clicked),
            ("Open Live Feed",    "🌐", COLORS["blue"], open_live_feed),
            ("Realtime Filtering","🪄", COLORS["peach"],  run_realtime_filtering),
            ("Run Inference", "🧠", COLORS["yellow"], lambda: render_inference_page(username)),
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

def render_inference_page(username):
    """
    Full-page inference that:
      - consumes filtered frames from get_filtered_frames(...)
      - passes them to run_inference_on_generator
      - displays annotated frames in a Tk label
    """

    clear_root()

    # -------- Sidebar --------
    sidebar = tk.Frame(root, bg="#bdbdbd", width=200)
    sidebar.pack(side="left", fill="y"); sidebar.pack_propagate(False)

    def sbtn(text, cmd, active=False):
        tk.Button(sidebar, text=text,
                  font=("Arial", 14, "bold" if active else "normal"),
                  bg="#eeeeee" if active else "#bdbdbd", relief="flat",
                  anchor="w", padx=14, pady=12, command=cmd).pack(fill="x", pady=2)

    sbtn("← Back to Camera", lambda: render_section("camera", username))
    sbtn("👤 Profile",   lambda: render_section("profile", username))
    sbtn("📦 Inventory", lambda: render_section("inventory", username))
    sbtn("🖼 Gallery",   lambda: render_section("gallery", username))

    # -------- Content --------
    content = tk.Frame(root, bg="white")
    content.pack(side="right", expand=True, fill="both")

    tk.Label(content, text="🧠 Inference (Filtered → Model)", font=SECTION_FONT, bg="white").pack(pady=(18, 6))
    tk.Label(content, text=f"Source: {STREAM_URL_RTSP}", font=("Arial", 11), bg="white", fg="#555").pack(pady=(0, 8))

    # Status bar
    status_bar = tk.Frame(content, bg="#f6f7f9", bd=1, relief="solid", height=56)
    status_bar.pack(fill="x", padx=18, pady=(0, 10))
    status_bar.pack_propagate(False)

    stats_inner = tk.Frame(status_bar, bg=status_bar["bg"]); stats_inner.pack(fill="both", expand=True, padx=10)
    for i in range(4):
        stats_inner.grid_columnconfigure(i, weight=1, uniform="stats")

    sv_state = tk.StringVar(value="Idle")
    sv_fps   = tk.StringVar(value="—")
    sv_det   = tk.StringVar(value="—")
    sv_err   = tk.StringVar(value="")

    def add_stat(col, label_text, var):
        tk.Label(stats_inner, text=label_text, bg=status_bar["bg"], fg="#333",
                 font=("Arial", 10, "bold")).grid(row=0, column=col, padx=6, pady=(6, 0), sticky="w")
        tk.Label(stats_inner, textvariable=var, bg=status_bar["bg"], fg="#111",
                 font=("Arial", 11)).grid(row=1, column=col, padx=6, pady=(0, 6), sticky="w")

    add_stat(0, "State", sv_state)
    add_stat(1, "FPS",   sv_fps)
    add_stat(2, "Detections", sv_det)
    add_stat(3, "Last Error", sv_err)

    def set_bar(level="info"):
        color = {"ok":"#e8f5e9", "warn":"#fff8e1", "err":"#ffebee", "info":"#f6f7f9"}.get(level, "#f6f7f9")
        status_bar.configure(bg=color)
        stats_inner.configure(bg=color)
        for w in stats_inner.winfo_children():
            w.configure(bg=color)

    # Controls
    controls = tk.Frame(content, bg="white")
    controls.pack(side="bottom", pady=(0, 18), padx=18, fill="x")


    # ------- Threads & state -------
    stop_event = threading.Event()
    running = {"v": False}
    frame_lock = threading.Lock()
    latest_frame = {"img": None}  # RGB annotated frame

    # FPS counters
    t0 = [time.time()]
    n  = [0]

    def push_frame_to_ui(annotated_bgr: np.ndarray, det_count: int | None = None):
        """Called by inference when a new annotated frame is ready."""
        try:
            rgb = cv2.cvtColor(annotated_bgr, cv2.COLOR_BGR2RGB)
        except Exception:
            rgb = annotated_bgr  # in case already RGB

        with frame_lock:
            latest_frame["img"] = rgb

        # fps
        n[0] += 1
        now = time.time()
        if now - t0[0] >= 1.0:
            sv_fps.set(str(n[0]))
            n[0] = 0
            t0[0] = now

        if det_count is not None:
            sv_det.set(str(det_count))

    def ui_refresh():
        if not video_lbl.winfo_exists():
            return
        img = None
        with frame_lock:
            img = latest_frame["img"]
        if img is not None:
            # fit to label
            h, w = img.shape[:2]
            win_w = max(1, video_lbl.winfo_width())
            win_h = max(1, video_lbl.winfo_height())
            scale = min(win_w / w, win_h / h)
            new_size = (max(1, int(w * scale)), max(1, int(h * scale)))
            vis = cv2.resize(img, new_size, interpolation=cv2.INTER_AREA)
            try:
                pil = Image.fromarray(vis)
                imgtk = ImageTk.PhotoImage(image=pil)
                video_lbl.imgtk = imgtk
                video_lbl.configure(image=imgtk)
            except Exception:
                pass
        video_lbl.after(33, ui_refresh)  # ~30 fps

    def filtered_generator():
        print("[DEBUG] Starting filtered_generator()")
        gen = get_filtered_frames(
            STREAM_URL_RTSP,
            enable_chroma_key=False,
            enable_grayscale=False,
            enable_binarization=False,
            enable_rotation=False,
            enable_zoom=False,
            threshold_val=200,
            stop_event=stop_event,
        )
        for f in gen:
            if stop_event.is_set():
                break
            print("[DEBUG] Yielding filtered frame")
            yield f

    def inference_worker():
        print("[DEBUG] Inference thread started")
        sv_state.set("Connecting…"); set_bar("info")
        try:
            result = run_inference_on_generator(
                frame_generator=filtered_generator(),
                on_frame=lambda img, det_count=None: push_frame_to_ui(img, det_count),
            )
        except TypeError:
            # older signature: try without 'device'
            result = run_inference_on_generator(
                filtered_generator(),
                on_frame=lambda img, det_count=None: push_frame_to_ui(img, det_count),
            )
            print("[DEBUG] run_inference_on_generator returned:", type(result))
            if result is not None:
                for annotated in result:
                    if stop_event.is_set():
                        break
                    if isinstance(annotated, tuple):
                        img, dets = annotated[0], annotated[1] if len(annotated) > 1 else None
                        push_frame_to_ui(img, dets)
                    else:
                        push_frame_to_ui(annotated, None)
            sv_state.set("Stopped"); set_bar("info")
        except Exception as e:
            sv_state.set("Error")
            sv_err.set(str(e))
            set_bar("err")
        finally:
            running["v"] = False
            start_btn.config(state="normal")
            stop_btn.config(state="disabled")

    def start_inference():
        print("[DEBUG] Start button pressed", flush=True)
        if running["v"]:
            return
        sv_err.set("")
        stop_event.clear()
        running["v"] = True
        sv_state.set("Running"); set_bar("ok")
        start_btn.config(state="disabled")
        stop_btn.config(state="normal")
        threading.Thread(target=inference_worker, daemon=True).start()

    def stop_inference():
        print("[DEBUG] Stop button pressed", flush=True)
        if not running["v"]:
            return
        sv_state.set("Stopping…"); set_bar("warn")
        stop_event.set()

    start_btn = tk.Button(
        controls, text="Start Inference",
        font=BTN_FONT, bg=COLORS["green"], fg="black",
        command=start_inference
    )
    stop_btn = tk.Button(
        controls, text="Stop Inference",
        font=BTN_FONT, bg=COLORS["pink"], fg="black",
        command=stop_inference
    )
    back_btn = tk.Button(
        controls, text="Back",
        font=BTN_FONT, bg=COLORS["grey"], fg="black",
        command=lambda: render_section("camera", username)
    )

    # align them to the right edge
    back_btn.pack(side="right", padx=6)
    stop_btn.pack(side="right", padx=6)
    start_btn.pack(side="right", padx=6)

    video_holder = tk.Frame(content, bg="#111", width=960, height=720)
    video_holder.pack(side="top", padx=18, pady=(0, 10), fill="both", expand=True)
    video_holder.pack_propagate(False)

    video_lbl = tk.Label(video_holder, bg="black")
    video_lbl.pack(fill="both", expand=True)


    ui_refresh()




# START 

def main():
    show_splash()
    root.mainloop()

if __name__ == "__main__":
    main()
