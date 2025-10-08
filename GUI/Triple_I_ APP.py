# Triple_I_app.py  — final

import tkinter as tk
import tkinter.messagebox as messagebox
import tkinter.font as tkfont
from PIL import Image, ImageTk
import webbrowser, subprocess, sys
from pathlib import Path

# ===================== LOOK & FEEL =====================

BG = "#ffffff"

# Enlarged, clean typography
TITLE_FONT     = ("Arial", 28, "bold")
SECTION_FONT   = ("Arial", 24, "bold")
LABEL_FONT     = ("Arial", 14, "bold")
BTN_FONT       = ("Arial", 16, "bold")

# Tile sizing (big icons everywhere)
TILE_W, TILE_H = 200, 180
TILE_PADX, TILE_PADY = 28, 20

# Original palette
COLORS = {
    "grey":   "#e0e0e0",
    "blue":   "#e8f5ff",
    "peach":  "#ffe0b2",
    "green":  "#c8e6c9",
    "pink":   "#ffcdd2",
}

# Logo file (used on splash, login, dashboard)
LOGO_PATH = r"C:\Users\Group8\Downloads\logo_final.jpg"

# Live feed URL (browser)
LIVE_FEED_URL = "http://localhost:5000/video_feed"

# ===================== SCRIPT PATHS =====================

SCRIPT_PATHS = {
    "execute":   Path(r"C:\Users\Group8\Desktop\Yolo Demo\ECTE351AI\AI\execute.py"),
    "train":     Path(r"C:\Users\Group8\Desktop\Yolo Demo\ECTE351AI\AI\train.py"),
    "auto":      Path(r"C:\Users\Group8\Desktop\Yolo Demo\ECTE351AI\VideoProcessing\AutomateKdenlive.py"),
    "xml2yolo":  Path(r"C:\Users\Group8\Desktop\Yolo Demo\ECTE351AI\VideoProcessing\KdenliveXMLtoYOLOv8.py"),
    "viz_one":   Path(r"C:\Users\Group8\Desktop\Yolo Demo\ECTE351AI\VideoProcessing\visualiseTXT.py"),
    "viz_batch": Path(r"C:\Users\Group8\Desktop\Yolo Demo\ECTE351AI\VideoProcessing\BatchVisualiseTXTBB.py"),
    # Optional: Start the local stream server if you have it
    "stream":    Path(r"C:\Users\Group8\Desktop\Yolo Demo\ECTE351AI\GUI\Video_Feed.py"),
}

# ===================== AUTH =====================

USERS = {
    "Joy Pasala": "7452408",
    "Jonathan Walsh": "pass1",
    "Tom Mchugh": "6413717",
    "Jacob Rhados": "8002812",
    "Jerome Eid": "7171316",
    "Jason Watson": "7678721",
}
REMEMBER_FILE = Path(__file__).resolve().parent / "remember_me.txt"

# ===================== HELPERS =====================

def run_script(path: Path):
    """Launch a Python script with its own working directory."""
    if not isinstance(path, Path):
        path = Path(path)
    if not path.exists():
        messagebox.showerror("Error", f"File not found:\n{path}")
        return
    try:
        subprocess.Popen([sys.executable, str(path)], cwd=str(path.parent))
    except Exception as e:
        messagebox.showerror("Error", f"Failed to run {path.name}:\n{e}")

def open_live_feed():
    try:
        webbrowser.open(LIVE_FEED_URL)
    except Exception as e:
        messagebox.showerror("Error", f"Failed to open live feed URL:\n{e}")

def start_stream_if_available():
    stream_path = SCRIPT_PATHS.get("stream")
    if stream_path and stream_path.exists():
        run_script(stream_path)
    else:
        messagebox.showinfo("Stream", "Stream script not found. Opening live feed URL instead.")
        open_live_feed()

def hex_shift(hex_color: str, pct: float) -> str:
    """Lighten/darken a hex color by pct (-0.4..+0.4)."""
    hex_color = hex_color.lstrip("#")
    r = int(hex_color[0:2], 16)
    g = int(hex_color[2:4], 16)
    b = int(hex_color[4:6], 16)
    def clamp(x): return max(0, min(255, int(x)))
    r = clamp(r + pct*255)
    g = clamp(g + pct*255)
    b = clamp(b + pct*255)
    return f"#{r:02x}{g:02x}{b:02x}"

# ===================== ROOT =====================

root = tk.Tk()
root.title("YourQualityCheck")
root.state("zoomed")
root.configure(bg=BG)
root.resizable(False, False)

# Fonts for hover transitions
_icon_normal = tkfont.Font(family="Arial", size=36, weight="normal")
_icon_bold   = tkfont.Font(family="Arial", size=36, weight="bold")

# ===================== TILE FACTORY (with hover) =====================

def make_tile(parent, title, icon_text, bg_color, command):
    """
    A large colored tile with hover effects (bold + gentle color shift),
    used consistently across dashboard and inner pages.
    """
    container = tk.Frame(parent, bg="white")

    tile = tk.Frame(container, bg=bg_color, width=TILE_W, height=TILE_H,
                    highlightthickness=1, highlightbackground="#b0b0b0", relief="flat", bd=2)
    tile.pack_propagate(False)
    tile.pack(padx=8, pady=(0, 8))

    icon_lbl = tk.Label(tile, text=icon_text, bg=bg_color, fg="black", font=_icon_normal)
    icon_lbl.pack(expand=True)

    title_lbl = tk.Label(container, text=title, font=LABEL_FONT, bg="white")
    title_lbl.pack()

    hover_bg = hex_shift(bg_color, -0.06)  # subtle darken

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

# ===================== SCREENS =====================

def show_splash():
    for w in root.winfo_children():
        w.destroy()
    splash = tk.Frame(root, bg=BG)
    splash.pack(fill="both", expand=True)

    tk.Label(splash, text="Welcome to YourQualityCheck", font=TITLE_FONT, bg=BG)\
      .pack(pady=(20, 10))

    # Big logo
    try:
        img = Image.open(LOGO_PATH).resize((320, 320), Image.Resampling.LANCZOS)
        logo_img = ImageTk.PhotoImage(img)
        lbl = tk.Label(splash, image=logo_img, bg=BG)
        lbl.image = logo_img #type: ignore[attr-defined]
        lbl.pack(pady=(0, 4))
    except Exception:
        tk.Label(splash, text="[Logo not found]", font=("Arial", 18), bg=BG).pack(pady=20)

    root.after(1000, show_login)

def show_login():
    for w in root.winfo_children():
        w.destroy()

    frame = tk.Frame(root, bg=BG)
    # Slightly higher than center (as requested)
    frame.place(relx=0.5, rely=0.42, anchor="center")

    # Logo a bit higher
    try:
        img = Image.open(LOGO_PATH).resize((320, 320), Image.Resampling.LANCZOS)
        login_logo = ImageTk.PhotoImage(img)
        l = tk.Label(frame, image=login_logo, bg=BG)
        l.image = login_logo #type: ignore[attr-defined]
        l.grid(row=0, column=0, columnspan=3, pady=(0, 8))
    except Exception:
        tk.Label(frame, text="Quack", font=("Arial", 22, "bold"), bg=BG).grid(row=0, column=0, columnspan=3, pady=(0, 12))

    # Inputs
    tk.Label(frame, text="User Name:", bg=BG, font=("Arial", 16)).grid(row=2, column=0, padx=12, pady=8, sticky="e")
    user_entry = tk.Entry(frame, font=("Arial", 16), width=28)
    user_entry.grid(row=2, column=1, columnspan=2, pady=8, sticky="w")

    tk.Label(frame, text="Password:", bg=BG, font=("Arial", 16)).grid(row=3, column=0, padx=12, pady=8, sticky="e")
    pass_entry = tk.Entry(frame, font=("Arial", 16), width=28, show="*")
    pass_entry.grid(row=3, column=1, columnspan=2, pady=8, sticky="w")

    remember_var = tk.BooleanVar(value=False)
    if REMEMBER_FILE.exists():
        try:
            remembered = REMEMBER_FILE.read_text(encoding="utf-8").strip()
            if remembered:
                user_entry.insert(0, remembered)
                remember_var.set(True)
        except Exception:
            pass

    def toggle_pw():
        pass_entry.config(show="" if pass_entry.cget("show") == "*" else "*")
        show_btn.config(text="Hide Password" if pass_entry.cget("show")=="" else "Show Password")

    show_btn = tk.Button(frame, text="Show Password", font=BTN_FONT, bg="#007ACC", fg="white", command=toggle_pw)
    show_btn.grid(row=4, column=0, columnspan=3, pady=(6, 4))

    tk.Checkbutton(frame, text="Remember Me", variable=remember_var, bg=BG, font=("Arial", 12)).grid(row=5, column=0, columnspan=3)

    def forgot_pw(_=None):
        uname = user_entry.get().strip()
        if uname in USERS:
            messagebox.showinfo("Password Recovery", f"Password for {uname}: {USERS[uname]}")
        else:
            messagebox.showerror("Error", "Enter a valid username first.")
    fp = tk.Label(frame, text="Forgot Password?", fg="blue", cursor="hand2", bg=BG, font=("Arial", 12))
    fp.grid(row=6, column=0, columnspan=3, pady=(0, 8))
    fp.bind("<Button-1>", forgot_pw)

    def do_login(_evt=None):
        uname = user_entry.get().strip()
        pwd = pass_entry.get()
        if uname in USERS and USERS[uname] == pwd:
            if remember_var.get():
                try: REMEMBER_FILE.write_text(uname, encoding="utf-8")
                except Exception: pass
            else:
                if REMEMBER_FILE.exists():
                    try: REMEMBER_FILE.unlink()
                    except Exception: pass
            show_dashboard(uname)
        else:
            messagebox.showerror("Access Denied", "Invalid username or password.")

    login_btn = tk.Button(frame, text="Login", font=("Arial", 18, "bold"),
                          bg="#007ACC", fg="white", width=14, command=do_login)
    login_btn.grid(row=7, column=0, columnspan=3, pady=(8, 4))

    root.bind("<Return>", do_login)
    user_entry.focus_set()

def show_dashboard(username: str):
    for w in root.winfo_children():
        w.destroy()

    main = tk.Frame(root, bg="white")
    main.pack(fill="both", expand=True)

    tk.Label(main, text="Welcome to YourQualityCheck", font=TITLE_FONT, bg="white").pack(pady=(18, 6))
    try:
        img = Image.open(LOGO_PATH).resize((220, 220), Image.Resampling.LANCZOS)
        logo = ImageTk.PhotoImage(img)
        lab = tk.Label(main, image=logo, bg="white")
        lab.image = logo #type: ignore[attr-defined]
        lab.pack()
    except Exception:
        tk.Label(main, text="QuAck", font=("Arial", 22, "bold"), bg="white").pack(pady=(2, 26))

    tiles_frame = tk.Frame(main, bg="white")
    tiles_frame.pack()

    tiles = [
        ("Profile",       "👤", COLORS["grey"],  lambda: render_section("profile", username)),
        ("Inventory",     "🧊", COLORS["blue"],  lambda: render_section("inventory", username)),
        ("Camera Feed",   "📷", COLORS["peach"], lambda: render_section("camera", username)),
        ("Photo Gallery", "🖼", COLORS["green"], lambda: render_section("gallery", username)),
        ("Logout",        "📱", COLORS["pink"],  logout),
    ]
    for i, (title, icon, color, cmd) in enumerate(tiles):
        make_tile(tiles_frame, title, icon, color, cmd).grid(row=0, column=i, padx=TILE_PADX, pady=TILE_PADY)

def logout():
    show_login()

def render_section(section: str, username: str):
    for w in root.winfo_children():
        w.destroy()

    # Sidebar (bigger)
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
    sbtn("🚪 Logout",    False,                logout)

    # Content
    content = tk.Frame(root, bg="white")
    content.pack(side="right", expand=True, fill="both")

    titles = {
        "profile":   "👤 Profile",
        "inventory": "📦 Model Management",
        "camera":    "📷 Camera Feed",
        "gallery":   "🖼 Dataset Tools",
    }
    tk.Label(content, text=titles.get(section, "Section"), font=SECTION_FONT, bg="white").pack(pady=(18, 10))

    tiles_frame = tk.Frame(content, bg="white")
    tiles_frame.pack(pady=10)

    if section == "profile":
        tk.Label(content, text=f"User: {username}", font=("Arial", 16), bg="white").pack(pady=(0, 8))
        make_tile(tiles_frame, "Back to Dashboard", "🏠", COLORS["grey"],
                  lambda: show_dashboard(username)).grid(row=0, column=0, padx=TILE_PADX, pady=TILE_PADY)

    elif section == "inventory":
        tiles = [
            ("Train Model",          "📚", COLORS["blue"],  lambda: run_script(SCRIPT_PATHS["train"])),
            ("Automate Annotations", "⚙", COLORS["green"], lambda: run_script(SCRIPT_PATHS["auto"])),
            ("Convert XML→YOLO",     "📂", COLORS["peach"], lambda: run_script(SCRIPT_PATHS["xml2yolo"])),
        ]
        for i, (title, icon, color, cmd) in enumerate(tiles):
            make_tile(tiles_frame, title, icon, color, cmd).grid(row=0, column=i, padx=TILE_PADX, pady=TILE_PADY)

    elif section == "camera":
        tiles = [
            ("Run Inference",  "🎯", COLORS["blue"],  lambda: run_script(SCRIPT_PATHS["execute"])),
            ("Start Stream",   "▶️", COLORS["green"], start_stream_if_available),
            ("Open Live Feed", "🌐", COLORS["peach"], open_live_feed),
        ]
        for i, (title, icon, color, cmd) in enumerate(tiles):
            make_tile(tiles_frame, title, icon, color, cmd).grid(row=0, column=i, padx=TILE_PADX, pady=TILE_PADY)

    elif section == "gallery":
        tiles = [
            ("Visualise (Single)", "🔍", COLORS["green"], lambda: run_script(SCRIPT_PATHS["viz_one"])),
            ("Batch Visualise",    "🖼", COLORS["pink"],  lambda: run_script(SCRIPT_PATHS["viz_batch"])),
        ]
        for i, (title, icon, color, cmd) in enumerate(tiles):
            make_tile(tiles_frame, title, icon, color, cmd).grid(row=0, column=i, padx=TILE_PADX, pady=TILE_PADY)

# ===================== START =====================

show_splash()
root.mainloop()