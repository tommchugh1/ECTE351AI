# Triple_I_app.py

import tkinter as tk
from PIL import Image, ImageTk
import tkinter.messagebox as messagebox
import webbrowser, subprocess, sys, time
from pathlib import Path

# =============== CONFIG ===============

# App look & feel
BG = "#ffffff"
TITLE_FONT = ("Arial", 26, "bold")
SECTION_FONT = ("Arial", 22, "bold")
ICON_FONT = ("Arial", 28, "bold")
LABEL_FONT = ("Arial", 13, "bold")
BTN_FONT = ("Arial", 14, "bold")

TILE_W, TILE_H = 170, 150  # tile size in px
TILE_PADX, TILE_PADY = 30, 18

# Tile colors (matching your original palette)
COLORS = {
    "grey":   "#e0e0e0",
    "blue":   "#e8f5ff",
    "peach":  "#ffe0b2",
    "green":  "#c8e6c9",
    "pink":   "#ffcdd2",
}

# Logo (used on splash & dashboard). Change if needed.
LOGO_PATH = r"C:\Users\Group8\Downloads\logo_final.jpg"

# Live feed URL
LIVE_FEED_URL = "http://localhost:5000/video_feed"

# Hard-wired tool paths (as requested)
SCRIPT_PATHS = {
    "execute":   Path(r"C:\Users\Group8\Desktop\Yolo Demo\ECTE351AI\AI\execute.py"),
    "train":     Path(r"C:\Users\Group8\Desktop\Yolo Demo\ECTE351AI\AI\train.py"),
    "auto":      Path(r"C:\Users\Group8\Desktop\Yolo Demo\ECTE351AI\VideoProcessing\AutomateKdenlive.py"),
    "xml2yolo":  Path(r"C:\Users\Group8\Desktop\Yolo Demo\ECTE351AI\VideoProcessing\KdenliveXMLtoYOLOv8.py"),
    "viz_one":   Path(r"C:\Users\Group8\Desktop\Yolo Demo\ECTE351AI\VideoProcessing\visualiseTXT.py"),
    "viz_batch": Path(r"C:\Users\Group8\Desktop\Yolo Demo\ECTE351AI\VideoProcessing\BatchVisualiseTXTBB.py"),
}

# Simple user store
USERS = {
    "Joy Pasala": "7452408",
    "Jonathan Walsh": "pass1",
    "Tom Mchugh": "6413717",
    "Jacob Rhados": "8002812",
    "Jerome Eid": "pass4",
    "Jason Watson": "7678721",
}

REMEMBER_FILE = Path(__file__).resolve().parent / "remember_me.txt"

# =============== UTILITIES ===============

def run_script(path: Path):
    """Launch a Python script in another process with its folder as CWD."""
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

def make_tile(parent, title, icon_text, bg_color, command):
    """Create a colored square tile with an icon-like text; click anywhere to trigger command."""
    container = tk.Frame(parent, bg="white")
    tile = tk.Frame(container, bg=bg_color, width=TILE_W, height=TILE_H,
                    highlightthickness=1, highlightbackground="#9e9e9e")
    tile.pack_propagate(False)
    tile.pack(padx=6, pady=(0, 6))

    icon_lbl = tk.Label(tile, text=icon_text, font=ICON_FONT, bg=bg_color, fg="black")
    icon_lbl.pack(expand=True)

    # Click behavior on tile and icon
    for widget in (tile, icon_lbl):
        widget.bind("<Button-1>", lambda _e: command())

    # Label
    tk.Label(container, text=title, font=LABEL_FONT, bg="white").pack()

    return container

# =============== APP ROOT ===============

root = tk.Tk()
root.title("YourQualityCheck")
root.state("zoomed")
root.configure(bg=BG)
root.resizable(False, False)

# =============== SPLASH ===============

def show_splash():
    for w in root.winfo_children():
        w.destroy()

    splash = tk.Frame(root, bg=BG)
    splash.pack(fill="both", expand=True)

    # Title
    tk.Label(splash, text="Welcome to YourQualityCheck", font=TITLE_FONT, bg=BG)\
      .pack(pady=(30, 10))

    # Logo big
    try:
        img = Image.open(LOGO_PATH).resize((320, 320), Image.Resampling.LANCZOS)
        logo_img = ImageTk.PhotoImage(img)
        logo_lbl = tk.Label(splash, image=logo_img, bg=BG)
        logo_lbl.image = logo_img  # keep ref
        logo_lbl.pack(pady=10)
        tk.Label(splash, text="Sponsored by Triple - I", font=("Arial", 12), bg=BG).pack()
    except Exception:
        tk.Label(splash, text="[Logo not found]", font=("Arial", 18), bg=BG).pack(pady=20)

    root.after(1200, show_login)

# =============== LOGIN ===============

def show_login():
    for w in root.winfo_children():
        w.destroy()

    frame = tk.Frame(root, bg=BG)
    frame.place(relx=0.5, rely=0.5, anchor="center")  # centered

    # Logo above login
    try:
        img = Image.open(LOGO_PATH).resize((180, 180), Image.Resampling.LANCZOS)
        login_logo = ImageTk.PhotoImage(img)
        logo_lbl = tk.Label(frame, image=login_logo, bg=BG)
        logo_lbl.image = login_logo
        logo_lbl.grid(row=0, column=0, columnspan=3, pady=(0, 15))
        tk.Label(frame, text="Sponsored by Triple - I", bg=BG, font=("Arial", 11)).grid(row=1, column=0, columnspan=3, pady=(0, 15))
    except Exception:
        tk.Label(frame, text="QuAck", font=("Arial", 22, "bold"), bg=BG).grid(row=0, column=0, columnspan=3, pady=(0, 15))

    # Inputs (larger)
    tk.Label(frame, text="User Name:", bg=BG, font=("Arial", 16)).grid(row=2, column=0, padx=12, pady=8, sticky="e")
    user_entry = tk.Entry(frame, font=("Arial", 16), width=28)
    user_entry.grid(row=2, column=1, columnspan=2, pady=8, sticky="w")

    tk.Label(frame, text="Password:", bg=BG, font=("Arial", 16)).grid(row=3, column=0, padx=12, pady=8, sticky="e")
    pass_entry = tk.Entry(frame, font=("Arial", 16), width=28, show="*")
    pass_entry.grid(row=3, column=1, columnspan=2, pady=8, sticky="w")

    # Remember Me
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
    show_btn.grid(row=4, column=0, columnspan=3, pady=(6, 6))

    tk.Checkbutton(frame, text="Remember Me", variable=remember_var, bg=BG, font=("Arial", 12)).grid(row=5, column=0, columnspan=3)

    def forgot_pw(_=None):
        uname = user_entry.get().strip()
        if uname in USERS:
            messagebox.showinfo("Password Recovery", f"Password for {uname}: {USERS[uname]}")
        else:
            messagebox.showerror("Error", "Enter a valid username first.")
    fp = tk.Label(frame, text="Forgot Password?", fg="blue", cursor="hand2", bg=BG, font=("Arial", 12))
    fp.grid(row=6, column=0, columnspan=3, pady=(0, 10))
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

    login_btn = tk.Button(frame, text="Login", font=("Arial", 18, "bold"), bg="#007ACC", fg="white",
                          width=14, command=do_login)
    login_btn.grid(row=7, column=0, columnspan=3, pady=(8, 4))

    # Enter-to-login
    root.bind("<Return>", do_login)
    user_entry.focus_set()

# =============== DASHBOARD & PAGES ===============

def show_dashboard(username: str):
    # clear window
    for w in root.winfo_children():
        w.destroy()

    main = tk.Frame(root, bg="white")
    main.pack(fill="both", expand=True)

    # Title + Logo
    tk.Label(main, text="Welcome to YourQualityCheck", font=TITLE_FONT, bg="white")\
      .pack(pady=(18, 8))
    try:
        img = Image.open(LOGO_PATH).resize((220, 220), Image.Resampling.LANCZOS)
        dlogo = ImageTk.PhotoImage(img)
        lab = tk.Label(main, image=dlogo, bg="white")
        lab.image = dlogo
        lab.pack()
        tk.Label(main, text="Sponsored by Triple - I", bg="white", font=("Arial", 12)).pack(pady=(4, 30))
    except Exception:
        tk.Label(main, text="QuAck", font=("Arial", 22, "bold"), bg="white").pack(pady=(4, 30))

    tiles_frame = tk.Frame(main, bg="white")
    tiles_frame.pack()

    def goto(section):
        render_section(section, username)

    # dashboard tiles (same look everywhere)
    tiles = [
        ("Profile",       "👤",  COLORS["grey"],  lambda: goto("profile")),
        ("Inventory",     "🧊",  COLORS["blue"],  lambda: goto("inventory")),
        ("Camera Feed",   "📷",  COLORS["peach"], lambda: goto("camera")),
        ("Photo Gallery", "🖼",  COLORS["green"], lambda: goto("gallery")),
        ("Logout",        "📱",  COLORS["pink"],  logout),
    ]

    for i, (title, icon, color, cmd) in enumerate(tiles):
        t = make_tile(tiles_frame, title, icon, color, cmd)
        t.grid(row=0, column=i, padx=TILE_PADX, pady=TILE_PADY)

def logout():
    show_login()

def render_section(section: str, username: str):
    for w in root.winfo_children():
        w.destroy()

    # Sidebar (bigger)
    sidebar = tk.Frame(root, bg="#bdbdbd", width=190)
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
    tk.Label(content, text=titles.get(section, "Section"), font=SECTION_FONT, bg="white")\
      .pack(pady=(18, 10))

    tiles_frame = tk.Frame(content, bg="white")
    tiles_frame.pack(pady=10)

    if section == "profile":
        # Just a big "Back to Dashboard" tile & user label to keep layout consistent
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
            ("Run Inference",   "🎯", COLORS["blue"],  lambda: run_script(SCRIPT_PATHS["execute"])),
            ("Open Live Feed",  "🌐", COLORS["peach"], open_live_feed),
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

# =============== STARTUP ===============
show_splash()
root.mainloop()
