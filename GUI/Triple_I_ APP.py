import tkinter as tk
from PIL import Image, ImageTk
import tkinter.messagebox as messagebox
import os, webbrowser, subprocess, sys, time
from pathlib import Path
import fnmatch

# ========= Project layout & assets =========
PROJECT_ROOT = Path(__file__).resolve().parent

# Expected relative locations (fallback to recursive search if not found)
CANDIDATES = {
    "execute":   ["AI/execute.py", "execute.py"],                          # YOLO OpenVINO runner
    "train":     ["AI/train.py", "train.py"],                              # YOLO training
    "auto":      ["VideoProcessing/AutomateKdenlive.py"],                  # Replace clip filename and open Kdenlive
    "xml2yolo":  ["VideoProcessing/KdenliveXMLtoYOLOv8.py"],               # Convert Kdenlive → YOLO labels
    "viz_one":   ["VideoProcessing/visualiseTXT.py"],                      # Visualise one .txt
    "viz_batch": ["VideoProcessing/BatchVisualiseTXTBB.py"],               # Batch visualise
    # optional local camera/Flask app (if your project has it)
    "video_feed": ["GUI/Video_Feed.py", "Video_Feed.py", "video_feed.py"]
}

# Logo (same look & feel)
LOGO_PATH = r"C:\Users\Group8\Downloads\logo_final.jpg"  # keep your existing path or change here

# ========= Auth / UI theme =========
users = {
    "Joy Pasala": "7452408",
    "Jonathan Walsh": "pass1",
    "Tom Mchugh": "6413717",
    "Jacob Rhados": "8002812",
    "Jerome Eid": "pass4",
    "Jason Watson": "7678721"
}

BG_COLOR = "#ffffff"
BTN_COLOR = "#007ACC"
BTN_TEXT_COLOR = "white"
ENTRY_BG = "white"
REMEMBER_FILE = PROJECT_ROOT / "remember_me.txt"

# ========= Utility: robust file resolver & launcher =========
def resolve_path(key: str) -> Path | None:
    """Return the first matching file path from CANDIDATES[key], or by recursive search."""
    if key not in CANDIDATES:
        return None

    # try preferred relative paths
    for rel in CANDIDATES[key]:
        p = (PROJECT_ROOT / rel).resolve()
        if p.exists():
            return p

    # fallback: recursive search for the basename(s)
    basenames = [Path(rel).name for rel in CANDIDATES[key]]
    for root, _, files in os.walk(PROJECT_ROOT):
        for name in basenames:
            matches = fnmatch.filter(files, name)
            if matches:
                return Path(root) / matches[0]
    return None

def run_script_by_key(key: str):
    """Find and start a script in a separate process with its dir as CWD."""
    p = resolve_path(key)
    if not p:
        messagebox.showerror("Error", f"Could not find script for '{key}'.\nLooked under: {CANDIDATES.get(key)}")
        return
    try:
        subprocess.Popen([sys.executable, str(p)], cwd=str(p.parent))
    except Exception as e:
        messagebox.showerror("Error", f"Failed to run {p.name}:\n{e}")

def start_live_feed_then_open():
    """
    Try to start a local Flask/OpenCV feed if a feed script exists,
    then open the browser at /video_feed. If no script is found, still open the URL
    (works if you already run the feed elsewhere).
    """
    feed_path = resolve_path("video_feed")
    if feed_path:
        try:
            # spawn the feed server/viewer in background
            subprocess.Popen([sys.executable, str(feed_path)], cwd=str(feed_path.parent))
            time.sleep(1.5)  # small warm-up
        except Exception as e:
            messagebox.showwarning("Live Feed", f"Started to open feed script failed:\n{e}\nOpening URL anyway…")

    stream_url = "http://localhost:5000/video_feed"
    try:
        webbrowser.open(stream_url)
    except Exception as e:
        messagebox.showerror("Error", f"Failed to open live feed URL:\n{e}")

# ========= Tk root & splash =========
root = tk.Tk()
root.title("YourQualityCheck")
root.state("zoomed")
root.configure(bg=BG_COLOR)
root.resizable(False, False)

splash = tk.Frame(root, bg=BG_COLOR)
splash.pack(fill="both", expand=True)

try:
    logo_img = Image.open(LOGO_PATH).resize((300, 300), Image.Resampling.LANCZOS)
    logo = ImageTk.PhotoImage(logo_img)
    tk.Label(splash, image=logo, bg=BG_COLOR).place(relx=0.5, rely=0.35, anchor="center")
except Exception:
    tk.Label(splash, text="Logo Not Found", font=("Arial", 20), bg=BG_COLOR).place(relx=0.5, rely=0.35, anchor="center")

# ========= Screens & navigation =========
page_history: list[str] = []
forward_stack: list[str] = []

def clear_root():
    for w in root.winfo_children():
        w.destroy()

def show_dashboard(username: str):
    clear_root()
    page_history.clear(); forward_stack.clear()

    main = tk.Frame(root, bg="white"); main.pack(fill="both", expand=True)
    tk.Label(main, text="Welcome to YourQualityCheck", font=("Arial", 20, "bold"), bg="white").pack(pady=(20, 10))

    buttons = tk.Frame(main, bg="white"); buttons.pack(expand=True)
    tiles = [
        ("👤", "Profile",       lambda: open_section("profile", username)),
        ("📦", "Inventory",     lambda: open_section("inventory", username)),
        ("📷", "Camera Feed",   lambda: open_section("camera", username)),
        ("🖼", "Photo Gallery", lambda: open_section("gallery", username)),
        ("🚪", "Logout",        do_logout),
    ]
    for col, (emoji, label, cmd) in enumerate(tiles):
        tk.Button(buttons, text=emoji, font=("Arial", 30), width=6, height=2, bg="#eeeeee", command=cmd)\
            .grid(row=0, column=col, padx=20, pady=20)
        tk.Label(buttons, text=label, font=("Arial", 13, "bold"), bg="white").grid(row=1, column=col)

def open_section(section: str, username: str):
    if not page_history or page_history[-1] != section:
        page_history.append(section); forward_stack.clear()
    render_section(section, username)

def do_logout():
    page_history.clear(); forward_stack.clear()
    clear_root(); show_login()

def render_section(section: str, username: str):
    clear_root()
    sidebar = tk.Frame(root, bg="#bdbdbd", width=170); sidebar.pack(side="left", fill="y"); sidebar.pack_propagate(False)
    tk.Button(sidebar, text="← Dashboard", font=("Arial", 12), bg="#eeeeee", anchor="w",
              command=lambda: show_dashboard(username)).pack(fill="x", pady=(5, 2))

    def sidebtn(key, label):
        tk.Button(sidebar, text=label, font=("Arial", 12, "bold" if section == key else "normal"),
                  bg="#eeeeee" if section == key else "#bdbdbd", anchor="w",
                  relief="flat", padx=10, pady=10,
                  command=(do_logout if key == "logout" else (lambda k=key: render_section(k, username))))\
            .pack(fill="x", pady=1)

    sidebtn("profile",  "👤 Profile")
    sidebtn("inventory","📦 Inventory")
    sidebtn("camera",   "📷 Camera Feed")
    sidebtn("gallery",  "🖼 Photo Gallery")
    sidebtn("logout",   "🚪 Logout")

    content = tk.Frame(root, bg="white"); content.pack(side="right", expand=True, fill="both")
    titles = {"profile":"👤 Profile","inventory":"📦 Model Management","camera":"📷 Camera Feed","gallery":"🖼 Dataset Tools"}
    tk.Label(content, text=titles.get(section, "Section"), font=("Arial", 18, "bold"), bg="white").pack(pady=(20,5))

    if section == "profile":
        tk.Label(content, text=f"User: {username}", font=("Arial", 14), bg="white").pack(pady=20)

    elif section == "inventory":
        tk.Button(content, text="📚 Train Model", bg="#90caf9", font=("Arial", 14),
                  command=lambda: run_script_by_key("train")).pack(pady=10)
        tk.Button(content, text="⚙ Automate Annotations", bg="#a5d6a7", font=("Arial", 14),
                  command=lambda: run_script_by_key("auto")).pack(pady=10)
        tk.Button(content, text="📂 Convert XML→YOLO", bg="#fbc02d", font=("Arial", 14),
                  command=lambda: run_script_by_key("xml2yolo")).pack(pady=10)

    elif section == "camera":
        tk.Button(content, text="📡 Run Inference", bg="#90caf9", font=("Arial", 14),
                  command=lambda: run_script_by_key("execute")).pack(pady=10)
        tk.Button(content, text="🌐 Open Live Feed (URL)", bg="#a5d6a7", font=("Arial", 14),
                  command=start_live_feed_then_open).pack(pady=10)

    elif section == "gallery":
        tk.Button(content, text="🔍 Visualise Single File", bg="#ce93d8", font=("Arial", 14),
                  command=lambda: run_script_by_key("viz_one")).pack(pady=10)
        tk.Button(content, text="🖼 Batch Visualise", bg="#ffab91", font=("Arial", 14),
                  command=lambda: run_script_by_key("viz_batch")).pack(pady=10)

# ========= Login screen (with Remember-me, Forgot-password & Enter-to-login) =========
def show_login():
    clear_root()
    login = tk.Frame(root, bg=BG_COLOR); login.pack(pady=20)

    # logo
    try:
        limg = Image.open(LOGO_PATH).resize((150, 150), Image.Resampling.LANCZOS)
        lphoto = ImageTk.PhotoImage(limg)
        tk.Label(login, image=lphoto, bg=BG_COLOR).grid(row=0, column=0, columnspan=3, pady=(10,20))
        login.image = lphoto  # prevent GC
    except Exception:
        tk.Label(login, text="Logo", bg=BG_COLOR, font=("Arial",14)).grid(row=0, column=0, columnspan=3, pady=(10,20))

    tk.Label(login, text="User Name:", bg=BG_COLOR, font=("Arial", 12)).grid(row=1, column=0, padx=10, pady=6, sticky="e")
    user_entry = tk.Entry(login, font=("Arial", 12), width=25, bg=ENTRY_BG); user_entry.grid(row=1, column=1, sticky="w", pady=6, columnspan=2)

    tk.Label(login, text="Password:", bg=BG_COLOR, font=("Arial", 12)).grid(row=2, column=0, padx=10, pady=6, sticky="e")
    pass_entry = tk.Entry(login, show="*", font=("Arial", 12), width=25, bg=ENTRY_BG); pass_entry.grid(row=2, column=1, sticky="w", pady=6, columnspan=2)

    # Remember me
    remember_var = tk.BooleanVar(value=False)
    if REMEMBER_FILE.exists():
        try:
            remembered = REMEMBER_FILE.read_text(encoding="utf-8").strip()
            if remembered:
                user_entry.insert(0, remembered)
                remember_var.set(True)
        except Exception:
            pass

    tk.Checkbutton(login, text="Remember Me", variable=remember_var, bg=BG_COLOR).grid(row=3, column=0, columnspan=3, pady=(0, 4))

    # Forgot password
    def forgot_password(_=None):
        uname = user_entry.get().strip()
        if uname in users:
            messagebox.showinfo("Password Recovery", f"Password for {uname}: {users[uname]}")
        else:
            messagebox.showerror("Error", "Enter a valid username first.")

    fp = tk.Label(login, text="Forgot Password?", fg="blue", cursor="hand2", bg=BG_COLOR, font=("Arial", 10))
    fp.grid(row=4, column=0, columnspan=3, pady=(0, 10))
    fp.bind("<Button-1>", forgot_password)

    # Login
    def do_login(_evt=None):
        uname = user_entry.get().strip()
        pwd = pass_entry.get()
        if uname in users and users[uname] == pwd:
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

    # Enter-to-login support
    login.bind_all("<Return>", do_login)

    tk.Button(login, text="Login", command=do_login, font=("Arial", 12),
              width=15, bg=BTN_COLOR, fg=BTN_TEXT_COLOR).grid(row=5, column=0, columnspan=3, pady=(0, 10))

# ========= Start app =========
root.after(1200, show_login)  # short splash
root.mainloop()
