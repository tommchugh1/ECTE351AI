import tkinter as tk
from PIL import Image, ImageTk
import tkinter.messagebox as messagebox
import os
import webbrowser
import subprocess
import sys
import pathlib

# -------------------
# App / Paths
# -------------------
PROJECT_ROOT = pathlib.Path(__file__).resolve().parent

# Absolute script paths (clean names as requested)
SCRIPT_PATHS = {
    "execute":   PROJECT_ROOT / "execute.py",
    "train":     PROJECT_ROOT / "train.py",
    "auto":      PROJECT_ROOT / "VideoProcessing" / "AutomateKdenlive.py",
    "xml2yolo":  PROJECT_ROOT / "VideoProcessing" / "KdenliveXMLtoYOLOv8.py",
    "viz_one":   PROJECT_ROOT / "VideoProcessing" / "visualiseTXT.py",
    "viz_batch": PROJECT_ROOT / "VideoProcessing" / "BatchVisualiseTXTBB.py",
}

# -------------------
# Login credentials
# -------------------
users = {
    "Joy Pasala": "7452408",
    "Jonathan Walsh": "pass1",
    "Tom Mchugh": "6413717",
    "Jacob Rhados": "8002812",
    "Jerome Eid": "pass4",
    "Jason Watson": "7678721"
}

# -------------------
# Theme Settings
# -------------------
BG_COLOR = "#ffffff"
BTN_COLOR = "#007ACC"
BTN_TEXT_COLOR = "white"
ENTRY_BG = "white"

REMEMBER_FILE = "remember_me.txt"

# -------------------
# Tk Root
# -------------------
root = tk.Tk()
root.title("YourQualityCheck")
root.state("zoomed")
root.configure(bg=BG_COLOR)
root.resizable(False, False)

# -------------------
# Remembered Username
# -------------------
remembered_username = ""
if os.path.exists(REMEMBER_FILE):
    with open(REMEMBER_FILE, "r") as f:
        remembered_username = f.read().strip()

# -------------------
# Splash Screen
# -------------------
splash_frame = tk.Frame(root, bg=BG_COLOR)
splash_frame.pack(fill="both", expand=True)

# Update to your actual logo path if needed
logo_path = r"C:\Users\joypa\Downloads\logo_final.jpg"

try:
    logo_img = Image.open(logo_path).resize((300, 300), Image.Resampling.LANCZOS)
    logo = ImageTk.PhotoImage(logo_img)
    logo_label = tk.Label(splash_frame, image=logo, bg=BG_COLOR)
    logo_label.place(relx=0.5, rely=0.4, anchor="center")
except Exception:
    logo_label = tk.Label(splash_frame, text="Logo Not Found", font=("Arial", 20), bg=BG_COLOR)
    logo_label.place(relx=0.5, rely=0.4, anchor="center")

# -------------------
# Helpers
# -------------------
def _run_script(path: pathlib.Path):
    """Launch a Python script in a separate process with its own working dir."""
    if not path.exists():
        messagebox.showerror("Error", f"{path.name} not found\n\n{path}")
        return
    try:
        subprocess.Popen([sys.executable, str(path)], cwd=str(path.parent))
    except Exception as e:
        messagebox.showerror("Error", f"Failed to run {path.name}:\n{e}")

def open_live_feed_url():
    # Adjust if your Flask/stream address differs
    stream_url = "http://localhost:5000/video_feed"
    try:
        webbrowser.open(stream_url)
    except Exception as e:
        messagebox.showerror("Error", f"Failed to open live feed URL:\n{e}")

# -------------------
# Login Screen
# -------------------
def show_login_screen():
    splash_frame.destroy()

    login_frame = tk.Frame(root, bg=BG_COLOR)
    login_frame.pack(pady=20)

    try:
        login_logo_img = Image.open(logo_path).resize((150, 150), Image.Resampling.LANCZOS)
        login_logo = ImageTk.PhotoImage(login_logo_img)
        logo_label = tk.Label(login_frame, image=login_logo, bg=BG_COLOR)
        logo_label.grid(row=0, column=0, columnspan=3, pady=(10, 20))
        # keep reference
        login_frame.image = login_logo
    except Exception:
        tk.Label(login_frame, text="Logo", bg=BG_COLOR, font=("Arial", 14)).grid(
            row=0, column=0, columnspan=3, pady=(10, 20)
        )

    tk.Label(login_frame, text="User Name:", bg=BG_COLOR, font=("Arial", 12)).grid(
        row=1, column=0, padx=10, pady=10, sticky="e"
    )
    user_entry = tk.Entry(login_frame, font=("Arial", 12), width=25, bg=ENTRY_BG)
    user_entry.grid(row=1, column=1, columnspan=2, sticky="w", pady=10)
    user_entry.insert(0, remembered_username)

    tk.Label(login_frame, text="Password:", bg=BG_COLOR, font=("Arial", 12)).grid(
        row=2, column=0, padx=10, pady=10, sticky="e"
    )
    pass_entry = tk.Entry(login_frame, show="*", font=("Arial", 12), width=25, bg=ENTRY_BG)
    pass_entry.grid(row=2, column=1, columnspan=2, sticky="w", pady=10)

    def login():
        username = user_entry.get()
        password = pass_entry.get()
        if username in users and users[username] == password:
            # remember the user if they had one previously saved
            with open(REMEMBER_FILE, "w") as f:
                f.write(username)
            login_frame.destroy()
            show_dashboard(username)
        else:
            messagebox.showerror("Access Denied", "Invalid username or password.")

    tk.Button(
        login_frame, text="Login", command=login, font=("Arial", 12),
        width=15, bg=BTN_COLOR, fg=BTN_TEXT_COLOR
    ).grid(row=6, column=0, columnspan=3, pady=(0, 15))

# -------------------
# Navigation State
# -------------------
page_history = []
forward_stack = []

def clear_root():
    for widget in root.winfo_children():
        widget.destroy()

def do_logout():
    page_history.clear()
    forward_stack.clear()
    clear_root()
    show_login_screen()

def open_section(section, username):
    if not page_history or page_history[-1] != section:
        page_history.append(section)
        forward_stack.clear()
    render_section(section, username)

# -------------------
# Dashboard
# -------------------
def show_dashboard(username):
    for widget in root.winfo_children():
        widget.destroy()
    page_history.clear()
    forward_stack.clear()

    main_frame = tk.Frame(root, bg="white")
    main_frame.pack(fill="both", expand=True)

    tk.Label(main_frame, text="Welcome to YourQualityCheck",
             font=("Arial", 20, "bold"), bg="white").pack(pady=(20, 10))

    buttons_frame = tk.Frame(main_frame, bg="white")
    buttons_frame.pack(expand=True)

    icons = [
        {"text": "👤", "label": "Profile",       "command": lambda: open_section("profile", username)},
        {"text": "📦", "label": "Inventory",     "command": lambda: open_section("inventory", username)},
        {"text": "📷", "label": "Camera Feed",   "command": lambda: open_section("camera", username)},
        {"text": "🖼", "label": "Photo Gallery", "command": lambda: open_section("gallery", username)},
        {"text": "🚪", "label": "Logout",        "command": do_logout},
    ]

    for col, item in enumerate(icons):
        tk.Button(buttons_frame, text=item["text"], font=("Arial", 30),
                  width=6, height=2, bg="#eeeeee",
                  command=item["command"]).grid(row=0, column=col, padx=20, pady=20)
        tk.Label(buttons_frame, text=item["label"], font=("Arial", 13, "bold"),
                 bg="white").grid(row=1, column=col)

# -------------------
# Sections
# -------------------
def render_section(section, username):
    clear_root()

    # Sidebar
    sidebar = tk.Frame(root, bg="#bdbdbd", width=170)
    sidebar.pack(side="left", fill="y")
    sidebar.pack_propagate(False)

    tk.Button(
        sidebar, text="← Dashboard", font=("Arial", 12),
        bg="#eeeeee", fg="black", relief="flat", anchor="w", padx=10, pady=10,
        activebackground="#d4d4d4", command=lambda: show_dashboard(username)
    ).pack(fill="x", pady=(5, 2))

    for key, label in [
        ("profile", "👤 Profile"),
        ("inventory", "📦 Inventory"),
        ("camera", "📷 Camera Feed"),
        ("gallery", "🖼 Photo Gallery"),
        ("logout", "🚪 Logout"),
    ]:
        tk.Button(
            sidebar, text=label, font=("Arial", 12, "bold" if key == section else "normal"),
            bg="#eeeeee" if key == section else "#bdbdbd", fg="black", relief="flat",
            anchor="w", padx=10, pady=10, activebackground="#d4d4d4",
            command=(do_logout if key == "logout" else (lambda k=key: render_section(k, username)))
        ).pack(fill="x", pady=1)

    # Content
    content = tk.Frame(root, bg="white")
    content.pack(side="right", expand=True, fill="both")

    titles = {
        "profile": "👤 Profile",
        "inventory": "📦 Model Management",
        "camera": "📷 Camera Feed",
        "gallery": "🖼 Dataset Tools",
    }
    tk.Label(content, text=titles.get(section, "Section"), font=("Arial", 18, "bold"),
             bg="white").pack(pady=(20, 5))

    if section == "profile":
        tk.Label(content, text=f"User: {username}", font=("Arial", 14), bg="white").pack(pady=20)

    elif section == "inventory":
        tk.Button(content, text="📚 Train Model", bg="#90caf9", font=("Arial", 14),
                  command=lambda: _run_script(SCRIPT_PATHS["train"])).pack(pady=10)
        tk.Button(content, text="⚙ Automate Annotations", bg="#a5d6a7", font=("Arial", 14),
                  command=lambda: _run_script(SCRIPT_PATHS["auto"])).pack(pady=10)
        tk.Button(content, text="📂 Convert XML→YOLO", bg="#fbc02d", font=("Arial", 14),
                  command=lambda: _run_script(SCRIPT_PATHS["xml2yolo"])).pack(pady=10)

    elif section == "camera":
        tk.Button(content, text="📡 Run Inference", bg="#90caf9", font=("Arial", 14),
                  command=lambda: _run_script(SCRIPT_PATHS["execute"])).pack(pady=10)
        tk.Button(content, text="🌐 Open Live Feed (URL)", bg="#a5d6a7", font=("Arial", 14),
                  command=open_live_feed_url).pack(pady=10)

    elif section == "gallery":
        tk.Button(content, text="🔍 Visualise Single File", bg="#ce93d8", font=("Arial", 14),
                  command=lambda: _run_script(SCRIPT_PATHS["viz_one"])).pack(pady=10)
        tk.Button(content, text="🖼 Batch Visualise", bg="#ffab91", font=("Arial", 14),
                  command=lambda: _run_script(SCRIPT_PATHS["viz_batch"])).pack(pady=10)

# -------------------
# Start App
# -------------------
root.after(2000, show_login_screen)
root.mainloop()
