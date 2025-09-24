import tkinter as tk
from PIL import Image, ImageTk
import tkinter.messagebox as messagebox
import os, webbrowser, subprocess, sys

# -------------------
# User Credentials
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
        logo_label.grid(row=0, column=0, columnspan=3, pady=(10,20))
        login_frame.image = login_logo
    except Exception:
        tk.Label(login_frame, text="Logo", bg=BG_COLOR, font=("Arial",14)).grid(row=0, column=0, columnspan=3)

    tk.Label(login_frame, text="User Name:", bg=BG_COLOR, font=("Arial", 12)).grid(row=1, column=0, padx=10, pady=10, sticky="e")
    user_entry = tk.Entry(login_frame, font=("Arial", 12), width=25, bg=ENTRY_BG)
    user_entry.grid(row=1, column=1, columnspan=2, sticky="w", pady=10)
    user_entry.insert(0, remembered_username)

    tk.Label(login_frame, text="Password:", bg=BG_COLOR, font=("Arial", 12)).grid(row=2, column=0, padx=10, pady=10, sticky="e")
    pass_entry = tk.Entry(login_frame, show="*", font=("Arial", 12), width=25, bg=ENTRY_BG)
    pass_entry.grid(row=2, column=1, columnspan=2, sticky="w", pady=10)

    def login():
        username = user_entry.get()
        password = pass_entry.get()
        if username in users and users[username] == password:
            if os.path.exists(REMEMBER_FILE) and not password:
                os.remove(REMEMBER_FILE)
            if remembered_username:
                with open(REMEMBER_FILE, "w") as f: f.write(username)
            login_frame.destroy()
            show_dashboard(username)
        else:
            messagebox.showerror("Access Denied", "Invalid username or password.")

    tk.Button(login_frame, text="Login", command=login,
              font=("Arial", 12), width=15, bg=BTN_COLOR, fg=BTN_TEXT_COLOR).grid(row=6, column=0, columnspan=3, pady=(0,15))

# -------------------
# Dashboard
# -------------------
page_history, forward_stack = [], []

def show_dashboard(username):
    for widget in root.winfo_children(): widget.destroy()
    page_history.clear(); forward_stack.clear()

    main_frame = tk.Frame(root, bg="white"); main_frame.pack(fill="both", expand=True)

    tk.Label(main_frame, text="Welcome to YourQualityCheck", font=("Arial", 20, "bold"), bg="white").pack(pady=(20, 10))

    # Icon Buttons
    icons = [
        {"text": "👤", "label": "Profile", "command": lambda: open_section("profile", username)},
        {"text": "📦", "label": "Inventory", "command": lambda: open_section("inventory", username)},
        {"text": "📷", "label": "Camera Feed", "command": lambda: open_section("camera", username)},
        {"text": "🖼", "label": "Photo Gallery", "command": lambda: open_section("gallery", username)},
        {"text": "🚪", "label": "Logout", "command": lambda: do_logout()}
    ]

    buttons_frame = tk.Frame(main_frame, bg="white"); buttons_frame.pack(expand=True)
    for col, item in enumerate(icons):
        tk.Button(buttons_frame, text=item["text"], font=("Arial", 30), width=6, height=2, bg="#eeeeee",
                  command=item["command"]).grid(row=0, column=col, padx=20, pady=20)
        tk.Label(buttons_frame, text=item["label"], font=("Arial", 13, "bold"), bg="white").grid(row=1, column=col)

# -------------------
# Section Rendering
# -------------------
def clear_root(): 
    for widget in root.winfo_children(): widget.destroy()

def do_logout():
    page_history.clear(); forward_stack.clear(); clear_root(); show_login_screen()

def open_section(section, username):
    if not page_history or page_history[-1] != section:
        page_history.append(section); forward_stack.clear()
    render_section(section, username)

def render_section(section, username):
    clear_root()
    sidebar = tk.Frame(root, bg="#bdbdbd", width=170); sidebar.pack(side="left", fill="y")

    tk.Button(sidebar, text="← Dashboard", font=("Arial", 12),
              bg="#eeeeee", anchor="w", command=lambda: show_dashboard(username)).pack(fill="x")

    content_panel = tk.Frame(root, bg="white"); content_panel.pack(side="right", expand=True, fill="both")

    if section == "profile":
        tk.Label(content_panel, text=f"User: {username}", font=("Arial", 14), bg="white").pack(pady=20)

    elif section == "inventory":
        tk.Label(content_panel, text="Model Management", font=("Arial", 16, "bold"), bg="white").pack(pady=20)
        tk.Button(content_panel, text="📚 Train Model", bg="#90caf9", font=("Arial", 14),
                  command=lambda: run_script("train (1).py")).pack(pady=10)
        tk.Button(content_panel, text="⚙ Automate Annotations", bg="#a5d6a7", font=("Arial", 14),
                  command=lambda: run_script("AutomateKdenlive (1).py")).pack(pady=10)
        tk.Button(content_panel, text="📂 Convert XML→YOLO", bg="#fbc02d", font=("Arial", 14),
                  command=lambda: run_script("KdenliveXMLtoYOLOv8 (1).py")).pack(pady=10)

    elif section == "camera":
        tk.Label(content_panel, text="Camera Feed", font=("Arial", 16, "bold"), bg="white").pack(pady=20)
        tk.Button(content_panel, text="📡 Run Inference", bg="#90caf9", font=("Arial", 14),
                  command=lambda: run_script("execute (2).py")).pack(pady=10)

    elif section == "gallery":
        tk.Label(content_panel, text="Dataset Tools", font=("Arial", 16, "bold"), bg="white").pack(pady=20)
        tk.Button(content_panel, text="🔍 Visualise Single File", bg="#ce93d8", font=("Arial", 14),
                  command=lambda: run_script("visualiseTXT.py")).pack(pady=10)
        tk.Button(content_panel, text="🖼 Batch Visualise", bg="#ffab91", font=("Arial", 14),
                  command=lambda: run_script("BatchVisualiseTXTBB.py")).pack(pady=10)

def run_script(script_name):
    header = os.path.dirname(os.path.realpath(__file__))
    script_path = os.path.join(header, script_name)
    if not os.path.exists(script_path):
        messagebox.showerror("Error", f"{script_name} not found")
        return
    try:
        subprocess.Popen([sys.executable, script_path])
    except Exception as e:
        messagebox.showerror("Error", f"Failed to run {script_name}:\n{e}")

# -------------------
# Start App
# -------------------
root.after(2000, show_login_screen)
root.mainloop()
