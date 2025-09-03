import tkinter as tk
from PIL import Image, ImageTk
import tkinter.messagebox as messagebox
import os
import webbrowser
import subprocess
import sys

# --- Login credentials ---

users = {
    "Joy Pasala": "7452408",
    "Jonathan Walsh": "pass1",
    "Tom Mchugh": "6413717",
    "Jacob Rhados": "8002812",
    "Jerome Eid": "pass4",
    "Jason Watson": "7678721"
}

# --- Theme Settings ---

BG_COLOR = "#ffffff"
BTN_COLOR = "#007ACC"
BTN_TEXT_COLOR = "white"
ENTRY_BG = "white"

# --- File to store remembered user ---

REMEMBER_FILE = "remember_me.txt"

# --- Main Window Setup ---

root = tk.Tk()
root.title("YourQualityCheck")
root.state("zoomed")
root.configure(bg=BG_COLOR)
root.resizable(False, False)

# --- Global state for remembered username ---

remembered_username = ""
if os.path.exists(REMEMBER_FILE):
    with open(REMEMBER_FILE, "r") as f:
        remembered_username = f.read().strip()

# --- Splash Frame ---

splash_frame = tk.Frame(root, bg=BG_COLOR)
splash_frame.pack(fill="both", expand=True)

logo_path = r"C:\Users\joypa\Downloads\logo_final.jpg"

try:
    logo_img = Image.open(logo_path)
    logo_img = logo_img.resize((300, 300), Image.Resampling.LANCZOS)
    logo = ImageTk.PhotoImage(logo_img)
    logo_label = tk.Label(splash_frame, image=logo, bg=BG_COLOR)
    logo_label.place(relx=0.5, rely=0.4, anchor="center")
except Exception:
    logo_label = tk.Label(splash_frame, text="Logo Not Found", font=("Arial", 20), bg=BG_COLOR)
    logo_label.place(relx=0.5, rely=0.4, anchor="center")

# --- Login Screen ---

def show_login_screen():
    splash_frame.destroy()

    login_frame = tk.Frame(root, bg=BG_COLOR)
    login_frame.pack(pady=20)

    try:
        login_logo_img = Image.open(logo_path)
        login_logo_img = login_logo_img.resize((150, 150), Image.Resampling.LANCZOS)
        login_logo = ImageTk.PhotoImage(login_logo_img)
        logo_label = tk.Label(login_frame, image=login_logo, bg=BG_COLOR)
        logo_label.grid(row=0, column=0, columnspan=3, pady=(10,20))
    except Exception:
        logo_label = tk.Label(login_frame, text="Logo", bg=BG_COLOR, font=("Arial",14))
        logo_label.grid(row=0, column=0, columnspan=3, pady=(10,20))

    tk.Label(login_frame, text="User Name:", bg=BG_COLOR, font=("Arial", 12)).grid(row=1, column=0, padx=10, pady=10, sticky="e")
    user_entry = tk.Entry(login_frame, font=("Arial", 12), width=25, bg=ENTRY_BG)
    user_entry.grid(row=1, column=1, columnspan=2, sticky="w", pady=10)
    user_entry.insert(0, remembered_username)

    tk.Label(login_frame, text="Password:", bg=BG_COLOR, font=("Arial", 12)).grid(row=2, column=0, padx=10, pady=10, sticky="e")
    pass_entry = tk.Entry(login_frame, show="*", font=("Arial", 12), width=25, bg=ENTRY_BG)
    pass_entry.grid(row=2, column=1, columnspan=2, sticky="w", pady=10)

    def toggle_password():
        if pass_entry.cget('show') == '*':
            pass_entry.config(show='')
            toggle_btn.config(text='Hide Password')
        else:
            pass_entry.config(show='*')
            toggle_btn.config(text='Show Password')

    toggle_btn = tk.Button(login_frame, text="Show Password", command=toggle_password, font=("Arial", 10), bg=BTN_COLOR, fg=BTN_TEXT_COLOR)
    toggle_btn.grid(row=3, column=0, columnspan=3, pady=(0, 15))

    remember_var = tk.BooleanVar()
    if remembered_username:
        remember_var.set(True)

    remember_check = tk.Checkbutton(login_frame, text="Remember Me", variable=remember_var, bg=BG_COLOR)
    remember_check.grid(row=4, column=0, columnspan=3, pady=(0, 5))

    def forgot_password():
        username = user_entry.get()
        if username in users:
            messagebox.showinfo("Password Recovery", f"Password for {username} is: {users[username]}")
        else:
            messagebox.showerror("Error", "Enter a valid username first.")

    forgot_btn = tk.Label(login_frame, text="Forgot Password?", fg="blue", cursor="hand2", bg=BG_COLOR, font=("Arial", 10))
    forgot_btn.grid(row=5, column=0, columnspan=3, pady=(0,20))
    forgot_btn.bind("<Button-1>", lambda e: forgot_password())

    def login():
        username = user_entry.get()
        password = pass_entry.get()
        if username in users and users[username] == password:
            if remember_var.get():
                with open(REMEMBER_FILE, "w") as f:
                    f.write(username)
            else:
                if os.path.exists(REMEMBER_FILE):
                    os.remove(REMEMBER_FILE)
            login_frame.destroy()
            show_dashboard(username)
        else:
            messagebox.showerror("Access Denied", "Invalid username or password.")

    login_button = tk.Button(login_frame, text="Login", command=login, font=("Arial", 12), width=15, bg=BTN_COLOR, fg=BTN_TEXT_COLOR)
    login_button.grid(row=6, column=0, columnspan=3, pady=(0,15))

    # Keep reference to image to prevent garbage collection
    login_frame.image = login_logo

# --- Global navigation state ---

page_history = []
forward_stack = []

# --- Dashboard ---

def show_dashboard(username):
    for widget in root.winfo_children():
        widget.destroy()
    page_history.clear()
    forward_stack.clear()

    main_frame = tk.Frame(root, bg="white")
    main_frame.pack(fill="both", expand=True)

    title_label = tk.Label(main_frame, text="Welcome to YourQualityCheck", font=("Arial", 20, "bold"), bg="white")
    title_label.pack(pady=(20, 10))

    try:
        logo_img = Image.open(logo_path)
        logo_img = logo_img.resize((250, 250), Image.Resampling.LANCZOS)
        logo = ImageTk.PhotoImage(logo_img)
        logo_label = tk.Label(main_frame, image=logo, bg="white")
        logo_label.image = logo
        logo_label.pack(pady=(0, 10))
    except Exception:
        logo_label = tk.Label(main_frame, text="Logo Not Found", font=("Arial", 16), bg="white")
        logo_label.pack(pady=(0, 10))

    icons = [
        {"text": "👤", "label": "Profile", "command": lambda: open_section("profile", username), "bg": "#e0e0e0"},
        {"text": "📦", "label": "Inventory", "command": lambda: open_section("inventory", username), "bg": "#e1f5fe"},
        {"text": "📷", "label": "Camera Feed", "command": lambda: open_section("camera", username), "bg": "#ffe0b2"},
        {"text": "🖼", "label": "Photo Gallery", "command": lambda: open_section("gallery", username), "bg": "#c8e6c9"},
        {"text": "🚪", "label": "Logout", "command": lambda: do_logout(), "bg": "#ffcdd2"}
    ]

    buttons_frame = tk.Frame(main_frame, bg="white")
    buttons_frame.pack(expand=True)

    for col, item in enumerate(icons):
        btn = tk.Button(buttons_frame, text=item["text"], font=("Arial", 30), width=6, height=2, bg=item["bg"], command=item["command"])
        btn.grid(row=0, column=col, padx=20, pady=20)
        tk.Label(buttons_frame, text=item["label"], font=("Arial", 13, "bold"), bg="white").grid(row=1, column=col)

def open_section(section, username):
    if not page_history or page_history[-1] != section:
        page_history.append(section)
        forward_stack.clear()
    render_section(section, username)

def go_back(username):
    if len(page_history) > 1:
        forward_stack.append(page_history.pop())
        render_section(page_history[-1], username)

def go_forward(username):
    if forward_stack:
        section = forward_stack.pop()
        page_history.append(section)
        render_section(section, username)

def clear_root():
    for widget in root.winfo_children():
        widget.destroy()

def do_logout():
    page_history.clear()
    forward_stack.clear()
    clear_root()
    show_login_screen()

def render_section(section, username):
    clear_root()

    sidebar = tk.Frame(root, bg="#bdbdbd", width=170)
    sidebar.pack(side="left", fill="y")
    sidebar.pack_propagate(False)

    def to_dashboard():
        show_dashboard(username)

    arrow_btn = tk.Button(
        sidebar, text="← Dashboard", font=("Arial", 12),
        bg="#eeeeee", fg="black",
        relief="flat", anchor="w", padx=10, pady=10,
        activebackground="#d4d4d4", command=to_dashboard
    )
    arrow_btn.pack(fill="x", pady=(5, 2))

    section_titles = {
        "profile": "👤 Profile",
        "inventory": "📦 Inventory",
        "camera": "📷 Camera Feed",
        "gallery": "🖼 Photo Gallery",
        "logout": "🚪 Logout"
    }

    icon_commands = {
        "profile": lambda: render_section("profile", username),
        "inventory": lambda: render_section("inventory", username),
        "camera": lambda: render_section("camera", username),
        "gallery": lambda: render_section("gallery", username),
        "logout": do_logout
    }

    for key in ["profile", "inventory", "camera", "gallery", "logout"]:
        bg_col = "#eeeeee" if key == section else "#bdbdbd"
        style = ("Arial", 12, "bold" if key == section else "normal")
        btn = tk.Button(
            sidebar,
            text=section_titles[key],
            font=style,
            bg=bg_col,
            fg="black",
            relief="flat",
            anchor="w",
            padx=10,
            pady=10,
            activebackground="#d4d4d4",
            command=icon_commands[key]
        )
        btn.pack(fill="x", pady=1)

    content_panel = tk.Frame(root, bg="white")
    content_panel.pack(side="right", expand=True, fill="both")

    title = section_titles.get(section, "Section")
    tk.Label(content_panel, text=title, font=("Arial", 18, "bold"), bg="white").pack(pady=(20, 5))

    def access_camera_section():
        live_feed_btn = tk.Button(
            content_panel, text="📡 Access Live Feed", font=("Arial", 16, "bold"),
            width=20, height=3, bg="#90caf9", fg="black",
            command=open_live_feed)
        live_feed_btn.pack(pady=(40, 20))

        local_camera_btn = tk.Button(
            content_panel, text="📷 Access Camera", font=("Arial", 16, "bold"),
            width=20, height=3, bg="#a5d6a7", fg="black",
            command=open_local_camera)
        local_camera_btn.pack(pady=(0, 20))

    section_func_map = {
        "profile": lambda: tk.Label(content_panel, text=f"User: {username}", font=("Arial", 14), bg="white").pack(pady=20),
        "inventory": lambda: tk.Label(content_panel, text="Track and manage all items here.", font=("Arial", 12), bg="white").pack(pady=20),
        "camera": access_camera_section,
        "gallery": lambda: tk.Label(content_panel, text="Feature coming soon...", font=("Arial", 12), bg="white").pack(pady=20),
    }

    if section in section_func_map:
        section_func_map[section]()

def open_live_feed():
    stream_url = "http://localhost:5000/video_feed"
    try:
        webbrowser.open(stream_url)
    except Exception as e:
        messagebox.showerror("Error", f"Failed to open live feed URL:\n{e}")

def open_local_camera():
    # Replace this with your actual Video_Feed.py location
    script_path = r"C:\Users\joypa\path_to\Video_Feed.py"
    try:
        subprocess.Popen([sys.executable, script_path])
    except Exception as e:
        messagebox.showerror("Error", f"Failed to launch camera app:\n{e}")

# --- Start with splash ---

root.after(2000, show_login_screen)
root.mainloop()
