import tkinter as tk
from PIL import Image, ImageTk
import tkinter.messagebox as messagebox
import os

# --- Login credentials
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

# --- Main Window Setup
root = tk.Tk()
root.title("YourQualityCheck")
root.state("zoomed")  
root.configure(bg=BG_COLOR)
root.resizable(False, False)

# --- Global state for remembered username
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
except:
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
        logo_label.grid(row=0, column=0, columnspan=3, pady=(10, 20))
    except:
        logo_label = tk.Label(login_frame, text="Logo", bg=BG_COLOR, font=("Arial", 14))
        logo_label.grid(row=0, column=0, columnspan=3, pady=(10, 20))

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
    forgot_btn.grid(row=5, column=0, columnspan=3, pady=(0, 20))
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
    login_button.grid(row=6, column=0, columnspan=3, pady=(0, 15))

    login_frame.image = login_logo

# --- Dashboard Screen ---
def show_dashboard(username):
    dashboard = tk.Frame(root, bg="white")
    dashboard.pack(fill="both", expand=True)

    # Left navigation panel
    nav_panel = tk.Frame(dashboard, bg="#1f78d1", width=170)
    nav_panel.pack(side="left", fill="y")
    nav_panel.pack_propagate(False)

    # Main content panel
    content_panel = tk.Frame(dashboard, bg="white")
    content_panel.pack(side="right", expand=True, fill="both")

    # Fixed title
    title_label = tk.Label(content_panel, text="Welcome to YourQualityCheck", font=("Arial", 20, "bold"), bg="white")
    title_label.pack(pady=(20, 5))

    # Content area for dynamic pages (this is key)
    content_area = tk.Frame(content_panel, bg="white")
    content_area.pack(expand=True, fill="both", padx=20, pady=20)

    def clear_content():
        for widget in content_area.winfo_children():
            widget.destroy()

    # Page render functions
    def show_profile():
        clear_content()
        tk.Label(content_area, text=f"👤  User: {username}", font=("Arial", 14), bg="white").pack(anchor="w", pady=5)
        tk.Label(content_area, text="This is your profile section.", font=("Arial", 12), bg="white").pack(anchor="w")

    def show_inventory():
        clear_content()
        tk.Label(content_area, text="📦  Inventory Section", font=("Arial", 14), bg="white").pack(anchor="w", pady=5)
        tk.Label(content_area, text="Track and manage all items here.", font=("Arial", 12), bg="white").pack(anchor="w")

    def show_camera():
        clear_content()
        tk.Label(content_area, text="📷  Camera Access", font=("Arial", 14), bg="white").pack(anchor="w", pady=5)
        tk.Label(content_area, text="Feature coming soon...", font=("Arial", 12), bg="white").pack(anchor="w")

    def show_gallery():
        clear_content()
        tk.Label(content_area, text="🖼️  Photo Gallery", font=("Arial", 14), bg="white").pack(anchor="w", pady=5)
        tk.Label(content_area, text="Feature coming soon...", font=("Arial", 12), bg="white").pack(anchor="w")

    def do_logout():
        dashboard.destroy()
        show_login_screen()

    def create_nav_button(text, command, color="#ffffff", fg="black"):
        btn = tk.Button(nav_panel, text=text, font=("Arial", 11), bg=color, fg=fg,
                        relief="flat", activebackground="#005fa3", activeforeground="white",
                        command=command, anchor="w", padx=15)
        btn.pack(fill="x", pady=2)
        return btn

    # Navigation Buttons
    create_nav_button("📋  My Profile", show_profile)
    create_nav_button("📦  Inventory", show_inventory)
    create_nav_button("📷  Access Camera", show_camera)
    create_nav_button("🖼️Photo Gallery", show_gallery)
    create_nav_button("🚪  Logout", do_logout, color="#dc3545", fg="white")

    # Bottom right logo
    try:
        small_logo_img = Image.open(r"C:\Users\joypa\Downloads\logo_small.png")
        small_logo_img = small_logo_img.resize((60, 60), Image.Resampling.LANCZOS)
        small_logo = ImageTk.PhotoImage(small_logo_img)

        logo_widget = tk.Label(content_panel, image=small_logo, bg="white")
        logo_widget.image = small_logo
        logo_widget.place(relx=1.0, rely=1.0, anchor="se", x=-10, y=-10)
    except:
        pass


# --- Start with splash ---
root.after(2000, show_login_screen)
root.mainloop()
