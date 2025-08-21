import tkinter as tk
from PIL import Image, ImageTk
import tkinter.messagebox as messagebox

# Function to show login screen after splash
def show_login():
    splash_frame.destroy()
    show_login_screen()

# Main window setup
root = tk.Tk()
root.title("YourQualityCheck")
root.geometry("500x500")
root.configure(bg="white")
root.resizable(False, False)

# Splash Frame
splash_frame = tk.Frame(root, bg="white")
splash_frame.pack(fill="both", expand=True)

# Correct Windows-compatible image path
logo_path = r"C:\Users\joypa\Downloads\test_logo.jpg"

try:
    logo_img = Image.open(logo_path)
    logo_img = logo_img.resize((300, 300), Image.Resampling.LANCZOS)
    logo = ImageTk.PhotoImage(logo_img)

    logo_label = tk.Label(splash_frame, image=logo, bg="white")
    logo_label.place(relx=0.5, rely=0.4, anchor="center")
except FileNotFoundError:
    logo_label = tk.Label(splash_frame, text="Logo Not Found", font=("Arial", 20), bg="white")
    logo_label.place(relx=0.5, rely=0.4, anchor="center")


# Fade-like delay before going to login screen
root.after(2000, show_login)

# Dummy login screen
def show_login_screen():
    login_frame = tk.Frame(root, bg="white")
    login_frame.pack(pady=20)

    tk.Label(login_frame, text="User Name:", bg="white").grid(row=0, column=0, padx=10, pady=10)
    user_entry = tk.Entry(login_frame)
    user_entry.grid(row=0, column=1)

    tk.Label(login_frame, text="Password:", bg="white").grid(row=1, column=0, padx=10, pady=10)
    pass_entry = tk.Entry(login_frame, show="*")
    pass_entry.grid(row=1, column=1)

    def login():
        username = user_entry.get()
        password = pass_entry.get()
        messagebox.showinfo("Mock Login", f"Welcome, {username}!")

    tk.Button(login_frame, text="Login", command=login).grid(row=2, columnspan=2, pady=10)

# Run GUI
root.mainloop()
