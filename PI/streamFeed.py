import tkinter as tk
from tkinter import ttk, messagebox
import subprocess
import threading
import socket
import os
import time


MEDIAMTX_HOST = "0.0.0.0"
RTSP_PORT = 8889
CHECK_INTERVAL_MS = 800

class MediaMTXApp:
    def __init__(self, root):
        self.root = root
        self.root.title("MediaMTX Controller")
        self.process = None

        self.script_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "mediamtx")
        self.binary_path = os.path.join(self.script_dir, "mediamtx")
        self.config_path = os.path.join(self.script_dir, "mediamtx.yml")

        frm = ttk.Frame(root, padding=12)
        frm.pack(fill="both", expand=True)

        self.start_btn = ttk.Button(frm, text="Start", command=self.start_server)
        self.stop_btn  = ttk.Button(frm, text="Stop", command=self.stop_server)
        self.start_btn.grid(row=0, column=0, padx=6, pady=6, sticky="ew")
        self.stop_btn.grid(row=0, column=1, padx=6, pady=6, sticky="ew")

        self.status_canvas = tk.Canvas(frm, width=14, height=14, highlightthickness=0)
        self.status_dot = self.status_canvas.create_oval(2, 2, 12, 12, fill="#c33", outline="#a11")
        self.status_canvas.grid(row=1, column=0, padx=(6,2), pady=(4,6), sticky="w")
        self.status_label = ttk.Label(frm, text="Stopped")
        self.status_label.grid(row=1, column=1, padx=(2,6), pady=(4,6), sticky="w")

        self.root.protocol("WM_DELETE_WINDOW", self.on_close)
        self._update_status()

    def start_server(self):
        if not os.path.isfile(self.binary_path):
            messagebox.showerror("MediaMTX", f"Binary not found:\n{self.binary_path}")
            return
        if not os.path.isfile(self.config_path):
            messagebox.showerror("MediaMTX", f"Config not found:\n{self.config_path}")
            return
        if self.process and self.process.poll() is None:
            messagebox.showinfo("MediaMTX", "Already running.")
            return

        def run():
            try:
                with open(os.devnull, "wb") as devnull:
                    self.process = subprocess.Popen(
                        [self.binary_path, self.config_path],
                        cwd=self.script_dir,
                        stdout=devnull,
                        stderr=devnull
                    )
            except Exception as e:
                messagebox.showerror("MediaMTX", f"Failed to start:\n{e}")

        threading.Thread(target=run, daemon=True).start()

    def stop_server(self):
        if self.process and self.process.poll() is None:
            try:
                self.process.terminate()
                deadline = time.time() + 2.5
                while time.time() < deadline and self.process.poll() is None:
                    time.sleep(0.1)
                if self.process.poll() is None:
                    self.process.kill()
            except Exception as e:
                messagebox.showerror("MediaMTX", f"Failed to stop:\n{e}")
            
    def _port_open(self, host, port, timeout=0.25):
        try:
            with socket.create_connection((host, port), timeout=timeout):
                return True
        except OSError:
            return False

    def _update_status(self):
        running = (self.process is not None and self.process.poll() is None)
        port_ok = self._port_open(MEDIAMTX_HOST, RTSP_PORT) if running else False

        if running and port_ok:
            self.status_canvas.itemconfig(self.status_dot, fill="#2aa745", outline="#1d7f33")
            self.status_label.config(text="Running")
        elif running:
            self.status_canvas.itemconfig(self.status_dot, fill="#e0a800", outline="#c28a00")
            self.status_label.config(text="Starting…")
        else:
            self.status_canvas.itemconfig(self.status_dot, fill="#c33", outline="#a11")
            self.status_label.config(text="Stopped")

        self.root.after(CHECK_INTERVAL_MS, self._update_status)

    def on_close(self):
        self.stop_server()
        self.root.destroy()

if __name__ == "__main__":
    root = tk.Tk()
    try:
        style = ttk.Style()
        if "clam" in style.theme_names():
            style.theme_use("clam")
    except Exception:
        pass
    app = MediaMTXApp(root)
    root.mainloop()