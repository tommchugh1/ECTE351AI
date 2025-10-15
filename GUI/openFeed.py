import tkinter as tk
from tkinter import ttk
from PIL import Image, ImageTk
import threading
import cv2
import time

class StreamPopup:
    def __init__(self, parent, url, title="Live Stream", max_width=640, max_height=640):
        self.parent = parent
        self.url = url
        self.max_w = max_width
        self.max_h = max_height
        self.running = True
        self.paused = False
        self.frame = None
        self.frame_lock = threading.Lock()

        print(f"[DEBUG] Starting StreamPopup for URL: {self.url}")

        # Create the popup window
        self.win = tk.Toplevel(parent)
        self.win.title(title)
        self.win.protocol("WM_DELETE_WINDOW", self._on_close)
        self.win.geometry("640x640")
        self.win.resizable(True, True)

        # UI layout
        top = ttk.Frame(self.win, padding=8)
        top.pack(fill="both", expand=True)

        self.video_lbl = tk.Label(top, bg="black")
        self.video_lbl.pack(fill="both", expand=True)

        controls = ttk.Frame(top)
        controls.pack(fill="x", pady=(6, 0))

        self.status_var = tk.StringVar(value="Connecting…")
        ttk.Label(controls, textvariable=self.status_var).pack(side="left")

        self.btn_toggle = ttk.Button(controls, text="Pause", command=self._toggle_pause)
        self.btn_toggle.pack(side="right", padx=(6, 0))
        ttk.Button(controls, text="Close", command=self._on_close).pack(side="right")

        # Start reader and UI updater
        self.reader_thread = threading.Thread(target=self._reader_worker, daemon=True)
        self.reader_thread.start()
        self._refresh_ui()

    def _reader_worker(self):
        cap = cv2.VideoCapture(self.url)

        if not cap or not cap.isOpened():
            self.status_var.set("❌ Failed to open stream.")
            return

        self.status_var.set("✅ Connected")

        while self.running:
            if self.paused:
                time.sleep(0.05)
                continue

            ok, frame = cap.read()
            if not ok:
                self.status_var.set("⏳ Waiting for stream…")
                time.sleep(0.1)
                continue

            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            with self.frame_lock:
                self.frame = frame
            self.status_var.set("📡 Streaming")

        cap.release()

    def _refresh_ui(self):
        if not self.running:
            return

        frame = None
        with self.frame_lock:
            if self.frame is not None:
                frame = self.frame.copy()

        if frame is not None:
            h, w, _ = frame.shape
            win_w = max(1, self.win.winfo_width() - 20)
            win_h = max(1, self.win.winfo_height() - 80)
            scale = min(win_w / w, win_h / h)
            new_w, new_h = int(w * scale), int(h * scale)
            frame_resized = cv2.resize(frame, (new_w, new_h))
            img = ImageTk.PhotoImage(Image.fromarray(frame_resized))

            self.video_lbl.imgtk = img
            self.video_lbl.config(image=img)

        self.win.after(33, self._refresh_ui)

    def _toggle_pause(self):
        self.paused = not self.paused
        self.btn_toggle.config(text="Resume" if self.paused else "Pause")
        self.status_var.set("⏸️ Paused" if self.paused else "📡 Streaming")

    def _on_close(self):
        if not self.running:
            return
        self.running = False
        self.win.after(100, self.win.destroy)


# Exportable wrapper for import
def open_stream_popup(parent, url, title="Live Stream"):
    return StreamPopup(parent, url, title=title)
