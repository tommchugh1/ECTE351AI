import tkinter as tk
from tkinter import ttk, messagebox
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

        # Create popup window
        self.win = tk.Toplevel(parent)
        self.win.title(title)
        self.win.protocol("WM_DELETE_WINDOW", self._on_close)
        self.win.resizable(True, True)

        # UI
        top = ttk.Frame(self.win, padding=8)
        top.pack(fill="both", expand=True)

        self.video_lbl = tk.Label(top, bg="black")
        self.video_lbl.pack(fill="both", expand=True)

        controls = ttk.Frame(top)
        controls.pack(fill="x", pady=(6, 0))

        self.status_var = tk.StringVar(value="Connecting…")
        ttk.Label(controls, textvariable=self.status_var).pack(side="left")

        self.pause = False
        self.btn_toggle = ttk.Button(controls, text="Pause", command=self._toggle_pause)
        self.btn_toggle.pack(side="right", padx=(6, 0))
        ttk.Button(controls, text="Close", command=self._on_close).pack(side="right")

        # Stream state
        self.cap = None
        self.running = True
        self.frame = None
        self.frame_lock = threading.Lock()

        # Start worker and UI refresher
        self.worker = threading.Thread(target=self._reader_worker, daemon=True)
        self.worker.start()
        self._refresh_ui()

    def _toggle_pause(self):
        self.pause = not self.pause
        self.btn_toggle.config(text="Resume" if self.pause else "Pause")

    def _reader_worker(self):
        try:
            self.cap = cv2.VideoCapture(self.url, cv2.CAP_FFMPEG)
            # second attempt without explicit backend (helps on some builds)
            if not self.cap.isOpened():
                self.cap.release()
                self.cap = cv2.VideoCapture(self.url)
        except Exception as e:
            self.status_var.set(f"OpenCV error: {e}")
            return

        if not self.cap or not self.cap.isOpened():
            self.status_var.set("Failed to open stream (RTSP/MJPEG only).")
            return

        self.status_var.set("Connected")
        last_ok = time.time()

        while self.running:
            if self.pause:
                time.sleep(0.05)
                continue

            ok, frame = self.cap.read()
            if not ok:
                # transient hiccup: allow a brief grace period
                if time.time() - last_ok > 2.0:
                    self.status_var.set("Reconnecting…")
                time.sleep(0.05)
                continue

            last_ok = time.time()
            self.status_var.set("Streaming")

            # Convert BGR->RGB and store latest frame
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            with self.frame_lock:
                self.frame = frame

        try:
            if self.cap:
                self.cap.release()
        except Exception:
            pass

    def _refresh_ui(self):
        if not self.running:
            return

        frame = None
        with self.frame_lock:
            if self.frame is not None:
                frame = self.frame.copy()

        if frame is not None:
            h, w, _ = frame.shape

            # Fit to window while preserving aspect ratio
            win_w = max(1, self.win.winfo_width() - 20)
            win_h = max(1, self.win.winfo_height() - 80)
            target_w = min(self.max_w, win_w)
            target_h = min(self.max_h, win_h)

            scale = min(target_w / w, target_h / h)
            if scale <= 0:
                scale = 1.0
            new_w = max(1, int(w * scale))
            new_h = max(1, int(h * scale))

            if new_w != w or new_h != h:
                frame_resized = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_AREA)
            else:
                frame_resized = frame

            img = Image.fromarray(frame_resized)
            imgtk = ImageTk.PhotoImage(image=img)
            # prevent GC
            self.video_lbl.imgtk = imgtk
            self.video_lbl.configure(image=imgtk)

        # Aim ~30 fps UI refresh
        self.win.after(33, self._refresh_ui)

    def _on_close(self):
        self.running = False
        # let worker exit cleanly
        self.win.after(100, self.win.destroy)

def open_stream_popup(parent, url, title="Live Stream"):
    """Convenience function to open the popup."""
    return StreamPopup(parent, url, title=title)

if __name__ == "__main__":
    # Minimal demo: run this file directly to test a URL
    import os
    STREAM_URL_HTTP = os.environ.get("STREAM_URL", "http://10.12.10.252:8889/cam1")
    root = tk.Tk()
    root.title("Popup Stream Demo")
    ttk.Button(root, text="Open Stream", command=lambda: open_stream_popup(root, STREAM_URL_HTTP)).pack(padx=12, pady=12)
    root.mainloop()
