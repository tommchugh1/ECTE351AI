import tkinter as tk
from tkinter import ttk
from PIL import Image, ImageTk
import cv2
import numpy as np
import threading
import time

class FilteredStreamPopup:
    def __init__(self, parent, url, title="Filtered Stream"):
        self.parent = parent
        self.url = url
        self.running = True
        self.paused = False
        self.frame = None
        self.frame_lock = threading.Lock()

        # Filters: enable or disable
        self.enable_chroma_key = True
        self.enable_grayscale = True
        self.enable_binarization = True
        self.enable_rotation = True
        self.enable_zoom = True

        # Filter settings
        self.rotate_angle = 0
        self.zoom_scale = 1
        self.threshold_val = 200

        # Setup popup window
        self.win = tk.Toplevel(parent)
        self.win.title(title)
        self.win.geometry("640x640")
        self.win.protocol("WM_DELETE_WINDOW", self._on_close)

        # Layout
        container = ttk.Frame(self.win, padding=10)
        container.pack(fill="both", expand=True)

        self.video_lbl = tk.Label(container, bg="black")
        self.video_lbl.pack(fill="both", expand=True)

        bottom = ttk.Frame(container)
        bottom.pack(fill="x", pady=(6, 0))

        self.status_var = tk.StringVar(value="Connecting…")
        ttk.Label(bottom, textvariable=self.status_var).pack(side="left")

        self.fps_var = tk.StringVar(value="")
        ttk.Label(bottom, textvariable=self.fps_var).pack(side="left", padx=(10, 0))

        self.btn_toggle = ttk.Button(bottom, text="Pause", command=self._toggle_pause)
        self.btn_toggle.pack(side="right", padx=(6, 0))
        ttk.Button(bottom, text="Close", command=self._on_close).pack(side="right")

        # Capture + UI
        self.cap = cv2.VideoCapture(self.url)
        if not self.cap.isOpened():
            self.status_var.set("Failed to open stream.")
            self.running = False
            return

        self.status_var.set("Connected")

        self.reader = threading.Thread(target=self._reader_loop, daemon=True)
        self.reader.start()
        self._update_ui()

    def _toggle_pause(self):
        self.paused = not self.paused
        self.btn_toggle.config(text="Resume" if self.paused else "Pause")
        self.status_var.set("Paused" if self.paused else "Streaming")

    def _reader_loop(self):
        last_ok = time.time()
        frame_count = 0
        fps_start = time.time()

        while self.running:
            if self.paused:
                time.sleep(0.05)
                continue

            ok, frame = self.cap.read()
            if not ok:
                if time.time() - last_ok > 2:
                    self.status_var.set("Reconnecting…")
                time.sleep(0.05)
                continue

            last_ok = time.time()
            self.status_var.set("Streaming")

            frame = self.apply_filters(frame)

            with self.frame_lock:
                self.frame = frame

            frame_count += 1
            if time.time() - fps_start > 1.0:
                self.fps_var.set(f"{frame_count} fps")
                frame_count = 0
                fps_start = time.time()

        if self.cap:
            self.cap.release()

    def _update_ui(self):
        if not self.running or not self.win.winfo_exists():
            return

        frame = None
        with self.frame_lock:
            if self.frame is not None:
                frame = self.frame.copy()

        if frame is not None:
            win_w = max(1, self.win.winfo_width() - 20)
            win_h = max(1, self.win.winfo_height() - 80)
            h, w = frame.shape[:2]
            scale = min(win_w / w, win_h / h)
            frame = cv2.resize(frame, (int(w * scale), int(h * scale)))

            img = Image.fromarray(frame)
            imgtk = ImageTk.PhotoImage(img)
            self.video_lbl.imgtk = imgtk
            self.video_lbl.config(image=imgtk)

        self.win.after(30, self._update_ui)

    def apply_filters(self, frame):
        try:
            if self.enable_chroma_key:
                hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
                green_mask = cv2.inRange(hsv, (35, 100, 100), (85, 255, 255))
                inv_mask = cv2.bitwise_not(green_mask)
                frame = cv2.bitwise_and(frame, frame, mask=inv_mask)

            if self.enable_grayscale:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            if self.enable_binarization:
                if len(frame.shape) == 3:
                    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                _, frame = cv2.threshold(frame, self.threshold_val, 255, cv2.THRESH_BINARY)

            if self.enable_rotation:
                h, w = frame.shape[:2]
                M = cv2.getRotationMatrix2D((w // 2, h // 2), self.rotate_angle, 1.0)
                frame = cv2.warpAffine(frame, M, (w, h))

            if self.enable_zoom:
                frame = cv2.resize(frame, None, fx=self.zoom_scale, fy=self.zoom_scale)

            if len(frame.shape) == 2:
                frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2RGB)
            else:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            return frame

        except Exception as e:
            print(f"[ERROR] Filter failed: {e}")
            return np.zeros((240, 320, 3), dtype=np.uint8)

    def _on_close(self):
        self.running = False
        self.win.after(100, self.win.destroy)


# 🔄 Helper function
def open_filtered_popup(parent, url, title="Filtered Stream"):
    return FilteredStreamPopup(parent, url, title=title)


# 🔧 Standalone test
if __name__ == "__main__":
    root = tk.Tk()
    root.title("Test Filtered Stream")
    test_url = "rtsp://10.12.10.242:8554/cam1"
    ttk.Button(root, text="Open Filtered View", command=lambda: open_filtered_popup(root, test_url)).pack(padx=12, pady=12)
    root.mainloop()
