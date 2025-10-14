import tkinter as tk
from tkinter import ttk, messagebox
from PIL import Image, ImageTk
import threading
import cv2
import time

class StreamPopup:
    def __init__(self, parent, url, title="Live Stream", max_width=640, max_height=640, ui_fps=25):
        self.parent = parent
        self.url = url
        self.max_w = max_width
        self.max_h = max_height
        self.ui_period_ms = max(10, int(1000 / ui_fps))

        # state
        self.cap = None
        self.running = True
        self.paused = False
        self.frame = None
        self.frame_lock = threading.Lock()
        self.last_ts = 0.0
        self._last_size = (0, 0)  # to avoid repeated resizes same size

        # window
        self.win = tk.Toplevel(parent)
        self.win.title(title)
        self.win.protocol("WM_DELETE_WINDOW", self._on_close)
        self.win.resizable(True, True)
        self.win.bind("<Escape>", lambda e: self._on_close())
        # optional: center a reasonable size
        try:
            self.win.geometry("800x600+120+80")
        except Exception:
            pass

        top = ttk.Frame(self.win, padding=8)
        top.pack(fill="both", expand=True)

        self.video_lbl = tk.Label(top, bg="black")
        self.video_lbl.pack(fill="both", expand=True)

        controls = ttk.Frame(top)
        controls.pack(fill="x", pady=(6, 0))

        self.status_var = tk.StringVar(value="Connecting…")
        ttk.Label(controls, textvariable=self.status_var).pack(side="left")

        self.fps_var = tk.StringVar(value="")
        ttk.Label(controls, textvariable=self.fps_var).pack(side="left", padx=(10, 0))

        self.btn_toggle = ttk.Button(controls, text="Pause", command=self._toggle_pause)
        self.btn_toggle.pack(side="right", padx=(6, 0))
        ttk.Button(controls, text="Close", command=self._on_close).pack(side="right")

        # open capture in a worker with timeout (prevents UI blocking)
        self.open_done = threading.Event()
        threading.Thread(target=self._open_capture_worker, daemon=True).start()

        # start the reader and UI refresher
        self.reader_thread = threading.Thread(target=self._reader_worker, daemon=True)
        self.reader_thread.start()
        self._refresh_ui()

    # ---------- open with fallback backends ----------
    def _open_capture_worker(self):
        """Try to open the stream with FFmpeg → GStreamer (RTSP pipeline) → default."""
        def try_open(obj):
            return obj is not None and obj.isOpened()

        # 1) Prefer FFmpeg (best path for RTSP + HTTP MJPEG)
        cap = cv2.VideoCapture(self.url, cv2.CAP_FFMPEG)
        if not try_open(cap):
            if cap: cap.release()
            cap = None

        # 2) If RTSP and GStreamer is available, try explicit pipeline
        if cap is None and self.url.lower().startswith("rtsp"):
            gst = (
                f"rtspsrc location={self.url} latency=0 ! "
                "rtph264depay ! h264parse ! avdec_h264 ! "
                "videoconvert ! appsink drop=true sync=false max-buffers=2"
            )
            cap = cv2.VideoCapture(gst, cv2.CAP_GSTREAMER)
            if not try_open(cap):
                if cap: cap.release()
                cap = None

        # 3) Plain OpenCV fallback (sometimes works for simple MJPEG)
        if cap is None:
            cap = cv2.VideoCapture(self.url)
            if not try_open(cap):
                if cap: cap.release()
                cap = None

        self.cap = cap

        if self.cap:
            # Low-latency preferences (best-effort; some backends ignore)
            try: self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 2)
            except Exception: pass
            try: self.cap.set(cv2.CAP_PROP_FPS, 30)
            except Exception: pass
            self.status_var.set("Connected")
        else:
            self.status_var.set("Failed to open stream (install ffmpeg and gstreamer plugins).")

        self.open_done.set()

    # ---------- reader loop ----------
    def _reader_worker(self):
        # Wait briefly for open to complete; stop early if window closed
        if not self.open_done.wait(timeout=6.0) or not self.running:
            return

        if not self.cap:
            return

        last_ok = time.time()
        frame_count = 0
        fps_window_start = time.time()

        while self.running:
            if self.paused:
                time.sleep(0.06)
                continue

            ok, frame = self.cap.read()
            if not ok:
                # transient hiccup → brief backoff
                if time.time() - last_ok > 2.0:
                    self.status_var.set("Reconnecting…")
                time.sleep(0.04)
                continue

            last_ok = time.time()
            self.status_var.set("Streaming")

            # Convert BGR->RGB
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            with self.frame_lock:
                self.frame = frame

            # simple FPS estimate (for info only)
            frame_count += 1
            if time.time() - fps_window_start >= 1.0:
                self.fps_var.set(f"{frame_count} fps")
                frame_count = 0
                fps_window_start = time.time()

        try:
            if self.cap:
                self.cap.release()
        except Exception:
            pass

    # ---------- UI loop ----------
    def _refresh_ui(self):
        if not self.running or not self.win.winfo_exists():
            return

        frame = None
        with self.frame_lock:
            if self.frame is not None:
                frame = self.frame

        if frame is not None:
            h, w, _ = frame.shape

            # Fit to current window while preserving AR
            win_w = max(1, self.win.winfo_width() - 20)
            win_h = max(1, self.win.winfo_height() - 80)
            target_w = min(self.max_w, win_w)
            target_h = min(self.max_h, win_h)

            scale = min(target_w / w, target_h / h)
            new_w = max(1, int(w * scale))
            new_h = max(1, int(h * scale))

            # Avoid re-resizing to the same size (saves CPU)
            if (new_w, new_h) != self._last_size:
                frame_resized = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_AREA)
                self._last_size = (new_w, new_h)
            else:
                frame_resized = frame

            try:
                img = Image.fromarray(frame_resized)
                imgtk = ImageTk.PhotoImage(image=img)
                # prevent GC
                self.video_lbl.imgtk = imgtk
                self.video_lbl.configure(image=imgtk)
            except Exception:
                # can happen during teardown race; ignore
                pass

        self.win.after(self.ui_period_ms, self._refresh_ui)

    # ---------- controls ----------
    def _toggle_pause(self):
        self.paused = not self.paused
        self.btn_toggle.config(text="Resume" if self.paused else "Pause")
        self.status_var.set("Paused" if self.paused else "Streaming")

    def _on_close(self):
        if not self.running:
            return
        self.running = False
        # give reader time to exit cleanly before destroying UI
        def _destroy():
            try:
                if self.win.winfo_exists():
                    self.win.destroy()
            except Exception:
                pass
        self.win.after(120, _destroy)

def open_stream_popup(parent, url, title="Live Stream"):
    return StreamPopup(parent, url, title=title)