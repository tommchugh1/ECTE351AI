import tkinter as tk
from PIL import Image, ImageTk
import cv2
import numpy as np

class VideoFilterApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Live Filtered Video")

        # === FILTER TOGGLES ===
        self.enable_chroma_key = True        # Remove green background
        self.enable_grayscale = True         # Convert to grayscale
        self.enable_binarization = True      # Convert to black & white
        self.enable_rotation = True          # Rotate image
        self.enable_zoom = True              # Scale image

        # === FILTER SETTINGS ===
        self.rotate_angle = 45               # Degrees
        self.zoom_scale = 1.5                # 1.0 = normal, >1 = zoom in
        self.threshold_val = 100             # For binarization (0–255)

        # === UI SETUP ===
        self.video_label = tk.Label(root)
        self.video_label.pack(fill=tk.BOTH, expand=True)

        self.cap = cv2.VideoCapture(0)

        # Start frame update loop
        self.update_frame()

        # Graceful exit
        self.root.protocol("WM_DELETE_WINDOW", self.on_close)

    def update_frame(self):
        ret, frame = self.cap.read()
        if not ret:
            return

        # ================================
        # 1. Chroma Key (Green Screen Removal)
        # ================================
        if self.enable_chroma_key:
            hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
            # Define green range in HSV
            green_mask = cv2.inRange(hsv, (35, 100, 100), (85, 255, 255))
            inv_mask = cv2.bitwise_not(green_mask)
            frame = cv2.bitwise_and(frame, frame, mask=inv_mask)

        # ================================
        # 2. Grayscale Conversion
        # ================================
        if self.enable_grayscale:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # ================================
        # 3. Binarization (Thresholding)
        # ================================
        if self.enable_binarization:
            if len(frame.shape) == 3:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            _, frame = cv2.threshold(frame, self.threshold_val, 255, cv2.THRESH_BINARY)

        # ================================
        # 4. Rotation
        # ================================
        if self.enable_rotation:
            h, w = frame.shape[:2]
            center = (w // 2, h // 2)
            M = cv2.getRotationMatrix2D(center, self.rotate_angle, 1.0)
            frame = cv2.warpAffine(frame, M, (w, h))

        # ================================
        # 5. Zoom (Scaling)
        # ================================
        if self.enable_zoom:
            frame = cv2.resize(frame, None, fx=self.zoom_scale, fy=self.zoom_scale)

        # Resize to fit the window
        screen_w = self.root.winfo_width()
        screen_h = self.root.winfo_height()

        frame = cv2.resize(frame, (screen_w, screen_h))

        # ================================
        # Convert to ImageTk and update label
        # ================================
        if len(frame.shape) == 2:
            img = Image.fromarray(frame)
        else:
            img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

        imgtk = ImageTk.PhotoImage(image=img)
        self.video_label.imgtk = imgtk
        self.video_label.config(image=imgtk)

        self.root.after(10, self.update_frame)

    def on_close(self):
        if self.cap:
            self.cap.release()
        self.root.destroy()

if __name__ == "__main__":
    root = tk.Tk()
    root.geometry("900x700")
    app = VideoFilterApp(root)
    root.mainloop()
