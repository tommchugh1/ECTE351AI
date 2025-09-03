import tkinter as tk
from PIL import Image, ImageTk
import cv2
import os

class FullViewVideoRecorderGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Full View Camera Recorder")
        self.root.configure(bg="#000000")
        self.root.state('zoomed')  # Fullscreen for Windows

        # Tracking sidebar visibility
        self.sidebar_visible = False
        self.sidebar_width = 240

        # Camera preview fills most of the window
        self.video_label = tk.Label(root, bg="#000000")
        self.video_label.place(x=0, y=0, width=self.root.winfo_screenwidth(), height=self.root.winfo_screenheight())

        # Sidebar frame initially hidden, placed off-screen right
        self.sidebar = tk.Frame(root, bg="#202020", width=self.sidebar_width, height=self.root.winfo_screenheight())
        self.sidebar.place(x=self.root.winfo_screenwidth(), y=0, height=self.root.winfo_screenheight())

        # Start button - top of sidebar
        self.start_button = tk.Button(
            self.sidebar,
            text="▶ START LIVE FEED",
            font=("Arial", 18, "bold"),
            bg="#228822",
            fg="white",
            width=16,
            height=2,
            command=self.start_feed
        )
        self.start_button.pack(pady=(60, 24))

        # Stop button - below start button
        self.stop_button = tk.Button(
            self.sidebar,
            text="■ STOP & SAVE",
            font=("Arial", 18, "bold"),
            bg="#cc2222",
            fg="white",
            width=16,
            height=2,
            command=self.stop_feed,
            state=tk.DISABLED
        )
        self.stop_button.pack(pady=(0, 24))

        # Camera and video
        self.cap = None
        self.out = None
        self.is_recording = False
        self.video_folder = r"C:\Users\joypa\Pictures\Live Video"
        os.makedirs(self.video_folder, exist_ok=True)
        self.out_path = os.path.join(self.video_folder, "live_recording.mp4")

        # Bind mouse motion on root to detect cursor position
        self.root.bind('<Motion>', self.track_mouse)

        # Periodically adjust video label size to not overlap sidebar
        self.adjust_video_label_size()

    def track_mouse(self, event):
        screen_width = self.root.winfo_screenwidth()
        cursor_x = event.x_root
        # Show sidebar if cursor near right edge (within 50 px)
        if cursor_x >= screen_width - 50 and not self.sidebar_visible:
            self.show_sidebar()
        # Hide sidebar if cursor away from sidebar area
        elif cursor_x < screen_width - self.sidebar_width - 50 and self.sidebar_visible:
            self.hide_sidebar()

    def show_sidebar(self):
        self.sidebar.place(x=self.root.winfo_screenwidth() - self.sidebar_width, y=0, height=self.root.winfo_screenheight())
        self.sidebar_visible = True
        self.adjust_video_label_size()

    def hide_sidebar(self):
        self.sidebar.place(x=self.root.winfo_screenwidth(), y=0, height=self.root.winfo_screenheight())
        self.sidebar_visible = False
        self.adjust_video_label_size()

    def adjust_video_label_size(self):
        # Set the video label to occupy all width minus sidebar if visible
        if self.sidebar_visible:
            self.video_label.place(x=0, y=0, width=self.root.winfo_width() - self.sidebar_width, height=self.root.winfo_height())
        else:
            self.video_label.place(x=0, y=0, width=self.root.winfo_width(), height=self.root.winfo_height())
        self.root.after(200, self.adjust_video_label_size)

    def start_feed(self):
        if not self.is_recording:
            self.cap = cv2.VideoCapture(0)
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            fps = 20.0
            frame_width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            frame_height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            self.out = cv2.VideoWriter(self.out_path, fourcc, fps, (frame_width, frame_height))
            self.is_recording = True
            self.start_button.config(state=tk.DISABLED)
            self.stop_button.config(state=tk.NORMAL)
            self.update_frame()

    def update_frame(self):
        if self.cap and self.is_recording:
            ret, frame = self.cap.read()
            if ret:
                self.out.write(frame)
                cv2image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                img = Image.fromarray(cv2image)
                w = self.video_label.winfo_width()
                h = self.video_label.winfo_height()
                if w > 0 and h > 0:
                    img = img.resize((w, h))
                imgtk = ImageTk.PhotoImage(image=img)
                self.video_label.imgtk = imgtk
                self.video_label.configure(image=imgtk)
            self.root.after(10, self.update_frame)

    def stop_feed(self):
        if self.cap and self.is_recording:
            self.is_recording = False
            self.cap.release()
            self.out.release()
            self.start_button.config(state=tk.NORMAL)
            self.stop_button.config(state=tk.DISABLED)

    def on_close(self):
        if self.cap:
            self.cap.release()
        if self.out:
            self.out.release()
        self.root.destroy()

if __name__ == "__main__":
    root = tk.Tk()
    app = FullViewVideoRecorderGUI(root)
    root.protocol("WM_DELETE_WINDOW", app.on_close)
    root.mainloop()
