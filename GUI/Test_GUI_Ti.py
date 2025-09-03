import tkinter as tk
from tkinter import ttk
from PIL import Image, ImageTk
import cv2
import csv
import os

class QualityFeedbackGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Triple I - Quality Feedback Camera")
        self.root.geometry("600x400")
        self.root.configure(bg="#f0f0f0")

        # Video feed frame
        self.video_label = tk.Label(root)
        self.video_label.pack(pady=10)

        # Result display
        self.result_label = tk.Label(root, text="Result: Pending", font=("Arial", 14), bg="#ffffff", width=30, relief="solid")
        self.result_label.pack(pady=5)

        # Label input
        label_frame = tk.Frame(root, bg="#f0f0f0")
        label_frame.pack(pady=5)
        tk.Label(label_frame, text="Label:", font=("Arial", 12), bg="#f0f0f0").pack(side=tk.LEFT)
        self.label_entry = tk.Entry(label_frame, width=30)
        self.label_entry.pack(side=tk.LEFT, padx=5)

        # Buttons
        button_frame = tk.Frame(root, bg="#f0f0f0")
        button_frame.pack(pady=10)
        tk.Button(button_frame, text="Run Model", width=15, command=self.run_model).grid(row=0, column=0, padx=5)
        tk.Button(button_frame, text="Collect Data", width=15, command=self.collect_data).grid(row=0, column=1, padx=5)
        tk.Button(root, text="Quit", width=40, command=self.quit_app).pack(pady=10)

        # Setup camera
        self.cap = cv2.VideoCapture(0)
        self.update_frame()

        # CSV file init
        self.csv_file = "collected_data.csv"
        if not os.path.exists(self.csv_file):
            with open(self.csv_file, mode='w', newline='') as file:
                writer = csv.writer(file)
                writer.writerow(["Frame", "Label"])

    def update_frame(self):
        ret, frame = self.cap.read()
        if ret:
            # Resize and convert frame for display
            cv2image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(cv2image).resize((480, 270))
            imgtk = ImageTk.PhotoImage(image=img)
            self.video_label.imgtk = imgtk
            self.video_label.configure(image=imgtk)
        self.root.after(10, self.update_frame)

    def run_model(self):
        # Simulated result (replace with real model logic)
        self.result_label.config(text="Result: Good", fg="green")

    def collect_data(self):
        label = self.label_entry.get()
        ret, frame = self.cap.read()
        if ret and label:
            filename = f"frame_{label}_{cv2.getTickCount()}.jpg"
            cv2.imwrite(filename, frame)
            with open(self.csv_file, mode='a', newline='') as file:
                writer = csv.writer(file)
                writer.writerow([filename, label])
            print(f"Saved frame as '{filename}' with label '{label}'")

    def quit_app(self):
        self.cap.release()
        self.root.destroy()

# Run GUI
if __name__ == "__main__":
    root = tk.Tk()
    app = QualityFeedbackGUI(root)
    root.mainloop()
