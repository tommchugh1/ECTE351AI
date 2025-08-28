import cv2
from ultralytics import YOLO
import os
import torch


# Load YOLOv8
model = YOLO("yolov8n.pt")
desktop = os.path.expanduser("~/Desktop")

# Capture a frame from the camera
frame = desktop + "\Yolo Demo/ECTE351AI/AI/example.webp"

    
# Run YOLO model on the captured frame and store the results
results = model.predict(source=frame)