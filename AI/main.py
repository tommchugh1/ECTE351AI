import cv2
from ultralytics import YOLO
import os

#Find local directory of file
header = os.path.dirname(os.path.realpath(__file__))
frame = header + "\\example.webp"

# Load YOLOv8
model = YOLO(header + "\\yolov8n.pt")
model.export(format="openvino")
ov_model = YOLO(header + "\\yolov8n_openvino_model/")

# Run YOLO model on the file using CPU
'''
results = ov_model(frame, device="intel:cpu")
results = ov_model(frame, device="intel:cpu")
results = ov_model(frame, device="intel:cpu")
results = ov_model(frame, device="intel:cpu")
results = ov_model(frame, device="intel:cpu")
'''

# Run YOLO model on the file using GPU
#'''
results = ov_model(frame, device="intel:gpu")
results = ov_model(frame, device="intel:gpu")
results = ov_model(frame, device="intel:gpu")
results = ov_model(frame, device="intel:gpu")
results = ov_model(frame, device="intel:gpu")
#'''


# Run YOLO model on the file using NPU
'''
results = ov_model(frame, device="intel:npu")
results = ov_model(frame, device="intel:npu")
results = ov_model(frame, device="intel:npu")
results = ov_model(frame, device="intel:npu")
results = ov_model(frame, device="intel:npu")
'''

