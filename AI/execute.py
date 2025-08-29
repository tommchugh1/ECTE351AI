from ultralytics import YOLO
import os

#Find local directory of file
header = os.path.dirname(os.path.realpath(__file__))
frame = header + "\\" + "example.webp"

# Load YOLOv8 and export to format compatible with Intel hardware
model = YOLO(header + "\\MODELS\\" + "yolov8n.pt")
model.export(format="openvino")

#Load OpenVino model
ov_model = YOLO(header + "\\MODELS\\" + "yolov8n_openvino_model")

# Run model on the file using CPU
'''
results = ov_model(frame, device="intel:cpu")
results = ov_model(frame, device="intel:cpu")
results = ov_model(frame, device="intel:cpu")
results = ov_model(frame, device="intel:cpu")
results = ov_model(frame, device="intel:cpu")
'''

# Run model on the file using GPU
#'''
results = ov_model(frame, device="intel:gpu")
results = ov_model(frame, device="intel:gpu")
results = ov_model(frame, device="intel:gpu")
results = ov_model(frame, device="intel:gpu")
results = ov_model(frame, device="intel:gpu")
#'''


# Run model on the file using NPU
'''
results = ov_model(frame, device="intel:npu")
results = ov_model(frame, device="intel:npu")
results = ov_model(frame, device="intel:npu")
results = ov_model(frame, device="intel:npu")
results = ov_model(frame, device="intel:npu")
'''

