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


#BENCHMARK
def testProcessor(processor, iterations):
    for i in range(iterations):
        results = ov_model(frame, device="intel:"+processor)


#PROCESSORS: cpu, gpu, npu
testProcessor("gpu", 10)


