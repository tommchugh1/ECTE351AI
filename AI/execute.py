from ultralytics import YOLO
import os
import cv2

#Find local directory of example file
header = os.path.dirname(os.path.realpath(__file__))
frame = header + "\\" + "example.webp"

# Load YOLOv8 and export to format compatible with Intel hardware
model = YOLO(header + "\\MODELS\\" + "yolov8n.pt")
model.export(format="openvino")

#Load OpenVino model
ov_model = YOLO(header + "\\MODELS\\" + "yolov8n_openvino_model")

#Select Processor: cpu, gpu, npu
processor = "npu"


#BENCHMARK
def testProcessor(iterations):
    for i in range(iterations):
        results = ov_model(frame, device="intel:"+processor)

#testProcessor(10)


#Provide video stream address
cap = cv2.VideoCapture('http://0.0.0.0:5000/video_feed')

if not cap.isOpened():
    print('Error: Could not access video stream')
    exit()


while True:
    ret, frame = cap.read()
    if not ret:
        print('Error: Failed to grab frame')
        break

    cv2.imshow('Processed Frame', frame)

    results = ov_model(frame, device="intel:"+processor)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break


cap.release()
cv2.destroyAllWindows()


