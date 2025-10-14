from ultralytics import YOLO
import os
import cv2
import time
from openvino.runtime import Core

core = Core()
devices = core.available_devices or []   # None-safe
device = "CPU"
if any("GPU" in d.upper() for d in devices):
    device = "GPU"

print(f"OpenVINO devices: {devices} — using: {device}")

# Auto-select best available OpenVINO device
if any("GPU" in d for d in devices):
    device = "GPU"
elif any("NPU" in d for d in devices):
    device = "NPU"
elif "CPU" in devices:
    device = "CPU"
else:
    print("No compatible OpenVINO devices found.")
    exit(1)

os.environ["OPENVINO_DEFAULT_DEVICE"] = device
print(f"Selected OpenVINO device: {device}")

#Find local directory of example file
header = os.path.dirname(os.path.realpath(__file__))

# Define paths
image_path = os.path.join(header, "RUNS", "train", "bolt_training", "image.jpg")
model_path = os.path.join(header, "RUNS", "train", "bolt_training", "weights", "best.pt")
openvino_model_path = os.path.join(header, "RUNS", "train", "bolt_training", "weights", "best_openvino_model")

frame = header + "\\RUNS\\train\\bolt_training\\image.jpg"

# Verify file paths
if not os.path.exists(image_path):
    print(f"Image file not found: {image_path}")
    exit(1)
if not os.path.exists(model_path):
    print(f"Model file not found: {model_path}")
    exit(1)

# Load YOLOv8 model
try:
    model = YOLO(model_path, task="detect")
    print("YOLOv8 model loaded successfully.")
except Exception as e:
    print(f"Failed to load YOLOv8 model: {e}")
    exit(1)

# Export to OpenVINO format if not already exported
try:
    if not os.path.exists(openvino_model_path):
        print("Exporting model to OpenVINO format...")
        model.export(format="openvino")
        print("Model exported to OpenVINO format.")
    else:
        print("OpenVINO model already exists, skipping export.")
except Exception as e:
    print(f"Failed to export model to OpenVINO: {e}")
    exit(1)

# Load OpenVINO model
try:
    ov_model = YOLO(openvino_model_path, task="detect")
    print("OpenVINO model loaded successfully.")
except Exception as e:
    print(f"Failed to load OpenVINO model: {e}")
    exit(1)

def testProcessor(iterations, model, image_path):
    print(f"Benchmarking on OpenVINO device for {iterations} iterations...")

    # Warm-up run
    try:
        model.predict(source=image_path, imgsz=640)
        print("Warm-up run completed.")
    except Exception as e:
        print(f"Warm-up run failed: {e}")
        return

    # Benchmark
    total_time = 0
    results = None
    for i in range(iterations):
        start_time = time.time()
        try:
            results = model.predict(source=image_path, imgsz=640, conf=0.1)
            end_time = time.time()
            inference_time = end_time - start_time
            total_time += inference_time
            print(f"Iteration {i+1}/{iterations}: {inference_time:.4f} seconds")
        except Exception as e:
            print(f"Inference failed on iteration {i+1}: {e}")
            return

    avg_time = total_time / iterations
    print(f"Average inference time: {avg_time:.4f} seconds")
    print(f"Selected OpenVINO device: {device}")

    # Load original image
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"Failed to load image at {image_path}")
    class_names = model.names  # get class names from model

    # Draw all boxes manually on the image
    results = model(image)  
    if results is None:
        results = []        # safe default

    for r in results:
        boxes = r.boxes
        for box in boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
            conf = float(box.conf[0])
            cls_id = int(box.cls[0])
            label = class_names[cls_id] if class_names else str(cls_id)
            cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(image, f"{label} {conf:.2f}", (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

            print(f"Class: {cls_id}, Confidence: {conf:.2f}, Box: ({x1}, {y1}, {x2}, {y2})")

    # Save and show annotated image
    output_path = os.path.join(header, "RUNS", "train", "bolt_training", "inference_image.jpg")
    cv2.imwrite(output_path, image)
    print(f"Saved annotated image to {output_path}")

    cv2.imshow("Inference Result", image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()



# Run benchmark
testProcessor(1, ov_model, image_path)

'''

#Provide video stream address
#cap = cv2.VideoCapture('http://0.0.0.0:5000/video_feed')
cap = cv2.VideoCapture('http://10.12.165.126:8889/cam1/')

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

'''