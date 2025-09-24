from ultralytics import YOLO
import os
import cv2
import time
from openvino.runtime import Core

# Disable CUDA to prevent PyTorch fallback
os.environ["CUDA_VISIBLE_DEVICES"] = ""

# Verify Intel GPU availability
core = Core()
devices = core.available_devices
print(f"Available devices: {devices}")
if "GPU" not in devices:
    print("Intel GPU not detected. Available devices: {}".format(devices))
    exit(1)

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
        model.export(format="openvino", device="GPU")
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


#Select Processor: CPU, GPU, NPU
processor = "GPU"

# Benchmark function with timing
def testProcessor(iterations, model, device, image_path):
    print(f"Benchmarking on {device} for {iterations} iterations...")
    
    # Warm-up run
    try:
        model(image_path, device=device)
        print("Warm-up run completed.")
    except Exception as e:
        print(f"Warm-up run failed: {e}")
        return

    # Benchmark
    total_time = 0
    for i in range(iterations):
        start_time = time.time()
        try:
            results = model(image_path, device=device, task="detect")
            end_time = time.time()
            inference_time = end_time - start_time
            total_time += inference_time
            print(f"Iteration {i+1}/{iterations}: {inference_time:.4f} seconds")
        except Exception as e:
            print(f"Inference failed on iteration {i+1}: {e}")
            return
    
    avg_time = total_time / iterations
    print(f"Average inference time: {avg_time:.4f} seconds")

# Run benchmark
testProcessor(5, ov_model, processor, image_path)

'''

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

'''