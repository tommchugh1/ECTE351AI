import os
import cv2
import numpy as np
from openvino.runtime import Core
import matplotlib.pyplot as plt
import matplotlib.patches as patches

# --- CONFIGURATION ---
input_folder = r"C:\Users\Group8\Documents\ModelTrainingData\OutputPhotosAndLabel\250916 1600"  # Folder containing .jpg or .png files
model_folder = r"C:\Users\Group8\Desktop\Yolo Demo\ECTE351AI\AI\RUNS\train\bolt_training\yolov8n_openvino_model"  # Folder containing yolov8n.xml
output_folder = r"C:\Users\Group8\Documents\ModelTrainingData\OutputPhotosAndLabel\250916 1600\labels\InferenceVisualisation"  # Output folder to save detection images
image_size = (640, 640)  # YOLOv8 default input size
device = "GPU"  # or "CPU", "MYRIAD"

class_colors = {
    0: "red", 1: "blue", 2: "green", 3: "yellow", 4: "magenta", 5: "cyan"
}

# --- SETUP OUTPUT FOLDER ---
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

# --- INITIALIZE OPENVINO ---
ie = Core()
model_path = os.path.join(model_folder, "yolov8n.xml")

try:
    model = ie.read_model(model=model_path)
    compiled_model = ie.compile_model(model=model, device_name=device)
except Exception as e:
    raise RuntimeError(f"Failed to load or compile model: {e}")

input_layer = compiled_model.input(0)
output_layer = compiled_model.output(0)

# --- PROCESS EACH IMAGE IN FOLDER ---
image_files = [f for f in os.listdir(input_folder) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]

if not image_files:
    raise FileNotFoundError(f"No image files found in {input_folder}")

for idx, image_name in enumerate(image_files):
    image_path = os.path.join(input_folder, image_name)
    image = cv2.imread(image_path)

    if image is None:
        print(f"❌ Could not read image: {image_path}")
        continue

    original_height, original_width = image.shape[:2]

    # --- PREPROCESSING ---
    resized_image = cv2.resize(image, image_size)
    input_data = resized_image.astype(np.float32) / 255.0
    input_data = np.transpose(input_data, (2, 0, 1))
    input_data = np.expand_dims(input_data, axis=0)

    # --- INFERENCE ---
    try:
        results = compiled_model.infer_new_request({input_layer: input_data})
        output = results[output_layer]
    except Exception as e:
        print(f"⚠️ Inference failed for {image_name}: {e}")
        continue

    # --- POSTPROCESSING ---
    def postprocess_output(output, conf_threshold=0.4, iou_threshold=0.5):
        boxes, scores, classes = [], [], []
        for detection in output[0]:
            confidence = detection[4]
            if confidence > conf_threshold:
                x, y, w, h = detection[:4]
                class_id = int(np.argmax(detection[5:]))
                score = detection[5 + class_id]
                if score > conf_threshold:
                    boxes.append([x, y, w, h])
                    scores.append(float(confidence))
                    classes.append(class_id)

        indices = cv2.dnn.NMSBoxes(boxes, scores, conf_threshold, iou_threshold)
        return [(boxes[i], scores[i], classes[i]) for i in indices.flatten()]

    detections = postprocess_output(output)

    # --- VISUALIZE WITH MATPLOTLIB ---
    fig, ax = plt.subplots(1, figsize=(12, 7))
    ax.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    ax.set_title(f"AI Detection: {image_name}")
    ax.axis("off")

    for box, score, class_id in detections:
        x, y, w, h = box

        # YOLOv8 gives normalized coordinates (scaled to input size)
        # Convert back to original image scale
        x *= original_width / image_size[0]
        y *= original_height / image_size[1]
        w *= original_width / image_size[0]
        h *= original_height / image_size[1]

        x1 = x - w / 2
        y1 = y - h / 2

        rect = patches.Rectangle(
            (x1, y1), w, h,
            linewidth=2,
            edgecolor=class_colors.get(class_id, "white"),
            facecolor="none"
        )
        ax.add_patch(rect)
        ax.text(x1, y1 - 10, f"Class {class_id} ({score:.2f})",
                color="white", fontsize=10,
                bbox=dict(facecolor=class_colors.get(class_id, "black"), alpha=0.6))

    # --- SAVE OUTPUT ---
    output_filename = f"Det_{idx+1:05d}_{os.path.splitext(image_name)[0]}.png"
    output_path = os.path.join(output_folder, output_filename)
    plt.savefig(output_path)
    plt.close(fig)
    print(f"✅ Saved: {output_path}")
