import cv2
import numpy as np
from openvino import Core
import os

# Initialize OpenVINO Core
ie = Core()

# Load the model
header = os.path.dirname(__file__)
model_path = os.path.join(header, "weights\\best_openvino_model", "best.xml")  # Path to your .xml file
model = ie.read_model(model=model_path)

# Use "GPU" or "CPU" if not available
try:
    compiled_model = ie.compile_model(model=model, device_name="GPU")
except:
    print("GPU not available, using CPU.")
    compiled_model = ie.compile_model(model=model, device_name="CPU")  

# Get input and output layer names
input_layer = compiled_model.input(0)
output_layer = compiled_model.output(0)

# Read input image
img_path = os.path.join(header, "image.jpg")
image = cv2.imread(img_path)

if image is None:
    raise FileNotFoundError(f"Could not read image at: {img_path}")

input_height, input_width = 640, 640  # Adjust based on metadata.yaml
resized_image = cv2.resize(image, (input_width, input_height))

# Preprocess the image (normalize to [0,1] and transpose to NCHW format)
input_data = resized_image.astype(np.float32) / 255.0
input_data = np.transpose(input_data, (2, 0, 1))  # Change HWC to CHW
input_data = np.expand_dims(input_data, axis=0)  # Add batch dimension

# Run inference
results = compiled_model.infer_new_request({input_layer: input_data})

# Get output
output = results[output_layer]
output = output.transpose(0, 2, 1)  # Now shape (1, 8400, 8)

# Postprocess the output (example for YOLOv8)
# Note: Adjust based on your model's output format (refer to metadata.yaml or YOLOv8 docs)
def postprocess_output(output, conf_threshold=0.5, iou_threshold=0.4):
    boxes, scores, classes = [], [], []
    for detection in output[0]:
        confidence = detection[4]  # Confidence score
        if confidence > conf_threshold:
            x, y, w, h = detection[:4]
            class_id = np.argmax(detection[5:])
            score = detection[5 + class_id]
            if score > conf_threshold:
                boxes.append([x, y, w, h])
                scores.append(confidence)
                classes.append(class_id)

    # Apply Non-Maximum Suppression (NMS)
    indices = cv2.dnn.NMSBoxes(boxes, scores, conf_threshold, iou_threshold)
    return [(boxes[i], scores[i], classes[i]) for i in indices]

# Process results
detections = postprocess_output(output)

# List classes
class_names = ['Bent', 'Bent And Dethreaded', 'Dethreaded', 'Normal']

# Print detection info
if detections:
    print(f"Detected {len(detections)} objects:")
    for i, (box, score, class_id) in enumerate(detections, 1):
        print(f" {i}. Class: {class_names[class_id]}, Score: {score:.2f}, Box: {box}")
else:
    print("No objects detected.")

# Draw bounding boxes on the image
for box, score, class_id in detections:
    x, y, w, h = box
    x1, y1 = int(x - w / 2), int(y - h / 2)
    x2, y2 = int(x + w / 2), int(y + h / 2)
    cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
    cv2.putText(image, f"{class_names[class_id]}: {score:.2f}", (x1, y1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

print("Output shape:", output.shape)
print("Example detection row:", output[0][:, 0])

# Save or display the result
cv2.imwrite(header + "\\output.jpg", image)
cv2.imshow("Result", image)
cv2.waitKey(0)
cv2.destroyAllWindows()
