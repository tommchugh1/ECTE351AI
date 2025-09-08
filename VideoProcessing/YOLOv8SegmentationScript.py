import os
import cv2
import numpy as np
from pathlib import Path


def create_yolo_seg_labels(images_dir, masks_dir, output_labels_dir, class_id=0):
    os.makedirs(output_labels_dir, exist_ok=True)

    for mask_name in os.listdir(masks_dir):
        mask_path = os.path.join(masks_dir, mask_name)
        img_name = os.path.splitext(mask_name)[0] + ".jpg"
        img_path = os.path.join(images_dir, img_name)

        # Read image for size
        img = cv2.imread(img_path)
        h, w = img.shape[:2]

        # Read mask and threshold
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        _, binary = cv2.threshold(mask, 1, 255, cv2.THRESH_BINARY)

        # Find contours
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Write YOLO label file
        label_lines = []
        for contour in contours:
            if len(contour) < 6:
                continue  # Skip small or bad contours

            # Bounding box
            x, y, box_w, box_h = cv2.boundingRect(contour)
            x_center = (x + box_w / 2) / w
            y_center = (y + box_h / 2) / h
            norm_w = box_w / w
            norm_h = box_h / h

            # Normalize polygon points
            polygon = contour.squeeze()
            if len(polygon.shape) != 2:
                continue
            polygon_norm = [(pt[0] / w, pt[1] / h) for pt in polygon]
            flat_polygon = " ".join([f"{x:.6f} {y:.6f}" for x, y in polygon_norm])

            line = f"{class_id} {x_center:.6f} {y_center:.6f} {norm_w:.6f} {norm_h:.6f} {flat_polygon}"
            label_lines.append(line)

        # Save label file
        label_file_path = os.path.join(output_labels_dir, os.path.splitext(mask_name)[0] + ".txt")
        with open(label_file_path, "w") as f:
            f.write("\n".join(label_lines))

    print("✅ Done creating YOLOv8 segmentation labels.")

# Example usage
create_yolo_seg_labels(
    images_dir="datasets/myproject/images/train",
    masks_dir="datasets/myproject/masks/train",
    output_labels_dir="datasets/myproject/labels/train"
)
