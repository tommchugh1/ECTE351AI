import os
import matplotlib.pyplot as plt
import matplotlib.patches as patches

# --- CONFIG ---
txt_file = r"C:\Users\Group8\Documents\ModelTrainingData\OutputPhotosAndLabel\250916 1600\labels\ImgSeq_00380.txt"
image_width = 1920
image_height = 1080
class_colors = {0:"red", 1:"blue", 2:"green", 3:"yellow"}

# --- Read YOLO labels ---
labels = []
if os.path.exists(txt_file):
    with open(txt_file, "r") as f:
        for line in f:
            line = line.strip()
            if not line: continue
            parts = line.split()
            if len(parts) != 5: continue
            class_id, x_center, y_center, w, h = map(float, parts)
            labels.append((int(class_id), x_center, y_center, w, h))
else:
    print(f"File not found: {txt_file}")
    exit(1)

# --- Visualization ---
fig, ax = plt.subplots(1, figsize=(12,7))
ax.set_xlim(0, image_width)
ax.set_ylim(0, image_height)
ax.invert_yaxis()
ax.set_title(f"YOLOv8 Labels: {os.path.basename(txt_file)}")
ax.set_xlabel("X pixels")
ax.set_ylabel("Y pixels")

for class_id, x_center, y_center, w, h in labels:
    x = (x_center - w/2) * image_width
    y = (y_center - h/2) * image_height
    width = w * image_width
    height = h * image_height
    rect = patches.Rectangle((x, y), width, height, linewidth=2,
                             edgecolor=class_colors.get(class_id,"white"), facecolor="none")
    ax.add_patch(rect)
    ax.text(x, y-5, str(class_id), color="white", fontsize=12, backgroundcolor="black")

plt.show()
