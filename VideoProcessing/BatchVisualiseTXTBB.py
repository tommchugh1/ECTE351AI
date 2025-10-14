import os
import matplotlib.pyplot as plt
import matplotlib.patches as patches

# --- CONFIG ---
input_folder = r"C:\Users\Group8\Documents\ModelTrainingData\OutputPhotosAndLabel\251012\labels"  # folder with .txt files
image_width = 1920
image_height = 1080
class_colors = {0: "red", 1: "blue", 2: "green", 3: "yellow"}
output_folder = r"C:\Users\Group8\Documents\ModelTrainingData\OutputPhotosAndLabel\251012\labels\BB Vis"  # Folder where images will be saved

# --- Create output directory if it doesn't exist ---
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

# --- Get all .txt files in the folder ---
txt_files = [f for f in os.listdir(input_folder) if f.endswith('.txt')]

# --- Process each .txt file ---
for idx, txt_file_name in enumerate(txt_files):
    txt_file_path = os.path.join(input_folder, txt_file_name)
    
    # --- Read YOLO labels ---
    labels = []
    if os.path.exists(txt_file_path):
        with open(txt_file_path, "r") as f:
            for line in f:
                line = line.strip()
                if not line: continue
                parts = line.split()
                if len(parts) != 5: continue
                class_id, x_center, y_center, w, h = map(float, parts)
                labels.append((int(class_id), x_center, y_center, w, h))
    else:
        print(f"File not found: {txt_file_path}")
        continue  # Skip to the next file if the current file is not found

    # --- Visualization ---
    fig, ax = plt.subplots(1, figsize=(12,7))
    ax.set_xlim(0, image_width)
    ax.set_ylim(0, image_height)
    ax.invert_yaxis()
    ax.set_title(f"YOLOv8 Labels: {txt_file_name}")
    ax.set_xlabel("X pixels")
    ax.set_ylabel("Y pixels")

    for class_id, x_center, y_center, w, h in labels:
        x = (x_center - w/2) * image_width
        y = (y_center - h/2) * image_height
        width = w * image_width
        height = h * image_height
        rect = patches.Rectangle((x, y), width, height, linewidth=2,
                                 edgecolor=class_colors.get(class_id, "white"), facecolor="none")
        ax.add_patch(rect)
        ax.text(x, y-5, str(class_id), color="white", fontsize=12, backgroundcolor="black")

    # --- Save the image ---
    output_image_name = f"ImgSeq_{idx+1:05d}_BB.png"  # Zero-padded filename
    output_image_path = os.path.join(output_folder, output_image_name)
    plt.savefig(output_image_path)
    print(f"Saved image: {output_image_path}")
    
    # Close the plot to avoid overlap with the next plot
    plt.close(fig)