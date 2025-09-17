import xml.etree.ElementTree as ET
import os
import re
import yaml

# Project Summary (YOLOv8 Annotation Conversion from Kdenlive XML)
# 1. Parsed a `.kdenlive` XML project to extract object annotations tied to video frames.
# 2. Interpolated bounding boxes across keyframes for smooth, frame-accurate labels.
# 3. Converted boxes to YOLOv8 format: `class_id x_center y_center width height`.
# 4. Matched frame numbers in `ImgSeq_XXXXX.png` to corresponding interpolated annotations.
# 5. Wrote each frame's annotations to `.txt` files in a `labels/` directory.
# 6. Ensured consistent inclusion of all class IDs across all frames, even if missing keyframes.
# 7. Ignored invalid or background entries (e.g., `producer0`, `black`).
# 8. Created `data.yaml` dynamically using `<clipname>` or fallback `<resource>` names.
# 9. Ensured proper YAML formatting using bracketed list style (`names: ['a', 'b', 'c']`).
# 10. Pipeline output is YOLOv8-ready for model training or validation.
# More info at https://roboflow.com/formats/yolov8-pytorch-txt and https://docs.ultralytics.com/usage/simple-utilities/#ultralytics-sweep-annotation 
#Ensure that the image output has names "ImgSeq" format

# === CONFIG ===
input_xml = r"C:\Users\Group8\Desktop\Yolo Demo\ECTE351AI\VideoProcessing\InputNormalOutputFiltered.kdenlive"
image_folder = r"C:\Users\Group8\Documents\ModelTrainingData\OutputPhotosAndLabel\250916 1600"
output_yaml = r"C:\Users\Group8\Desktop\Yolo Demo\ECTE351AI\AI\dataset\data.yaml"
output_label_folder = os.path.join(image_folder, "labels")  # Save .txt here
image_width = 1920
image_height = 1080
frame_rate = 25  # Adjust if different

# === CLASS MAPPING ===
producer_class_map = {
    "producer1": 0,  # Bent
    "producer2": 1,  # Bent and Dethreaded
    "producer3": 2,  # Dethreaded
    "producer4": 3,  # Normal
}

os.makedirs(output_label_folder, exist_ok=True)

def timecode_to_seconds(timecode: str) -> float:
    """Convert HH:MM:SS.mmm to total seconds as float."""
    match = re.match(r"(\d+):(\d+):(\d+).(\d+)", timecode)
    if not match:
        return -1.0
    h, m, s, ms = map(int, match.groups())
    return h * 3600 + m * 60 + s + ms / 1000

def seconds_to_frame(seconds: float) -> int:
    """Convert seconds to frame index (1-based)."""
    return int(seconds * frame_rate) + 1

def interpolate_bbox(t1, bbox1, t2, bbox2, t):
    """Linear interpolation of bbox between t1 and t2 for time t."""
    # bbox = (x, y, w, h)
    if t2 == t1:
        return bbox1
    ratio = (t - t1) / (t2 - t1)
    return tuple(b1 + ratio * (b2 - b1) for b1, b2 in zip(bbox1, bbox2))

def parse_bbox_string(bbox_str):
    """Parse bbox string 'x y w h ...' and return (x,y,w,h) floats."""
    parts = bbox_str.split()
    if len(parts) < 4:
        return None
    try:
        x, y, w, h = map(float, parts[:4])
        return (x, y, w, h)
    except ValueError:
        return None

# === Step 1: Parse XML to collect keyframe data per frame for each entry ===

# Data structure:
# frame_annotations[frame_idx] = list of (class_id, bbox) for that frame
frame_annotations = {}
image_files = sorted(f for f in os.listdir(image_folder) if re.match(r"ImgSeq_\d+\.png", f))

tree = ET.parse(input_xml)
root = tree.getroot()

for entry in root.iter("entry"):
    producer = entry.get("producer")
    if producer not in producer_class_map:
        continue
    class_id = producer_class_map[producer]

    for prop in entry.iter("property"):
        if prop.get("name") == "rect":
            rect_data = prop.text.strip()
            keyframes = rect_data.split(";")

            kf_list = []
            for kf in keyframes:
                if "=" not in kf:
                    continue
                timecode, bbox_str = kf.split("=")
                secs = timecode_to_seconds(timecode)
                bbox = parse_bbox_string(bbox_str)
                if secs >= 0 and bbox is not None:
                    kf_list.append((secs, bbox))
            if not kf_list:
                continue

            kf_list.sort(key=lambda x: x[0])

            # Get all frame indices from your images
            all_frames = sorted(int(re.match(r"ImgSeq_(\d+)\.png", f).group(1)) for f in image_files)

            for fidx in all_frames:
                current_sec = (fidx - 1) / frame_rate

                # Find the interval for interpolation or nearest bbox
                interp_bbox = None
                for i in range(len(kf_list) - 1):
                    t1, bbox1 = kf_list[i]
                    t2, bbox2 = kf_list[i + 1]
                    if t1 <= current_sec <= t2:
                        interp_bbox = interpolate_bbox(t1, bbox1, t2, bbox2, current_sec)
                        break

                if interp_bbox is None:
                    # Outside keyframe range, pick nearest bbox
                    if current_sec < kf_list[0][0]:
                        interp_bbox = kf_list[0][1]
                    else:
                        interp_bbox = kf_list[-1][1]

                x, y, w, h = interp_bbox
                x_center = (x + w / 2) / image_width
                y_center = (y + h / 2) / image_height
                w_norm = w / image_width
                h_norm = h / image_height

                yolo_label = f"{class_id} {x_center:.6f} {y_center:.6f} {w_norm:.6f} {h_norm:.6f}"

                if fidx not in frame_annotations:
                    frame_annotations[fidx] = []
                frame_annotations[fidx].append(yolo_label)



# === Step 2: For every image in folder, write corresponding txt file with annotations ===


for img_filename in image_files:
    match = re.match(r"ImgSeq_(\d+)\.png", img_filename)
    if not match:
        continue
    frame_idx = int(match.group(1))

    labels = frame_annotations.get(frame_idx, [])
    txt_filename = f"ImgSeq_{frame_idx:05d}.txt"
    txt_path = os.path.join(output_label_folder, txt_filename)

    with open(txt_path, "w") as f:
        if labels:
            f.write("\n".join(labels))
        else:
            # Write empty file or skip if you want
            f.write("")

print(f"✅ Wrote {len(image_files)} label files to '{output_label_folder}'")