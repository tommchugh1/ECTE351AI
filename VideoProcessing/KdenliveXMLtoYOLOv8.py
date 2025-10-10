import xml.etree.ElementTree as ET
import os
import re

# === CONFIG ===
input_xml = r"C:\Users\Group8\Desktop\Yolo Demo\ECTE351AI\VideoProcessing\InputNormalOutputFiltered.kdenlive"
image_folder = r"C:\Users\Group8\Documents\ModelTrainingData\OutputPhotosAndLabel\250916 1600"
output_label_folder = os.path.join(image_folder, "labels")
image_width = 1920
image_height = 1080
frame_rate = 25

os.makedirs(output_label_folder, exist_ok=True)

# === UTILS ===

def hhmmss_decimal_to_seconds(timecode: str) -> float:
    """Convert HH:MM:SS.xxx format to seconds (float)."""
    m = re.match(r"(\d+):(\d+):(\d+)\.(\d+)", timecode)
    if not m:
        raise ValueError(f"Cannot parse decimal timecode: '{timecode}'")
    h, mm, ss, frac = m.groups()
    return int(h) * 3600 + int(mm) * 60 + int(ss) + float("0." + frac)

def hhmmss_frames_to_seconds(timecode: str) -> float:
    """Convert HH:MM:SS:FF (frames) to seconds."""
    m = re.match(r"(\d+):(\d+):(\d+):(\d+)", timecode)
    if not m:
        raise ValueError(f"Cannot parse frame-based timecode: '{timecode}'")
    h, mm, ss, ff = map(int, m.groups())
    return h * 3600 + mm * 60 + ss + (ff / frame_rate)

def seconds_to_frame(seconds: float) -> int:
    """Convert seconds to frame index (1-based)."""
    return int(seconds * frame_rate) + 1

def interpolate_bbox(t1, bbox1, t2, bbox2, t):
    """Linear interpolation between two bounding boxes."""
    if t2 == t1:
        return bbox1
    ratio = (t - t1) / (t2 - t1)
    return tuple(b1 + ratio * (b2 - b1) for b1, b2 in zip(bbox1, bbox2))

def parse_bbox_string(bbox_str):
    """Convert 'x y w h ... rest' to (x, y, w, h) floats."""
    parts = bbox_str.strip().split()
    if len(parts) < 4:
        return None
    try:
        x, y, w, h = map(float, parts[:4])
        return (x, y, w, h)
    except:
        return None

# === LOAD XML ===

tree = ET.parse(input_xml)
root = tree.getroot()

# === Find playlist blank lengths & tractor out times ===

playlist_blank_length = {}
for playlist in root.findall(".//playlist"):
    pid = playlist.get("id")
    blank = playlist.find("blank")
    if blank is not None and "length" in blank.attrib:
        try:
            playlist_blank_length[pid] = hhmmss_decimal_to_seconds(blank.attrib["length"])
        except ValueError:
            playlist_blank_length[pid] = 0.0
    else:
        playlist_blank_length[pid] = 0.0

# Map playlist IDs to their tractor out time (XmlEnd)
# We'll need to detect which tractor belongs to which playlist context
tractor_out = {}
for tractor in root.findall(".//tractor"):
    out_tc = tractor.get("out")
    if out_tc:
        try:
            out_sec = hhmmss_decimal_to_seconds(out_tc)
        except ValueError:
            out_sec = 0.0
        # Now we need to find connected playlists
        for track in tractor.findall("track"):
            pid = track.get("producer")  # this is playlist ID in your setup
            if pid:
                # Multiple tracks might map — but keep the maximum out time
                existing = tractor_out.get(pid, 0.0)
                if out_sec > existing:
                    tractor_out[pid] = out_sec

# === Collect all frames we have images for ===

image_files = sorted([f for f in os.listdir(image_folder) if re.match(r"ImgSeq_(\d+)\.png", f)])
all_frames = [int(re.match(r"ImgSeq_(\d+)\.png", f).group(1)) for f in image_files]

frame_annotations = {}

# === Process each entry ===

for playlist in root.findall(".//playlist"):
    pid = playlist.get("id")
    base_start = playlist_blank_length.get(pid, 0.0)

    for entry in playlist.findall("entry"):
        producer = entry.get("producer")
        match = re.search(r"producer(\d+)", producer or "")
        if not match:
            continue
        class_id = int(match.group(1)) - 1

        # Entry in/out (clip-specific times)
        in_tc = entry.get("in", "00:00:00.000")
        out_tc = entry.get("out", None)
        if out_tc is None:
            continue  # if no out, skip

        try:
            entry_in_sec = hhmmss_decimal_to_seconds(in_tc)
        except ValueError:
            # May be frames format in some entries — you can adapt if needed
            entry_in_sec = 0.0

        try:
            entry_out_sec = hhmmss_decimal_to_seconds(out_tc)
        except ValueError:
            # fallback
            continue

        # Global start / end for this producer clip
        xml_start = base_start + entry_in_sec
        xml_end = base_start + entry_out_sec

        # Grab rect keyframes for this entry
        rect_prop = None
        for filt in entry.findall("filter"):
            for prop in filt.findall("property"):
                if prop.get("name") == "rect":
                    rect_prop = prop.text.strip()
                    break
            if rect_prop:
                break

        if not rect_prop:
            continue

        # Parse all keyframes from rect
        keyframes = []
        for kf_item in rect_prop.split(";"):
            if "=" not in kf_item:
                continue
            tcode, bbox_str = kf_item.split("=", 1)
            tcode = tcode.strip()
            try:
                t_sec = hhmmss_decimal_to_seconds(tcode)
            except ValueError:
                # skip invalid
                continue
            bbox = parse_bbox_string(bbox_str)
            if bbox is None:
                continue
            # Store the keyframe time *relative to global video*, i.e. add base_start
            global_t = base_start + t_sec
            # Only consider keyframes that fall within xml_start/xml_end (or slightly outside to allow interpolation)
            keyframes.append((global_t, bbox))

        if not keyframes:
            continue

        keyframes.sort(key=lambda x: x[0])

        # === Now loop through frames and generate annotations for this entry

        for fidx in all_frames:
            current_sec = (fidx - 1) / frame_rate
            if current_sec < xml_start or current_sec > xml_end:
                continue

            # Find two keyframes around current_sec
            bbox_interp = None
            for i in range(len(keyframes) - 1):
                t1, b1 = keyframes[i]
                t2, b2 = keyframes[i + 1]
                if t1 <= current_sec <= t2:
                    bbox_interp = interpolate_bbox(t1, b1, t2, b2, current_sec)
                    break

            if bbox_interp is None:
                # If before first keyframe, or after last, pick nearest
                if current_sec < keyframes[0][0]:
                    bbox_interp = keyframes[0][1]
                else:
                    bbox_interp = keyframes[-1][1]

            x, y, w, h = bbox_interp
            # Normalize to YOLO format: center_x, center_y, width, height (all relative 0–1)
            x_center = (x + w / 2.0) / image_width
            y_center = (y + h / 2.0) / image_height
            w_rel = w / image_width
            h_rel = h / image_height

            # Clip or skip boxes that are totally outside or weird
            if w_rel <= 0 or h_rel <= 0 or x_center < 0 or x_center > 1 or y_center < 0 or y_center > 1:
                continue

            label = f"{class_id} {x_center:.6f} {y_center:.6f} {w_rel:.6f} {h_rel:.6f}"

            frame_annotations.setdefault(fidx, []).append(label)

# === Write .txt files ===

for img_filename in image_files:
    m = re.match(r"ImgSeq_(\d+)\.png", img_filename)
    if not m:
        continue
    fidx = int(m.group(1))
    labels = frame_annotations.get(fidx, [])
    txt_path = os.path.join(output_label_folder, f"ImgSeq_{fidx:05d}.txt")
    with open(txt_path, "w") as f:
        if labels:
            f.write("\n".join(labels))
        else:
            f.write("")

print(f"✅ Finished writing {len(image_files)} annotation files to {output_label_folder}")
