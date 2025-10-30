import os
import time
import cv2
import numpy as np
from functools import lru_cache
from ultralytics import YOLO
from openvino import Core 

# Resolve paths relative to this file (portable; no hard-coded absolute paths)
HERE = os.path.dirname(os.path.realpath(__file__))
IMAGE_PATH = os.path.join(HERE, "RUNS", "train", "bolt_training", "image.jpg")
PT_MODEL_PATH = os.path.join(HERE, "RUNS", "train", "bolt_training", "weights", "best.pt")
OV_MODEL_PATH = os.path.join(HERE, "RUNS", "train", "bolt_training", "weights", "best_openvino_model")

# ---- Lazy, cached accessors (run once on first call) ----
@lru_cache(maxsize=1)
def get_core():
    return Core()

def pick_device(preferred="CPU"):
    # keep it simple; you can auto-select here if you want
    devices = get_core().available_devices or []
    print(f"[execute] OpenVINO devices: {devices} — using: {preferred}")
    os.environ["OPENVINO_DEFAULT_DEVICE"] = preferred
    return preferred

@lru_cache(maxsize=1)
def get_pt_model():
    if not os.path.exists(PT_MODEL_PATH):
        raise FileNotFoundError(f"Model file not found: {PT_MODEL_PATH}")
    m = YOLO(PT_MODEL_PATH)
    # For PyTorch backends, .to('cpu'|'cuda') is relevant; OpenVINO uses export or runtime
    try:
        m.to("cpu")
    except Exception:
        pass
    print("[execute] YOLOv8 (PyTorch) loaded.")
    return m

@lru_cache(maxsize=1)
def get_ov_model():
    # Only use if you actually exported an OpenVINO model
    if not os.path.exists(OV_MODEL_PATH):
        raise FileNotFoundError(f"OpenVINO model not found: {OV_MODEL_PATH}")
    m = YOLO(OV_MODEL_PATH)
    print("[execute] YOLOv8 (OpenVINO) loaded.")
    return m

# ---- Public API ----
def run_inference_on_generator(frame_generator, on_frame=None, device="CPU", use_openvino=False):
    """
    Runs YOLO inference on frames from a generator.
      - If on_frame is provided: pushes annotated frames via callback.
      - Else: yields annotated frames (iterator mode).
    """
    print("[execute] entered run_inference_on_generator", flush=True)
    pick_device(device)
    print("[execute] after device probe; loading model…", flush=True)
    # Choose one model (don’t load both)
    model = get_ov_model() if use_openvino else get_pt_model()
    print("[execute] model loaded; entering frame loop", flush=True)
    fps_t0 = time.time()
    fps_n = 0
    print("[execute] YOLO loaded; entering frame loop", flush=True)
    for frame in frame_generator:
        if frame is None or not isinstance(frame, np.ndarray):
            continue

        # Ultralytics predict (quiet)
        results = model.predict(frame, verbose=False)
        det_count = len(results[0].boxes) if results and results[0].boxes is not None else 0
        annotated = results[0].plot()

        fps_n += 1
        if time.time() - fps_t0 >= 1.0:
            fps_n = 0
            fps_t0 = time.time()

        if on_frame:
            on_frame(annotated, det_count)
        else:
            yield annotated

    print("[execute] Inference stopped.")
    

# Benchmarks or demos behind a main guard only
if __name__ == "__main__":
    # Example: quick sanity check without impacting importers
    if os.path.exists(IMAGE_PATH):
        _ = get_pt_model()  # or get_ov_model()
        print("[execute] Sanity check OK.")
        model = YOLO(OV_MODEL_PATH)
        model.predict(IMAGE_PATH)
    else:
        print(f"[execute] Missing sample image: {IMAGE_PATH}")