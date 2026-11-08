import os
import sys
import time
import cv2
import numpy as np
import threading
import queue
import torch
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config
from ultralytics import YOLO
from transformers import AutoImageProcessor, AutoModelForDepthEstimation

print("Testing Asynchronous Multi-Rate Reflex Engine (30 FPS Lock)...")

device = "cuda" if torch.cuda.is_available() else "cpu"
detector = YOLO("yolo11n.pt")
processor = AutoImageProcessor.from_pretrained(config.METRIC_DEPTH_MODEL, local_files_only=True)
depth_model = AutoModelForDepthEstimation.from_pretrained(config.METRIC_DEPTH_MODEL, local_files_only=True).to(device).eval()

depth_queue = queue.Queue(maxsize=1)
latest_depth = np.full((480, 640), 2.5, dtype=np.float32)
running = True

def depth_worker():
    global latest_depth, running
    while running:
        try:
            frame = depth_queue.get(timeout=0.2)
            if frame is None:
                break
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            inputs = processor(images=Image.fromarray(rgb), return_tensors="pt").to(device)
            with torch.inference_mode():
                outputs = depth_model(**inputs)
            d_map = torch.nn.functional.interpolate(
                outputs.predicted_depth.unsqueeze(1),
                size=(480, 640),
                mode="bicubic",
                align_corners=False,
            ).squeeze().cpu().numpy()
            latest_depth = d_map
            time.sleep(0.01) # Yield GIL
        except queue.Empty:
            continue
        except Exception as e:
            time.sleep(0.05)

t = threading.Thread(target=depth_worker, daemon=True)
t.start()

dummy_frame = np.zeros((480, 640, 3), dtype=np.uint8)

# Simulate 30 frames of the main loop
fps_list = []
last_t = time.perf_counter()

for i in range(30):
    t0 = time.perf_counter()

    # 1. Non-blocking push to depth queue
    if depth_queue.empty():
        depth_queue.put(dummy_frame.copy())

    # 2. YOLO tracking (640x480)
    res = detector.track(dummy_frame, persist=True, tracker="bytetrack.yaml", verbose=False)[0]

    # 3. Read latest depth map
    d = latest_depth[240, 320]

    t1 = time.perf_counter()
    loop_time = (t1 - t0) * 1000.0
    dt = t1 - last_t
    last_t = t1
    fps = 1.0 / dt if dt > 0 else 30.0
    fps_list.append(fps)
    print(f"Frame {i+1}: Loop Time = {loop_time:.1f}ms | Instant FPS = {fps:.1f}")

    # Small sleep to simulate 30 FPS ceiling (33ms period)
    remain = max(0.0, 0.0333 - (time.perf_counter() - t0))
    time.sleep(remain)

running = False
avg_fps = np.mean(fps_list[1:])
print(f"\nAverage Main Loop FPS: {avg_fps:.1f}")

