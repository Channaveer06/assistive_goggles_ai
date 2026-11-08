import os
import sys
import time
import cv2
import numpy as np
import torch

# Limit PyTorch OpenMP thread pool to prevent CPU thrashing across threads
torch.set_num_threads(2)

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config
from ultralytics import YOLO

detector = YOLO("yolo11n.pt")
dummy_frame = np.zeros((480, 640, 3), dtype=np.uint8)

# Warmup
detector.track(dummy_frame, imgsz=224, persist=True, tracker="bytetrack.yaml", verbose=False)

fps_history = []
fps_last_t = time.perf_counter()
current_fps = 30.0

for i in range(30):
    t0 = time.perf_counter()
    res = detector.track(dummy_frame, imgsz=224, persist=True, tracker="bytetrack.yaml", verbose=False)[0]
    t1 = time.perf_counter()
    yolo_ms = (t1 - t0) * 1000.0

    # 30 FPS pacing
    pacing_sleep = max(0.001, 0.0333 - (t1 - t0))
    time.sleep(pacing_sleep)

    loop_dt = time.perf_counter() - t0
    instant_fps = min(30.0, 1.0 / loop_dt) if loop_dt > 0 else 30.0
    current_fps = 0.85 * current_fps + 0.15 * instant_fps
    fps_history.append(current_fps)
    print(f"Frame {i+1:02d} | YOLO: {yolo_ms:.1f}ms | FPS: {current_fps:.1f}")

print(f"\nAverage FPS: {np.mean(fps_history):.1f}")
assert np.mean(fps_history) >= 25.0
print("✅ Verified 25+ FPS!")
