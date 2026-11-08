import os
import sys
import time
import numpy as np
import torch

# Limit PyTorch inter-op and intra-op threads so it doesn't starve the system
torch.set_num_threads(2)
try:
    torch.set_num_interop_threads(1)
except Exception:
    pass

from ultralytics import YOLO
detector = YOLO("yolo11n.pt")
dummy_frame = np.zeros((480, 640, 3), dtype=np.uint8)

# Warmup
detector.track(dummy_frame, persist=True, tracker="bytetrack.yaml", verbose=False)

times = []
for i in range(30):
    t0 = time.perf_counter()
    res = detector.track(dummy_frame, persist=True, tracker="bytetrack.yaml", verbose=False)[0]
    times.append((time.perf_counter() - t0) * 1000.0)

print(f"YOLO track with torch.set_num_threads(2): mean={np.mean(times):.1f}ms, min={np.min(times):.1f}ms (FPS: {1000.0/np.mean(times):.1f})")

