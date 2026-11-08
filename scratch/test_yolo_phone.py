import os
import sys
import time
import numpy as np
import cv2
from ultralytics import YOLO

model = YOLO("yolo11n.pt")
dummy = np.zeros((480, 640, 3), dtype=np.uint8)

# Warmup
for _ in range(5):
    _ = model(dummy, imgsz=192, verbose=False)

for size in [192, 256, 320, 384]:
    times = []
    for _ in range(20):
        t0 = time.perf_counter()
        _ = model(dummy, imgsz=size, verbose=False)
        times.append((time.perf_counter() - t0) * 1000)
    print(f"imgsz={size}: Mean Latency = {np.mean(times):.1f}ms (Max = {np.max(times):.1f}ms)")
