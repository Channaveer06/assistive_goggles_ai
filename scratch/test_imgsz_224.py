import time
import cv2
import numpy as np
from ultralytics import YOLO

detector = YOLO("yolo11n.pt")
dummy_frame = np.zeros((480, 640, 3), dtype=np.uint8)

for size in [256, 224, 192]:
    times = []
    for _ in range(15):
        t0 = time.perf_counter()
        detector.track(dummy_frame, imgsz=size, persist=True, tracker="bytetrack.yaml", verbose=False)
        times.append((time.perf_counter() - t0) * 1000.0)
    print(f"imgsz={size}: mean={np.mean(times):.1f}ms, min={np.min(times):.1f}ms (FPS: {1000.0/np.mean(times):.1f})")

