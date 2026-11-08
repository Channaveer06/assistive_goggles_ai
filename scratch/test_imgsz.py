import time
import numpy as np
from ultralytics import YOLO

detector = YOLO("yolo11n.pt")
dummy_frame = np.zeros((480, 640, 3), dtype=np.uint8)

# Warmup
detector.track(dummy_frame, imgsz=320, persist=True, tracker="bytetrack.yaml", verbose=False)

for size in [640, 480, 384, 320, 256]:
    times = []
    for _ in range(10):
        t0 = time.perf_counter()
        detector.track(dummy_frame, imgsz=size, persist=True, tracker="bytetrack.yaml", verbose=False)
        times.append((time.perf_counter() - t0) * 1000.0)
    print(f"imgsz={size}: mean={np.mean(times):.1f}ms (FPS: {1000.0/np.mean(times):.1f})")

