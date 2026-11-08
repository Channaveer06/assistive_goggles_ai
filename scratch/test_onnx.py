import time
import os
import numpy as np
from ultralytics import YOLO

onnx_path = os.path.join("project_data", "models_onnx", "yolo11n.onnx")
print(f"Testing ONNX path: {onnx_path}, exists={os.path.exists(onnx_path)}")
if os.path.exists(onnx_path):
    detector = YOLO(onnx_path, task="detect")
    dummy = np.zeros((480, 640, 3), dtype=np.uint8)
    detector.track(dummy, imgsz=320, persist=True, tracker="bytetrack.yaml", verbose=False)
    times = []
    for _ in range(10):
        t0 = time.perf_counter()
        detector.track(dummy, imgsz=320, persist=True, tracker="bytetrack.yaml", verbose=False)
        times.append((time.perf_counter() - t0) * 1000.0)
    print(f"ONNX imgsz=320: mean={np.mean(times):.1f}ms (FPS: {1000.0/np.mean(times):.1f})")

