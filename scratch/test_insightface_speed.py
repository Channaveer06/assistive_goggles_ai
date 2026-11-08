import time
import cv2
import numpy as np
from insightface.app import FaceAnalysis

print("Initializing InsightFace...")
app = FaceAnalysis(name="buffalo_l", providers=["CPUExecutionProvider"])
app.prepare(ctx_id=0, det_size=(320, 320))

dummy = np.zeros((480, 640, 3), dtype=np.uint8)

# Test 1: Full frame
t0 = time.perf_counter()
faces1 = app.get(dummy)
t1 = time.perf_counter()
print(f"Full Frame (480x640) took: {(t1 - t0)*1000:.1f}ms")

# Test 2: Resized frame (240x320)
small = cv2.resize(dummy, (320, 240))
t0 = time.perf_counter()
faces2 = app.get(small)
t1 = time.perf_counter()
print(f"Small Frame (240x320) took: {(t1 - t0)*1000:.1f}ms")
