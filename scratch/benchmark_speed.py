import os
import sys
import time
import cv2
import numpy as np
import torch
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config
from transformers import AutoImageProcessor, AutoModelForDepthEstimation

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Device: {device}")
processor = AutoImageProcessor.from_pretrained(config.METRIC_DEPTH_MODEL, local_files_only=True)
model = AutoModelForDepthEstimation.from_pretrained(config.METRIC_DEPTH_MODEL, local_files_only=True).to(device).eval()

dummy_bgr = np.zeros((480, 640, 3), dtype=np.uint8)

@torch.inference_mode()
def test_depth(frame_bgr, downsample=1):
    h, w = frame_bgr.shape[:2]
    if downsample > 1:
        small = cv2.resize(frame_bgr, (w // downsample, h // downsample))
        rgb = cv2.cvtColor(small, cv2.COLOR_BGR2RGB)
    else:
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    inputs = processor(images=Image.fromarray(rgb), return_tensors="pt").to(device)
    outputs = model(**inputs)
    depth_map = torch.nn.functional.interpolate(
        outputs.predicted_depth.unsqueeze(1),
        size=(h, w),
        mode="bicubic",
        align_corners=False,
    ).squeeze().cpu().numpy()
    return depth_map

# Warmup
test_depth(dummy_bgr)

# Benchmark standard
t0 = time.perf_counter()
test_depth(dummy_bgr, downsample=1)
d1 = (time.perf_counter() - t0) * 1000.0
print(f"Full resolution depth time: {d1:.1f}ms")

# Benchmark 2x downsample
t0 = time.perf_counter()
test_depth(dummy_bgr, downsample=2)
d2 = (time.perf_counter() - t0) * 1000.0
print(f"2x downsample depth time: {d2:.1f}ms")
