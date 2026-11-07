import os
import sys
import time
import numpy as np
import cv2

# Set UTF-8 encoding for console output on Windows
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

# Ensure project root is in sys.path
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

print("🧪 [TEST] Starting Dual Logging & Benchmarking Verification Test...")

# 1. Test Logger Configuration
from run_live_local import setup_dual_logger, LOG_FILE

test_log_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), "test_execution_logs.txt")
if os.path.exists(test_log_file):
    os.remove(test_log_file)

logger = setup_dual_logger(log_file=test_log_file)

# Emit sample logs
logger.info("[Main Thread] YOLOv8n Inference Time: 28ms | FPS: 30.2")
logger.info("[Daemon Thread] DeepFace Processing: Starting recognition for Track #1...")
logger.info("[Daemon Thread] DeepFace Inference Time: 135ms | Recognized: 'Channaveer' (Track #1)")
logger.info("[Daemon Thread] EasyOCR Processing: Scanning spatial frame crop...")
logger.info("[Daemon Thread] EasyOCR Inference Time: 88ms | Detected: ['EXIT']")

# Flush handlers
for handler in logger.handlers:
    handler.flush()

assert os.path.exists(test_log_file), "test_execution_logs.txt was not created!"
with open(test_log_file, "r", encoding="utf-8") as f:
    log_content = f.read()

assert "[Main Thread] YOLOv8n Inference Time: 28ms | FPS: 30.2" in log_content
assert "[Daemon Thread] DeepFace Processing:" in log_content
assert "[Daemon Thread] DeepFace Inference Time: 135ms" in log_content
assert "[Daemon Thread] EasyOCR Inference Time: 88ms" in log_content
print("✅ [TEST] Dual Logging & Formatting Verification Passed!")

# 2. Test HUD Live FPS Overlay
dummy_frame = np.zeros((480, 640, 3), dtype=np.uint8)
from run_live_local import LiveLocalGogglesApp

app = LiveLocalGogglesApp(headless=True, test_mode=True, disable_mic=True)
hud = app.draw_hud_overlays(dummy_frame, fps=30.2, frame_count=1)

assert hud is not None
assert hud.shape[1] == 640
assert hud.shape[0] == 480 + 46 + 64
print("✅ [TEST] Live FPS HUD Overlay Verification Passed!")

# 3. Test Benchmarking attributes on modules
assert hasattr(app.reflex, "last_yolo_time_ms")
assert hasattr(app.reflex, "last_depth_time_ms")
assert hasattr(app.face_recognizer, "last_inference_time_ms")
assert hasattr(app.ocr_reader, "last_inference_time_ms")
assert hasattr(app.ocr, "last_inference_time_ms")
assert hasattr(app.social, "last_inference_time_ms")
print("✅ [TEST] Inference Benchmarking Attributes Verified on all models!")

print("🎉 ALL TESTS PASSED SUCCESSFULLY!")
