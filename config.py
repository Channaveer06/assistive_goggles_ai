import os

# === CENTRAL CONFIGURATION (Shared by Colab & Local Hardware) ===
ENV_MODE = "LOCAL"               # Switched to "LOCAL" for local hardware

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Safety & Distance Thresholds (in real meters)
CRITICAL_DANGER_M = 1.2          # Immediate stop/hazard warning (under 1.2 meters)
WARNING_DISTANCE_M = 2.5         # Standard obstacle awareness (under 2.5 meters)
TRACK_COOLDOWN_SEC = 4.0         # Wait 4 seconds before re-announcing the same Track ID

# VizWiz Frame Quality Thresholds (Framing & Lighting Coach)
MIN_BRIGHTNESS = 35.0            # Mean pixel luminance (0-255) before warning "Too dark"
MIN_SHARPNESS_LAPLACIAN = 20.0   # Variance of Laplacian before warning "Camera blurry"

# Model Checkpoints
YOLO_ONNX_PATH = os.path.join(BASE_DIR, "project_data", "models_onnx", "yolo11n.onnx")
METRIC_DEPTH_MODEL = "depth-anything/Depth-Anything-V2-Metric-Indoor-Small-hf"

