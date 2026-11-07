import os
import sys
import time
import cv2
import numpy as np
from deepface import DeepFace

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_DB_PATH = os.path.join(BASE_DIR, "registered_faces")

def _safe_print(msg):
    """Safely prints messages to standard output without crashing on Windows cp1252 consoles."""
    try:
        print(msg)
    except UnicodeEncodeError:
        try:
            print(msg.encode("ascii", errors="replace").decode("ascii"))
        except Exception:
            pass

class FaceRecognizer:
    """
    Asynchronous Local Facial Recognition using DeepFace.
    Pre-warms face recognition models on boot for zero-lag CPU/GPU inference.
    Matches person crops against local image references in registered_faces/.
    """
    def __init__(self, db_path=None):
        self.db_path = db_path if db_path is not None else DEFAULT_DB_PATH
        self.last_inference_time_ms = 0.0
        os.makedirs(self.db_path, exist_ok=True)

        _safe_print("[FaceRecognizer] Initializing DeepFace local facial recognition engine...")
        try:
            # Build and load VGG-Face model into RAM on boot
            self.model = DeepFace.build_model("VGG-Face")
            _safe_print("[FaceRecognizer] DeepFace model preloaded into RAM.")
        except Exception as e:
            _safe_print(f"[FaceRecognizer] Model preload notice: {e}")
            self.model = None

    def identify_face(self, frame_crop):
        """
        Runs DeepFace.find to match frame_crop against registered_faces/.
        Returns the recognized name (extracted from filename and capitalized) or 'Unknown'.
        """
        if frame_crop is None or not isinstance(frame_crop, np.ndarray) or frame_crop.size == 0:
            return "Unknown"

        try:
            valid_exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
            has_images = any(
                os.path.splitext(f)[1].lower() in valid_exts
                for f in os.listdir(self.db_path)
                if os.path.isfile(os.path.join(self.db_path, f))
            )
            if not has_images:
                return "Unknown"

            t_start = time.perf_counter()
            results = DeepFace.find(
                img_path=frame_crop,
                db_path=self.db_path,
                enforce_detection=False,
                silent=True
            )
            self.last_inference_time_ms = (time.perf_counter() - t_start) * 1000.0

            if results and len(results) > 0 and len(results[0]) > 0:
                matched_path = results[0]["identity"].iloc[0]
                filename = os.path.basename(matched_path)
                name = os.path.splitext(filename)[0].capitalize()
                return name
        except Exception:
            pass

        return "Unknown"
