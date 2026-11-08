import os
import time
import cv2
import numpy as np
import easyocr
import torch

class SpatialOCR:
    """
    Asynchronous Continuous Spatial OCR Engine using EasyOCR.
    Preloads English text recognition models into RAM on boot.
    Extracts text strings from frame crops with confidence score > 0.5.
    """
    def __init__(self, languages=None, gpu=None):
        self.last_inference_time_ms = 0.0
        self.languages = languages if languages is not None else ["en"]
        self.gpu = gpu if gpu is not None else torch.cuda.is_available()
        self.reader = None

    def _ensure_reader(self):
        """Lazily initializes EasyOCR reader only if fallback execution is needed."""
        if self.reader is None:
            try:
                print(f"[SpatialOCR] Lazily initializing EasyOCR fallback engine (gpu={self.gpu})...")
                self.reader = easyocr.Reader(self.languages, gpu=self.gpu, verbose=False)
            except Exception as e:
                print(f"[SpatialOCR] CPU fallback for EasyOCR ({e})...")
                self.reader = easyocr.Reader(self.languages, gpu=False, verbose=False)

    def read_text(self, frame_crop):
        """
        Runs self.reader.readtext(frame_crop) and returns a list of detected
        text strings that have a confidence score > 0.5.
        """
        if frame_crop is None or not isinstance(frame_crop, np.ndarray) or frame_crop.size == 0:
            return []

        self._ensure_reader()
        try:
            t0 = time.perf_counter()
            time.sleep(0.01)  # Forcefully yield Python GIL to main YOLO thread
            results = self.reader.readtext(frame_crop)
            time.sleep(0.01)  # Yield GIL after CPU-heavy inference
            self.last_inference_time_ms = (time.perf_counter() - t0) * 1000.0
            detected_texts = [
                text for (bbox, text, prob) in results
                if prob > 0.5
            ]
            return detected_texts
        except Exception as e:
            print(f"[SpatialOCR] OCR reading error: {e}")
            return []
