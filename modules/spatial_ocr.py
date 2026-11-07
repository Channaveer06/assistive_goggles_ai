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
        if languages is None:
            languages = ["en"]
        if gpu is None:
            gpu = torch.cuda.is_available()

        try:
            print(f"[SpatialOCR] Initializing EasyOCR Spatial Text Engine (gpu={gpu})...")
            self.reader = easyocr.Reader(languages, gpu=gpu, verbose=False)
            print("[SpatialOCR] EasyOCR Reader initialized successfully.")
        except Exception as e:
            print(f"[SpatialOCR] Notice: Initialization with gpu={gpu} failed ({e}). Falling back to CPU...")
            self.reader = easyocr.Reader(languages, gpu=False, verbose=False)
            print("[SpatialOCR] EasyOCR Reader initialized on CPU.")

    def read_text(self, frame_crop):
        """
        Runs self.reader.readtext(frame_crop) and returns a list of detected
        text strings that have a confidence score > 0.5.
        """
        if frame_crop is None or not isinstance(frame_crop, np.ndarray) or frame_crop.size == 0:
            return []

        try:
            t0 = time.perf_counter()
            results = self.reader.readtext(frame_crop)
            self.last_inference_time_ms = (time.perf_counter() - t0) * 1000.0
            detected_texts = [
                text for (bbox, text, prob) in results
                if prob > 0.5
            ]
            return detected_texts
        except Exception as e:
            print(f"[SpatialOCR] OCR reading error: {e}")
            return []
