import os
import sys
import time
import cv2
import numpy as np
from concurrent.futures import ProcessPoolExecutor

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from modules.async_workers import process_recognize_face, process_read_ocr

if __name__ == "__main__":
    print("Testing Multiprocessing AI Workers...")

    # Create dummy frame with synthetic text
    dummy_ocr = np.full((200, 400, 3), 255, dtype=np.uint8)
    cv2.putText(dummy_ocr, "EMERGENCY", (20, 100), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 0), 3)

    dummy_face = np.full((120, 120, 3), 150, dtype=np.uint8)

    with ProcessPoolExecutor(max_workers=2) as executor:
        f_ocr = executor.submit(process_read_ocr, np.ascontiguousarray(dummy_ocr.copy()))
        f_face = executor.submit(process_recognize_face, np.ascontiguousarray(dummy_face.copy()), "registered_faces")

        ocr_res, ocr_time = f_ocr.result(timeout=20)
        face_res, face_time = f_face.result(timeout=20)

        print(f"OCR Result: {ocr_res} in {ocr_time:.1f}ms")
        print(f"Face Result: {face_res} in {face_time:.1f}ms")

        assert isinstance(ocr_res, list)
        assert isinstance(face_res, str)
        print("✅ Multiprocessing Workers Executed Successfully!")

