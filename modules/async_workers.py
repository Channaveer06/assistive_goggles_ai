import os
import sys
import time
import numpy as np

# Lazy global model singletons inside each worker process
_worker_deepface_model = None
_worker_easyocr_reader = None

def init_worker():
    """Optional initializer for worker processes."""
    pass

def face_worker_loop(input_q, output_q, db_path="registered_faces"):
    """
    Persistent dedicated worker process for DeepFace facial recognition.
    Runs in a dedicated OS process with its own memory space, completely bypassing the Python GIL.
    """
    try:
        from deepface import DeepFace
    except Exception as e:
        output_q.put(("ERROR", str(e)))
        return

    output_q.put(("READY", 0.0))
    while True:
        try:
            task = input_q.get()
            if task is None:  # Sentinel to terminate gracefully
                break
            track_id, crop_np = task
            t0 = time.perf_counter()
            matched_name = "Unknown"
            valid_exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
            if os.path.exists(db_path):
                has_images = any(
                    os.path.splitext(f)[1].lower() in valid_exts
                    for f in os.listdir(db_path)
                    if os.path.isfile(os.path.join(db_path, f))
                )
                if has_images:
                    results = DeepFace.find(
                        img_path=crop_np,
                        db_path=db_path,
                        enforce_detection=False,
                        silent=True
                    )
                    if results and len(results) > 0 and len(results[0]) > 0:
                        matched_path = results[0]["identity"].iloc[0]
                        filename = os.path.basename(matched_path)
                        raw_name = os.path.splitext(filename)[0]
                        clean_name = raw_name.replace("WIN_", "").replace("_Pro", "").replace("_", " ").strip()
                        digits_cnt = sum(c.isdigit() for c in clean_name)
                        if digits_cnt >= 3 or clean_name.lower().startswith("win") or len(clean_name) < 2:
                            matched_name = "Person"
                        else:
                            matched_name = clean_name.capitalize() if clean_name else "Person"
            duration_ms = (time.perf_counter() - t0) * 1000.0
            output_q.put((track_id, matched_name, duration_ms))
        except Exception:
            try:
                output_q.put((track_id, "Unknown", 0.0))
            except Exception:
                pass


def ocr_worker_loop(input_q, output_q):
    """
    Persistent dedicated worker process for EasyOCR text recognition.
    Runs in a dedicated OS process with its own memory space, completely bypassing the Python GIL.
    """
    try:
        import easyocr
        reader = easyocr.Reader(['en'], gpu=False, verbose=False)
    except Exception as e:
        output_q.put(("ERROR", str(e)))
        return

    output_q.put(("READY", 0.0))
    while True:
        try:
            task = input_q.get()
            if task is None:  # Sentinel to terminate gracefully
                break
            crop_np, timestamp = task
            t0 = time.perf_counter()
            results = reader.readtext(crop_np)
            detected = [
                text.strip() for (_, text, prob) in results
                if prob > 0.5 and len(text.strip()) >= 3
            ]
            duration_ms = (time.perf_counter() - t0) * 1000.0
            output_q.put((detected, duration_ms, timestamp))
        except Exception:
            try:
                output_q.put(([], 0.0, 0.0))
            except Exception:
                pass


def process_recognize_face(crop_np, db_path="registered_faces"):
    """
    Runs DeepFace facial recognition in an isolated process to completely bypass the Python GIL.
    crop_np: cleanly detached NumPy array (np.ascontiguousarray).
    db_path: path to registered faces reference directory.
    Returns: (matched_name: str, duration_ms: float)
    """
    if crop_np is None or not isinstance(crop_np, np.ndarray) or crop_np.size == 0:
        return "Unknown", 0.0

    # Ensure memory is completely detached and contiguous
    crop_clean = np.ascontiguousarray(crop_np.copy())
    t0 = time.perf_counter()
    matched_name = "Unknown"

    try:
        from deepface import DeepFace
        time.sleep(0.005) # Yield GIL

        valid_exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
        if os.path.exists(db_path):
            has_images = any(
                os.path.splitext(f)[1].lower() in valid_exts
                for f in os.listdir(db_path)
                if os.path.isfile(os.path.join(db_path, f))
            )
            if has_images:
                results = DeepFace.find(
                    img_path=crop_clean,
                    db_path=db_path,
                    enforce_detection=False,
                    silent=True
                )
                if results and len(results) > 0 and len(results[0]) > 0:
                    matched_path = results[0]["identity"].iloc[0]
                    filename = os.path.basename(matched_path)
                    raw_name = os.path.splitext(filename)[0]
                    # Strip any Windows camera prefixes if present
                    clean_name = raw_name.replace("WIN_", "").replace("_Pro", "").replace("_", " ").strip()
                    matched_name = clean_name.capitalize() if clean_name else "Friend"
    except Exception as e:
        matched_name = "Unknown"

    duration_ms = (time.perf_counter() - t0) * 1000.0
    return matched_name, duration_ms


def process_read_ocr(crop_np):
    """
    Runs EasyOCR text recognition in an isolated process to completely bypass the Python GIL.
    crop_np: cleanly detached NumPy array (np.ascontiguousarray).
    Returns: (detected_texts: list of str, duration_ms: float)
    """
    global _worker_easyocr_reader

    if crop_np is None or not isinstance(crop_np, np.ndarray) or crop_np.size == 0:
        return [], 0.0

    # Ensure memory is completely detached and contiguous
    crop_clean = np.ascontiguousarray(crop_np.copy())
    t0 = time.perf_counter()
    detected_texts = []

    try:
        if _worker_easyocr_reader is None:
            import easyocr
            _worker_easyocr_reader = easyocr.Reader(["en"], gpu=False, verbose=False)

        time.sleep(0.005) # Yield GIL
        results = _worker_easyocr_reader.readtext(crop_clean)
        detected_texts = [
            text.strip() for (bbox, text, prob) in results
            if prob > 0.5 and len(text.strip()) >= 3
        ]
    except Exception as e:
        detected_texts = []

    duration_ms = (time.perf_counter() - t0) * 1000.0
    return detected_texts, duration_ms

