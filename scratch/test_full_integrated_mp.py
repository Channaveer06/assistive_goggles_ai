import os
import sys
import time
import numpy as np
import multiprocessing as mp
import threading
import queue
import cv2
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from ultralytics import YOLO

# Limit PyTorch OpenMP threads
torch.set_num_threads(2)

def face_worker_loop(input_q, output_q, db_path="registered_faces"):
    """Dedicated background process for DeepFace with its own GIL."""
    from deepface import DeepFace
    output_q.put(("READY", 0.0))
    while True:
        try:
            task = input_q.get()
            if task is None:
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
                        raw_name = os.path.splitext(os.path.basename(matched_path))[0]
                        clean_name = raw_name.replace("WIN_", "").replace("_Pro", "").replace("_", " ").strip()
                        matched_name = clean_name.capitalize() if clean_name else "Friend"
            output_q.put((track_id, matched_name, (time.perf_counter() - t0) * 1000.0))
        except Exception:
            output_q.put((track_id, "Unknown", 0.0))

def ocr_worker_loop(input_q, output_q):
    """Dedicated background process for EasyOCR with its own GIL."""
    import easyocr
    reader = easyocr.Reader(['en'], gpu=False, verbose=False)
    output_q.put(("READY", 0.0))
    while True:
        try:
            task = input_q.get()
            if task is None:
                break
            crop_np, timestamp = task
            t0 = time.perf_counter()
            results = reader.readtext(crop_np)
            detected = [text.strip() for (_, text, prob) in results if prob > 0.5 and len(text.strip()) >= 3]
            output_q.put((detected, (time.perf_counter() - t0) * 1000.0))
        except Exception:
            output_q.put(([], 0.0))

if __name__ == "__main__":
    print("Testing Full Integrated Multiprocessing Architecture...")

    # Start workers
    face_in_q, face_out_q = mp.Queue(), mp.Queue()
    ocr_in_q, ocr_out_q = mp.Queue(), mp.Queue()

    p_face = mp.Process(target=face_worker_loop, args=(face_in_q, face_out_q, "registered_faces"), daemon=True)
    p_ocr = mp.Process(target=ocr_worker_loop, args=(ocr_in_q, ocr_out_q), daemon=True)
    p_face.start()
    p_ocr.start()

    # Wait for workers to be ready
    face_ready, _ = face_out_q.get(timeout=30)
    ocr_ready, _ = ocr_out_q.get(timeout=30)
    print(f"Workers Ready: Face={face_ready}, OCR={ocr_ready}")

    detector = YOLO("yolo11n.pt")
    dummy_frame = np.zeros((480, 640, 3), dtype=np.uint8)
    detector.track(dummy_frame, imgsz=224, persist=True, tracker="bytetrack.yaml", verbose=False)

    fps_list = []
    current_fps = 30.0

    # Simulate 50 frames while sending continuous background tasks
    for i in range(50):
        t0 = time.perf_counter()

        # Send background tasks at intervals
        if i == 5:
            print(f"[Frame {i}] Submitting face recognition to separate process...")
            crop = np.ascontiguousarray(dummy_frame[:120, :120].copy())
            face_in_q.put((1, crop))

        if i % 15 == 0:
            print(f"[Frame {i}] Submitting OCR scan to separate process...")
            crop = np.ascontiguousarray(dummy_frame[100:300, 100:400].copy())
            ocr_in_q.put((crop, time.time()))

        # Non-blocking check for completed worker results
        try:
            tid, name, d_ms = face_out_q.get_nowait()
            print(f"🎉 [Result] Face Track #{tid} recognized as '{name}' in {d_ms:.0f}ms")
        except queue.Empty:
            pass

        try:
            detected, ocr_ms = ocr_out_q.get_nowait()
            print(f"📖 [Result] OCR Detected: {detected} in {ocr_ms:.0f}ms")
        except queue.Empty:
            pass

        # Main thread YOLO detection
        res = detector.track(dummy_frame, imgsz=224, persist=True, tracker="bytetrack.yaml", verbose=False)[0]

        loop_dt = time.perf_counter() - t0
        instant_fps = min(30.0, 1.0 / loop_dt) if loop_dt > 0 else 30.0
        current_fps = 0.85 * current_fps + 0.15 * instant_fps
        fps_list.append(current_fps)

        # 30 FPS pacing
        pacing = max(0.001, 0.0333 - loop_dt)
        time.sleep(pacing)

    p_face.terminate()
    p_ocr.terminate()

    avg_fps = np.mean(fps_list[5:])
    print(f"\nFinal Average Main Thread FPS: {avg_fps:.1f}")
    assert avg_fps >= 25.0, f"Average FPS too low: {avg_fps}"
    print("✅ Full Architecture Verified: 25+ FPS Guaranteed!")
