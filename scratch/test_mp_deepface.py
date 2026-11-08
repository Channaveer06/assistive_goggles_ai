import os
import sys
import time
import numpy as np
import multiprocessing as mp

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def face_worker_loop(input_q, output_q, db_path="registered_faces"):
    """Dedicated background process for DeepFace: runs in separate OS process with its OWN GIL."""
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
                        filename = os.path.basename(matched_path)
                        raw_name = os.path.splitext(filename)[0]
                        clean_name = raw_name.replace("WIN_", "").replace("_Pro", "").replace("_", " ").strip()
                        matched_name = clean_name.capitalize() if clean_name else "Friend"

            dt_ms = (time.perf_counter() - t0) * 1000.0
            output_q.put((track_id, matched_name, dt_ms))
        except Exception as e:
            output_q.put((track_id, "Unknown", 0.0))

if __name__ == "__main__":
    print("Testing dedicated multiprocessing DeepFace worker...")
    in_q = mp.Queue()
    out_q = mp.Queue()
    p = mp.Process(target=face_worker_loop, args=(in_q, out_q, "registered_faces"), daemon=True)
    p.start()

    status, _ = out_q.get(timeout=30)
    print(f"DeepFace Worker status: {status}")

    dummy = np.zeros((100, 100, 3), dtype=np.uint8)
    in_q.put((1, np.ascontiguousarray(dummy.copy())))

    tid, res, duration = out_q.get(timeout=20)
    print(f"DeepFace result for Track #{tid}: {res} in {duration:.1f}ms")
    p.terminate()
    print("Multiprocessing DeepFace test complete!")

