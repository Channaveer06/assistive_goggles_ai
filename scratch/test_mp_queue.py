import os
import sys
import time
import numpy as np
import multiprocessing as mp

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def ocr_worker_loop(input_q, output_q):
    """Dedicated background process for EasyOCR: runs in separate OS process with its OWN GIL."""
    # Only import heavy libraries inside the worker process
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
            # Crop is a detached NumPy array
            results = reader.readtext(crop_np)
            detected = [text.strip() for (_, text, prob) in results if prob > 0.5 and len(text.strip()) >= 3]
            dt_ms = (time.perf_counter() - t0) * 1000.0
            output_q.put((detected, dt_ms))
        except Exception as e:
            output_q.put(([], 0.0))

if __name__ == "__main__":
    print("Testing dedicated multiprocessing queue worker...")
    in_q = mp.Queue()
    out_q = mp.Queue()
    p = mp.Process(target=ocr_worker_loop, args=(in_q, out_q), daemon=True)
    p.start()

    # Wait for ready signal
    status, _ = out_q.get(timeout=30)
    print(f"Worker status: {status}")

    # Send a detached crop
    dummy = np.zeros((100, 200, 3), dtype=np.uint8)
    in_q.put((np.ascontiguousarray(dummy.copy()), time.time()))

    res, duration = out_q.get(timeout=10)
    print(f"OCR result: {res} in {duration:.1f}ms")
    p.terminate()
    print("Multiprocessing worker test complete!")

