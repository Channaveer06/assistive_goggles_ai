import os
import sys
import time
import numpy as np
from concurrent.futures import ProcessPoolExecutor

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def worker_test_fn(crop_np):
    # Ensure detached NumPy array
    arr = np.ascontiguousarray(crop_np.copy())
    time.sleep(0.05)
    return arr.shape, "Success"

if __name__ == "__main__":
    print("Testing ProcessPoolExecutor...")
    dummy = np.zeros((100, 100, 3), dtype=np.uint8)
    with ProcessPoolExecutor(max_workers=2) as executor:
        future = executor.submit(worker_test_fn, dummy)
        res = future.result(timeout=5)
        print(f"ProcessPoolExecutor result: {res}")

