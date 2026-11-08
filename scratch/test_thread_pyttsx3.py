import threading
import time
import pyttsx3
import pythoncom

def worker():
    print("[Thread] Starting worker...")
    pythoncom.CoInitialize()
    try:
        engine = pyttsx3.init()
        print("[Thread] pyttsx3.init() success")
        for i in range(3):
            msg = f"Test message number {i+1}"
            print(f"[Thread] Saying: '{msg}'...")
            t0 = time.time()
            engine.say(msg)
            engine.runAndWait()
            print(f"[Thread] Done saying message {i+1} in {time.time()-t0:.2f}s")
            time.sleep(1)
    except Exception as e:
        print(f"[Thread] Error: {e}")
    finally:
        pythoncom.CoUninitialize()

t = threading.Thread(target=worker)
t.start()
t.join()
print("All done!")

