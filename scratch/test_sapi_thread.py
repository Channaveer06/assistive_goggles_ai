import threading
import time
import win32com.client
import pythoncom

def worker():
    print("[Thread] Starting SAPI worker...")
    pythoncom.CoInitialize()
    try:
        speaker = win32com.client.Dispatch("SAPI.SpVoice")
        print("[Thread] SAPI.SpVoice initialized successfully.")
        for i in range(3):
            msg = f"Testing message number {i+1} from native SAPI"
            print(f"[Thread] Speaking: '{msg}'...")
            t0 = time.time()
            speaker.Speak(msg)
            print(f"[Thread] Finished speaking message {i+1} in {time.time()-t0:.2f}s")
            time.sleep(0.5)
    except Exception as e:
        print(f"[Thread] Error: {e}")
    finally:
        pythoncom.CoUninitialize()

t = threading.Thread(target=worker)
t.start()
t.join()
print("All done!")

