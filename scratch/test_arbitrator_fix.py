import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config
config.ENV_MODE = "LOCAL"

import pythoncom
import win32com.client

print("Testing direct SAPI in arbitrator pattern...")
pythoncom.CoInitialize()
speaker = win32com.client.Dispatch("SAPI.SpVoice")
speaker.Rate = 1
speaker.Volume = 100

for i in range(3):
    text = f"Assistive Goggles sound check message {i+1}."
    print(f"Speaking: '{text}'...")
    t0 = time.time()
    speaker.Speak(text, 0) # Synchronous speak
    print(f"Finished in {time.time()-t0:.2f}s")

pythoncom.CoUninitialize()
print("All messages spoken successfully!")
