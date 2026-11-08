import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config
config.ENV_MODE = "LOCAL"

import pythoncom
import win32com.client

class FastArbitrator:
    def __init__(self):
        pythoncom.CoInitialize()
        self.speaker = win32com.client.Dispatch("SAPI.SpVoice")
        self.speaker.Rate = 1
        self.speaker.Volume = 100
        voices = self.speaker.GetVoices()
        for i in range(voices.Count):
            v = voices.Item(i)
            desc = v.GetDescription().lower()
            if any(k in desc for k in ["heera", "ravi", "india"]):
                self.speaker.Voice = v
                break
        print(f"Active Voice: {self.speaker.Voice.GetDescription()}")

    def speak(self, text, purge=False):
        flags = 2 if purge else 0
        print(f"Speaking: '{text}' (purge={purge})")
        self.speaker.Speak(text, flags)

arb = FastArbitrator()
arb.speak("Assistive Goggles initialized. System active.")
arb.speak("Person moving closer, now 1.8 meters.")
pythoncom.CoUninitialize()
print("All spoken perfectly!")
