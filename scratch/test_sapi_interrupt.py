import time
import win32com.client
import pythoncom

pythoncom.CoInitialize()
speaker = win32com.client.Dispatch("SAPI.SpVoice")
# SVSFlagsAsync = 1, SVSFPurgeBeforeSpeak = 2
SVSFlagsAsync = 1
SVSFPurgeBeforeSpeak = 2

print("Speaking long sentence asynchronously...")
speaker.Speak("This is a very long sentence that is intended to describe something in great detail and take many seconds to finish.", SVSFlagsAsync)
time.sleep(1.0)
print("Interrupting with Priority 1 hazard alert!")
speaker.Speak("Stop! Critical hazard ahead!", SVSFPurgeBeforeSpeak | SVSFlagsAsync)

time.sleep(2.5)
pythoncom.CoUninitialize()
print("Done interruption test.")

