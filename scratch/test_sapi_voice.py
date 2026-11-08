import win32com.client
import pythoncom

pythoncom.CoInitialize()
speaker = win32com.client.Dispatch("SAPI.SpVoice")
# Set rate (-10 to 10)
speaker.Rate = 1 # fast conversational
speaker.Volume = 100

voices = speaker.GetVoices()
print(f"Total installed SAPI voices: {voices.Count}")
selected_idx = 0
for i in range(voices.Count):
    v = voices.Item(i)
    name = v.GetDescription()
    print(f"Voice {i}: {name}")
    if any(k in name.lower() for k in ["heera", "ravi", "india"]):
        selected_idx = i

speaker.Voice = voices.Item(selected_idx)
print(f"Selected voice: {voices.Item(selected_idx).GetDescription()}")
speaker.Speak("Hello, this is the Assistive Goggles voice engine.")
pythoncom.CoUninitialize()
print("Voice test completed successfully!")

