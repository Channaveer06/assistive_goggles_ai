import os
import sys
import speech_recognition as sr

print("Listing all microphone devices on system:")
for index, name in enumerate(sr.Microphone.list_microphone_names()):
    print(f"[{index}] {name}")

print("\nDefault microphone test:")
r = sr.Recognizer()
try:
    with sr.Microphone() as source:
        print(f"Opened default mic. Initial energy_threshold: {r.energy_threshold}")
        r.adjust_for_ambient_noise(source, duration=1.0)
        print(f"Energy threshold after ambient noise adjustment: {r.energy_threshold}")
        print("Say 'help' or anything now (listening for 5 seconds)...")
        audio = r.listen(source, timeout=5.0, phrase_time_limit=4.0)
        print("Audio captured! Running Google Speech Recognition...")
        text = r.recognize_google(audio)
        print(f"Recognized text: '{text}'")
except Exception as e:
    print(f"Mic test error: {e}")
