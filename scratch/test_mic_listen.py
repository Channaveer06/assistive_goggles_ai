import time
import speech_recognition as sr

r = sr.Recognizer()
r.energy_threshold = 200
r.dynamic_energy_threshold = True

def callback(recognizer, audio):
    try:
        print("[Thread] Recognizing voice...")
        text = recognizer.recognize_google(audio).lower().strip()
        print(f"🗣️ Recognized: '{text}'")
    except Exception as e:
        print(f"[Thread] Speech not recognized or error: {e}")

mic = sr.Microphone()
with mic as source:
    r.adjust_for_ambient_noise(source, duration=0.8)
print(f"Calibrated energy_threshold: {r.energy_threshold}")

stop_listening = r.listen_in_background(mic, callback, phrase_time_limit=4)
print("Listening in background for 6 seconds. Please speak or wait...")
time.sleep(6)
stop_listening(wait_for_stop=False)
print("Background listener stopped.")
