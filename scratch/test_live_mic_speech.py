import time
import speech_recognition as sr

r = sr.Recognizer()
r.energy_threshold = 120 # Moderate sensitivity
r.dynamic_energy_threshold = True
r.pause_threshold = 0.5

mic = sr.Microphone()
with mic as source:
    r.adjust_for_ambient_noise(source, duration=0.5)

print(f"Calibrated threshold: {r.energy_threshold}")

def callback(recognizer, audio):
    try:
        text = recognizer.recognize_google(audio).lower().strip()
        print(f"\n[MIC CALLBACK] 🗣️ Detected Speech: '{text}'")
        if any(w in text for w in ["help", "sos", "emergency"]):
            print("🚨 SOS TRIGGER MATCHED!")
    except sr.UnknownValueError:
        pass # Unclear sound / click / breath
    except Exception as e:
        print(f"[MIC ERROR] {e}")

stop_listening = r.listen_in_background(mic, callback, phrase_time_limit=4)
print("Background listener started. Listening continuously for 5 seconds...")
time.sleep(5)
stop_listening(wait_for_stop=False)
print("Stopped.")
