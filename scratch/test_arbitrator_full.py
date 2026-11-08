import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config
config.ENV_MODE = "LOCAL"

from audio_arbitrator import PriorityAudioArbitrator

print("Initializing PriorityAudioArbitrator...")
arbitrator = PriorityAudioArbitrator()

print("Emitting alert 1: Boot...")
arbitrator.emit_alert(0, "Assistive Goggles initialized. System active.", time.time(), key="boot")
time.sleep(3.5)

print("Emitting alert 2: Obstacle...")
arbitrator.emit_alert(2, "Chair detected ahead, 2 meters.", time.time(), key="obs_1")
time.sleep(3.0)

print("Emitting alert 3: Approach...")
arbitrator.emit_alert(1, "Person moving closer, now 1.2 meters.", time.time(), key="person_approach")
time.sleep(3.5)

print("Stopping arbitrator...")
arbitrator.stop()
print("All audio alerts played successfully!")

