import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config
config.ENV_MODE = "LOCAL"

from audio_arbitrator import PriorityAudioArbitrator

print("Testing PriorityAudioArbitrator...")
arbitrator = PriorityAudioArbitrator()
print("Initialized arbitrator. Emitting alert...")
arbitrator.emit_alert(1, "Testing priority one alert. Can you hear this?", time.time())
print("Alert emitted. Waiting 5 seconds for speech...")
time.sleep(5)
print("Stopping arbitrator...")
arbitrator.stop()
print("Done test.")
