import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config
config.ENV_MODE = "LOCAL"

from audio_arbitrator import PriorityAudioArbitrator
from io_bridge import HardwareIOBridge

print("Initializing test arbitrator & IO bridge...")
arbitrator = PriorityAudioArbitrator()
io = HardwareIOBridge(arbitrator, video_source=0)

print("Triggering SOS...")
msg = io.trigger_guardian_sos(current_room="living_room", timestamp=time.time())
print(f"Triggered SOS result: {msg}")

print("Waiting 5 seconds for background WebRTC server & Telegram dispatch...")
time.sleep(5)

print("Releasing IO...")
io.release()
arbitrator.stop()
print("Test completed.")
