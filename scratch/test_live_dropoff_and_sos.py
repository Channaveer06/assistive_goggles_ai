import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config
config.ENV_MODE = "LOCAL"

from run_live_local import LiveLocalGogglesApp

print("Initializing LiveLocalGogglesApp in test mode...")
app = LiveLocalGogglesApp(camera_idx=0, disable_mic=True, test_mode=True, headless=True)

# Queue 's' in voice_command_queue so frame 2 will process SOS
app.voice_command_queue.put('s')

print("Running live test for 15 frames with queued ['s'] SOS command...")
app.run(max_frames=15)

print("Live test completed.")
