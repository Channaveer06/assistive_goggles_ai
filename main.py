import argparse
import time
import cv2
import numpy as np
import config
from audio_arbitrator import PriorityAudioArbitrator
from io_bridge import HardwareIOBridge
from modules.reflex_depth import ReflexDepthEngine
from modules.face_social import FaceAndSocialEngine
from modules.episodic_memory import EpisodicObjectMemory
from modules.finger_ocr import FingerOCRAndUtilities
from modules.indoor_nav import IndoorTopologicalNavigator

class AssistiveGogglesBrain:
    def __init__(self):
        print("🧠 Booting Unified Assistive Goggles Brain...")
        self.arbitrator = PriorityAudioArbitrator()
        self.io = HardwareIOBridge(self.arbitrator)
        self.reflex = ReflexDepthEngine(self.arbitrator)
        self.social = FaceAndSocialEngine(self.arbitrator)
        self.memory = EpisodicObjectMemory()
        self.ocr = FingerOCRAndUtilities(self.arbitrator)
        self.navigator = IndoorTopologicalNavigator(self.arbitrator)
        self.current_room = "study_room"
        print("✅ All 6 Independence Modules Online!")

    def handle_voice_command(self, command_text, frame_bgr=None, fingertip_xy=None, timestamp=0.0):
        cmd = command_text.lower().strip()

        if "go to" in cmd or "take me to" in cmd or "bathroom" in cmd or "washroom" in cmd:
            target = "bathroom"
            for room in self.navigator.graph.nodes():
                if room.replace("_", " ") in cmd:
                    target = room
                    break
            _, speech = self.navigator.plan_voice_route(self.current_room, target, timestamp=timestamp)
            return speech

        if "where is" in cmd or "where did i leave" in cmd or "find my" in cmd:
            for token in ["cell phone", "bottle", "backpack", "laptop", "cup", "book", "umbrella", "remote"]:
                if token in cmd:
                    ans = self.memory.query_item(token)
                    self.arbitrator.emit_alert(2, ans, timestamp, key=f"mem_{token}")
                    return ans

        if ("read" in cmd or "medicine" in cmd or "currency" in cmd or "note" in cmd) and frame_bgr is not None:
            _, speech = self.ocr.read_scene_or_fingertip(frame_bgr, fingertip_xy=fingertip_xy, timestamp=timestamp)
            return speech

        if ("light" in cmd or "color" in cmd) and frame_bgr is not None:
            return self.ocr.check_room_light_and_color(frame_bgr, timestamp=timestamp)

        if "save this person as" in cmd and frame_bgr is not None:
            name = command_text.split("as")[-1].strip()
            _, msg = self.social.enroll_known_person(frame_bgr, name)
            self.arbitrator.emit_alert(2, msg, timestamp, key="enroll")
            return msg

        if "sos" in cmd or "guardian" in cmd or "help me" in cmd:
            return self.io.trigger_guardian_sos(self.current_room, timestamp=timestamp)

        return "Command not recognized."

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--env", default="COLAB", choices=["COLAB", "LOCAL"])
    args = parser.parse_args()
    config.ENV_MODE = args.env
    brain = AssistiveGogglesBrain()
