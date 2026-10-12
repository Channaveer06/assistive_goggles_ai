import cv2
import time
import config

class HardwareIOBridge:
    """
    Unified I/O Bridge inspired by SightlineAI & OpenGlass.
    - In COLAB mode: Reads uploaded POV videos or test frames and outputs browser audio.
    - In LOCAL mode: Connects directly to USB/Laptop Webcam (cv2.VideoCapture(0)) and offline TTS speaker.
    """
    def __init__(self, arbitrator, video_source=0):
        self.arbitrator = arbitrator
        self.video_source = video_source
        self.cap = None
        self.sos_active = False

    def start_camera(self):
        if config.ENV_MODE == "LOCAL" or isinstance(self.video_source, str):
            self.cap = cv2.VideoCapture(self.video_source)
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
            self.cap.set(cv2.CAP_PROP_FPS, 30)
            return self.cap.isOpened()
        return True

    def read_frame(self):
        if self.cap is not None and self.cap.isOpened():
            return self.cap.read()
        return False, None

    def trigger_guardian_sos(self, current_room="unknown_location", timestamp=0.0):
        """One-Command Emergency Fallback: Alerts family guardian with current room node & live stream link."""
        self.sos_active = True
        msg = (
            f"Emergency Guardian SOS activated. Sharing live camera stream and current location "
            f"({current_room.replace('_', ' ')}) with your registered guardian."
        )
        self.arbitrator.emit_alert(1, msg, timestamp, key="sos_alert")
        return msg

    def release(self):
        if self.cap is not None:
            self.cap.release()
