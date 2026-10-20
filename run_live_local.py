"""
Assistive Smart Goggles AI — Live Local Real-Time Execution Engine
------------------------------------------------------------------
Runs the complete assistive vision & spatial audio pipeline locally on laptop hardware:
1. 30 FPS camera loop (cv2.VideoCapture(0)) with combined RGB + Metric Depth HUD.
2. Continuous ReflexDepthEngine (YOLOv11 + Metric Depth Indoors in meters).
3. Continuous EpisodicObjectMemory (records personal items near room landmarks).
4. Asynchronous FaceAndSocialEngine every 10 frames (InsightFace 512-D + YOLOv11-Pose cues).
5. Unified Person + Face announcement ("Channaveer 1.5 meters straight ahead, looking at you").
6. Non-blocking SAPI.SpVoice / PowerShell zero-lag offline audio arbitration.
7. Smart Stationary-Object Silence (no repetitive announcements for stationary objects).
8. Real Name Face Enrollment with 5-second non-blocking console prompt.
9. Hands-Free Microphone Voice Commands alongside Dual Keyboard Hotkeys (Terminal + HUD):
   [e] Enroll live face in front of webcam into ChromaDB (custom name)
   [r] Fingertip Point-to-Read OCR (Medicine / Currency / Text)
   [f] Find where I last left my cell phone or bottle
   [b] Speak turn-by-turn Indoor Navigation route to the bathroom
   [l] Check Room Light & Center Object Color
   [s] Trigger Guardian SOS
   [c] Call Guardian Two-Way Intercom (Voice Walkie-Talkie)
   [q] Quit cleanly
"""

import os
import sys
import time
import warnings
import argparse
import threading
import queue

# Silence InsightFace numpy lstsq FutureWarning
warnings.filterwarnings("ignore", category=FutureWarning)

try:
    import msvcrt
except ImportError:
    msvcrt = None

try:
    import speech_recognition as sr
except ImportError:
    sr = None

import cv2
import numpy as np

# Ensure local environment mode
import config
config.ENV_MODE = "LOCAL"

from audio_arbitrator import PriorityAudioArbitrator
from io_bridge import HardwareIOBridge
from modules.reflex_depth import ReflexDepthEngine
from modules.face_social import FaceAndSocialEngine
from modules.episodic_memory import EpisodicObjectMemory
from modules.finger_ocr import FingerOCRAndUtilities
from modules.indoor_nav import IndoorTopologicalNavigator


class LiveLocalGogglesApp:
    def __init__(self, camera_idx=0, headless=False, test_mode=False, disable_mic=False):
        self.camera_idx = camera_idx
        self.headless = headless
        self.test_mode = test_mode
        self.disable_mic = disable_mic
        self.current_room = "study_room"
        self.search_target = None
        self.is_running = False
        self.window_name = "Assistive Smart Goggles AI — Live HUD"

        print("=" * 65)
        print("👓 BOOTING ASSISTIVE SMART GOGGLES AI — LOCAL HARDWARE ENGINE")
        print("=" * 65)

        # 1. Initialize Arbitrator with offline non-blocking SAPI.SpVoice / pyttsx3
        self.arbitrator = PriorityAudioArbitrator()

        # 2. Initialize IO Bridge with WebRTC Guardian Live Stream
        self.io = HardwareIOBridge(self.arbitrator, video_source=camera_idx)
        self.guardian = self.io.guardian_server

        # 3. Initialize Core Vision & Independence Modules
        print("\n[1/5] Loading Face & Social Micro-Cue Engine...")
        self.social = FaceAndSocialEngine(self.arbitrator)

        print("\n[2/5] Loading Reflex Depth Engine (YOLOv11 + Metric Depth)...")
        # Connect social engine for Unified Person + Face Announcements
        self.reflex = ReflexDepthEngine(self.arbitrator, social_engine=self.social)

        print("\n[3/5] Loading Episodic Object Memory...")
        self.memory = EpisodicObjectMemory()

        print("\n[4/5] Loading Fingertip OCR, Currency & Lighting Coach...")
        self.ocr = FingerOCRAndUtilities(self.arbitrator)

        print("\n[5/5] Loading Indoor Topological Navigator...")
        self.navigator = IndoorTopologicalNavigator(self.arbitrator)

        # Asynchronous worker for FaceAndSocialEngine (runs every 10 frames)
        self.face_task_queue = queue.Queue(maxsize=1)
        self.face_worker_thread = threading.Thread(target=self._async_face_worker, daemon=True)
        self.face_worker_thread.start()

        # Asynchronous worker for on-demand OCR (so hotkey [r] never freezes camera)
        self.ocr_task_queue = queue.Queue(maxsize=1)
        self.ocr_worker_thread = threading.Thread(target=self._async_ocr_worker, daemon=True)
        self.ocr_worker_thread.start()

        # Hands-free microphone voice listener queue
        self.voice_command_queue = queue.Queue(maxsize=5)
        self.mic_active = False
        if sr is not None and not self.disable_mic and not self.test_mode:
            self._start_voice_listener()

        # Last status announcement to display on HUD
        self.last_hud_message = "Ready. Press [h] for hotkeys or [q] to quit."
        self.last_hud_message_time = time.time()

        print("\n" + "=" * 65)
        print("✅ ALL SYSTEMS ONLINE — ZERO-LAG ASSISTIVE GOGGLES READY")
        print("   HOTKEYS (Work in Terminal & HUD Window):")
        print("   [e] Enroll live face into ChromaDB (name prompt)")
        print("   [r] Fingertip Point-to-Read OCR (Medicine / Currency / Text)")
        print("   [f] Intent-Driven 'Find [Object]' Scan Mode (e.g. 'find cell phone')")
        print("   [b] Turn-by-turn Navigation route to the bathroom")
        print("   [l] Check Room Light & Center Object Color")
        print("   [s] Trigger Guardian Emergency SOS")
        print("   [c] Call Guardian Two-Way Intercom (Voice Walkie-Talkie)")
        print("   [q] Quit cleanly")
        if self.mic_active:
            print("   🎙️ HANDS-FREE VOICE COMMANDS ACTIVE: Speak 'find [item]', 'where is [item]', 'call guardian', 'sos', 'bathroom', 'read', 'light'")
        print("=" * 65 + "\n")

    def _start_voice_listener(self):
        """Starts background daemon thread for hands-free microphone voice recognition."""
        voice_thread = threading.Thread(target=self._voice_listener_worker, daemon=True)
        voice_thread.start()

    def _voice_listener_worker(self):
        """Continuously listens for user voice commands without blocking OpenCV loop."""
        recognizer = sr.Recognizer()
        recognizer.energy_threshold = 300
        recognizer.dynamic_energy_threshold = True
        recognizer.pause_threshold = 0.6

        try:
            mic = sr.Microphone()
            with mic as source:
                recognizer.adjust_for_ambient_noise(source, duration=0.8)
            self.mic_active = True
            print("🎙️ [MIC ONLINE] Laptop microphone listening for voice commands...")
        except Exception as e:
            print(f"⚠️ Microphone notice: {e}. Keyboard hotkeys remain 100% active.")
            return

        while self.is_running:
            try:
                with mic as source:
                    audio = recognizer.listen(source, timeout=3.0, phrase_time_limit=4.0)
                text = recognizer.recognize_google(audio).lower().strip()
                print(f"🗣️ [VOICE DETECTED]: \"{text}\"")

                # Detect search intent commands: "find [item]", "where is [item]", "search for [item]"
                search_item = None
                if "find " in text:
                    search_item = text.split("find ", 1)[1].strip()
                elif "where is " in text:
                    search_item = text.split("where is ", 1)[1].strip()
                elif "search for " in text:
                    search_item = text.split("search for ", 1)[1].strip()

                if search_item:
                    clean_item = search_item.replace("my ", "").replace("the ", "").replace("a ", "").strip(" .?!")
                    if clean_item in ["phone", "cellphone"]:
                        clean_item = "cell phone"
                    self.voice_command_queue.put(('find', clean_item))
                elif any(phrase in text for phrase in ["call guardian", "call channaveer", "call for help", "voice call"]):
                    self.voice_command_queue.put('c')
                elif any(w in text for w in ["bathroom", "washroom", "toilet", "nav"]):
                    self.voice_command_queue.put('b')
                elif any(w in text for w in ["find", "where is", "phone", "bottle", "item"]):
                    self.voice_command_queue.put('f')
                elif any(w in text for w in ["read", "medicine", "currency", "money", "rupee"]):
                    self.voice_command_queue.put('r')
                elif any(w in text for w in ["light", "color", "dark"]):
                    self.voice_command_queue.put('l')
                elif any(w in text for w in ["enroll", "save person", "save face"]):
                    self.voice_command_queue.put('e')
                elif any(w in text for w in ["sos", "help", "emergency", "guardian"]):
                    self.voice_command_queue.put('s')
                elif any(w in text for w in ["quit", "exit", "stop program"]):
                    self.voice_command_queue.put('q')
            except (sr.WaitTimeoutError, sr.UnknownValueError):
                continue
            except Exception:
                time.sleep(0.5)

    def _async_face_worker(self):
        """Processes InsightFace + YOLOv11-Pose asynchronously and feeds unified announcements."""
        while True:
            try:
                frame_bgr, timestamp = self.face_task_queue.get()
                if frame_bgr is None:
                    break

                # emit_audio=False so ReflexDepthEngine handles unified name + distance + cues
                annotated_faces, announcements = self.social.analyze_people(
                    frame_bgr, timestamp=timestamp, emit_audio=False
                )
                if announcements:
                    self._update_hud_status(announcements[-1])

                self.face_task_queue.task_done()
            except Exception as e:
                print(f"Face worker error: {e}")

    def _async_ocr_worker(self):
        """Executes EasyOCR in the background upon hotkey or voice press."""
        while True:
            try:
                frame_bgr, timestamp = self.ocr_task_queue.get()
                if frame_bgr is None:
                    break

                _, speech = self.ocr.read_scene_or_fingertip(frame_bgr, timestamp=timestamp, priority=0)
                self._update_hud_status(speech)
                self.ocr_task_queue.task_done()
            except Exception as e:
                print(f"OCR worker error: {e}")

    def _update_hud_status(self, text):
        self.last_hud_message = text
        self.last_hud_message_time = time.time()

    def prompt_enroll_name(self, timeout=5.0, default="Friend"):
        """Prompts for real name in terminal with a 5-second non-blocking timeout."""
        print("\n" + "=" * 55)
        print(f"👤 [FACE ENROLLMENT] Type person's name & press Enter:")
        print(f"   (Auto-defaulting to '{default}' in {int(timeout)}s if no input)... ", end="", flush=True)

        if msvcrt is None or self.test_mode:
            print(f"\n✅ Enrolling as: '{default}'")
            return default

        start_t = time.time()
        chars = []
        while time.time() - start_t < timeout:
            if msvcrt.kbhit():
                ch = msvcrt.getch()
                if ch in [b'\r', b'\n']:
                    break
                elif ch == b'\x08':  # Backspace
                    if chars:
                        chars.pop()
                        sys.stdout.write("\b \b")
                        sys.stdout.flush()
                else:
                    try:
                        c = ch.decode("utf-8")
                        chars.append(c)
                        sys.stdout.write(c)
                        sys.stdout.flush()
                    except Exception:
                        pass
            time.sleep(0.04)

        entered = "".join(chars).strip()
        chosen_name = entered if entered else default
        print(f"\n✅ Enrolling as: '{chosen_name}'")
        return chosen_name

    def handle_hotkey(self, key_char, current_frame, timestamp):
        """Dispatches keyboard controls and voice triggers with Priority 0 immediate preemption."""
        key_char = key_char.lower().strip()

        if key_char == 'e':
            # [e] Enroll live face in front of webcam with Real Name prompt
            print("📸 [ACTION] Live Face Enrollment Triggered...")
            self.arbitrator.emit_alert(0, "Face enrollment triggered. Please look directly at the camera.", timestamp, key="enroll_hint")
            person_name = self.prompt_enroll_name(timeout=5.0, default="Friend")
            success, msg = self.social.enroll_known_person(current_frame, person_name=person_name)
            self.arbitrator.emit_alert(0, msg, timestamp, key="enroll_result")
            self._update_hud_status(msg)

        elif key_char == 'r':
            # [r] Fingertip Point-to-Read OCR (Medicine / Currency / Text)
            print("📖 [ACTION] Fingertip / Scene OCR Triggered...")
            self.arbitrator.emit_alert(0, "Reading scene text...", timestamp, key="ocr_start")
            if self.ocr_task_queue.empty():
                self.ocr_task_queue.put((current_frame.copy(), timestamp))
            self._update_hud_status("Analyzing text, currency & medicine labels...")

        elif (isinstance(key_char, tuple) and key_char[0] == 'find') or key_char == 'f':
            # Intent-Driven "Find [Object]" Scan Mode
            item = key_char[1] if isinstance(key_char, tuple) else "cell phone"
            print(f"🔍 [ACTION] Intent-Driven Scan Mode Triggered for: {item}...")
            self.search_target = item
            msg = f"Searching for {item}. Please scan the room slowly."
            self.arbitrator.emit_alert(0, msg, timestamp, key="find_mode_start")
            self._update_hud_status(msg)

        elif key_char == 'b':
            # [b] Speak turn-by-turn Indoor Navigation route to the bathroom
            print("🧭 [ACTION] Indoor Navigation Route to Bathroom Triggered...")
            _, speech = self.navigator.plan_voice_route(self.current_room, "bathroom", timestamp=timestamp, priority=0)
            self._update_hud_status(speech)

        elif key_char == 'l':
            # [l] Check Room Light & Center Object Color
            print("💡 [ACTION] Room Light & Center Object Color Check Triggered...")
            report = self.ocr.check_room_light_and_color(current_frame, timestamp=timestamp, priority=0)
            self._update_hud_status(report)

        elif key_char == 's':
            # [s] Trigger Guardian SOS
            print("🚨 [ACTION] Guardian Emergency SOS Triggered!")
            msg = self.io.trigger_guardian_sos(self.current_room, timestamp=timestamp)
            self._update_hud_status(msg)

        elif key_char == 'c':
            # [c] Call Guardian Two-Way Voice Intercom
            print("📞 [ACTION] Call Guardian Voice Request Triggered...")
            msg = self.io.trigger_call_guardian(self.current_room, timestamp=timestamp)
            self._update_hud_status(msg)

    def draw_hud_overlays(self, combined_hud, fps, frame_count):
        """Draws a high-visibility, accessible HUD overlay banner on the video stream."""
        h, w = combined_hud.shape[:2]

        # Top banner for telemetry & room info
        top_bar = np.zeros((46, w, 3), dtype=np.uint8)
        mic_status = "MIC: ON" if self.mic_active else "MIC: OFF"
        sos_status = " | 🚨 SOS / CALL STREAMING" if (getattr(self, "guardian", None) and self.guardian.is_running) else ""
        scan_status = f" | 🔍 SCANNING: {self.search_target.upper()}" if self.search_target else ""
        title = f"ASSISTIVE SMART GOGGLES AI  |  FPS: {fps:04.1f}  |  ROOM: {self.current_room.upper()}  |  {mic_status}{sos_status}{scan_status}"
        cv2.putText(top_bar, title, (16, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 255), 2, cv2.LINE_AA)

        # Bottom banner for hotkeys and recent speech
        bottom_bar = np.zeros((64, w, 3), dtype=np.uint8)
        hotkeys_text = "[e] Enroll  [r] OCR/Medicine  [f] Find  [b] Nav  [l] Light  [s] SOS  [c] Call  [q] Quit"
        cv2.putText(bottom_bar, hotkeys_text, (16, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (180, 180, 180), 1, cv2.LINE_AA)

        # Display latest spoken message
        msg_display = f"SPEECH: {self.last_hud_message}"
        if len(msg_display) > 110:
            msg_display = msg_display[:107] + "..."
        cv2.putText(bottom_bar, msg_display, (16, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 0), 2, cv2.LINE_AA)

        return np.vstack((top_bar, combined_hud, bottom_bar))

    def run(self, max_frames=None):
        """Main real-time 30 FPS camera loop."""
        self.is_running = True

        # Open webcam or mock video source
        opened = self.io.start_camera()
        mock_source = False

        if not opened or self.io.cap is None or not self.io.cap.isOpened():
            print("⚠️ Hardware webcam could not be opened. Using test mock scene for execution.")
            mock_source = True
            base_dir = os.path.dirname(os.path.abspath(__file__))
            test_img_path = os.path.join(base_dir, "test_scene.jpg")
            if os.path.exists(test_img_path):
                mock_frame = cv2.imread(test_img_path)
                mock_frame = cv2.resize(mock_frame, (640, 480))
            else:
                mock_frame = np.full((480, 640, 3), 120, dtype=np.uint8)

        # Initialize and bring OpenCV HUD window to the front
        if not self.headless:
            cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)
            try:
                cv2.setWindowProperty(self.window_name, cv2.WND_PROP_TOPMOST, 1)
            except Exception:
                pass

        frame_count = 0
        start_time = time.time()
        fps_timer = time.time()
        current_fps = 30.0

        print("🚀 Real-Time 30 FPS Processing Loop Active!")
        self.arbitrator.emit_alert(2, "Assistive Goggles initialized. System active.", time.time(), key="boot_alert")

        try:
            while self.is_running:
                now = time.time() - start_time
                frame_count += 1

                # 1. Grab Frame (Webcam or Mock Frame)
                if mock_source:
                    frame = mock_frame.copy()
                    shift_x = int(5 * np.sin(frame_count * 0.1))
                    frame = np.roll(frame, shift_x, axis=1)
                else:
                    ret, frame = self.io.read_frame()
                    if not ret or frame is None:
                        print("⚠️ Camera frame drop. Retrying...")
                        time.sleep(0.01)
                        continue

                # 2. Continuous Reflex & Metric Depth (Inner 50% median meters) with Unified Person Announcements
                combined_hud, speech = self.reflex.process_frame(frame, timestamp=now)
                if speech:
                    self._update_hud_status(speech)

                # Intent-Driven "Find [Object]" Scan Mode Detection
                if self.search_target is not None and hasattr(self.reflex, "last_detections") and self.reflex.last_detections:
                    target_norm = self.search_target.lower().strip()
                    if target_norm in ["phone", "cellphone"]:
                        target_norm = "cell phone"

                    matched_det = None
                    furniture_dets = []

                    for det in self.reflex.last_detections:
                        lbl = det["label"].lower().strip()
                        if lbl in ["dining table", "desk", "chair", "couch", "bed", "table"]:
                            furniture_dets.append(det)
                        if lbl == target_norm or (target_norm in lbl) or (lbl in target_norm):
                            matched_det = det

                    if matched_det is not None:
                        # Determine horizontal orientation: Left (< width/3), Center (width/3 to 2*width/3), Right (> 2*width/3)
                        cx = matched_det["center_x"]
                        w_frame = frame.shape[1]
                        if cx < (w_frame / 3.0):
                            dir_str = "left"
                        elif cx > (2.0 * w_frame / 3.0):
                            dir_str = "right"
                        else:
                            dir_str = "center"

                        # Check vertical overlap against furniture bounding boxes ("dining table", "desk", "chair")
                        tx1, ty1, tx2, ty2 = matched_det.get("box", (cx - 10, matched_det["center_y"] - 10, cx + 10, matched_det["center_y"] + 10))
                        found_furniture = None
                        for f_det in furniture_dets:
                            fx1, fy1, fx2, fy2 = f_det.get("box", (0, 0, 0, 0))
                            overlap_x = max(0, min(tx2, fx2) - max(tx1, fx1))
                            if overlap_x > 0.2 * max(1, tx2 - tx1):
                                if ty2 >= fy1 - 40 and ty1 <= fy2:
                                    found_furniture = f_det["label"].replace("dining table", "table")
                                    break

                        target_depth = matched_det["dist_m"]
                        if found_furniture:
                            found_msg = f"Found {self.search_target} on your {dir_str}, resting on the {found_furniture}, {target_depth:.1f} meters away."
                        else:
                            found_msg = f"Found {self.search_target} on your {dir_str}, {target_depth:.1f} meters away."

                        self.arbitrator.emit_alert(0, found_msg, now, key="found_search_target")
                        self._update_hud_status(found_msg)
                        self.search_target = None

                # 3. Continuous Episodic Object Memory Logging
                if hasattr(self.reflex, "last_detections") and self.reflex.last_detections:
                    self.memory.log_scene_detections(self.reflex.last_detections, current_room=self.current_room)

                # 4. Asynchronous Face & Social Micro-Cues (Every 10 frames)
                if frame_count % 10 == 0:
                    if self.face_task_queue.empty():
                        self.face_task_queue.put((frame.copy(), now))

                # Calculate live FPS
                if frame_count % 15 == 0:
                    elapsed = time.time() - fps_timer
                    if elapsed > 0:
                        current_fps = 15.0 / elapsed
                    fps_timer = time.time()

                # 5. Composite Final Live HUD
                hud_display = self.draw_hud_overlays(combined_hud, current_fps, frame_count)

                # Thread-safely supply live annotated HUD frame to WebRTC Guardian stream
                self.io.update_stream_frame(hud_display)

                # 6. Check for Hotkey / Voice Input (from Microphone, Terminal, and OpenCV Window)
                key_char = None

                # (a) Check Hands-free Microphone Voice Command Queue
                try:
                    key_char = self.voice_command_queue.get_nowait()
                except queue.Empty:
                    pass

                # (b) Check Windows Console Input via msvcrt (non-blocking)
                if key_char is None and msvcrt is not None:
                    while msvcrt.kbhit():
                        raw_ch = msvcrt.getch()
                        try:
                            key_char = raw_ch.decode("utf-8", errors="ignore").lower()
                        except Exception:
                            key_char = None

                # (c) Check OpenCV Window Input (if not headless)
                if not self.headless:
                    cv2.imshow(self.window_name, hud_display)
                    cv_key = cv2.waitKey(1) & 0xFF
                    if cv_key != 255 and key_char is None:
                        key_char = chr(cv_key).lower()

                # (d) Dispatch hotkey
                if key_char:
                    if key_char == 'q':
                        print("🛑 User initiated clean quit via [q] key.")
                        break
                    self.handle_hotkey(key_char, frame, now)

                # Test mode auto-triggers for verification
                if self.test_mode and frame_count in [5, 10, 15, 20, 25, 30, 32]:
                    test_keys = {5: 'f', 10: 'b', 15: 'l', 20: 's', 25: 'r', 30: 'e', 32: 'c'}
                    k = test_keys[frame_count]
                    print(f"\n🧪 [TEST VERIFICATION] Auto-triggering key [{k}] at frame {frame_count}...")
                    self.handle_hotkey(k, frame, now)

                if max_frames is not None and frame_count >= max_frames:
                    print(f"✅ Completed target frame count: {max_frames} frames.")
                    break

        except KeyboardInterrupt:
            print("\n🛑 Program interrupted by user (Ctrl+C).")
        finally:
            self.shutdown()

    def shutdown(self):
        """Releases camera and terminates background threads cleanly."""
        print("Cleaning up resources...")
        self.is_running = False
        self.io.release()

        # Stop face and OCR queues
        try:
            self.face_task_queue.put((None, 0.0))
            self.ocr_task_queue.put((None, 0.0))
        except Exception:
            pass

        # Stop audio arbitrator TTS thread
        self.arbitrator.stop()

        if not self.headless:
            cv2.destroyAllWindows()

        print("✅ Assistive Smart Goggles shutdown cleanly.")


def main():
    parser = argparse.ArgumentParser(description="Assistive Smart Goggles AI — Live Local Hardware Runner")
    parser.add_argument("--camera", type=int, default=0, help="Webcam device index (default: 0)")
    parser.add_argument("--headless", action="store_true", help="Run without cv2.imshow GUI window")
    parser.add_argument("--test", action="store_true", help="Run automated test check with synthetic triggers")
    parser.add_argument("--disable-mic", action="store_true", help="Disable microphone voice listener")
    parser.add_argument("--frames", type=int, default=None, help="Maximum frames to run before auto-exiting")
    args = parser.parse_args()

    app = LiveLocalGogglesApp(
        camera_idx=args.camera,
        headless=args.headless,
        test_mode=args.test,
        disable_mic=args.disable_mic
    )
    app.run(max_frames=args.frames)


if __name__ == "__main__":
    main()
