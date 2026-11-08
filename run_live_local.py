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
import logging
import warnings
import argparse
import threading
import queue

# Silence InsightFace numpy lstsq FutureWarning
warnings.filterwarnings("ignore", category=FutureWarning)

LOG_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "Execution_Logs.txt")

def setup_dual_logger(log_file=LOG_FILE):
    """
    Dual Logging System:
    Outputs all thread activity simultaneously to:
    1. Terminal console (sys.stdout) - for screen recording visibility
    2. Execution_Logs.txt - saved text file for benchmarking and technical review
    """
    logger = logging.getLogger("AssistiveGoggles")
    logger.setLevel(logging.INFO)
    logger.propagate = False

    if logger.hasHandlers():
        logger.handlers.clear()

    formatter = logging.Formatter(
        fmt="%(asctime)s.%(msecs)03d [%(levelname)s] %(message)s",
        datefmt="%H:%M:%S"
    )

    # 1. Console Stream Handler (stdout)
    c_handler = logging.StreamHandler(sys.stdout)
    c_handler.setLevel(logging.INFO)
    c_handler.setFormatter(formatter)
    logger.addHandler(c_handler)

    # 2. File Handler (Execution_Logs.txt)
    f_handler = logging.FileHandler(log_file, mode="a", encoding="utf-8")
    f_handler.setLevel(logging.INFO)
    f_handler.setFormatter(formatter)
    logger.addHandler(f_handler)

    return logger

logger = setup_dual_logger()

try:
    import msvcrt
except ImportError:
    msvcrt = None

try:
    import ctypes
    ctypes.windll.winmm.timeBeginPeriod(1)
except Exception:
    pass

try:
    import speech_recognition as sr
except ImportError:
    sr = None

import cv2
import numpy as np
import multiprocessing as mp
import torch

# Limit PyTorch OpenMP worker threads to prevent CPU core saturation
torch.set_num_threads(2)

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
from modules.face_ident import FaceRecognizer
from modules.spatial_ocr import SpatialOCR
from modules.async_workers import (
    face_worker_loop,
    ocr_worker_loop,
    process_recognize_face,
    process_read_ocr
)


class LiveLocalGogglesApp:
    def __init__(self, camera_idx=0, headless=False, test_mode=False, disable_mic=False):
        self.camera_idx = camera_idx
        self.headless = headless
        self.test_mode = test_mode
        self.disable_mic = disable_mic
        self.current_room = "study_room"
        self.search_target = None
        self.memory_query = None
        self.episodic_memory = {}
        self.last_dropoff_warning = -999.0
        self.last_memory_log_time = -999.0
        self.dropoff_active = False
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
        print("\n[DeepFace] Initializing Local Facial Recognition Engine...")
        self.face_recognizer = FaceRecognizer()
        self.track_identities = {}

        print("\n[1/5] Loading Face & Social Micro-Cue Engine...")
        self.social = FaceAndSocialEngine(self.arbitrator)

        print("\n[2/5] Loading Reflex Depth Engine (YOLOv11 + Metric Depth)...")
        # Connect social engine and track identities for Unified Person + Face Announcements
        self.reflex = ReflexDepthEngine(
            self.arbitrator,
            social_engine=self.social,
            track_identities=self.track_identities,
            async_recognize_cb=self._async_recognize
        )
        self.yolo_model_name = getattr(self.reflex, "model_name", "YOLOv8n")

        print("\n[3/5] Loading Episodic Object Memory...")
        self.memory = EpisodicObjectMemory()

        print("\n[4/5] Loading Fingertip OCR, Currency & Lighting Coach...")
        self.ocr = FingerOCRAndUtilities(self.arbitrator)

        print("\n[5/5] Loading Indoor Topological Navigator...")
        self.navigator = IndoorTopologicalNavigator(self.arbitrator)

        # Asynchronous worker for FaceAndSocialEngine (runs when person detected, throttled to 2.5s)
        self.face_task_queue = queue.Queue(maxsize=1)
        self.last_social_scan_time = 0.0
        self.face_worker_thread = threading.Thread(target=self._async_face_worker, daemon=True)
        self.face_worker_thread.start()

        # Asynchronous worker for on-demand OCR (so hotkey [r] never freezes camera)
        self.ocr_task_queue = queue.Queue(maxsize=1)
        self.ocr_worker_thread = threading.Thread(target=self._async_ondemand_ocr_worker, daemon=True)
        self.ocr_worker_thread.start()

        # Asynchronous Continuous Spatial OCR (EasyOCR loaded on boot)
        print("\n[EasyOCR] Initializing Continuous Spatial OCR Engine...")
        self.ocr_reader = SpatialOCR()
        self.ocr_cooldowns = {}
        self.ocr_thread_active = False
        self.last_ocr_scan_time = 0.0
        self.ocr_scan_interval = 1.5  # Throttle continuous OCR to 1.5s interval to avoid CPU starvation

        # Persistent Multiprocessing Queues & Workers to completely bypass Python GIL
        self.face_in_q = mp.Queue()
        self.face_out_q = mp.Queue()
        self.ocr_in_q = mp.Queue()
        self.ocr_out_q = mp.Queue()
        self.mp_active = False

        try:
            self.p_face = mp.Process(
                target=face_worker_loop,
                args=(self.face_in_q, self.face_out_q, "registered_faces"),
                daemon=True
            )
            self.p_ocr = mp.Process(
                target=ocr_worker_loop,
                args=(self.ocr_in_q, self.ocr_out_q),
                daemon=True
            )
            self.p_face.start()
            self.p_ocr.start()
            self.mp_active = True
            logger.info("[Architecture] Persistent Multiprocessing Workers launched (DeepFace & EasyOCR running in isolated memory spaces to bypass GIL).")
        except Exception as e:
            logger.warning(f"[Architecture] Multiprocessing initialization notice: {e}. Using GIL-yielding threading fallback.")
            self.mp_active = False

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
        print("   [m] Retrospective Visual Memory Recall (e.g. 'where did I leave my cell phone')")
        print("   [b] Turn-by-turn Navigation route to the bathroom")
        print("   [l] Check Room Light & Center Object Color")
        print("   [s] Trigger Guardian Emergency SOS")
        print("   [c] Call Guardian Two-Way Intercom (Voice Walkie-Talkie)")
        print("   [q] Quit cleanly")
        if self.mic_active:
            print("   🎙️ HANDS-FREE VOICE COMMANDS ACTIVE: Speak 'where did I leave my [item]', 'find [item]', 'call guardian', 'sos', 'bathroom', 'read', 'light'")
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

                # 1. Detect Retrospective Memory Queries: "where did i leave my [item]", "where did i put my [item]"
                memory_triggers = [
                    "where did i leave my ",
                    "where did i leave ",
                    "where did i put my ",
                    "where did i put ",
                    "where did i place my ",
                    "where did i place ",
                    "where did i keep my ",
                    "where did i keep "
                ]
                mem_item = None
                for trigger in memory_triggers:
                    if trigger in text:
                        mem_item = text.split(trigger, 1)[1].strip()
                        break

                if mem_item:
                    clean_item = mem_item.replace("my ", "").replace("the ", "").replace("a ", "").strip(" .?!")
                    if clean_item in ["phone", "cellphone"]:
                        clean_item = "cell phone"
                    self.voice_command_queue.put(('memory_query', clean_item))
                # 2. Detect search intent commands: "find [item]", "where is [item]", "search for [item]"
                elif any(p in text for p in ["find ", "where is ", "search for "]):
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
                elif any(phrase in text for phrase in ["where did i leave", "where did i put", "recall", "last seen"]):
                    self.voice_command_queue.put('m')
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

    def _async_recognize(self, track_id, crop):
        """Asynchronously runs DeepFace face recognition on detached crop bypassing the GIL."""
        logger.info(f"[Daemon Thread] DeepFace Processing: Starting recognition for Track #{track_id}...")
        t_start = time.perf_counter()
        crop_detached = np.ascontiguousarray(crop.copy())

        # 1. Primary: Persistent Multiprocessing Worker Process (zero GIL contention)
        if getattr(self, "mp_active", False):
            try:
                self.face_in_q.put((track_id, crop_detached))
                return
            except Exception as e:
                logger.error(f"[Daemon Thread] Multiprocessing queue error ({e}), using fallback...")

        # 2. Fallback: Background thread with explicit GIL yielding
        def _fallback_face():
            try:
                time.sleep(0.01) # Forcefully yield GIL to main YOLO thread
                matched_name = self.face_recognizer.identify_face(crop_detached)
                inference_ms = (time.perf_counter() - t_start) * 1000.0
                if matched_name and matched_name != "Unknown":
                    self.track_identities[track_id] = matched_name
                    logger.info(f"[Daemon Thread] DeepFace Inference Time: {inference_ms:.0f}ms | Recognized: '{matched_name}' (Track #{track_id})")
                else:
                    self.track_identities[track_id] = "Unknown person"
                    logger.info(f"[Daemon Thread] DeepFace Inference Time: {inference_ms:.0f}ms | Result: No match found (Track #{track_id})")
            except Exception as e:
                inference_ms = (time.perf_counter() - t_start) * 1000.0
                self.track_identities[track_id] = "Unknown person"
                logger.error(f"[Daemon Thread] DeepFace Inference Time: {inference_ms:.0f}ms | Error: {e}")

        threading.Thread(target=_fallback_face, daemon=True).start()

    def _async_face_worker(self):
        """Processes InsightFace + YOLOv11-Pose asynchronously in background daemon thread with GIL yielding."""
        while True:
            try:
                frame_bgr, timestamp = self.face_task_queue.get()
                if frame_bgr is None:
                    break

                logger.info("[Daemon Thread] InsightFace & Social Engine: Analyzing facial micro-cues...")
                time.sleep(0.01) # Forcefully yield GIL to main YOLO thread
                orig_threads = torch.get_num_threads()
                torch.set_num_threads(1)
                t_start = time.perf_counter()
                annotated_faces, announcements = self.social.analyze_people(
                    frame_bgr, timestamp=timestamp, emit_audio=False
                )
                inference_ms = (time.perf_counter() - t_start) * 1000.0
                torch.set_num_threads(orig_threads)
                logger.info(f"[Daemon Thread] InsightFace Inference Time: {inference_ms:.0f}ms | Cues: {len(announcements)}")
                if announcements:
                    self._update_hud_status(announcements[-1])

                self.face_task_queue.task_done()
                time.sleep(0.05) # Yield GIL
            except Exception as e:
                logger.error(f"[Daemon Thread] Face worker error: {e}")

    def _async_ondemand_ocr_worker(self):
        """Executes EasyOCR in the background upon hotkey [r] or voice command with GIL yielding."""
        while True:
            try:
                frame_bgr, timestamp = self.ocr_task_queue.get()
                if frame_bgr is None:
                    break

                logger.info("[Daemon Thread] EasyOCR Processing: On-Demand Fingertip/Scene OCR triggered...")
                time.sleep(0.01) # Yield GIL
                t_start = time.perf_counter()
                _, speech = self.ocr.read_scene_or_fingertip(frame_bgr, timestamp=timestamp, priority=0)
                inference_ms = (time.perf_counter() - t_start) * 1000.0
                logger.info(f"[Daemon Thread] EasyOCR On-Demand Inference Time: {inference_ms:.0f}ms | Result: '{speech}'")
                self._update_hud_status(speech)
                self.ocr_task_queue.task_done()
            except Exception as e:
                logger.error(f"[Daemon Thread] On-demand OCR worker error: {e}")

    def _async_ocr_worker(self, center_crop):
        """Asynchronously runs EasyOCR on center 60% spatial frame crop bypassing the GIL."""
        self.ocr_thread_active = True
        logger.info("[Daemon Thread] EasyOCR Processing: Scanning spatial frame crop...")
        t_start = time.perf_counter()
        crop_detached = np.ascontiguousarray(center_crop.copy())

        # 1. Primary: Persistent Multiprocessing Worker Process (zero GIL contention)
        if getattr(self, "mp_active", False):
            try:
                self.ocr_in_q.put((crop_detached, time.time()))
                return
            except Exception as e:
                logger.error(f"[Daemon Thread] Multiprocessing queue error ({e}), using fallback...")

        # 2. Fallback: Background thread with explicit GIL yielding
        def _fallback_ocr():
            try:
                time.sleep(0.01) # Forcefully yield GIL to main YOLO thread
                detected_texts = self.ocr_reader.read_text(crop_detached)
                inference_ms = (time.perf_counter() - t_start) * 1000.0
                self._process_detected_texts(detected_texts, inference_ms)
            except Exception as e:
                inference_ms = (time.perf_counter() - t_start) * 1000.0
                logger.error(f"[Daemon Thread] EasyOCR Inference Error ({inference_ms:.0f}ms): {e}")
            finally:
                self.ocr_thread_active = False

        threading.Thread(target=_fallback_ocr, daemon=True).start()

    def _process_detected_texts(self, detected_texts, inference_ms):
        """Processes OCR results and emits Priority 2 alerts with 20.0s anti-repeat cooldown."""
        current_time = time.time()
        if detected_texts:
            logger.info(f"[Daemon Thread] EasyOCR Inference Time: {inference_ms:.0f}ms | Detected: {detected_texts}")
        else:
            logger.info(f"[Daemon Thread] EasyOCR Inference Time: {inference_ms:.0f}ms | No text detected")

        for raw_text in detected_texts:
            clean_text = raw_text.strip()
            if len(clean_text) < 3:
                continue
            last_seen = self.ocr_cooldowns.get(clean_text, 0.0)
            if (current_time - last_seen) >= 20.0:
                self.ocr_cooldowns[clean_text] = current_time
                alert_msg = f"Text detected: {clean_text}"
                self.arbitrator.emit_alert(2, alert_msg, current_time, key=f"spatial_ocr_{clean_text.lower()}")
                self._update_hud_status(alert_msg)

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

        elif (isinstance(key_char, tuple) and key_char[0] == 'memory_query') or key_char == 'm':
            # Retrospective Visual Memory Recall Query
            item = key_char[1] if isinstance(key_char, tuple) else "cell phone"
            print(f"🧠 [ACTION] Retrospective Visual Memory Query Triggered for: {item}...")
            self.memory_query = item
            self._update_hud_status(f"Recalling memory of {item}...")

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
        recall_status = f" | 🧠 RECALL: {self.memory_query.upper()}" if self.memory_query else ""
        dropoff_status = " | [!] DROP-OFF DETECTED" if self.dropoff_active else ""
        title = f"ASSISTIVE SMART GOGGLES AI  |  FPS: {fps:04.1f}  |  ROOM: {self.current_room.upper()}  |  {mic_status}{sos_status}{scan_status}{recall_status}{dropoff_status}"
        title_color = (0, 0, 255) if self.dropoff_active else (0, 255, 255)
        cv2.putText(top_bar, title, (16, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.65, title_color, 2, cv2.LINE_AA)

        # Bottom banner for hotkeys and recent speech
        bottom_bar = np.zeros((64, w, 3), dtype=np.uint8)
        hotkeys_text = "[e] Enroll  [r] OCR  [f] Find  [m] Recall  [b] Nav  [l] Light  [s] SOS  [c] Call  [q] Quit"
        cv2.putText(bottom_bar, hotkeys_text, (16, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (180, 180, 180), 1, cv2.LINE_AA)

        # Display latest spoken message
        msg_display = f"SPEECH: {self.last_hud_message}"
        if len(msg_display) > 110:
            msg_display = msg_display[:107] + "..."
        cv2.putText(bottom_bar, msg_display, (16, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 0), 2, cv2.LINE_AA)

        composite = np.vstack((top_bar, combined_hud, bottom_bar))

        # Live Running FPS Overlay in Top-Right Corner (cv2.putText badge for demo recordings)
        fps_text = f"FPS: {fps:04.1f}"
        badge_w = 145
        badge_h = 32
        x1 = max(10, w - badge_w - 12)
        y1 = 7
        x2 = min(w - 12, x1 + badge_w)
        y2 = y1 + badge_h
        cv2.rectangle(composite, (x1, y1), (x2, y2), (20, 20, 20), -1)
        cv2.rectangle(composite, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(composite, fps_text, (x1 + 10, y1 + 23), cv2.FONT_HERSHEY_SIMPLEX, 0.72, (0, 255, 0), 2, cv2.LINE_AA)

        return composite

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
        fps_last_t = time.perf_counter()
        current_fps = 30.0

        logger.info("=" * 65)
        logger.info("🚀 Real-Time 30 FPS Processing Loop Active!")
        logger.info("   Dual Logging Active: Console (stdout) & Execution_Logs.txt")
        logger.info("=" * 65)
        self.arbitrator.emit_alert(2, "Assistive Goggles initialized. System active.", time.time(), key="boot_alert")

        # Warm up YOLO detector to eliminate frame-1 initialization spike
        try:
            dummy_warm = np.zeros((480, 640, 3), dtype=np.uint8)
            self.reflex.detector.track(dummy_warm, imgsz=192, persist=True, tracker="bytetrack.yaml", verbose=False)
        except Exception:
            pass

        try:
            while self.is_running:
                loop_start_t = time.perf_counter()
                now = time.time() - start_time
                frame_count += 1

                # Non-blocking poll for persistent multiprocessing worker results
                if getattr(self, "mp_active", False):
                    while True:
                        try:
                            res = self.face_out_q.get_nowait()
                            if res[0] == "READY":
                                logger.info("[Daemon Thread] DeepFace Worker Process is READY.")
                                continue
                            elif res[0] == "ERROR":
                                logger.error(f"[Daemon Thread] DeepFace Worker Process error: {res[1]}")
                                continue
                            tid, matched_name, duration_ms = res
                            if matched_name and matched_name != "Unknown":
                                self.track_identities[tid] = matched_name
                                logger.info(f"[Daemon Thread] DeepFace Inference Time: {duration_ms:.0f}ms | Recognized: '{matched_name}' (Track #{tid})")
                            else:
                                self.track_identities[tid] = "Unknown person"
                                logger.info(f"[Daemon Thread] DeepFace Inference Time: {duration_ms:.0f}ms | Result: No match found (Track #{tid})")
                        except queue.Empty:
                            break

                    while True:
                        try:
                            res = self.ocr_out_q.get_nowait()
                            if res[0] == "READY":
                                logger.info("[Daemon Thread] EasyOCR Worker Process is READY.")
                                continue
                            elif res[0] == "ERROR":
                                logger.error(f"[Daemon Thread] EasyOCR Worker Process error: {res[1]}")
                                self.ocr_thread_active = False
                                continue
                            detected_texts, duration_ms, timestamp = res
                            self._process_detected_texts(detected_texts, duration_ms)
                            self.ocr_thread_active = False
                        except queue.Empty:
                            break

                # 1. Grab Frame (Webcam or Mock Frame)
                if mock_source:
                    frame = mock_frame.copy()
                    shift_x = int(5 * np.sin(frame_count * 0.1))
                    frame = np.roll(frame, shift_x, axis=1)
                else:
                    ret, frame = self.io.read_frame()
                    if not ret or frame is None:
                        self._cam_fail_count = getattr(self, "_cam_fail_count", 0) + 1
                        if self._cam_fail_count > 5:
                            print("⚠️ Camera frame drops detected. Switching to mock scene for execution.")
                            mock_source = True
                            base_dir = os.path.dirname(os.path.abspath(__file__))
                            test_img = os.path.join(base_dir, "test_scene.jpg")
                            if os.path.exists(test_img):
                                mock_frame = cv2.resize(cv2.imread(test_img), (640, 480))
                            else:
                                mock_frame = np.full((480, 640, 3), 120, dtype=np.uint8)
                        time.sleep(0.01)
                        continue
                    else:
                        self._cam_fail_count = 0

                # 2. Continuous Reflex & Metric Depth with Negative Obstacle Drop-off Detection
                res = self.reflex.process_frame(frame, timestamp=now)
                if len(res) == 3:
                    combined_hud, speech, dropoff_detected = res
                else:
                    combined_hud, speech = res
                    dropoff_detected = getattr(self.reflex, "dropoff_detected", False)

                # Benchmark Live Running FPS (EMA) & Main Thread YOLO Inference
                t_frame_end = time.perf_counter()
                loop_dt = t_frame_end - loop_start_t
                if frame_count > 1 and loop_dt > 0:
                    instant_fps = min(30.0, 1.0 / loop_dt)
                    current_fps = 0.70 * current_fps + 0.30 * instant_fps
                elif frame_count <= 1:
                    current_fps = 28.0

                yolo_ms = getattr(self.reflex, "last_yolo_time_ms", 28.0)
                logger.info(f"[Main Thread] {self.yolo_model_name} Inference Time: {yolo_ms:.0f}ms | FPS: {current_fps:.1f}")

                # Asynchronous Local Facial Recognition via DeepFace
                if hasattr(self.reflex, "last_results") and self.reflex.last_results is not None:
                    res_boxes = self.reflex.last_results.boxes
                    if res_boxes is not None and len(res_boxes) > 0:
                        boxes_coords = res_boxes.xyxy.cpu().numpy()
                        classes_arr = res_boxes.cls.cpu().numpy().astype(int)
                        track_ids_arr = (
                            res_boxes.id.cpu().numpy().astype(int)
                            if res_boxes.id is not None
                            else np.arange(len(boxes_coords))
                        )
                        h_frame, w_frame = frame.shape[:2]
                        for b_box, c_id, t_id in zip(boxes_coords, classes_arr, track_ids_arr):
                            c_lbl = self.reflex.detector.names[c_id]
                            if c_lbl == "person" and t_id not in self.track_identities:
                                self.track_identities[t_id] = "Person"
                                bx1, by1, bx2, by2 = map(int, b_box)
                                bx1, by1 = max(0, bx1), max(0, by1)
                                bx2, by2 = min(w_frame, bx2), min(h_frame, by2)
                                person_crop = frame[by1:by2, bx1:bx2].copy()
                                if person_crop.size > 0:
                                    self._async_recognize(t_id, person_crop)

                self.dropoff_active = dropoff_detected
                if speech:
                    self._update_hud_status(speech)

                # Emergency Priority 0 Drop-off / Descending Stairs Warning with 4.0s cooldown
                if dropoff_detected:
                    if (now - self.last_dropoff_warning) >= 4.0:
                        self.last_dropoff_warning = now
                        msg = "Warning: Drop-off or descending stairs ahead."
                        self.arbitrator.emit_alert(0, msg, now, key="dropoff_warning")
                        self._update_hud_status(msg)

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

                # Retrospective Visual Memory Retrieval Logic
                if self.memory_query is not None:
                    query_item = self.memory_query
                    target_norm = query_item.lower().strip()
                    if target_norm in ["phone", "cellphone"]:
                        target_norm = "cell phone"

                    matched_det = None
                    furniture_dets = []

                    if hasattr(self.reflex, "last_detections") and self.reflex.last_detections:
                        for det in self.reflex.last_detections:
                            lbl = det["label"].lower().strip()
                            if lbl in ["dining table", "desk", "chair", "couch", "bed", "table"]:
                                furniture_dets.append(det)
                            if lbl == target_norm or (target_norm in lbl) or (lbl in target_norm):
                                matched_det = det

                    if matched_det is not None:
                        # Item is currently visible in live YOLO frame -> announce live location
                        cx = matched_det["center_x"]
                        w_frame = frame.shape[1]
                        if cx < (w_frame / 3.0):
                            dir_str = "left"
                        elif cx > (2.0 * w_frame / 3.0):
                            dir_str = "right"
                        else:
                            dir_str = "center"

                        tx1, ty1, tx2, ty2 = matched_det.get("box", (cx - 10, matched_det["center_y"] - 10, cx + 10, matched_det["center_y"] + 10))
                        found_furniture = None
                        for f_det in furniture_dets:
                            fx1, fy1, fx2, fy2 = f_det.get("box", (0, 0, 0, 0))
                            overlap_x = max(0, min(tx2, fx2) - max(tx1, fx1))
                            if overlap_x > 0.2 * max(1, tx2 - tx1):
                                if ty2 >= fy1 - 40 and ty1 <= fy2:
                                    found_furniture = f_det["label"].replace("dining table", "dining table")
                                    break

                        target_depth = matched_det["dist_m"]
                        if found_furniture:
                            found_msg = f"Found {query_item} on your {dir_str}, resting on the {found_furniture}, {target_depth:.1f} meters away."
                        else:
                            found_msg = f"Found {query_item} on your {dir_str}, {target_depth:.1f} meters away."

                        self.arbitrator.emit_alert(0, found_msg, now, key="found_memory_target")
                        self._update_hud_status(found_msg)
                    else:
                        # Item NOT in current frame -> check self.episodic_memory
                        mem = self.episodic_memory.get(target_norm) or self.episodic_memory.get(query_item)
                        if mem:
                            elapsed_sec = max(0.0, time.time() - mem["timestamp"])
                            minutes_ago = int(elapsed_sec / 60)
                            time_str = "just now" if minutes_ago < 1 else f"{minutes_ago} minutes ago"
                            location = mem.get("location", "nearby")
                            msg = f"I don't see the {query_item} right now, but I saw it {location} {time_str}."
                            self.arbitrator.emit_alert(0, msg, now, key="recall_memory_target")
                            self._update_hud_status(msg)
                        else:
                            msg = f"I have no recent memory of the {query_item}."
                            self.arbitrator.emit_alert(0, msg, now, key="recall_memory_none")
                            self._update_hud_status(msg)

                    self.memory_query = None

                # 3. Continuous Episodic Object Memory Logging (Throttled to once every 2.0s to eliminate SQLite disk write latency)
                if hasattr(self.reflex, "last_detections") and self.reflex.last_detections and (now - self.last_memory_log_time >= 2.0):
                    self.last_memory_log_time = now
                    self.memory.log_scene_detections(self.reflex.last_detections, current_room=self.current_room)

                    # Retrospective Visual Memory: Background Logging of passive objects resting on furniture
                    furniture_dets = []
                    passive_dets = []
                    furniture_classes = {"dining table", "desk", "chair", "couch", "bed", "table"}
                    passive_classes = {
                        "bottle", "cell phone", "cup", "book", "laptop", "mouse", "keyboard",
                        "backpack", "handbag", "remote", "clock", "vase", "bowl", "keys", "glasses"
                    }

                    for det in self.reflex.last_detections:
                        lbl = det["label"].lower().strip()
                        if lbl in furniture_classes:
                            furniture_dets.append(det)
                        elif lbl in passive_classes or (lbl not in {"person", "car", "motorcycle", "bus", "truck"} and lbl not in furniture_classes):
                            passive_dets.append(det)

                    if furniture_dets and passive_dets:
                        current_wall_time = time.time()
                        for p_det in passive_dets:
                            p_lbl = p_det["label"].lower().strip()
                            if p_lbl in ["phone", "cellphone"]:
                                p_lbl = "cell phone"

                            cx = p_det["center_x"]
                            cy = p_det["center_y"]
                            tx1, ty1, tx2, ty2 = p_det.get("box", (cx - 10, cy - 10, cx + 10, cy + 10))

                            for f_det in furniture_dets:
                                fx1, fy1, fx2, fy2 = f_det.get("box", (0, 0, 0, 0))
                                overlap_x = max(0, min(tx2, fx2) - max(tx1, fx1))
                                if overlap_x > 0.2 * max(1, tx2 - tx1):
                                    if ty2 >= fy1 - 40 and ty1 <= fy2:
                                        furniture_label = f_det["label"].lower().strip()
                                        self.episodic_memory[p_lbl] = {
                                            "location": f"resting on the {furniture_label}",
                                            "timestamp": current_wall_time
                                        }
                                        if p_det["label"].lower().strip() != p_lbl:
                                            self.episodic_memory[p_det["label"].lower().strip()] = self.episodic_memory[p_lbl]
                                        break

                # 4. Asynchronous Face & Social Micro-Cues (Triggered when person detected, throttled to 2.5s)
                has_person = any(det.get("label") == "person" for det in self.reflex.last_detections) if hasattr(self.reflex, "last_detections") else False
                if has_person and (time.time() - self.last_social_scan_time >= 2.5):
                    if self.face_task_queue.empty():
                        self.last_social_scan_time = time.time()
                        self.face_task_queue.put((frame.copy(), now))

                # Asynchronous Continuous Spatial OCR in Daemon Thread (Center 60% of frame with 1.5s throttle)
                now_wall = time.time()
                if not self.ocr_thread_active and (now_wall - self.last_ocr_scan_time >= self.ocr_scan_interval):
                    self.last_ocr_scan_time = now_wall
                    h_f, w_f = frame.shape[:2]
                    cy1, cy2 = int(h_f * 0.2), int(h_f * 0.8)
                    cx1, cx2 = int(w_f * 0.2), int(w_f * 0.8)
                    center_crop = frame[cy1:cy2, cx1:cx2].copy()
                    if center_crop.size > 0:
                        self._async_ocr_worker(center_crop)

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
                if self.test_mode and frame_count in [5, 8, 10, 15, 20, 25, 30, 32]:
                    test_keys = {5: 'f', 8: 'm', 10: 'b', 15: 'l', 20: 's', 25: 'r', 30: 'e', 32: 'c'}
                    k = test_keys[frame_count]
                    print(f"\n🧪 [TEST VERIFICATION] Auto-triggering key [{k}] at frame {frame_count}...")
                    self.handle_hotkey(k, frame, now)

                # Pacing sleep to maintain steady 30 FPS ceiling and yield CPU/GIL
                elapsed_loop = time.perf_counter() - loop_start_t
                sleep_budget = max(0.001, 0.0333 - elapsed_loop)
                time.sleep(sleep_budget)

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

        # Shutdown multiprocessing processes cleanly
        if getattr(self, "mp_active", False):
            try:
                self.face_in_q.put(None)
                self.ocr_in_q.put(None)
            except Exception:
                pass
            try:
                if hasattr(self, "p_face") and self.p_face.is_alive():
                    self.p_face.terminate()
                if hasattr(self, "p_ocr") and self.p_ocr.is_alive():
                    self.p_ocr.terminate()
            except Exception:
                pass

        # Stop audio arbitrator TTS thread
        self.arbitrator.stop()

        if not self.headless:
            cv2.destroyAllWindows()

        try:
            import ctypes
            ctypes.windll.winmm.timeEndPeriod(1)
        except Exception:
            pass

        logger.info("🛑 Assistive Smart Goggles shutdown cleanly. Execution session ended.")
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
