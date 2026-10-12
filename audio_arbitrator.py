import time
import queue
import threading
import config

try:
    from gtts import gTTS
except ImportError:
    gTTS = None

try:
    from IPython.display import Audio, display
except ImportError:
    Audio, display = None, None

try:
    import pyttsx3
except ImportError:
    pyttsx3 = None

class PriorityAudioArbitrator:
    """
    ISANA & OSSG-inspired Priority Audio Queue with Track-ID Cooldown Memory.
    Priority 1: Critical Collision Hazard (<1.2m) -> Immediate Override
    Priority 2: Navigation & Approaching Obstacle (<2.5m)
    Priority 3: Social / Environmental Coaching (Lighting, Blur, Scene Info)
    """
    def __init__(self):
        self.track_memory = {}       # {track_id: last_announced_timestamp}
        self.general_cooldown = {}   # {message_type: last_announced_timestamp}
        self.spoken_log = []         # History log for Colab/Local benchmarking
        self.speech_queue = queue.PriorityQueue()
        self._msg_counter = 0
        self._stop_event = threading.Event()
        self._tts_thread = None

        if config.ENV_MODE == "LOCAL":
            self._start_speech_worker()

    def _start_speech_worker(self):
        """Starts a dedicated non-blocking background daemon thread for offline pyttsx3 voice."""
        if self._tts_thread is not None and self._tts_thread.is_alive():
            return
        self._tts_thread = threading.Thread(target=self._speech_worker_loop, daemon=True)
        self._tts_thread.start()

    def _speech_worker_loop(self):
        """Dedicated background thread worker executing offline zero-lag TTS."""
        engine = None
        if pyttsx3 is not None:
            try:
                engine = pyttsx3.init()
                engine.setProperty("rate", 175)  # Responsive, natural speech speed
                engine.setProperty("volume", 1.0)
            except Exception as e:
                print(f"⚠️ Warning: Could not initialize pyttsx3 engine: {e}")
                engine = None

        while not self._stop_event.is_set():
            try:
                item = self.speech_queue.get(timeout=0.4)
            except queue.Empty:
                continue

            priority, _, message = item
            if engine is not None:
                try:
                    engine.say(message)
                    engine.runAndWait()
                except Exception as e:
                    print(f"TTS Speech error: {e}")

            self.speech_queue.task_done()

    def get_spatial_direction(self, x_center_ratio):
        """Maps horizontal bounding box center (0.0 to 1.0) to human clock/ear directions."""
        if x_center_ratio < 0.35:
            return "on your left"
        elif x_center_ratio > 0.65:
            return "on your right"
        return "straight ahead"

    def should_announce_track(self, track_id, distance_m, current_time):
        """Prevents repeating 'Chair ahead' 30 times a second for the same object."""
        last_time, last_dist = self.track_memory.get(track_id, (0.0, 99.0))
        # Always announce immediately if the object suddenly moved much closer (<1.2m)
        if distance_m <= config.CRITICAL_DANGER_M and last_dist > config.CRITICAL_DANGER_M:
            self.track_memory[track_id] = (current_time, distance_m)
            return True
        # Otherwise enforce the cooldown timer
        if (current_time - last_time) >= config.TRACK_COOLDOWN_SEC:
            self.track_memory[track_id] = (current_time, distance_m)
            return True
        return False

    def emit_alert(self, priority, message, current_time, key="general"):
        """Queues and synthesizes speech without flooding user ears or blocking 30 FPS camera loop."""
        if key in self.general_cooldown and (current_time - self.general_cooldown[key]) < 5.0:
            return None
        self.general_cooldown[key] = current_time
        log_entry = f"[{current_time:05.2f}s | P{priority}] 🔊 \"{message}\""
        self.spoken_log.append((current_time, priority, message))
        print(log_entry)

        if config.ENV_MODE == "LOCAL":
            if self._tts_thread is None or not self._tts_thread.is_alive():
                self._start_speech_worker()

            # Priority 1 Critical Hazard immediate override: clear pending queue
            if priority == 1:
                with self.speech_queue.mutex:
                    self.speech_queue.queue.clear()

            self._msg_counter += 1
            self.speech_queue.put((priority, self._msg_counter, message))

        return message

    def play_audio_in_colab(self, text):
        """Synthesizes and plays an audio alert directly inside the Google Colab browser."""
        if gTTS is not None and Audio is not None and display is not None:
            tts = gTTS(text=text, lang="en", slow=False)
            tts.save("latest_alert.mp3")
            display(Audio("latest_alert.mp3", autoplay=True))

    def stop(self):
        """Signals the background TTS thread to stop."""
        self._stop_event.set()
