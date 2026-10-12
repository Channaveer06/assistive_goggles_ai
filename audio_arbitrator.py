import time
import queue
import threading
import subprocess
import config

try:
    from gTTS import gTTS
except ImportError:
    try:
        from gtts import gTTS
    except ImportError:
        gTTS = None

try:
    from IPython.display import Audio, display
except ImportError:
    Audio, display = None, None

class PriorityAudioArbitrator:
    """
    ISANA & OSSG-inspired Priority Audio Queue with Track-ID Cooldown Memory.
    Priority 0: User Direct Hotkey Request (Navigation / OCR / Item / Light / Face / SOS) -> Immediate Preemption
    Priority 1: Critical Collision Hazard (<1.2m) -> Immediate Override
    Priority 2: Approaching Obstacle (<2.5m)
    Priority 3: Environmental Coaching (Lighting, Blur)
    """
    def __init__(self):
        self.track_memory = {}       # {track_id: last_announced_timestamp}
        self.general_cooldown = {}   # {message_type: last_announced_timestamp}
        self.spoken_log = []         # History log for Colab/Local benchmarking
        self.speech_queue = queue.PriorityQueue(maxsize=2)
        self._msg_counter = 0
        self._stop_event = threading.Event()
        self._tts_thread = None

        if config.ENV_MODE == "LOCAL":
            self._start_speech_worker()

    def _start_speech_worker(self):
        """Starts a dedicated non-blocking background daemon thread for offline SAPI5/PowerShell voice."""
        if self._tts_thread is not None and self._tts_thread.is_alive():
            return
        self._tts_thread = threading.Thread(target=self._speech_worker_loop, daemon=True)
        self._tts_thread.start()

    def _speech_worker_loop(self):
        """Dedicated background thread worker executing offline zero-lag TTS via native SAPI.SpVoice."""
        speaker = None
        use_com = False

        try:
            import pythoncom
            import win32com.client
            pythoncom.CoInitialize()
            speaker = win32com.client.Dispatch("SAPI.SpVoice")
            speaker.Rate = 2  # Natural, responsive speech rate
            speaker.Volume = 100
            use_com = True
        except Exception as e:
            print(f"⚠️ SAPI.SpVoice COM initialization warning: {e}. Falling back to PowerShell TTS.")
            speaker = None

        while not self._stop_event.is_set():
            try:
                item = self.speech_queue.get(timeout=0.25)
            except queue.Empty:
                continue

            priority, _, message = item
            if not message:
                self.speech_queue.task_done()
                continue

            spoken = False
            if use_com and speaker is not None:
                try:
                    # SVSFDefault = 0 (synchronous speak inside background worker thread)
                    speaker.Speak(message, 0)
                    spoken = True
                except Exception as e:
                    print(f"⚠️ SAPI Speak error: {e}")

            if not spoken:
                try:
                    clean_msg = message.replace("'", "''").replace('"', '`"')
                    ps_cmd = (
                        f"Add-Type -AssemblyName System.Speech; "
                        f"$synth = New-Object System.Speech.Synthesis.SpeechSynthesizer; "
                        f"$synth.Rate = 2; $synth.Speak('{clean_msg}')"
                    )
                    subprocess.run(
                        ["powershell", "-NoProfile", "-Command", ps_cmd],
                        check=False,
                        stdout=subprocess.DEVNULL,
                        stderr=subprocess.DEVNULL
                    )
                except Exception as e:
                    print(f"⚠️ PowerShell speech synthesis error: {e}")

            self.speech_queue.task_done()

        if use_com:
            try:
                import pythoncom
                pythoncom.CoUninitialize()
            except Exception:
                pass

    def get_spatial_direction(self, x_center_ratio):
        """Maps horizontal bounding box center (0.0 to 1.0) to human clock/ear directions."""
        if x_center_ratio < 0.35:
            return "on your left"
        elif x_center_ratio > 0.65:
            return "on your right"
        return "straight ahead"

    def should_announce_track(self, track_id, distance_m, current_time):
        """
        Smart Stationary-Object Silence:
        Prevents repeating the same stationary obstacle every 4 seconds.
        Once announced, only re-announces if:
        (a) It moved closer by at least 0.35m (approach detected), OR
        (b) Inside CRITICAL_DANGER_M (<=1.2m) and at least 6.0s passed, OR
        (c) More than 30.0s have passed since it was last mentioned.
        """
        if track_id not in self.track_memory:
            self.track_memory[track_id] = (current_time, distance_m)
            return True

        last_time, last_dist = self.track_memory[track_id]
        time_elapsed = current_time - last_time

        # Immediate alert if object suddenly breached critical danger threshold (<1.2m)
        if distance_m <= config.CRITICAL_DANGER_M and last_dist > config.CRITICAL_DANGER_M:
            self.track_memory[track_id] = (current_time, distance_m)
            return True

        # Condition (a): Object moved closer by at least 0.35 meters
        if (last_dist - distance_m) >= 0.35:
            self.track_memory[track_id] = (current_time, distance_m)
            return True

        # Condition (b): Inside critical danger zone and >= 6.0 seconds have passed
        if distance_m <= config.CRITICAL_DANGER_M and time_elapsed >= 6.0:
            self.track_memory[track_id] = (current_time, distance_m)
            return True

        # Condition (c): Periodic refresh after 30.0 seconds
        if time_elapsed >= 30.0:
            self.track_memory[track_id] = (current_time, distance_m)
            return True

        return False

    def emit_alert(self, priority, message, current_time, key="general"):
        """Queues and synthesizes speech with immediate hotkey preemption & small queue buffer."""
        # Only enforce cooldown on priority >= 2 (environmental/obstacle coaching)
        if priority >= 2:
            if key in self.general_cooldown and (current_time - self.general_cooldown[key]) < 4.0:
                return None
            self.general_cooldown[key] = current_time

        log_entry = f"[{current_time:05.2f}s | P{priority}] [AUDIO] \"{message}\""
        self.spoken_log.append((current_time, priority, message))
        try:
            print(log_entry)
        except UnicodeEncodeError:
            print(log_entry.encode("ascii", errors="replace").decode("ascii"))

        if config.ENV_MODE == "LOCAL":
            if self._tts_thread is None or not self._tts_thread.is_alive():
                self._start_speech_worker()

            self._msg_counter += 1

            # Priority 0 (User Hotkey) & Priority 1 (Hazard Stop): Preempt all pending queued speech
            if priority <= 1:
                with self.speech_queue.mutex:
                    self.speech_queue.queue.clear()
                try:
                    self.speech_queue.put_nowait((priority, self._msg_counter, message))
                except queue.Full:
                    pass
            else:
                # For obstacle/coaching alerts, drop stale alerts if queue is already full
                try:
                    self.speech_queue.put_nowait((priority, self._msg_counter, message))
                except queue.Full:
                    # Queue full -> discard stale obstacle alert
                    pass

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

