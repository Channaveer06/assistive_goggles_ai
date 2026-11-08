import time
import queue
import threading
import subprocess
import config

try:
    import pythoncom
    import win32com.client
except ImportError:
    win32com = None
    pythoncom = None

try:
    import pyttsx3
except ImportError:
    pyttsx3 = None

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
        self.speech_queue = queue.PriorityQueue(maxsize=5)
        self._msg_counter = 0
        self._stop_event = threading.Event()
        self._tts_thread = None
        self.guardian_speaking = False
        self.selected_voice_id = None
        self.selected_voice_name = "Default"
        self.sapi_voice_index = None

        # 1. Inspect & Select Native Offline Indian English Voice
        if win32com is not None and pythoncom is not None:
            try:
                pythoncom.CoInitialize()
                sp = win32com.client.Dispatch("SAPI.SpVoice")
                voices = sp.GetVoices()
                installed_names = [voices.Item(i).GetDescription() for i in range(voices.Count)]
                print(f"[AUDIO] Installed system SAPI voices: {installed_names}")

                for i in range(voices.Count):
                    desc = voices.Item(i).GetDescription().lower()
                    if any(k in desc for k in ["heera", "ravi", "india"]):
                        self.sapi_voice_index = i
                        self.selected_voice_name = voices.Item(i).GetDescription()
                        print(f"[AUDIO] Active SAPI voice set to Indian English: {self.selected_voice_name}")
                        break

                if self.sapi_voice_index is None and voices.Count > 0:
                    self.sapi_voice_index = 0
                    self.selected_voice_name = voices.Item(0).GetDescription()
                    print(
                        f"[AUDIO] Indian voice pack not detected in Windows. Using default: {self.selected_voice_name}. "
                        "To install, go to Windows Settings -> Time & Language -> Speech -> Add Voices -> English (India)."
                    )
                del voices
                del sp
                pythoncom.CoUninitialize()
            except Exception as e:
                print(f"⚠️ SAPI voice inspection notice: {e}")

        elif pyttsx3 is not None:
            try:
                engine = pyttsx3.init()
                voices = engine.getProperty('voices')
                for v in (voices or []):
                    v_name = getattr(v, "name", "").lower()
                    v_id = getattr(v, "id", "").lower()
                    if any(k in v_name or k in v_id for k in ["heera", "ravi", "india"]):
                        self.selected_voice_id = v.id
                        self.selected_voice_name = getattr(v, "name", str(v))
                        break
            except Exception as e:
                print(f"⚠️ pyttsx3 voice selection notice: {e}")

        if config.ENV_MODE == "LOCAL":
            self._start_speech_worker()

    def clear_telemetry_queue(self):
        """Clears all queued telemetry and obstacle alerts (priority >= 1)."""
        with self.speech_queue.mutex:
            p0_items = [item for item in self.speech_queue.queue if item[0] == 0]
            self.speech_queue.queue.clear()
            for it in p0_items:
                self.speech_queue.queue.append(it)

    def _start_speech_worker(self):
        """Starts a dedicated non-blocking background daemon thread for offline pyttsx3/SAPI5 voice."""
        if self._tts_thread is not None and self._tts_thread.is_alive():
            return
        self._tts_thread = threading.Thread(target=self._speech_worker_loop, daemon=True)
        self._tts_thread.start()

    def _speech_worker_loop(self):
        """Dedicated background thread worker executing offline zero-lag TTS via native SAPI/pyttsx3/PowerShell."""
        sapi_voice = None
        worker_engine = None
        use_com = False

        if win32com is not None and pythoncom is not None:
            try:
                pythoncom.CoInitialize()
                use_com = True
                sapi_voice = win32com.client.Dispatch("SAPI.SpVoice")
                sapi_voice.Rate = 1
                sapi_voice.Volume = 100
                if self.sapi_voice_index is not None:
                    voices = sapi_voice.GetVoices()
                    if self.sapi_voice_index < voices.Count:
                        sapi_voice.Voice = voices.Item(self.sapi_voice_index)
            except Exception as e:
                print(f"⚠️ SAPI voice initialization notice: {e}")
                sapi_voice = None

        if sapi_voice is None and pyttsx3 is not None:
            try:
                worker_engine = pyttsx3.init()
                if self.selected_voice_id:
                    worker_engine.setProperty('voice', self.selected_voice_id)
                worker_engine.setProperty('rate', 165)
                worker_engine.setProperty('volume', 1.0)
            except Exception as e:
                print(f"⚠️ Worker pyttsx3 initialization notice: {e}")
                worker_engine = None

        while not self._stop_event.is_set():
            try:
                item = self.speech_queue.get(timeout=0.25)
            except queue.Empty:
                continue

            priority, _, message = item
            if not message:
                self.speech_queue.task_done()
                continue

            # Walkie-Talkie Lock: If guardian is speaking, suppress all Priority >= 1 (telemetry) announcements
            if self.guardian_speaking and priority >= 1:
                self.speech_queue.task_done()
                continue

            spoken = False

            # Tier 1: Native Windows SAPI.SpVoice (Direct COM, 0ms lag, supports purge interruption)
            if sapi_voice is not None:
                try:
                    # SVSFPurgeBeforeSpeak = 2 if priority <= 1 else 0 (synchronous)
                    flags = 2 if priority <= 1 else 0
                    sapi_voice.Speak(message, flags)
                    spoken = True
                except Exception as e:
                    print(f"⚠️ SAPI speech error: {e}")
                    spoken = False

            # Tier 2: pyttsx3 fallback
            if not spoken and worker_engine is not None:
                try:
                    worker_engine.say(message)
                    worker_engine.runAndWait()
                    spoken = True
                except Exception as e:
                    print(f"⚠️ pyttsx3 speech error: {e}")

            # Tier 3: PowerShell System.Speech fallback
            if not spoken:
                try:
                    clean_msg = message.replace("'", "''").replace('"', '`"')
                    ps_cmd = (
                        f"Add-Type -AssemblyName System.Speech; "
                        f"$synth = New-Object System.Speech.Synthesis.SpeechSynthesizer; "
                        f"$synth.Rate = 1; "
                        f"$synth.Speak('{clean_msg}')"
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

        if sapi_voice is not None:
            try:
                # SVSFPurgeBeforeSpeak = 2
                sapi_voice.Speak("", 2)
            except Exception:
                pass
            del sapi_voice

        if worker_engine is not None:
            try:
                worker_engine.stop()
            except Exception:
                pass
            del worker_engine

        if use_com and pythoncom is not None:
            try:
                pythoncom.CoUninitialize()
            except Exception:
                pass

    _play_loop = _speech_worker_loop

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
        # Walkie-Talkie Lock: If guardian is speaking, suppress all Priority >= 1 (telemetry) announcements
        if self.guardian_speaking and priority >= 1:
            return None

        # Only enforce cooldown on priority >= 2 (environmental/obstacle coaching)
        if priority >= 2:
            if key in self.general_cooldown and (current_time - self.general_cooldown[key]) < 4.0:
                return None
            self.general_cooldown[key] = current_time

        log_entry = f"[{current_time:05.2f}s | P{priority}] [AUDIO] \"{message}\""
        self.spoken_log.append((current_time, priority, message))
        try:
            import logging
            logging.getLogger("AssistiveGoggles").info(f"[Audio Arbitrator] Priority {priority} Alert: \"{message}\"")
        except Exception:
            pass
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
        if hasattr(self, 'engine') and self.engine is not None:
            try:
                self.engine.stop()
            except Exception:
                pass
        if self._tts_thread is not None and self._tts_thread.is_alive():
            self._tts_thread.join(timeout=1.5)

