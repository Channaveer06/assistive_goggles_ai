
import os
import sys
import time
import socket
import asyncio
import threading
import numpy as np
import cv2

from aiohttp import web
from aiortc import VideoStreamTrack, RTCPeerConnection, RTCSessionDescription, RTCConfiguration, RTCIceServer
from av import VideoFrame

try:
    from pyngrok import ngrok, conf
except ImportError:
    ngrok = None

import config

INDEX_HTML = """<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Assistive Smart Goggles AI — Guardian Emergency Live Stream</title>
    <style>
        * {
            box-sizing: border-box;
            margin: 0;
            padding: 0;
        }
        body {
            font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
            background: #0b0f19;
            color: #f1f5f9;
            min-height: 100vh;
            display: flex;
            flex-direction: column;
            align-items: center;
            padding: 20px 16px;
        }
        .header {
            width: 100%;
            max-width: 960px;
            display: flex;
            justify-content: space-between;
            align-items: center;
            padding: 14px 20px;
            background: #151d30;
            border-radius: 12px;
            border: 1px solid #1e293b;
            box-shadow: 0 4px 20px rgba(0, 0, 0, 0.4);
            margin-bottom: 18px;
        }
        .brand {
            display: flex;
            align-items: center;
            gap: 12px;
        }
        .brand-icon {
            font-size: 26px;
        }
        .brand-title {
            font-size: 1.15rem;
            font-weight: 700;
            letter-spacing: 0.5px;
            color: #ffffff;
        }
        .brand-sub {
            font-size: 0.78rem;
            color: #94a3b8;
        }
        .sos-badge {
            display: flex;
            align-items: center;
            gap: 8px;
            background: rgba(239, 68, 68, 0.15);
            border: 1px solid #ef4444;
            color: #fca5a5;
            padding: 6px 14px;
            border-radius: 20px;
            font-size: 0.82rem;
            font-weight: 700;
            text-transform: uppercase;
            letter-spacing: 1px;
        }
        .pulse-dot {
            width: 10px;
            height: 10px;
            background-color: #ef4444;
            border-radius: 50%;
            box-shadow: 0 0 0 rgba(239, 68, 68, 0.7);
            animation: pulse 1.6s infinite;
        }
        @keyframes pulse {
            0% { box-shadow: 0 0 0 0 rgba(239, 68, 68, 0.7); }
            70% { box-shadow: 0 0 0 10px rgba(239, 68, 68, 0); }
            100% { box-shadow: 0 0 0 0 rgba(239, 68, 68, 0); }
        }
        .main-card {
            width: 100%;
            max-width: 960px;
            background: #111827;
            border-radius: 16px;
            border: 1px solid #1f2937;
            overflow: hidden;
            box-shadow: 0 10px 30px rgba(0, 0, 0, 0.6);
            display: flex;
            flex-direction: column;
        }
        .video-container {
            position: relative;
            width: 100%;
            background: #000000;
            aspect-ratio: 4 / 3;
            max-height: 640px;
            display: flex;
            align-items: center;
            justify-content: center;
        }
        video {
            width: 100%;
            height: 100%;
            object-fit: contain;
            background: #000;
        }
        .overlay-status {
            position: absolute;
            top: 14px;
            left: 14px;
            background: rgba(15, 23, 42, 0.85);
            backdrop-filter: blur(8px);
            padding: 6px 12px;
            border-radius: 8px;
            border: 1px solid rgba(255, 255, 255, 0.1);
            font-size: 0.78rem;
            font-weight: 600;
            color: #38bdf8;
            display: flex;
            align-items: center;
            gap: 6px;
        }
        .info-bar {
            padding: 16px 20px;
            display: flex;
            flex-wrap: wrap;
            justify-content: space-between;
            align-items: center;
            gap: 12px;
            background: #131b2e;
            border-top: 1px solid #1e293b;
        }
        .meta-group {
            display: flex;
            gap: 20px;
            font-size: 0.85rem;
        }
        .meta-item {
            display: flex;
            flex-direction: column;
        }
        .meta-label {
            font-size: 0.72rem;
            color: #64748b;
            text-transform: uppercase;
            font-weight: 600;
            letter-spacing: 0.5px;
        }
        .meta-value {
            color: #e2e8f0;
            font-weight: 600;
        }
        .btn-reconnect {
            background: #2563eb;
            color: white;
            border: none;
            padding: 8px 18px;
            border-radius: 8px;
            font-size: 0.85rem;
            font-weight: 600;
            cursor: pointer;
            transition: all 0.2s ease;
        }
        .btn-reconnect:hover {
            background: #1d4ed8;
        }
        .instructions {
            width: 100%;
            max-width: 960px;
            margin-top: 18px;
            padding: 16px 20px;
            background: #0f172a;
            border-radius: 12px;
            border: 1px solid #1e293b;
            font-size: 0.83rem;
            color: #94a3b8;
            line-height: 1.5;
        }
        .instructions h3 {
            color: #f1f5f9;
            font-size: 0.92rem;
            margin-bottom: 6px;
        }
        .btn-talkback {
            width: 100%;
            margin-top: 14px;
            padding: 18px 24px;
            background: linear-gradient(135deg, #1e293b, #0f172a);
            border: 2px solid #3b82f6;
            border-radius: 12px;
            color: #ffffff;
            font-size: 1.05rem;
            font-weight: 700;
            letter-spacing: 0.5px;
            cursor: pointer;
            user-select: none;
            -webkit-user-select: none;
            touch-action: none;
            display: flex;
            align-items: center;
            justify-content: center;
            gap: 10px;
            box-shadow: 0 4px 15px rgba(59, 130, 246, 0.25);
            transition: all 0.15s ease;
        }
        .btn-talkback:hover {
            border-color: #60a5fa;
            box-shadow: 0 6px 20px rgba(59, 130, 246, 0.4);
        }
        .btn-talkback.recording {
            background: linear-gradient(135deg, #dc2626, #991b1b) !important;
            border-color: #ef4444 !important;
            box-shadow: 0 0 25px rgba(239, 68, 68, 0.7) !important;
            transform: scale(0.98);
        }
        .btn-talkback.sent {
            background: linear-gradient(135deg, #16a34a, #15803d) !important;
            border-color: #22c55e !important;
            box-shadow: 0 0 20px rgba(34, 197, 94, 0.5) !important;
        }
    </style>
</head>
<body>
    <header class="header">
        <div class="brand">
            <span class="brand-icon">👓</span>
            <div>
                <div class="brand-title">Assistive Smart Goggles AI</div>
                <div class="brand-sub">Real-Time Zero-Lag Guardian Portal</div>
            </div>
        </div>
        <div class="sos-badge">
            <span class="pulse-dot"></span>
            <span>SOS ACTIVE</span>
        </div>
    </header>

    <main class="main-card">
        <div class="video-container">
            <img id="video" src="/mjpeg" style="width: 100%; border-radius: 8px; object-fit: cover; min-height: 250px; background: #000;" />
        </div>

        <!-- Guardian Two-Way Walkie-Talkie Button -->
        <div style="padding: 0 16px 14px 16px;">
            <button id="btnTalk" class="btn-talkback">🎙️ Hold to Speak (Walkie-Talkie)</button>
        </div>

        <div class="info-bar">
            <div class="meta-group">
                <div class="meta-item">
                    <span class="meta-label">Protocol</span>
                    <span class="meta-value">MJPEG Stream + 2-Way Voice</span>
                </div>
                <div class="meta-item">
                    <span class="meta-label">Stream Feed</span>
                    <span class="meta-value">Live POV + Metric HUD</span>
                </div>
                <div class="meta-item">
                    <span class="meta-label">Status</span>
                    <span id="connStatus" class="meta-value" style="color: #4ade80;">Live Streaming</span>
                </div>
            </div>
            <button id="btnReconnect" class="btn-reconnect" onclick="reloadStream()">Reload Stream</button>
        </div>
    </main>

    <section class="instructions">
        <h3>🛡️ Guardian Emergency Guidance</h3>
        <p>This secure live stream delivers the POV perspective of the blind user along with real-time AI computer vision telemetry (metric depth, obstacle bounding boxes, and indoor location). Hold the walkie-talkie button below to talk directly into the goggles speakers.</p>
    </section>

    <script>
        function reloadStream() {
            const img = document.getElementById('video');
            if (img) {
                img.src = '/mjpeg?t=' + new Date().getTime();
            }
        }

        let mediaRecorder = null;
        let audioChunks = [];
        let isRecording = false;

        const btnTalk = document.getElementById('btnTalk');

        async function startRecording(e) {
            if (e) e.preventDefault();
            if (isRecording) return;

            try {
                const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
                audioChunks = [];

                let mimeType = 'audio/webm';
                if (MediaRecorder.isTypeSupported('audio/webm;codecs=opus')) {
                    mimeType = 'audio/webm;codecs=opus';
                } else if (MediaRecorder.isTypeSupported('audio/mp4')) {
                    mimeType = 'audio/mp4';
                }

                mediaRecorder = new MediaRecorder(stream, { mimeType });
                mediaRecorder.ondataavailable = (event) => {
                    if (event.data.size > 0) audioChunks.push(event.data);
                };

                mediaRecorder.onstop = async () => {
                    stream.getTracks().forEach(track => track.stop());
                    if (audioChunks.length === 0) return;

                    const audioBlob = new Blob(audioChunks, { type: mimeType });
                    btnTalk.className = 'btn-talkback sent';
                    btnTalk.innerText = '✅ Sent to Channaveer!';

                    try {
                        await fetch('/guardian_audio', {
                            method: 'POST',
                            headers: {
                                'Content-Type': audioBlob.type,
                                'ngrok-skip-browser-warning': 'true'
                            },
                            body: audioBlob
                        });
                    } catch (err) {
                        console.error('Audio send error:', err);
                    }

                    setTimeout(() => {
                        btnTalk.className = 'btn-talkback';
                        btnTalk.innerText = '🎙️ Hold to Speak (Walkie-Talkie)';
                    }, 1800);
                };

                mediaRecorder.start();
                isRecording = true;
                btnTalk.className = 'btn-talkback recording';
                btnTalk.innerText = '🔴 Recording / Transmitting... (Release to Send)';
            } catch (err) {
                console.error('Microphone access denied:', err);
                btnTalk.innerText = '⚠️ Microphone Access Denied';
                setTimeout(() => {
                    btnTalk.className = 'btn-talkback';
                    btnTalk.innerText = '🎙️ Hold to Speak (Walkie-Talkie)';
                }, 2000);
            }
        }

        function stopRecording(e) {
            if (e) e.preventDefault();
            if (!isRecording) return;
            isRecording = false;
            if (mediaRecorder && mediaRecorder.state !== 'inactive') {
                mediaRecorder.stop();
            }
        }

        if (btnTalk) {
            btnTalk.addEventListener('mousedown', startRecording);
            btnTalk.addEventListener('mouseup', stopRecording);
            btnTalk.addEventListener('mouseleave', stopRecording);
            btnTalk.addEventListener('touchstart', startRecording);
            btnTalk.addEventListener('touchend', stopRecording);
            btnTalk.addEventListener('touchcancel', stopRecording);
        }
    </script>
</body>
</html>
"""


class GogglesVideoStreamTrack(VideoStreamTrack):
    """
    Custom WebRTC Video Stream Track reading the latest annotated HUD frames
    from the main OpenCV thread safely without blocking the 30 FPS camera loop.
    """
    kind = "video"

    def __init__(self, frame_getter):
        super().__init__()
        self.frame_getter = frame_getter
        self._blank_frame = np.zeros((480, 640, 3), dtype=np.uint8)
        cv2.putText(self._blank_frame, "Assistive Goggles AI - Live POV Standby", (50, 240),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 255), 2, cv2.LINE_AA)

    async def recv(self):
        pts, time_base = await self.next_timestamp()
        frame = self.frame_getter()

        if frame is None:
            frame = self._blank_frame.copy()

        # Ensure contiguous memory for libav encoder
        if not frame.flags['C_CONTIGUOUS']:
            frame = np.ascontiguousarray(frame)

        # Convert OpenCV BGR to PyAV VideoFrame
        video_frame = VideoFrame.from_ndarray(frame, format="bgr24")
        video_frame.pts = pts
        video_frame.time_base = time_base
        return video_frame


def _safe_print(text):
    """Prints safely across all Windows command line codepages (cp1252, utf-8)."""
    try:
        print(text)
    except UnicodeEncodeError:
        try:
            print(text.encode("ascii", errors="replace").decode("ascii"))
        except Exception:
            pass


class WebRTCGuardianServer:
    """
    Manages the background aiohttp WebRTC signaling server and pyngrok secure tunnel.
    Operates in a completely detached daemon thread so the main 30 FPS loop is never impacted.
    """
    def __init__(self, arbitrator, io_bridge=None, port=8080):
        self.arbitrator = arbitrator
        self.io_bridge = io_bridge or getattr(arbitrator, "io", None)
        self.port = port
        self.latest_frame = None
        self.frame_lock = threading.Lock()

        self.server_thread = None
        self.is_running = False
        self.public_url = None
        self.local_url = None
        self.current_room = "study_room"
        self.call_request = False

        self.pcs = set()
        self.runner = None
        self.loop = None

    def update_frame(self, frame_bgr):
        """Thread-safe update of the latest annotated HUD frame."""
        if frame_bgr is None:
            return
        with self.frame_lock:
            self.latest_frame = frame_bgr

    def get_latest_frame(self):
        """Thread-safe retrieval of the latest frame."""
        with self.frame_lock:
            if self.latest_frame is not None:
                return self.latest_frame.copy()
            return None

    def _get_local_ip(self):
        """Returns the local LAN IP address."""
        try:
            s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            s.connect(("8.8.8.8", 80))
            ip = s.getsockname()[0]
            s.close()
            return ip
        except Exception:
            return "127.0.0.1"

    def start_stream(self, current_room="study_room", timestamp=0.0, call_request=False):
        """
        Launches the WebRTC/MJPEG server and pyngrok tunnel in a background daemon thread.
        Non-blocking: returns immediately while the server initializes asynchronously.
        """
        self.current_room = current_room
        self.call_request = call_request

        if self.is_running and (self.public_url or self.local_url):
            url = self.public_url or self.local_url
            _safe_print(f"\n[GUARDIAN SOS] Live stream is already active: {url}")
            if call_request:
                self.arbitrator.emit_alert(
                    0,
                    "Calling registered guardian. Opening live audio and video stream.",
                    timestamp or time.time(),
                    key="call_guardian"
                )
                if hasattr(self.arbitrator, "io") and self.arbitrator.io is not None:
                    self.arbitrator.io.notify_guardian(current_room, url, call_request=True)
            else:
                self.arbitrator.emit_alert(
                    0,
                    f"Guardian live stream is already active at {url}.",
                    timestamp or time.time(),
                    key="sos_already_active"
                )
            return url

        if self.server_thread is not None and self.server_thread.is_alive():
            return self.public_url or self.local_url

        self.server_thread = threading.Thread(
            target=self._run_server_thread,
            args=(current_room, timestamp),
            daemon=True,
            name="WebRTCGuardianThread"
        )
        self.server_thread.start()
        return None

    def _run_server_thread(self, current_room, timestamp):
        """Background thread target setting up the asyncio event loop and aiohttp server."""
        self.loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self.loop)

        local_ip = self._get_local_ip()
        self.local_url = f"http://{local_ip}:{self.port}"

        try:
            # 1. Start aiohttp server on local port first
            self.loop.run_until_complete(self._start_aiohttp_server())
            self.is_running = True

            # 2. Open pyngrok public HTTPS tunnel to local port
            self._setup_ngrok()

            # 3. Prominently display clickable URL in terminal
            _safe_print("\n" + "=" * 70)
            _safe_print(">> [EMERGENCY GUARDIAN SOS] LIVE STREAM ACTIVE <<")
            if self.public_url:
                _safe_print(f"PUBLIC HTTPS STREAM LINK : {self.public_url}")
            _safe_print(f"LOCAL NETWORK LINK       : {self.local_url} (http://localhost:{self.port})")
            _safe_print("=" * 70 + "\n")

            # 4. Automatically dispatch live stream link to guardian via webhook
            dispatch_url = self.public_url if self.public_url else self.local_url
            if hasattr(self.arbitrator, "io") and self.arbitrator.io is not None:
                self.arbitrator.io.notify_guardian(current_room, dispatch_url, call_request=self.call_request)
            elif self.io_bridge is not None:
                self.io_bridge.notify_guardian(current_room, dispatch_url, call_request=self.call_request)

            # 5. Speak notification via priority audio arbitrator
            if self.call_request:
                self.arbitrator.emit_alert(
                    0,
                    "Calling registered guardian. Opening live audio and video stream.",
                    timestamp or time.time(),
                    key="call_guardian"
                )
            else:
                self.arbitrator.emit_alert(
                    0,
                    "Emergency Guardian SOS active. Secure live stream link generated.",
                    timestamp or time.time(),
                    key="sos_link_ready"
                )

            # 6. Keep loop running to handle incoming requests
            self.loop.run_forever()
        except Exception as e:
            _safe_print(f"WebRTC Guardian server error: {e}")
        finally:
            self._cleanup_loop()

    def _setup_ngrok(self):
        """Attempts to open a public pyngrok tunnel to the local port."""
        if ngrok is None:
            return

        try:
            token = getattr(config, "NGROK_AUTHTOKEN", "") or os.environ.get("NGROK_AUTHTOKEN", "")
            if token:
                ngrok.set_auth_token(token)

            tunnel = ngrok.connect(self.port, "http")
            pub_url = tunnel.public_url
            if pub_url.startswith("http://"):
                pub_url = pub_url.replace("http://", "https://")
            self.public_url = pub_url
        except Exception as e:
            err_msg = str(e)
            _safe_print(f"[pyngrok] Public tunnel not opened ({err_msg[:80]}...). Using local network.")
            if "ERR_NGROK_4018" in err_msg or "authentication failed" in err_msg:
                _safe_print("   TIP: To enable public mobile streaming, run: ngrok config add-authtoken <TOKEN>")

    async def _start_aiohttp_server(self):
        """Creates the aiohttp application with HTML, MJPEG, Talkback, and status endpoints."""
        app = web.Application()
        self.app = app
        app.router.add_get("/", self._handle_index)
        app.router.add_get("/mjpeg", self._handle_mjpeg)
        app.router.add_post("/guardian_audio", self._handle_guardian_audio)
        app.router.add_post("/offer", self._handle_offer)
        app.router.add_get("/status", self._handle_status)

        self.runner = web.AppRunner(app)
        await self.runner.setup()
        site = web.TCPSite(self.runner, "0.0.0.0", self.port)
        await site.start()

    async def _handle_mjpeg(self, request):
        """
        Streams live frames as Motion JPEG (MJPEG) over standard HTTP multipart.
        Bypasses strict 5G / CGNAT mobile firewalls that block WebRTC UDP traffic.
        Optimized for ultra-low latency on cellular networks (480p, quality 45, ~14 FPS).
        """
        response = web.StreamResponse(
            status=200,
            reason="OK",
            headers={
                "Content-Type": "multipart/x-mixed-replace; boundary=frame",
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "ngrok-skip-browser-warning": "true"
            }
        )
        await response.prepare(request)

        try:
            while True:
                frame = self.get_latest_frame()
                if frame is None:
                    await asyncio.sleep(0.07)
                    continue

                # Downscale streaming frames to 480p width to eliminate mobile buffer bloat
                h, w = frame.shape[:2]
                stream_frame = cv2.resize(frame, (480, int(480 * h / w)))

                # Set JPEG encoding quality to 45
                ret, buffer = cv2.imencode(".jpg", stream_frame, [cv2.IMWRITE_JPEG_QUALITY, 45])
                if not ret:
                    await asyncio.sleep(0.07)
                    continue

                await response.write(
                    b"--frame\r\n"
                    b"Content-Type: image/jpeg\r\n\r\n" + buffer.tobytes() + b"\r\n"
                )
                await asyncio.sleep(0.07)
        except Exception:
            pass

        return response

    async def _handle_guardian_audio(self, request):
        """
        Receives incoming walkie-talkie audio from the guardian web portal.
        Saves and plays audio through the blind user's speakers in a background thread.
        """
        try:
            audio_bytes = await request.read()
            if not audio_bytes:
                return web.json_response({"status": "empty"}, status=400)

            temp_path = os.path.join(config.BASE_DIR, "temp_guardian_voice.webm")
            with open(temp_path, "wb") as f:
                f.write(audio_bytes)

            _safe_print(f"\n[TALKBACK] Received {len(audio_bytes)} bytes of guardian voice audio.")

            # Play in background thread so HTTP response returns immediately
            threading.Thread(
                target=self._play_guardian_voice_worker,
                args=(temp_path,),
                daemon=True,
                name="GuardianVoicePlayer"
            ).start()

            return web.json_response(
                {"status": "received", "bytes": len(audio_bytes)},
                headers={"ngrok-skip-browser-warning": "true"}
            )
        except Exception as e:
            _safe_print(f"[TALKBACK ERROR] {e}")
            return web.json_response({"status": "error", "error": str(e)}, status=500)

    def _play_guardian_voice_worker(self, audio_path):
        """Plays the guardian voice clip through the user's speakers with an alert announcement."""
        try:
            # 1. Announce incoming message via audio arbitrator
            self.arbitrator.emit_alert(
                0,
                "Guardian voice message incoming.",
                time.time(),
                key="guardian_voice_alert"
            )
            time.sleep(1.8)

            # 2. Decode and play audio via PyAV and sounddevice
            played = False
            try:
                import av
                import sounddevice as sd
                container = av.open(audio_path)
                if container.streams.audio:
                    audio_stream = container.streams.audio[0]
                    sample_rate = audio_stream.rate or 44100
                    resampler = av.AudioResampler(format="s16", layout="mono", rate=sample_rate)
                    frames = []
                    for frame in container.decode(audio_stream):
                        rf = resampler.resample(frame)
                        if rf:
                            if isinstance(rf, list):
                                for f in rf:
                                    frames.append(f.to_ndarray())
                            else:
                                frames.append(rf.to_ndarray())
                    container.close()
                    if frames:
                        data = np.concatenate(frames, axis=1).squeeze()
                        sd.play(data, sample_rate)
                        sd.wait()
                        played = True
            except Exception as av_err:
                _safe_print(f"[AUDIO] PyAV/sounddevice playback notice: {av_err}")

            # Fallback playback via pygame
            if not played:
                try:
                    import pygame
                    if not pygame.mixer.get_init():
                        pygame.mixer.init()
                    pygame.mixer.music.load(audio_path)
                    pygame.mixer.music.play()
                    while pygame.mixer.music.get_busy():
                        time.sleep(0.1)
                    played = True
                except Exception:
                    pass

            # Fallback playback via Windows PowerShell media player
            if not played and sys.platform == "win32":
                try:
                    import subprocess
                    cmd = f'powershell -c "$p = New-Object System.Windows.Media.MediaPlayer; $p.Open(\'{audio_path}\'); $p.Play(); Start-Sleep -Seconds 5"'
                    subprocess.run(cmd, shell=True, capture_output=True, timeout=8)
                except Exception:
                    pass

        except Exception as e:
            _safe_print(f"[AUDIO PLAYBACK ERROR] {e}")

    async def _handle_index(self, request):
        return web.Response(
            content_type="text/html",
            text=INDEX_HTML,
            headers={"ngrok-skip-browser-warning": "true"}
        )

    async def _handle_status(self, request):
        return web.json_response({
            "status": "online",
            "sos_active": True,
            "room": self.current_room,
            "active_connections": len(self.pcs),
            "public_url": self.public_url,
            "local_url": self.local_url
        }, headers={"ngrok-skip-browser-warning": "true"})

    async def _handle_offer(self, request):
        params = await request.json()
        offer = RTCSessionDescription(sdp=params["sdp"], type=params["type"])

        # Configure STUN & TURN servers for 5G cellular / CGNAT traversal
        ice_servers = [
            RTCIceServer(urls="stun:stun.l.google.com:19302"),
            RTCIceServer(urls="turn:openrelay.metered.ca:80", username="openrelayproject", credential="openrelayproject"),
            RTCIceServer(urls="turn:openrelay.metered.ca:443", username="openrelayproject", credential="openrelayproject"),
        ]
        config = RTCConfiguration(iceServers=ice_servers)
        pc = RTCPeerConnection(configuration=config)
        self.pcs.add(pc)

        @pc.on("connectionstatechange")
        async def on_connectionstatechange():
            if pc.connectionState in ["failed", "closed"]:
                await pc.close()
                self.pcs.discard(pc)

        # Attach custom video track that reads latest OpenCV frames
        video_track = GogglesVideoStreamTrack(self.get_latest_frame)
        pc.addTrack(video_track)

        await pc.setRemoteDescription(offer)
        answer = await pc.createAnswer()
        await pc.setLocalDescription(answer)

        return web.json_response(
            {
                "sdp": pc.localDescription.sdp,
                "type": pc.localDescription.type
            },
            headers={"ngrok-skip-browser-warning": "true"}
        )

    def _cleanup_loop(self):
        """Cleans up asyncio event loop and active peer connections."""
        if self.loop is None:
            return

        async def _close():
            for pc in list(self.pcs):
                await pc.close()
            self.pcs.clear()
            if self.runner is not None:
                await self.runner.cleanup()

        try:
            self.loop.run_until_complete(_close())
        except Exception:
            pass

    def stop(self):
        """Stops the guardian server and disconnects ngrok tunnels."""
        self.is_running = False
        try:
            if ngrok is not None:
                ngrok.kill()
        except Exception:
            pass

        if self.loop is not None and self.loop.is_running():
            self.loop.call_soon_threadsafe(self.loop.stop)
