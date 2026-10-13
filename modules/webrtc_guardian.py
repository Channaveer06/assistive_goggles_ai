
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
            <div id="overlayStatus" class="overlay-status">
                <span>🔄 Connecting to goggles WebRTC feed...</span>
            </div>
            <video id="video" autoplay playsinline muted controls style="width: 100%; border-radius: 8px;"></video>
        </div>
        <div class="info-bar">
            <div class="meta-group">
                <div class="meta-item">
                    <span class="meta-label">Protocol</span>
                    <span class="meta-value">WebRTC Low Latency (&lt;100ms)</span>
                </div>
                <div class="meta-item">
                    <span class="meta-label">Stream Feed</span>
                    <span class="meta-value">Live POV + Metric HUD</span>
                </div>
                <div class="meta-item">
                    <span class="meta-label">Status</span>
                    <span id="connStatus" class="meta-value" style="color: #fbbf24;">Negotiating</span>
                </div>
            </div>
            <button id="btnReconnect" class="btn-reconnect" onclick="startStream()">Reconnect Stream</button>
        </div>
    </main>

    <section class="instructions">
        <h3>🛡️ Guardian Emergency Guidance</h3>
        <p>This secure live stream delivers the POV perspective of the blind user along with real-time AI computer vision telemetry (metric depth, obstacle bounding boxes, and indoor location). Keep this portal open to monitor surroundings and provide voice assistance if needed.</p>
    </section>

    <script>
        let pc = null;

        async function startStream() {
            const overlay = document.getElementById('overlayStatus');
            const statusEl = document.getElementById('connStatus');
            const videoEl = document.getElementById('video') || document.getElementById('remoteVideo');

            overlay.innerHTML = '<span>🔄 Negotiating WebRTC peer connection...</span>';
            statusEl.innerText = 'Connecting...';
            statusEl.style.color = '#fbbf24';

            if (pc) {
                try {
                    pc.close();
                } catch (e) {}
            }

            const config = {
                iceServers: [
                    { urls: 'stun:stun.l.google.com:19302' },
                    { urls: 'turn:openrelay.metered.ca:80', username: 'openrelayproject', credential: 'openrelayproject' },
                    { urls: 'turn:openrelay.metered.ca:443', username: 'openrelayproject', credential: 'openrelayproject' }
                ]
            };
            pc = new RTCPeerConnection(config);

            pc.addEventListener('track', (evt) => {
                if (evt.track.kind === 'video') {
                    videoEl.srcObject = evt.streams[0];
                    videoEl.play().catch(e => console.log('Autoplay hint:', e));
                    overlay.innerHTML = '<span>🟢 LIVE ZERO-LAG POV FEED</span>';
                    overlay.style.color = '#4ade80';
                    statusEl.innerText = 'Live Streaming';
                    statusEl.style.color = '#4ade80';
                }
            });

            pc.addEventListener('connectionstatechange', () => {
                if (pc.connectionState === 'connected') {
                    statusEl.innerText = 'Connected';
                    statusEl.style.color = '#4ade80';
                } else if (pc.connectionState === 'failed' || pc.connectionState === 'disconnected') {
                    statusEl.innerText = 'Disconnected';
                    statusEl.style.color = '#f87171';
                    overlay.innerHTML = '<span>⚠️ Connection lost. Click Reconnect.</span>';
                }
            });

            pc.addTransceiver('video', { direction: 'recvonly' });

            try {
                const offer = await pc.createOffer();
                await pc.setLocalDescription(offer);

                // Wait for local ICE gathering to complete before posting offer
                await new Promise((resolve) => {
                    if (pc.iceGatheringState === 'complete') {
                        resolve();
                    } else {
                        function check() {
                            if (pc.iceGatheringState === 'complete') {
                                pc.removeEventListener('icegatheringstatechange', check);
                                resolve();
                            }
                        }
                        pc.addEventListener('icegatheringstatechange', check);
                        setTimeout(resolve, 1500); // 1.5s timeout fallback
                    }
                });

                const response = await fetch('/offer', {
                    method: 'POST',
                    headers: {
                        'Content-Type': 'application/json',
                        'ngrok-skip-browser-warning': 'true'
                    },
                    body: JSON.stringify({
                        sdp: pc.localDescription.sdp,
                        type: pc.localDescription.type
                    })
                });

                if (!response.ok) {
                    throw new Error('Server rejected SDP offer: ' + response.statusText);
                }

                const answer = await response.json();
                await pc.setRemoteDescription(new RTCSessionDescription(answer));
            } catch (err) {
                console.error('WebRTC error:', err);
                overlay.innerHTML = '<span>❌ Stream Error: ' + err.message + '</span>';
                overlay.style.color = '#f87171';
                statusEl.innerText = 'Error';
                statusEl.style.color = '#f87171';
            }
        }

        window.addEventListener('load', () => {
            startStream();
        });
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

    def start_stream(self, current_room="study_room", timestamp=0.0):
        """
        Launches the WebRTC server and pyngrok tunnel in a background daemon thread.
        Non-blocking: returns immediately while the server initializes asynchronously.
        """
        self.current_room = current_room

        if self.is_running and (self.public_url or self.local_url):
            url = self.public_url or self.local_url
            _safe_print(f"\n[GUARDIAN SOS] Live stream is already active: {url}")
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
                self.arbitrator.io.notify_guardian(current_room, dispatch_url)
            elif self.io_bridge is not None:
                self.io_bridge.notify_guardian(current_room, dispatch_url)

            # 5. Speak notification via priority audio arbitrator
            self.arbitrator.emit_alert(
                0,
                "Emergency Guardian SOS active. Secure live stream link generated.",
                timestamp or time.time(),
                key="sos_link_ready"
            )

            # 6. Keep loop running to handle WebRTC requests
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
        """Creates the aiohttp application with HTML & WebRTC endpoints."""
        app = web.Application()
        app.router.add_get("/", self._handle_index)
        app.router.add_post("/offer", self._handle_offer)
        app.router.add_get("/status", self._handle_status)

        self.runner = web.AppRunner(app)
        await self.runner.setup()
        site = web.TCPSite(self.runner, "0.0.0.0", self.port)
        await site.start()

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
