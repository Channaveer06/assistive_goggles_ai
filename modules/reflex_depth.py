import os
import time
import queue
import threading
import cv2
import torch
import numpy as np
from PIL import Image
from ultralytics import YOLO
from transformers import AutoImageProcessor, AutoModelForDepthEstimation
import config

# Pinhole Optical Depth Prior Parameters (Canonical physical dimensions in meters)
CANONICAL_SIZES = {
    "person": 1.65,
    "cell phone": 0.15,
    "laptop": 0.28,
    "bottle": 0.22,
    "cup": 0.10,
    "chair": 0.85,
    "tv": 0.60,
    "book": 0.24,
    "keyboard": 0.15,
    "mouse": 0.10,
}
FOCAL_LENGTH_PX = 540.0

def sanitize_person_name(name):
    """
    Sanitizes person identity announcements.
    If the name resembles a camera timestamp, UUID, or track ID (e.g. 'WIN_20261107...', '20261107 20 54 56'),
    simply returns 'person' so TTS never announces raw file digits or IDs.
    """
    if not name or not isinstance(name, str):
        return "person"
    clean = name.replace("WIN_", "").replace("_Pro", "").replace("_win", "").replace(" win", "").replace("_2", "").replace("_", " ").strip()
    digits = sum(c.isdigit() for c in clean)
    if digits >= 3 or clean.lower().startswith("win") or len(clean) < 2:
        return "person"
    if clean.lower() in ("unknown", "unknown person", "verifying...", "id"):
        return "person"
    return clean

class ReflexDepthEngine:
    ALERT_CLASSES = {"person", "car", "motorcycle", "bus", "truck"}

    def __init__(self, arbitrator, social_engine=None, track_identities=None, async_recognize_cb=None):
        self.arbitrator = arbitrator
        self.social_engine = social_engine
        self.track_identities = track_identities if track_identities is not None else {}
        self.async_recognize_cb = async_recognize_cb
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Initializing Reflex & Metric Depth Engine on {self.device.upper()}...")

        # 1. Load YOLOv11 with BoT-SORT / ByteTrack tracking support
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        model_path = os.path.join(base_dir, "yolo11n.pt")
        self.detector = YOLO(model_path if os.path.exists(model_path) else "yolo11n.pt")
        self.last_detections = []
        self.last_yolo_time_ms = 0.0
        self.last_depth_time_ms = 0.0
        self.model_name = "YOLOv8n"

        # Tracking cooldowns, trajectory memory, and monotonic area history
        self.person_cooldowns = {}     # {track_id: last_announced_timestamp}
        self.track_depth_history = {}  # {track_id: [(timestamp, depth), ...]}
        self.track_box_area = {}       # {track_id: (area, dist_m)} for monotonic distance enforcement
        self.motion_cooldowns = {}     # {track_id: (last_motion_timestamp, status)}
        self.dropoff_detected = False
        self._dropoff_consecutive_count = 0

        # 2. Load Depth Anything V2 Metric Indoor (ViT-Small) -> Returns real METERS
        try:
            self.depth_processor = AutoImageProcessor.from_pretrained(config.METRIC_DEPTH_MODEL, local_files_only=True)
            self.depth_model = AutoModelForDepthEstimation.from_pretrained(
                config.METRIC_DEPTH_MODEL, local_files_only=True
            ).to(self.device).eval()
        except Exception:
            self.depth_processor = AutoImageProcessor.from_pretrained(config.METRIC_DEPTH_MODEL)
            self.depth_model = AutoModelForDepthEstimation.from_pretrained(
                config.METRIC_DEPTH_MODEL
            ).to(self.device).eval()

        # Decoupled Asynchronous Depth Worker: Runs Depth Anything V2 in background
        # to ensure main YOLO reflex loop maintains a stable 25+ FPS without starvation.
        self.latest_depth_map = None
        self.depth_queue = queue.Queue(maxsize=1)
        self._depth_worker_running = True
        self._depth_worker_thread = threading.Thread(target=self._async_depth_worker, daemon=True)
        self._depth_worker_thread.start()

    def _async_depth_worker(self):
        """Continuously updates metric depth in background daemon thread with GIL yielding."""
        while self._depth_worker_running:
            try:
                frame_bgr = self.depth_queue.get(timeout=0.2)
                if frame_bgr is None:
                    break
                t0 = time.perf_counter()
                orig_threads = torch.get_num_threads()
                torch.set_num_threads(1)
                d_map = self.estimate_metric_depth(frame_bgr)
                torch.set_num_threads(orig_threads)
                self.latest_depth_map = d_map
                self.last_depth_time_ms = (time.perf_counter() - t0) * 1000.0
                time.sleep(0.2) # Fast yield without CPU starvation, refreshes depth smoothly
            except queue.Empty:
                continue
            except Exception:
                time.sleep(0.1)

    def set_social_engine(self, social_engine):
        """Connects social engine for unified person identity & micro-cue announcements."""
        self.social_engine = social_engine

    def detect_negative_obstacle(self, depth_map):
        """
        Negative Obstacle (Drop-off & Descending Stairs) Detection via Depth Gradients.
        Extracts bottom 30% center ground ROI and compares top vs bottom band metric depths.
        Returns: (dropoff_detected: bool, top_band_depth: float, bottom_band_depth: float)
        """
        # Guard against uninitialized or placeholder depth map
        if getattr(self, "latest_depth_map", None) is None or np.all(depth_map == 2.5):
            self._dropoff_consecutive_count = 0
            return False, 0.0, 0.0

        h, w = depth_map.shape[:2]
        ground_roi = depth_map[int(h * 0.7):h, int(w * 0.3):int(w * 0.7)]
        gh = ground_roi.shape[0]
        if gh < 4 or ground_roi.shape[1] < 4:
            self._dropoff_consecutive_count = 0
            return False, 0.0, 0.0

        half_gh = int(gh * 0.5)
        top_band = ground_roi[:half_gh, :]
        bottom_band = ground_roi[half_gh:, :]

        top_band_depth = float(np.median(top_band)) if top_band.size > 0 else 0.0
        bottom_band_depth = float(np.median(bottom_band)) if bottom_band.size > 0 else 0.0

        raw_dropoff = False
        if bottom_band.size > 0 and top_band.size > 0:
            # On flat ground / floor, bottom_band (near feet) is strictly CLOSER than top_band.
            # A true negative obstacle (drop-off / descending stairs) only occurs when bottom_band
            # is significantly DEEPER than top_band by at least 0.75m, indicating the ground falls away abruptly.
            if top_band_depth > 0.4 and (bottom_band_depth > top_band_depth + 0.75):
                raw_dropoff = True

        if raw_dropoff:
            self._dropoff_consecutive_count += 1
        else:
            self._dropoff_consecutive_count = 0

        # Require at least 2 consecutive frames of negative depth gradient to prevent single-frame noise
        dropoff_detected = (self._dropoff_consecutive_count >= 2)
        return dropoff_detected, top_band_depth, bottom_band_depth

    def check_frame_quality(self, frame, timestamp):
        """VizWiz-inspired pre-check for dark rooms or severe camera blur."""
        small_gray = cv2.resize(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), (160, 120))
        brightness = float(np.mean(small_gray))
        sharpness = float(cv2.Laplacian(small_gray, cv2.CV_32F).var())

        if brightness < config.MIN_BRIGHTNESS:
            self.arbitrator.emit_alert(3, "Room is very dark, visibility reduced", timestamp, key="dark")
        elif sharpness < config.MIN_SHARPNESS_LAPLACIAN:
            self.arbitrator.emit_alert(3, "Camera motion blur detected, hold steady", timestamp, key="blur")
        return brightness, sharpness

    @torch.inference_mode()
    def estimate_metric_depth(self, frame_bgr):
        """Computes a full-frame metric depth map where pixel values = distance in meters."""
        h, w = frame_bgr.shape[:2]
        small_frame = cv2.resize(frame_bgr, (320, 240))
        rgb = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)
        inputs = self.depth_processor(images=Image.fromarray(rgb), return_tensors="pt").to(self.device)
        outputs = self.depth_model(**inputs)

        depth_map = torch.nn.functional.interpolate(
            outputs.predicted_depth.unsqueeze(1),
            size=(h, w),
            mode="bicubic",
            align_corners=False,
        ).squeeze().cpu().numpy()
        return depth_map

    def process_frame(self, frame, timestamp=0.0):
        """Runs 1) Quality Check, 2) BoT-SORT Tracking, 3) Inner-Box Meter Depth, 4) Drop-off Detection, 5) Audio Alert with Movement Trajectory."""
        h, w = frame.shape[:2]
        self.check_frame_quality(frame, timestamp)

        # Enqueue frame for background depth worker without blocking main 30 FPS loop
        if self.depth_queue.empty():
            try:
                self.depth_queue.put_nowait(frame.copy())
            except Exception:
                pass

        # Use latest asynchronous depth map (or fast fallback baseline if initial boot)
        if self.latest_depth_map is None:
            self.latest_depth_map = np.full((h, w), 2.5, dtype=np.float32)

        depth_map = self.latest_depth_map

        # Negative Obstacle (Drop-off & Descending Stairs) Detection via Ground ROI Depth Gradients
        dropoff_detected, top_depth, bottom_depth = self.detect_negative_obstacle(depth_map)
        self.dropoff_detected = dropoff_detected

        # Run YOLO with imgsz=256 for sharp smartphone/small-object recognition while sustaining >25 FPS
        t_yolo_0 = time.perf_counter()
        try:
            results = self.detector.track(frame, imgsz=256, persist=True, tracker="bytetrack.yaml", verbose=False)[0]
        except Exception:
            results = self.detector.track(frame, imgsz=256, persist=True, tracker="botsort.yaml", verbose=False)[0]
        self.last_yolo_time_ms = (time.perf_counter() - t_yolo_0) * 1000.0

        self.last_results = results
        annotated = frame.copy()
        triggered_speech = None
        self.last_detections = []

        if results.boxes is not None and len(results.boxes) > 0:
            boxes = results.boxes.xyxy.cpu().numpy()
            classes = results.boxes.cls.cpu().numpy().astype(int)
            track_ids = (
                results.boxes.id.cpu().numpy().astype(int)
                if results.boxes.id is not None
                else np.arange(len(boxes))
            )

            # Sort detections by closest distance first so nearest hazard always wins
            detections = []
            for box, cls_id, track_id in zip(boxes, classes, track_ids):
                x1, y1, x2, y2 = map(int, box)
                bw, bh = x2 - x1, y2 - y1
                # Sample center 30% of bounding box to get pure object core (avoids table/wall boundary bleed)
                ix1, iy1 = max(0, x1 + int(bw * 0.35)), max(0, y1 + int(bh * 0.35))
                ix2, iy2 = min(w, x2 - int(bw * 0.35)), min(h, y2 - int(bh * 0.35))

                roi_depth = depth_map[iy1:iy2, ix1:ix2]
                raw_metric_m = float(np.median(roi_depth)) if roi_depth.size > 0 else 9.9
                detections.append((raw_metric_m, x1, y1, x2, y2, cls_id, track_id))

            detections.sort(key=lambda item: item[0])

            # Prune stale tracking depth history if older than 5 seconds
            stale_keys = [k for k in self.track_depth_history if not any(t == k for _, _, _, _, _, _, t in detections)]
            if len(stale_keys) > 50:
                for k in stale_keys[:25]:
                    self.track_depth_history.pop(k, None)
                    self.track_box_area.pop(k, None)

            for raw_metric_m, x1, y1, x2, y2, cls_id, track_id in detections:
                label_name = self.detector.names[cls_id]
                bw, bh = x2 - x1, y2 - y1
                box_area = float(bw * bh)
                frame_area = float(w * h)
                area_ratio = box_area / max(1.0, frame_area)
                aspect = max(bw, bh) / max(1.0, min(bw, bh))

                # Disambiguate smartphone vs laptop misclassification
                # Smartphones are handheld, smaller area (< 25% of frame) or elongated (aspect >= 1.45) or held portrait (bh >= bw * 0.95)
                if label_name == "laptop":
                    if bh >= (bw * 0.95) or (aspect >= 1.45 and area_ratio < 0.28) or area_ratio < 0.18:
                        label_name = "cell phone"

                # Pinhole Optical Depth Prior & Monotonic Fusion
                # Bounding box expansion physically guarantees distance decreases as user approaches
                dist_m = raw_metric_m
                if label_name in CANONICAL_SIZES:
                    target_dim = CANONICAL_SIZES[label_name]
                    ref_px = max(bh, bw) if label_name in ("cell phone", "bottle", "cup") else max(bw, bh)
                    if ref_px > 15:
                        d_optical = float(np.clip((FOCAL_LENGTH_PX * target_dim) / ref_px, 0.2, 5.0))
                        # In close proximity or on glossy surfaces (screens), fuse optical & metric
                        if d_optical < 1.8:
                            dist_m = float(0.65 * d_optical + 0.35 * min(raw_metric_m, d_optical * 1.25))
                        else:
                            dist_m = float(0.35 * d_optical + 0.65 * raw_metric_m)

                # Strict track-level monotonic distance enforcement:
                # If the bounding box expanded by >8%, distance MUST NOT increase
                prev_area, prev_dist = self.track_box_area.get(track_id, (0.0, dist_m))
                if prev_area > 0 and box_area > prev_area * 1.08:
                    max_phys_dist = prev_dist * np.sqrt(prev_area / box_area) * 1.05
                    dist_m = min(dist_m, max_phys_dist)
                self.track_box_area[track_id] = (box_area, dist_m)

                x_center_ratio = ((x1 + x2) / 2.0) / w
                direction = self.arbitrator.get_spatial_direction(x_center_ratio)

                self.last_detections.append({
                    "label": label_name,
                    "dist_m": dist_m,
                    "direction": direction,
                    "center_x": (x1 + x2) / 2.0,
                    "center_y": (y1 + y2) / 2.0,
                    "box": (x1, y1, x2, y2),
                    "track_id": track_id
                })

                # Maintain 1.0-second rolling window depth history
                history = self.track_depth_history.setdefault(track_id, [])
                history.append((timestamp, dist_m))
                self.track_depth_history[track_id] = [pt for pt in history if (timestamp - pt[0]) <= 1.0]
                history = self.track_depth_history[track_id]

                # Movement trajectory analysis (delta across rolling window)
                motion_status = None
                if len(history) >= 2 and (history[-1][0] - history[0][0]) >= 0.2:
                    delta = history[-1][1] - history[0][1]
                    if delta < -0.35:
                        motion_status = "moving closer"
                    elif delta > 0.35:
                        motion_status = "moving away"

                # Color code: Red for Critical (<1.2m), Orange for Warning (<2.5m), Green for Safe
                color = (0, 255, 0)
                if dist_m <= config.CRITICAL_DANGER_M:
                    color = (0, 0, 255)
                elif dist_m <= config.WARNING_DISTANCE_M:
                    color = (0, 165, 255)

                face_name, face_cues = None, None
                if label_name == "person":
                    if self.social_engine is not None:
                        face_name, face_cues = self.social_engine.get_face_for_person_box((x1, y1, x2, y2))
                    if self.track_identities and track_id in self.track_identities:
                        stored_id = self.track_identities[track_id]
                        if stored_id and stored_id not in ["Person", "Unknown person"]:
                            face_name = stored_id

                clean_face = sanitize_person_name(face_name) if face_name else "person"
                if label_name == "person":
                    announced_name = clean_face
                    cues_str = f", {' and '.join(face_cues)}" if face_cues else ""
                else:
                    announced_name = label_name
                    cues_str = ""

                resolved_name = sanitize_person_name(self.track_identities.get(track_id, announced_name)) if label_name == "person" else announced_name
                cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
                motion_tag = f" [{motion_status}]" if motion_status else ""
                tag = f"ID#{track_id} {resolved_name.capitalize()}: {dist_m:.1f}m ({direction}){motion_tag}"
                cv2.putText(annotated, tag, (x1, max(20, y1 - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2)

                # Clutter Filter: Only ALERT_CLASSES trigger unprompted telemetry
                # Everyday items (bottles, cups, chairs, phones) are suppressed unless explicitly queried in Find Mode
                if label_name in self.ALERT_CLASSES and dist_m <= config.WARNING_DISTANCE_M:
                    last_motion_t, last_motion_type = self.motion_cooldowns.get(track_id, (0.0, None))
                    person_label = resolved_name.capitalize() if label_name == "person" else announced_name.capitalize()

                    if motion_status == "moving closer":
                        # Safety override: moving closer preempts the 20-second cooldown immediately
                        if (timestamp - last_motion_t) >= 3.0 or last_motion_type != "moving closer":
                            self.motion_cooldowns[track_id] = (timestamp, "moving closer")
                            self.person_cooldowns[track_id] = timestamp
                            msg = f"{person_label} moving closer, now {dist_m:.1f} meters."
                            speech_res = self.arbitrator.emit_alert(1, msg, timestamp, key=f"trk_motion_{track_id}")
                            if speech_res:
                                triggered_speech = speech_res
                    elif motion_status == "moving away":
                        if (timestamp - last_motion_t) >= 4.0 or last_motion_type != "moving away":
                            self.motion_cooldowns[track_id] = (timestamp, "moving away")
                            self.person_cooldowns[track_id] = timestamp
                            msg = f"{person_label} moving away, {dist_m:.1f} meters."
                            speech_res = self.arbitrator.emit_alert(2, msg, timestamp, key=f"trk_motion_{track_id}")
                            if speech_res:
                                triggered_speech = speech_res
                    else:
                        # 20-second cooldown per track ID
                        is_new = track_id not in self.person_cooldowns
                        time_since_last = timestamp - self.person_cooldowns.get(track_id, 0.0)

                        if is_new or time_since_last >= 20.0:
                            self.person_cooldowns[track_id] = timestamp
                            priority = 1 if dist_m <= config.CRITICAL_DANGER_M else 2
                            prefix = "Stop! " if priority == 1 else ""
                            cooldown_label = resolved_name if label_name == "person" else announced_name
                            msg = f"{prefix}{cooldown_label} {dist_m:.1f} meters {direction}{cues_str}"
                            speech_res = self.arbitrator.emit_alert(priority, msg, timestamp, key=f"trk_{track_id}")
                            if speech_res:
                                triggered_speech = speech_res

        # Draw ground plane ROI overlay on annotated HUD
        gx1, gy1 = int(w * 0.3), int(h * 0.7)
        gx2, gy2 = int(w * 0.7), h - 2
        if dropoff_detected:
            cv2.rectangle(annotated, (gx1, gy1), (gx2, gy2), (0, 0, 255), 3)
            cv2.putText(annotated, "WARNING: DROP-OFF / STAIRS", (gx1, max(20, gy1 - 8)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 0, 255), 2, cv2.LINE_AA)
        else:
            cv2.rectangle(annotated, (gx1, gy1), (gx2, gy2), (0, 180, 0), 1)

        # Create colorized metric depth visualization for side-by-side inspection (cached when depth hasn't changed)
        if getattr(self, "_last_depth_colormap_map", None) is not depth_map:
            depth_norm = np.clip((depth_map / 6.0) * 255.0, 0, 255).astype(np.uint8)
            self._cached_depth_colormap = cv2.applyColorMap(255 - depth_norm, cv2.COLORMAP_INFERNO)
            self._last_depth_colormap_map = depth_map

        depth_colormap = self._cached_depth_colormap
        combined_hud = np.hstack((annotated, depth_colormap))
        return combined_hud, triggered_speech, dropoff_detected
