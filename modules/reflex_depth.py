import os
import cv2
import torch
import numpy as np
from PIL import Image
from ultralytics import YOLO
from transformers import AutoImageProcessor, AutoModelForDepthEstimation
import config

class ReflexDepthEngine:
    ALERT_CLASSES = {"person", "car", "motorcycle", "bus", "truck"}

    def __init__(self, arbitrator, social_engine=None):
        self.arbitrator = arbitrator
        self.social_engine = social_engine
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Initializing Reflex & Metric Depth Engine on {self.device.upper()}...")

        # 1. Load YOLOv11 with BoT-SORT / ByteTrack tracking support
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        model_path = os.path.join(base_dir, "yolo11n.pt")
        self.detector = YOLO(model_path if os.path.exists(model_path) else "yolo11n.pt")
        self.last_detections = []

        # Tracking cooldowns and trajectory memory
        self.person_cooldowns = {}     # {track_id: last_announced_timestamp}
        self.track_depth_history = {}  # {track_id: [(timestamp, depth), ...]}
        self.motion_cooldowns = {}     # {track_id: (last_motion_timestamp, status)}

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

    def set_social_engine(self, social_engine):
        """Connects social engine for unified person identity & micro-cue announcements."""
        self.social_engine = social_engine

    def check_frame_quality(self, frame, timestamp):
        """VizWiz-inspired pre-check for dark rooms or severe camera blur."""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        brightness = float(np.mean(gray))
        sharpness = float(cv2.Laplacian(gray, cv2.CV_64F).var())

        if brightness < config.MIN_BRIGHTNESS:
            self.arbitrator.emit_alert(3, "Room is very dark, visibility reduced", timestamp, key="dark")
        elif sharpness < config.MIN_SHARPNESS_LAPLACIAN:
            self.arbitrator.emit_alert(3, "Camera motion blur detected, hold steady", timestamp, key="blur")
        return brightness, sharpness

    @torch.inference_mode()
    def estimate_metric_depth(self, frame_bgr):
        """Computes a full-frame metric depth map where pixel values = distance in meters."""
        h, w = frame_bgr.shape[:2]
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
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
        """Runs 1) Quality Check, 2) BoT-SORT Tracking, 3) Inner-Box Meter Depth, 4) Audio Alert with Movement Trajectory."""
        h, w = frame.shape[:2]
        self.check_frame_quality(frame, timestamp)

        # Compute metric depth map (meters)
        depth_map = self.estimate_metric_depth(frame)

        # Run YOLOv11 + BoT-SORT persistent tracking
        try:
            results = self.detector.track(frame, persist=True, tracker="botsort.yaml", verbose=False)[0]
        except Exception:
            results = self.detector.track(frame, persist=True, tracker="bytetrack.yaml", verbose=False)[0]

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
                # Crop inner 50% of bounding box to get pure object depth (ignores background edges)
                bw, bh = x2 - x1, y2 - y1
                ix1, iy1 = max(0, x1 + int(bw * 0.25)), max(0, y1 + int(bh * 0.25))
                ix2, iy2 = min(w, x2 - int(bw * 0.25)), min(h, y2 - int(bh * 0.25))

                roi_depth = depth_map[iy1:iy2, ix1:ix2]
                dist_m = float(np.median(roi_depth)) if roi_depth.size > 0 else 9.9
                detections.append((dist_m, x1, y1, x2, y2, cls_id, track_id))

            detections.sort(key=lambda item: item[0])

            # Prune stale tracking depth history if older than 5 seconds
            stale_keys = [k for k in self.track_depth_history if not any(t == k for _, _, _, _, _, _, t in detections)]
            if len(stale_keys) > 50:
                for k in stale_keys[:25]:
                    self.track_depth_history.pop(k, None)

            for dist_m, x1, y1, x2, y2, cls_id, track_id in detections:
                label_name = self.detector.names[cls_id]
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
                    if delta < -0.4:
                        motion_status = "moving closer"
                    elif delta > 0.4:
                        motion_status = "moving away"

                # Color code: Red for Critical (<1.2m), Orange for Warning (<2.5m), Green for Safe
                color = (0, 255, 0)
                if dist_m <= config.CRITICAL_DANGER_M:
                    color = (0, 0, 255)
                elif dist_m <= config.WARNING_DISTANCE_M:
                    color = (0, 165, 255)

                face_name, face_cues = None, None
                if label_name == "person" and self.social_engine is not None:
                    face_name, face_cues = self.social_engine.get_face_for_person_box((x1, y1, x2, y2))

                if label_name == "person" and face_name and not face_name.startswith("Verifying"):
                    announced_name = face_name if face_name != "Unknown person" else "person"
                    cues_str = f", {' and '.join(face_cues)}" if face_cues else ""
                else:
                    announced_name = label_name
                    cues_str = ""

                cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
                motion_tag = f" [{motion_status}]" if motion_status else ""
                tag = f"ID#{track_id} {announced_name}: {dist_m:.1f}m ({direction}){motion_tag}"
                cv2.putText(annotated, tag, (x1, max(20, y1 - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2)

                # Clutter Filter: Only ALERT_CLASSES trigger unprompted telemetry
                # Everyday items (bottles, cups, chairs, phones) are suppressed unless explicitly queried in Find Mode
                if label_name in self.ALERT_CLASSES and dist_m <= config.WARNING_DISTANCE_M:
                    last_motion_t, last_motion_type = self.motion_cooldowns.get(track_id, (0.0, None))

                    if motion_status == "moving closer":
                        # Safety override: moving closer preempts the 20-second cooldown immediately
                        if (timestamp - last_motion_t) >= 3.0 or last_motion_type != "moving closer":
                            self.motion_cooldowns[track_id] = (timestamp, "moving closer")
                            self.person_cooldowns[track_id] = timestamp
                            msg = f"{announced_name.capitalize()} moving closer, now {dist_m:.1f} meters."
                            speech_res = self.arbitrator.emit_alert(1, msg, timestamp, key=f"trk_motion_{track_id}")
                            if speech_res:
                                triggered_speech = speech_res
                    elif motion_status == "moving away":
                        if (timestamp - last_motion_t) >= 4.0 or last_motion_type != "moving away":
                            self.motion_cooldowns[track_id] = (timestamp, "moving away")
                            self.person_cooldowns[track_id] = timestamp
                            msg = f"{announced_name.capitalize()} moving away, {dist_m:.1f} meters."
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
                            msg = f"{prefix}{announced_name} {dist_m:.1f} meters {direction}{cues_str}"
                            speech_res = self.arbitrator.emit_alert(priority, msg, timestamp, key=f"trk_{track_id}")
                            if speech_res:
                                triggered_speech = speech_res

        # Create colorized metric depth visualization for side-by-side inspection
        depth_norm = np.clip((depth_map / 6.0) * 255.0, 0, 255).astype(np.uint8)
        depth_colormap = cv2.applyColorMap(255 - depth_norm, cv2.COLORMAP_INFERNO)
        combined_hud = np.hstack((annotated, depth_colormap))
        return combined_hud, triggered_speech
