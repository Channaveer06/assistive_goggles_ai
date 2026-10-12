import os
import cv2
import torch
import numpy as np
from PIL import Image
from ultralytics import YOLO
from transformers import AutoImageProcessor, AutoModelForDepthEstimation
import config

class ReflexDepthEngine:
    def __init__(self, arbitrator):
        self.arbitrator = arbitrator
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Initializing Reflex & Metric Depth Engine on {self.device.upper()}...")

        # 1. Load YOLOv11 with ByteTrack support
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        model_path = os.path.join(base_dir, "yolo11n.pt")
        self.detector = YOLO(model_path if os.path.exists(model_path) else "yolo11n.pt")
        self.last_detections = []

        # 2. Load Depth Anything V2 Metric Indoor (ViT-Small) -> Returns real METERS
        self.depth_processor = AutoImageProcessor.from_pretrained(config.METRIC_DEPTH_MODEL)
        self.depth_model = AutoModelForDepthEstimation.from_pretrained(
            config.METRIC_DEPTH_MODEL
        ).to(self.device).eval()

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
        """Runs 1) Quality Check, 2) ByteTrack Detection, 3) Inner-Box Meter Depth, 4) Audio Alert."""
        h, w = frame.shape[:2]
        self.check_frame_quality(frame, timestamp)

        # Compute metric depth map (meters)
        depth_map = self.estimate_metric_depth(frame)

        # Run YOLOv11 + ByteTrack persistent tracking
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

            for dist_m, x1, y1, x2, y2, cls_id, track_id in detections:
                label_name = self.detector.names[cls_id]
                x_center_ratio = ((x1 + x2) / 2.0) / w
                direction = self.arbitrator.get_spatial_direction(x_center_ratio)

                self.last_detections.append({
                    "label": label_name,
                    "dist_m": dist_m,
                    "direction": direction,
                    "center_x": (x1 + x2) / 2.0,
                    "center_y": (y1 + y2) / 2.0
                })

                # Color code: Red for Critical (<1.2m), Orange for Warning (<2.5m), Green for Safe
                color = (0, 255, 0)
                if dist_m <= config.CRITICAL_DANGER_M:
                    color = (0, 0, 255)
                elif dist_m <= config.WARNING_DISTANCE_M:
                    color = (0, 165, 255)

                cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
                tag = f"ID#{track_id} {label_name}: {dist_m:.1f}m ({direction})"
                cv2.putText(annotated, tag, (x1, max(20, y1 - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2)

                # Trigger audio if inside warning zone and cooldown has expired
                if dist_m <= config.WARNING_DISTANCE_M:
                    if self.arbitrator.should_announce_track(track_id, dist_m, timestamp):
                        priority = 1 if dist_m <= config.CRITICAL_DANGER_M else 2
                        prefix = "Stop! " if priority == 1 else ""
                        msg = f"{prefix}{label_name} {dist_m:.1f} meters {direction}"
                        triggered_speech = self.arbitrator.emit_alert(priority, msg, timestamp, key=f"trk_{track_id}")

        # Create colorized metric depth visualization for side-by-side inspection
        depth_norm = np.clip((depth_map / 6.0) * 255.0, 0, 255).astype(np.uint8)
        depth_colormap = cv2.applyColorMap(255 - depth_norm, cv2.COLORMAP_INFERNO)
        combined_hud = np.hstack((annotated, depth_colormap))
        return combined_hud, triggered_speech
