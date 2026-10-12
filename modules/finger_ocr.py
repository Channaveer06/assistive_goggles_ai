import cv2
import re
import numpy as np
import easyocr
import config

class FingerOCRAndUtilities:
    """
    Phase 3 Independence Utilities:
    1. VizWiz Framing Coach (checks blur, darkness, and edge cutoff before reading).
    2. Fingertip Point-to-Read ROI cropping (reads text right above the user's pointing finger, or full frame).
    3. Indian Currency Note (₹10 to ₹500) & Medicine Dosage / Expiry parser.
    4. Room Light Status ("Is the light on?") & Dominant Clothing Color Identifier.
    """
    def __init__(self, arbitrator):
        self.arbitrator = arbitrator
        print("Initializing GPU EasyOCR Engine for Text, Currency & Medicine Reading...")
        self.reader = easyocr.Reader(["en"], gpu=True, verbose=False)

        # Common Indian currency denominations and medicine keywords
        self.currency_patterns = {
            "500": "500 Rupee note",
            "200": "200 Rupee note",
            "100": "100 Rupee note",
            "50": "50 Rupee note",
            "20": "20 Rupee note",
            "10": "10 Rupee note",
        }
        self.medicine_keywords = ["mg", "ml", "tablet", "capsule", "paracetamol", "amoxicillin", "exp", "dosage", "daily"]

    def check_room_light_and_color(self, frame_bgr, timestamp=0.0, priority=0):
        """Answers 'Is the room light on?' and identifies the dominant color in the center of the camera."""
        h, w = frame_bgr.shape[:2]
        gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
        mean_lum = float(np.mean(gray))

        if mean_lum < 40:
            light_status = "The room is dark, lights appear to be off"
        elif mean_lum < 110:
            light_status = "The room has dim indoor lighting"
        else:
            light_status = "The room is brightly lit"

        # Analyze center 30% patch for clothing/object color
        cx1, cy1, cx2, cy2 = int(w * 0.35), int(h * 0.35), int(w * 0.65), int(h * 0.65)
        center_crop = frame_bgr[cy1:cy2, cx1:cx2]
        hsv = cv2.cvtColor(center_crop, cv2.COLOR_BGR2HSV)
        mean_h, mean_s, mean_v = np.mean(hsv.reshape(-1, 3), axis=0)

        if mean_v < 45:
            color_name = "black"
        elif mean_s < 35 and mean_v > 190:
            color_name = "white"
        elif mean_s < 40:
            color_name = "grey"
        elif mean_h < 10 or mean_h >= 165:
            color_name = "red"
        elif mean_h < 25:
            color_name = "orange or brown"
        elif mean_h < 35:
            color_name = "yellow"
        elif mean_h < 85:
            color_name = "green"
        elif mean_h < 130:
            color_name = "blue"
        else:
            color_name = "purple or pink"

        report = f"{light_status}. Center object color is {color_name}."
        self.arbitrator.emit_alert(priority, report, timestamp, key="light_color")
        return report

    def read_scene_or_fingertip(self, frame_bgr, fingertip_xy=None, timestamp=0.0, priority=0):
        """
        Crops region above pointing fingertip (if provided) or full frame,
        applies VizWiz quality check, reads text, and classifies Currency or Medicine.
        """
        h, w = frame_bgr.shape[:2]
        annotated = frame_bgr.copy()

        # 1. If fingertip coordinate (x, y) is detected, crop the text region directly above the fingertip
        if fingertip_xy is not None:
            fx, fy = map(int, fingertip_xy)
            rx1, ry1 = max(0, fx - int(w * 0.45)), max(0, fy - int(h * 0.35))
            rx2, ry2 = min(w, fx + int(w * 0.45)), max(50, fy)
            roi = frame_bgr[ry1:ry2, rx1:rx2]
            cv2.circle(annotated, (fx, fy), 10, (0, 0, 255), -1)
            cv2.rectangle(annotated, (rx1, ry1), (rx2, ry2), (255, 200, 0), 2)
            offset_x, offset_y = rx1, ry1
        else:
            roi = frame_bgr
            offset_x, offset_y = 0, 0

        # 2. VizWiz Quality & Framing Gate
        gray_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        sharpness = float(cv2.Laplacian(gray_roi, cv2.CV_64F).var())
        if sharpness < 15.0:
            msg = "Hold steady, text is too blurry to read clearly."
            self.arbitrator.emit_alert(2, msg, timestamp, key="ocr_blur")
            return annotated, msg

        # 3. Run GPU OCR
        ocr_results = self.reader.readtext(roi)
        if not ocr_results:
            msg = "No readable text found. Tilt camera slightly."
            self.arbitrator.emit_alert(2, msg, timestamp, key="ocr_empty")
            return annotated, msg

        extracted_words = []
        edge_cutoff_warning = False

        for bbox, text, conf in ocr_results:
            if conf < 0.30:
                continue
            extracted_words.append(text.strip())
            pts = np.array(bbox, dtype=np.int32)
            pts[:, 0] += offset_x
            pts[:, 1] += offset_y

            # Check if text touches frame border (VizWiz edge-cutoff check)
            if np.any(pts[:, 0] <= 4) or np.any(pts[:, 0] >= w - 4):
                edge_cutoff_warning = True

            cv2.polylines(annotated, [pts], isClosed=True, color=(0, 255, 0), thickness=2)
            cv2.putText(annotated, text, (pts[0][0], max(20, pts[0][1] - 6)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 0), 2)

        full_text = " ".join(extracted_words)
        lower_text = full_text.lower()

        # 4. Smart Classification: Currency vs. Medicine vs. General Text
        prefix = ""
        if "reserve bank" in lower_text or "rupees" in lower_text or "₹" in lower_text:
            for denom, label in self.currency_patterns.items():
                if re.search(rf"\b{denom}\b", lower_text):
                    prefix = f"Currency detected: {label}. "
                    break
        elif any(kw in lower_text for kw in self.medicine_keywords):
            prefix = "Medicine label detected: "

        framing_tip = " (Note: text near right edge may be cut off)" if edge_cutoff_warning else ""
        spoken_output = f"{prefix}Reading text: {full_text}{framing_tip}"
        self.arbitrator.emit_alert(priority, spoken_output, timestamp, key="ocr_read")
        return annotated, spoken_output
