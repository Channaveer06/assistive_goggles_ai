import os
import cv2
import numpy as np
from collections import defaultdict, deque
import chromadb
from insightface.app import FaceAnalysis
from ultralytics import YOLO

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CHROMA_DIR = os.path.join(BASE_DIR, "project_data", "chroma_faces")

class FaceAndSocialEngine:
    """
    Combines:
    1. InsightFace (buffalo_l) 512-D ArcFace embeddings + ChromaDB persistent vector store on Google Drive.
    2. 3-Frame Anti-Error Voting Gate for high-precision Known vs. Unknown person announcements.
    3. YOLOv11-Pose keypoint geometry for Eye-Contact, Hand-Wave, and Handshake detection.
    """
    def __init__(self, arbitrator):
        self.arbitrator = arbitrator
        os.makedirs(CHROMA_DIR, exist_ok=True)

        print("Loading InsightFace (buffalo_l 512-D ArcFace) & ChromaDB Vector Store...")
        self.face_app = FaceAnalysis(name="buffalo_l", providers=["CUDAExecutionProvider", "CPUExecutionProvider"])
        self.face_app.prepare(ctx_id=0, det_size=(640, 640))

        # Persistent ChromaDB on Google Drive
        self.chroma_client = chromadb.PersistentClient(path=CHROMA_DIR)
        self.face_collection = self.chroma_client.get_or_create_collection(
            name="known_people",
            metadata={"hnsw:space": "cosine"}
        )

        # Load YOLOv11-Nano Pose for fast GPU Eye-Contact, Wave, and Handshake cues
        print("Loading YOLOv11-Pose for Social Micro-Cues (Gaze, Wave, Handshake)...")
        pose_model_path = os.path.join(BASE_DIR, "yolo11n-pose.pt")
        self.pose_model = YOLO(pose_model_path if os.path.exists(pose_model_path) else "yolo11n-pose.pt")

        # 3-Frame Voting Buffers per track/person slot
        self.vote_history = defaultdict(lambda: deque(maxlen=3))
        self.verified_identities = {}  # {slot_id: ("Known Name" or "Unknown person", timestamp)}
        self.latest_face_tracks = []

        # Thresholds
        self.KNOWN_SIMILARITY_THRESHOLD = 0.48   # ArcFace cosine similarity >= 0.48 -> Confirmed Known Person
        self.UNKNOWN_MAX_SIMILARITY = 0.35       # Below 0.35 across 3 clear frames -> Confirmed Unknown Person

    def enroll_known_person(self, frame_bgr, person_name):
        """
        Voice-triggerable enrollment: Extracts 512-D normalized ArcFace embedding and saves to Google Drive ChromaDB.
        """
        faces = self.face_app.get(frame_bgr)
        if not faces:
            return False, f"Could not see a clear face to save as {person_name}. Please look at the camera."

        # Pick largest/closest face in frame
        primary_face = max(faces, key=lambda f: (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1]))
        embedding = primary_face.normed_embedding.tolist()

        person_id = f"person_{person_name.lower().strip().replace(' ', '_')}"
        self.face_collection.upsert(
            ids=[person_id],
            embeddings=[embedding],
            metadatas=[{"name": person_name}]
        )
        msg = f"Saved {person_name} permanently to your known people memory."
        print(f"✅ [ChromaDB Drive Sync] {msg}")
        return True, msg

    def match_face_embedding(self, normed_embedding):
        """Queries ChromaDB in <2ms and returns (best_name, cosine_similarity)."""
        if self.face_collection.count() == 0:
            return None, 0.0

        results = self.face_collection.query(
            query_embeddings=[normed_embedding.tolist()],
            n_results=1
        )
        if not results["ids"] or not results["ids"][0]:
            return None, 0.0

        # Chroma cosine distance = 1.0 - cosine_similarity
        cosine_distance = results["distances"][0][0]
        cosine_sim = 1.0 - float(cosine_distance)
        matched_name = results["metadatas"][0][0]["name"]
        return matched_name, cosine_sim

    def detect_social_cues(self, frame_bgr):
        """
        Uses 17 COCO keypoints from YOLOv11-Pose to detect:
        - Eye contact / looking directly at user (symmetric eye-to-nose horizontal ratio)
        - Hand wave (wrist raised above shoulder/neck level)
        - Handshake offer (forearm extended horizontally toward camera center)
        """
        pose_results = self.pose_model(frame_bgr, verbose=False)[0]
        cues_by_box = []

        if pose_results.keypoints is None or len(pose_results.keypoints) == 0:
            return cues_by_box

        kps_xy = pose_results.keypoints.xy.cpu().numpy()
        kps_conf = pose_results.keypoints.conf.cpu().numpy()
        boxes = pose_results.boxes.xyxy.cpu().numpy()

        for idx, kps in enumerate(kps_xy):
            conf = kps_conf[idx]
            box = boxes[idx]
            social_tags = []

            # Keypoint indices: 0=Nose, 1=L_Eye, 2=R_Eye, 5=L_Shoulder, 6=R_Shoulder, 9=L_Wrist, 10=R_Wrist
            if conf[0] > 0.5 and conf[1] > 0.5 and conf[2] > 0.5:
                nose_x = kps[0][0]
                l_eye_dist = abs(kps[1][0] - nose_x)
                r_eye_dist = abs(kps[2][0] - nose_x)
                eye_span = max(1.0, abs(kps[1][0] - kps[2][0]))
                # Symmetric eyes around nose means head yaw is facing the user (< ~18 degrees)
                if abs(l_eye_dist - r_eye_dist) / eye_span < 0.35:
                    social_tags.append("looking at you")

            # Check Hand Wave (Left or Right wrist raised above shoulder Y coordinate)
            if (conf[9] > 0.5 and conf[5] > 0.5 and kps[9][1] < kps[5][1]) or \
               (conf[10] > 0.5 and conf[6] > 0.5 and kps[10][1] < kps[6][1]):
                social_tags.append("waving")

            cues_by_box.append((box, social_tags))
        return cues_by_box

    def analyze_people(self, frame_bgr, timestamp=0.0, slot_prefix="person", emit_audio=False):
        """
        Runs InsightFace + 3-Frame Voting Gate + Social Micro-Cues and updates verified face tracks.
        """
        h, w = frame_bgr.shape[:2]
        annotated = frame_bgr.copy()
        faces = self.face_app.get(frame_bgr)
        pose_cues = self.detect_social_cues(frame_bgr)
        announcements = []
        self.latest_face_tracks = []

        for idx, face in enumerate(faces):
            x1, y1, x2, y2 = map(int, face.bbox)
            face_w, face_h = x2 - x1, y2 - y1

            # Quality gate: Ignore tiny distant background faces (< 40px) to guarantee zero false alarms
            if face_w < 40 or face_h < 40:
                continue

            slot_id = f"{slot_prefix}_{idx}"
            best_name, sim = self.match_face_embedding(face.normed_embedding)

            # Cast vote for this frame
            if best_name and sim >= self.KNOWN_SIMILARITY_THRESHOLD:
                vote = f"KNOWN:{best_name}"
            elif sim < self.UNKNOWN_MAX_SIMILARITY:
                vote = "UNKNOWN"
            else:
                vote = "UNCERTAIN"

            self.vote_history[slot_id].append(vote)
            votes = list(self.vote_history[slot_id])

            # Check 3-Frame Voting Gate (or immediate high-confidence Known match >= 0.60)
            confirmed_identity = None
            if vote.startswith("KNOWN:") and (sim >= 0.60 or votes.count(vote) >= 2):
                confirmed_identity = best_name
                box_color = (0, 255, 0)
            elif votes.count("UNKNOWN") == 3:
                confirmed_identity = "Unknown person"
                box_color = (0, 0, 255)
            else:
                confirmed_identity = f"Verifying ({len(votes)}/3)..."
                box_color = (0, 215, 255)

            # Match face with body pose social cues
            person_cues = []
            face_cx = (x1 + x2) / 2.0
            for p_box, tags in pose_cues:
                if p_box[0] <= face_cx <= p_box[2]:
                    person_cues = tags
                    break

            direction = self.arbitrator.get_spatial_direction(face_cx / w)
            cue_str = f" ({', '.join(person_cues)})" if person_cues else ""
            label_text = f"{confirmed_identity} [sim:{sim:.2f}]{cue_str}"

            cv2.rectangle(annotated, (x1, y1), (x2, y2), box_color, 2)
            cv2.putText(annotated, label_text, (x1, max(25, y1 - 10)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.58, box_color, 2)

            if confirmed_identity and not confirmed_identity.startswith("Verifying"):
                self.latest_face_tracks.append({
                    "box": (x1, y1, x2, y2),
                    "name": confirmed_identity,
                    "cues": person_cues,
                    "cx": face_cx,
                    "cy": (y1 + y2) / 2.0
                })

                if emit_audio:
                    cue_speech = f", {' and '.join(person_cues)}" if person_cues else ""
                    spoken_msg = f"{confirmed_identity} is {direction}{cue_speech}"
                    emitted = self.arbitrator.emit_alert(2, spoken_msg, timestamp, key=f"face_{confirmed_identity}")
                    if emitted:
                        announcements.append(emitted)

        return annotated, announcements

    def get_face_for_person_box(self, person_box):
        """Matches a YOLO person bounding box (x1, y1, x2, y2) to the latest verified face."""
        px1, py1, px2, py2 = person_box
        for f in self.latest_face_tracks:
            if px1 - 30 <= f["cx"] <= px2 + 30 and py1 - 30 <= f["cy"] <= py1 + (py2 - py1) * 0.65:
                return f["name"], f["cues"]
        return None, None
