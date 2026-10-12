# Assistive Smart Goggles AI — Antigravity IDE Workspace Rules

## Project Mission
Real-time, zero-lag (<150ms walking reflex) assistive computer vision and spatial audio software enabling blind users to live independently.

## Locked-In Architecture (Do Not Break)
1. **`config.py`**: Central thresholds (`CRITICAL_DANGER_M = 1.2`, `WARNING_DISTANCE_M = 2.5`, `ENV_MODE = "LOCAL"` on laptop).
2. **`audio_arbitrator.py`**: Priority interrupt queue (`P1` Collision Hazard > `P2` Navigation/Face/OCR > `P3` Coaching) + 4-second per-Track-ID anti-repeat cooldown.
3. **`modules/reflex_depth.py`**: YOLOv11n (`yolo11n.onnx` / `.pt`) + ByteTrack + Depth Anything V2 Metric Indoor (`vits`) inner-50% bounding-box median distance in real meters + VizWiz blur/darkness check.
4. **`modules/face_social.py`**: InsightFace (`buffalo_l` 512-D ArcFace) + ChromaDB persistent vector store (`project_data/chroma_faces`) + 3-Frame Anti-Error Voting Gate + YOLOv11-Pose Eye-Contact & Wave cues.
5. **`modules/episodic_memory.py`**: SQLite spatial logger (`project_data/episodic_items.db`) recording personal items near room furniture landmarks.
6. **`modules/finger_ocr.py`**: EasyOCR + Fingertip ROI cropper + Indian Currency (`₹10`–`₹500`) & Medicine parser + HSV Room Light/Color check.
7. **`modules/indoor_nav.py`**: ISANA Semantic Room-Door Graph (`project_data/indoor_map.json`) + ORB visual relocalization + Takerhythm Piecewise-Straight A* path simplifier.
8. **`modules/checkpoint_manager.py`**: Atomic `.tmp` per-epoch checkpointer (`save_period=1`) for fault-tolerant model fine-tuning.

## Local Execution Target in Antigravity IDE
- Switch `config.ENV_MODE = "LOCAL"`.
- Connect `io_bridge.py` to local webcam (`cv2.VideoCapture(0)`), offline microphone (`faster-whisper` or `speech_recognition`), and non-blocking local TTS (`pyttsx3` or `piper-tts`) in a background daemon thread so the 30 FPS OpenCV loop never blocks.
