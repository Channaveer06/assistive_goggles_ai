# 🕶️ Assistive Smart Goggles AI

> **Real-Time, Zero-Lag (<150ms Walking Reflex) Assistive Computer Vision & Spatial Audio for Independent Living**

[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![YOLOv11](https://img.shields.io/badge/Vision-YOLOv11-green.svg)](https://docs.ultralytics.com/)
[![Depth Anything V2](https://img.shields.io/badge/Metric%20Depth-Depth%20Anything%20V2-orange.svg)](https://github.com/DepthAnything/Depth-Anything-V2)
[![InsightFace](https://img.shields.io/badge/Face%20ID-InsightFace%20ArcFace-red.svg)](https://github.com/deepinsight/insightface)
[![ChromaDB](https://img.shields.io/badge/Vector%20Store-ChromaDB-purple.svg)](https://www.trychroma.com/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

---

## 🎯 Project Mission
Assistive Smart Goggles AI is a wearable computer vision and non-blocking spatial audio assistant engineered to empower blind and visually impaired individuals to navigate indoor environments, recognize familiar people, locate misplaced personal items, read medicines and currency notes, and receive immediate collision warnings without latency.

---

## 🏗️ System Architecture

```
                    [ 📹 30 FPS Camera Input ]
                                |
        +-----------------------+-----------------------+
        |                                               |
  [ Reflex & Depth ]                             [ Social Engine ]
  YOLOv11 + ByteTrack                             InsightFace 512-D
  Depth Anything V2 (Metric M)                    YOLOv11-Pose Cues
        |                                               |
  [ Spatial Logger ]                             [ ChromaDB Store ]
  SQLite Furniture Landmarks                      Persistent Face DB
        |                                               |
        +-----------------------+-----------------------+
                                |
                    [ Priority Audio Arbitrator ]
                    P1: Hazard Override (<1.2m)
                    P2: Nav / Face / OCR
                    P3: Environmental Coaching
                                |
                    [ 🔊 Non-Blocking pyttsx3 ]
                    Offline Zero-Lag Voice Thread
```

---

## ⚡ The 6 Independence Modules

### 1. Reflex & Metric Depth Engine (`modules/reflex_depth.py`)
- **Real-Time Tracking**: YOLOv11n paired with ByteTrack persistent identity tracker.
- **Metric Distance in Real Meters**: Depth Anything V2 Metric Indoor Small estimates depth maps; distance is calculated from the inner-50% median of the bounding box to avoid boundary artifacts.
- **VizWiz Quality Coach**: Automatically warns if ambient light is insufficient (`< 35.0` luminance) or if camera motion blur occurs.

### 2. Face & Social Micro-Cue Engine (`modules/face_social.py`)
- **ArcFace 512-D Embeddings**: InsightFace (`buffalo_l`) generates high-precision facial identity vectors.
- **Persistent ChromaDB Vector Store**: Saves and retrieves known individuals in `< 2ms` with cosine similarity search.
- **3-Frame Anti-Error Voting Gate**: Ensures zero false positives before announcing identity.
- **Social Micro-Cues**: YOLOv11-Pose keypoints detect eye contact (`looking at you`), waving hands, and handshake invitations.

### 3. Episodic Spatial Object Memory (`modules/episodic_memory.py`)
- **Ego4D ObjectNLQ-Inspired Spatial Logger**: SQLite database (`project_data/episodic_items.db`) continuously tracks personal belongings (`bottle`, `cell phone`, `keys`, `laptop`).
- **Landmark Anchoring**: Correlates personal objects with stationary room furniture (e.g., *"Your cell phone was last seen 15 minutes ago near the dining table, 1.7 meters on your right"*).

### 4. Fingertip Point-to-Read OCR & Utilities (`modules/finger_ocr.py`)
- **Fingertip ROI Cropping**: Reads text directly above where the user is pointing.
- **Indian Currency & Medicine Parser**: Accurately recognizes ₹10, ₹20, ₹50, ₹100, ₹200, and ₹500 notes, plus dosage information (`mg`, `tablet`, `expiry`).
- **Room Light & Clothing Color**: Computes lighting status and central object color in HSV space.

### 5. Indoor Topological Navigator (`modules/indoor_nav.py`)
- **ISANA Semantic Indoor Graph**: Room and junction topological graph (`project_data/indoor_map.json`).
- **ORB Visual Relocalization**: Matches live camera features against room landmarks.
- **Takerhythm Piecewise-Straight Planner**: Collapses micro-segments along corridors into straight walks and clear 90° turn instructions.

### 6. Atomic Checkpoint Manager (`modules/checkpoint_manager.py`)
- **Fault-Tolerant Training**: Atomic `.tmp` per-epoch saving (`save_period=1`) to prevent model corruption during fine-tuning.

---

## 🔊 Zero-Lag Priority Audio Arbitrator (`audio_arbitrator.py`)
- **Asynchronous Worker**: Runs an offline `pyttsx3` speech engine in a dedicated daemon thread. The 30 FPS camera loop **never freezes** while speech is being generated.
- **Priority Queue**:
  - `P1`: Critical Hazard (`< 1.2m`) — Immediately purges pending audio to yell *"Stop!"*.
  - `P2`: Navigation directions, known faces, and OCR text.
  - `P3`: Environmental feedback (darkness, motion blur).
- **Track Cooldown**: Enforces a 4-second anti-repeat cooldown per object tracking ID.

---

## 🎮 Live HUD Hotkeys (`run_live_local.py`)

When running `run_live_local.py`, the live interactive HUD window responds to immediate keyboard triggers:

| Hotkey | Action | Description |
| :---: | :--- | :--- |
| **`[e]`** | **Enroll Live Face** | Captures and embeds face directly into ChromaDB vector memory. |
| **`[r]`** | **Fingertip OCR** | Reads medicine labels, Indian currency notes, or scene text. |
| **`[f]`** | **Find Lost Item** | Answers where you last left your cell phone or bottle. |
| **`[b]`** | **Bathroom Navigation** | Speaks turn-by-turn route to the bathroom. |
| **`[l]`** | **Light & Color Check** | Reports whether room lights are on and identifies center object color. |
| **`[s]`** | **Guardian Emergency SOS** | Alerts designated guardian with current room and emergency status. |
| **`[q]`** | **Quit** | Shuts down camera and background threads cleanly. |

---

## 🚀 Quickstart & Installation

### 1. Clone the Repository
```bash
git clone https://github.com/Channaveer06/assistive_goggles_ai.git
cd assistive_goggles_ai
```

### 2. Set Up Virtual Environment
```bash
# Windows
python -m venv .venv
.\.venv\Scripts\activate

# Linux / macOS
python3 -m venv .venv
source .venv/bin/activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Run Live with Webcam
```bash
python run_live_local.py
```

### 5. Automated Verification / Test Mode
To test all modules on synthetic test frames without a physical webcam:
```bash
python run_live_local.py --headless --test --frames 32
```

---

## 📁 Repository Structure
```
assistive_goggles_ai/
├── config.py                 # Central configurations & thresholds
├── audio_arbitrator.py       # Asynchronous pyttsx3 audio priority queue
├── io_bridge.py              # Hardware video capture & SOS trigger
├── run_live_local.py         # Main 30 FPS live execution HUD runner
├── main.py                   # Unified Colab & Local brain CLI
├── requirements.txt          # Python package dependencies
├── modules/
│   ├── reflex_depth.py       # YOLOv11 + Metric Depth Anything V2
│   ├── face_social.py        # InsightFace + ChromaDB + Pose Cues
│   ├── episodic_memory.py    # SQLite spatial object memory
│   ├── finger_ocr.py         # EasyOCR, Currency & Lighting Coach
│   ├── indoor_nav.py         # ISANA Semantic Graph & A* Router
│   └── checkpoint_manager.py # Atomic model checkpoint manager
└── project_data/
    ├── indoor_map.json       # Indoor room and hallway graph
    ├── episodic_items.db     # SQLite item memory store
    ├── chroma_faces/         # ChromaDB face vector database
    ├── models_onnx/          # ONNX model exports
    └── room_landmarks/       # Visual landmark descriptor cache
```

---

## 📜 License
Distributed under the MIT License. See `LICENSE` for more information.
