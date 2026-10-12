import os
import json
import math
import cv2
import numpy as np
import networkx as nx

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MAP_JSON_PATH = os.path.join(BASE_DIR, "project_data", "indoor_map.json")
LANDMARKS_DIR = os.path.join(BASE_DIR, "project_data", "room_landmarks")

class IndoorTopologicalNavigator:
    """
    Combines:
    1. ISANA Semantic Indoor Graph (rooms, hallways, and doors stored in indoor_map.json on Google Drive).
    2. ORB Visual Keypoint Relocalization (identifies which room/node the user is currently standing in).
    3. Takerhythm Piecewise-Straight A* Path Planner (collapses jagged segments into straight walks + 90° turns).
    """
    def __init__(self, arbitrator):
        self.arbitrator = arbitrator
        os.makedirs(LANDMARKS_DIR, exist_ok=True)
        self.orb = cv2.ORB_create(nfeatures=500)
        self.bf_matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
        self.graph = nx.Graph()
        self.landmark_descriptors = {}  # {node_name: orb_descriptors}
        self._load_or_init_default_home_map()

    def _load_or_init_default_home_map(self):
        """Creates a realistic home/college floor graph on Google Drive if none exists yet."""
        if os.path.exists(MAP_JSON_PATH):
            with open(MAP_JSON_PATH, "r") as f:
                data = json.load(f)
        else:
            # Default 2D meter coordinates (x, y) for indoor rooms and corridor junctions
            data = {
                "nodes": {
                    "study_room":    {"x": 0.0, "y": 0.0, "door_sign": "STUDY"},
                    "hallway_start": {"x": 0.0, "y": 3.0, "door_sign": "CORRIDOR"},
                    "hallway_mid":   {"x": 0.0, "y": 7.0, "door_sign": "WATER COOLER"},
                    "hallway_end":   {"x": 0.0, "y": 10.0, "door_sign": "HALLWAY END"},
                    "bathroom":      {"x": -3.0, "y": 10.0, "door_sign": "WASHROOM / BATHROOM"},
                    "bedroom":       {"x": 3.5, "y": 7.0, "door_sign": "BEDROOM"},
                    "main_exit":     {"x": 0.0, "y": 14.0, "door_sign": "EXIT"}
                },
                "edges": [
                    ["study_room", "hallway_start"],
                    ["hallway_start", "hallway_mid"],
                    ["hallway_mid", "bedroom"],
                    ["hallway_mid", "hallway_end"],
                    ["hallway_end", "bathroom"],
                    ["hallway_end", "main_exit"]
                ]
            }
            with open(MAP_JSON_PATH, "w") as f:
                json.dump(data, f, indent=2)

        self.graph.clear()
        for name, attrs in data["nodes"].items():
            self.graph.add_node(name, **attrs)
        for u, v in data["edges"]:
            ux, uy = data["nodes"][u]["x"], data["nodes"][u]["y"]
            vx, vy = data["nodes"][v]["x"], data["nodes"][v]["y"]
            dist = round(math.hypot(vx - ux, vy - uy), 1)
            self.graph.add_edge(u, v, weight=dist)

    def register_room_visual_landmark(self, node_name, frame_bgr):
        """Extracts ORB visual keypoints from a room/door frame so the camera recognizes this location later."""
        gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
        keypoints, descriptors = self.orb.detectAndCompute(gray, None)
        if descriptors is not None:
            self.landmark_descriptors[node_name] = descriptors
            np.save(os.path.join(LANDMARKS_DIR, f"{node_name}_orb.npy"), descriptors)
            return True
        return False

    def localize_current_room(self, frame_bgr, fallback_node="study_room"):
        """Matches live camera ORB features against saved room landmarks to determine user's starting location."""
        if not self.landmark_descriptors:
            return fallback_node, 0

        gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
        _, live_des = self.orb.detectAndCompute(gray, None)
        if live_des is None:
            return fallback_node, 0

        best_node, best_good_matches = fallback_node, 0
        for node_name, saved_des in self.landmark_descriptors.items():
            matches = self.bf_matcher.match(live_des, saved_des)
            good_matches = [m for m in matches if m.distance < 45]
            if len(good_matches) > best_good_matches:
                best_good_matches = len(good_matches)
                best_node = node_name
        return best_node, best_good_matches

    def _simplify_piecewise_straight(self, node_path):
        """
        Takerhythm-inspired Piecewise-Straight Path Simplifier:
        Merges collinear hallway nodes (e.g., study_room -> hallway_start -> hallway_mid -> hallway_end)
        into a single straight walk so the blind user isn't interrupted every 2 meters.
        """
        if len(node_path) <= 2:
            return node_path

        simplified = [node_path[0]]
        for i in range(1, len(node_path) - 1):
            p_prev = self.graph.nodes[simplified[-1]]
            p_curr = self.graph.nodes[node_path[i]]
            p_next = self.graph.nodes[node_path[i + 1]]

            ang1 = math.atan2(p_curr["y"] - p_prev["y"], p_curr["x"] - p_prev["x"])
            ang2 = math.atan2(p_next["y"] - p_curr["y"], p_next["x"] - p_curr["x"])

            # Keep node only if heading angle changes by more than 15 degrees (a real turn!)
            if abs(math.degrees(ang2 - ang1)) > 15.0:
                simplified.append(node_path[i])
        simplified.append(node_path[-1])
        return simplified

    def plan_voice_route(self, start_node, target_node, timestamp=0.0):
        """Computes A* path, simplifies collinear segments, and generates human turn-by-turn audio instructions."""
        start_key = start_node.lower().strip().replace(" ", "_")
        target_key = target_node.lower().strip().replace(" ", "_")

        if target_key not in self.graph:
            return [], f"Destination {target_node} is not in your indoor map."

        raw_path = nx.astar_path(
            self.graph,
            start_key,
            target_key,
            heuristic=lambda u, v: math.hypot(
                self.graph.nodes[u]["x"] - self.graph.nodes[v]["x"],
                self.graph.nodes[u]["y"] - self.graph.nodes[v]["y"]
            ),
            weight="weight"
        )
        straight_path = self._simplify_piecewise_straight(raw_path)

        steps_speech = []
        prev_heading = None

        for i in range(len(straight_path) - 1):
            u, v = straight_path[i], straight_path[i + 1]
            ux, uy = self.graph.nodes[u]["x"], self.graph.nodes[u]["y"]
            vx, vy = self.graph.nodes[v]["x"], self.graph.nodes[v]["y"]
            dist = round(math.hypot(vx - ux, vy - uy), 1)
            heading = math.degrees(math.atan2(vy - uy, vx - ux))

            if prev_heading is None:
                steps_speech.append(f"Walk straight ahead {dist} meters to {v.replace('_', ' ')}")
            else:
                turn_angle = (heading - prev_heading + 180) % 360 - 180
                if turn_angle > 25:
                    turn_word = "Turn left"
                elif turn_angle < -25:
                    turn_word = "Turn right"
                else:
                    turn_word = "Continue straight"
                steps_speech.append(f"{turn_word} and walk {dist} meters to reach {v.replace('_', ' ')}")
            prev_heading = heading

        door_label = self.graph.nodes[target_key].get("door_sign", target_key.upper())
        steps_speech.append(f"Look for the {door_label} door sign straight ahead.")
        full_guidance = f"Route from {start_node.replace('_', ' ')} to {target_node.replace('_', ' ')}: " + ". Then, ".join(steps_speech)
        self.arbitrator.emit_alert(2, full_guidance, timestamp, key=f"nav_{target_key}")
        return straight_path, full_guidance
