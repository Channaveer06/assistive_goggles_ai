import sqlite3
import os
import time
from datetime import datetime

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DB_PATH = os.path.join(BASE_DIR, "project_data", "episodic_items.db")

# Personal items a blind user commonly searches for vs. room furniture landmarks
PERSONAL_ITEMS = {
    "bottle", "cup", "cell phone", "backpack", "handbag", "suitcase",
    "umbrella", "book", "laptop", "mouse", "remote", "keyboard",
    "scissors", "toothbrush", "wine glass", "bowl", "fork", "spoon", "knife"
}
LANDMARK_OBJECTS = {
    "chair", "couch", "bed", "dining table", "tv", "refrigerator",
    "sink", "oven", "microwave", "bench", "toilet", "potted plant"
}

class EpisodicObjectMemory:
    """
    Ego4D ObjectNLQ-inspired zero-cost spatial memory logger stored on Google Drive.
    Automatically records personal objects alongside the nearest room landmark and direction.
    """
    def __init__(self, db_path=DB_PATH):
        self.db_path = db_path
        os.makedirs(os.path.dirname(self.db_path), exist_ok=True)
        self._init_db()

    def _init_db(self):
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS item_memory (
                    item_name TEXT PRIMARY KEY,
                    direction TEXT,
                    distance_m REAL,
                    nearest_landmark TEXT,
                    room_node TEXT,
                    last_seen_epoch REAL,
                    last_seen_readable TEXT
                )
            """)
            conn.commit()

    def log_scene_detections(self, enriched_detections, current_room="living_area"):
        """
        Receives detections from ReflexDepthEngine:
        list of dicts: {'label', 'dist_m', 'direction', 'center_x', 'center_y'}
        """
        if not enriched_detections:
            return

        landmarks = [d for d in enriched_detections if d["label"] in LANDMARK_OBJECTS]
        items = [d for d in enriched_detections if d["label"] in PERSONAL_ITEMS]

        if not items:
            return

        now_epoch = time.time()
        now_str = datetime.now().strftime("%I:%M %p")

        with sqlite3.connect(self.db_path) as conn:
            for item in items:
                # Find closest furniture landmark in the same frame
                nearest_lm = "open floor"
                if landmarks:
                    closest = min(
                        landmarks,
                        key=lambda lm: abs(lm["center_x"] - item["center_x"]) + abs(lm["center_y"] - item["center_y"])
                    )
                    nearest_lm = f"near the {closest['label']}"

                conn.execute("""
                    INSERT INTO item_memory (
                        item_name, direction, distance_m, nearest_landmark, room_node, last_seen_epoch, last_seen_readable
                    ) VALUES (?, ?, ?, ?, ?, ?, ?)
                    ON CONFLICT(item_name) DO UPDATE SET
                        direction=excluded.direction,
                        distance_m=excluded.distance_m,
                        nearest_landmark=excluded.nearest_landmark,
                        room_node=excluded.room_node,
                        last_seen_epoch=excluded.last_seen_epoch,
                        last_seen_readable=excluded.last_seen_readable
                """, (
                    item["label"],
                    item["direction"],
                    round(item["dist_m"], 1),
                    nearest_lm,
                    current_room,
                    now_epoch,
                    now_str
                ))
            conn.commit()

    def query_item(self, item_name):
        """Answers 'Where did I leave my [item]?' in < 1 millisecond."""
        item_key = item_name.lower().strip()
        with sqlite3.connect(self.db_path) as conn:
            row = conn.execute(
                "SELECT direction, distance_m, nearest_landmark, room_node, last_seen_epoch, last_seen_readable "
                "FROM item_memory WHERE item_name = ?",
                (item_key,)
            ).fetchone()

        if not row:
            return f"I have not seen your {item_key} yet."

        direction, dist_m, nearest_lm, room_node, seen_epoch, seen_str = row
        mins_ago = max(0, int((time.time() - seen_epoch) // 60))
        time_phrase = "just now" if mins_ago == 0 else f"{mins_ago} minutes ago"
        return (
            f"Your {item_key} was last seen {time_phrase} at {seen_str} in the {room_node}, "
            f"{nearest_lm}, about {dist_m} meters {direction}."
        )
