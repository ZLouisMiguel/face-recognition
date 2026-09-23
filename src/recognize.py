import os
import json
import numpy as np
import cv2

from .camera import Camera
from .detect import FaceDetector
from .landmarks import LandmarkDetector
from .align import FaceAligner
from .embed import FaceEmbedder
from .face_tracking import TargetTracker, TrackCandidate
from .face_signals import OutputSignalManager


def display_identity(identity_name):
    """Return the user-facing label for a recognition result."""
    if not identity_name or identity_name == "Unknown":
        return "Stranger"
    return identity_name


class FaceRecognizer:
    def __init__(self, db_path="data/database.json", threshold=0.9):
        """
        Face Recognizer using L2 (Euclidean) Distance on L2-normalized embeddings.

        Args:
            db_path (str): Path to database JSON file.
            threshold (float): L2 distance cutoff. Values BELOW this threshold
                               are considered a match. With real ArcFace
                               embeddings (unit vectors), L2 distance relates
                               to cosine similarity by L2 = sqrt(2*(1-cos)).
                               threshold=0.9 corresponds to roughly cos > 0.6,
                               matching the book's "same person: 0.6-0.99"
                               guidance. Tune with +/- while running, or with
                               evaluate.py once you have 2+ enrolled people.
        """
        self.db_path = db_path
        self.threshold = threshold
        self.database = self.load_database()

    def load_database(self):
        """Loads and parses stored face embeddings from a JSON file."""
        if os.path.exists(self.db_path):
            with open(self.db_path, "r") as f:
                data = json.load(f)
                parsed_db = {}
                for name, vecs in data.items():
                    if not isinstance(vecs, list) or not vecs:
                        parsed_db[name] = []
                    elif isinstance(vecs[0], list):
                        parsed_db[name] = [np.array(v, dtype=np.float32) for v in vecs]
                    else:
                        parsed_db[name] = [np.array(vecs, dtype=np.float32)]
                return parsed_db
        return {}

    def save_database(self):
        """Saves current memory database back to JSON."""
        serializable_db = {}
        for name, vecs in self.database.items():
            serializable_db[name] = [v.tolist() for v in vecs]

        database_dir = os.path.dirname(os.path.abspath(self.db_path))
        os.makedirs(database_dir, exist_ok=True)
        with open(self.db_path, "w") as f:
            json.dump(serializable_db, f, indent=2)

    def l2_distance(self, vec1, vec2):
        """
        Calculates the Euclidean (L2) distance between two L2-normalized vectors.
        Returns float("inf") if dimensions do not align (e.g. an old entry
        enrolled with a different embedder).
        """
        v1 = np.asarray(vec1, dtype=np.float32)
        v2 = np.asarray(vec2, dtype=np.float32)

        if v1.shape != v2.shape:
            return float("inf")

        # L2-normalize vectors to unit length
        norm1 = np.linalg.norm(v1)
        norm2 = np.linalg.norm(v2)

        if norm1 == 0 or norm2 == 0:
            return float("inf")

        v1_norm = v1 / norm1
        v2_norm = v2 / norm2

        # Compute Euclidean distance
        return float(np.linalg.norm(v1_norm - v2_norm))

    def calculate_confidence(self, distance, max_distance=1.4):
        """
        Converts L2 distance into an intuitive 0% - 100% match confidence score.
        - Distance 0.00 -> 100% match
        - Distance >= max_distance -> 0% match
        """
        if distance >= max_distance:
            return 0.0

        confidence = (1.0 - (distance / max_distance)) * 100.0
        return round(confidence, 2)

    def identify(self, query_embedding):
        """
        Matches a query face embedding against all database entries.

        Returns:
            tuple: (name, confidence_percentage)
        """
        if not self.database or query_embedding is None:
            return "Unknown", 0.0

        best_match = "Unknown"
        min_dist = float("inf")

        # Search database for closest match
        for name, vecs in self.database.items():
            for db_embedding in vecs:
                dist = self.l2_distance(query_embedding, db_embedding)
                if dist < min_dist:
                    min_dist = dist
                    best_match = name

        if not np.isfinite(min_dist):
            return "Unknown", 0.0

        # Convert raw distance to user-friendly confidence %
        confidence = self.calculate_confidence(min_dist)

        # Console logging for live debugging
        print(
            f"[DEBUG] Best Match: {best_match} | Distance: {min_dist:.4f} | Confidence: {confidence}%"
        )

        # Check threshold (Lower distance = stronger match)
        if min_dist <= self.threshold:
            return best_match, confidence

        return "Unknown", confidence

    def add_identity(self, name, embedding):
        """Registers a new face embedding to database and saves to disk."""
        vec = np.array(embedding, dtype=np.float32)
        if name in self.database:
            self.database[name].append(vec)
        else:
            self.database[name] = [vec]
        self.save_database()


def run_recognition(db_path="data/database.json", cam_source=2, lock_threshold=0.4):
    """
    Live loop: detect -> landmark -> align -> embed -> identify -> lock -> display.

    Once a known identity is confidently recognized once, TargetTracker keeps
    the box locked onto that same face (by embedding similarity) even across
    frames where recognition wavers, so the "lock" stays stable.

    Controls:
        q   quit
        +/- loosen/tighten the recognition threshold live

    NOTE: this is the vision-only version. To trigger the Arduino servo,
    add a serial write where marked below once the board is connected:
        if is_locked:
            arduino.write(b'1')
    """
    camera = Camera(source=cam_source)
    detector = FaceDetector()
    landmarker = LandmarkDetector()
    aligner = FaceAligner()
    embedder = FaceEmbedder()
    recognizer = FaceRecognizer(db_path=db_path)
    tracker = TargetTracker(threshold=lock_threshold)
    signals = OutputSignalManager()

    print("Press 'q' to quit, '+' / '-' to adjust the match threshold.")

    while True:
        frame = camera.get_frame()
        if frame is None:
            print("Camera returned no frame. Stopping.")
            break

        detections = detector.detect(frame)
        candidates = []
        face_annotations = []

        for detection in detections:
            box = detection.box
            pts = detection.landmarks
            if pts is None:
                pts = landmarker.extract_5pt(frame, bbox=box)
            aligned = aligner.align_face(frame, pts)
            emb = embedder.extract_embedding(aligned)
            if emb is None:
                face_annotations.append((box, "Stranger", (0, 0, 255)))
                continue
            name, confidence = recognizer.identify(emb)
            label = display_identity(name)
            color = (0, 0, 255) if name == "Unknown" else (255, 200, 0)
            if name != "Unknown":
                label = f"{label} ({confidence:.1f}%)"
            face_annotations.append((box, label, color))
            candidates.append(
                TrackCandidate(
                    box=box,
                    embedding=emb,
                    identity=name,
                    confidence=confidence,
                )
            )

        track = tracker.update(candidates)

        for box, label, color in face_annotations:
            x, y, width, height = box
            cv2.rectangle(frame, (x, y), (x + width, y + height), color, 2)
            cv2.putText(
                frame,
                label,
                (x, max(20, y - 10)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                color,
                2,
            )

        # --- Arduino trigger goes here once the board is connected ---
        # if track.is_locked:
        #     arduino.write(b'1')
        # ----------------------------------------------------------------

        signals.update_status(track.is_locked, track.identity, track.state)
        display_box = track.box
        frame = signals.draw_overlay(frame, display_box)

        cv2.putText(
            frame,
            f"match threshold: {recognizer.threshold:.2f} (+/- to adjust)",
            (20, frame.shape[0] - 15),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (200, 200, 200),
            1,
        )

        cv2.imshow("Face Tracking - Identity Lock", frame)
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
        elif key in (ord("+"), ord("=")):
            recognizer.threshold = min(1.4, recognizer.threshold + 0.05)
        elif key == ord("-"):
            recognizer.threshold = max(0.1, recognizer.threshold - 0.05)

    camera.release()


if __name__ == "__main__":
    run_recognition()
