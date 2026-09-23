import os
import json
import numpy as np
import cv2

from .camera import Camera
from .detect import FaceDetector
from .landmarks import LandmarkDetector
from .align import FaceAligner
from .embed import FaceEmbedder


class FaceEnroller:
    def __init__(self, db_path="data/database.json"):
        self.db_path = db_path
        database_dir = os.path.dirname(os.path.abspath(self.db_path))
        os.makedirs(database_dir, exist_ok=True)
        self.database = self.load_database()

    def load_database(self):
        if os.path.exists(self.db_path):
            with open(self.db_path, "r") as f:
                return json.load(f)
        return {}

    def save_database(self):
        database_dir = os.path.dirname(os.path.abspath(self.db_path))
        os.makedirs(database_dir, exist_ok=True)
        with open(self.db_path, "w") as f:
            json.dump(self.database, f, indent=4)

    def enroll_user(self, name, embeddings):
        if not embeddings:
            return False

        normalized_embeddings = []
        for embedding in embeddings:
            vector = np.asarray(embedding, dtype=np.float32).reshape(-1)
            norm = np.linalg.norm(vector)
            if vector.size == 0 or not np.isfinite(norm) or norm < 1e-6:
                continue
            normalized_embeddings.append((vector / norm).tolist())

        if not normalized_embeddings:
            return False

        # Store every sample so pose and expression variation remain available
        # to nearest-sample recognition.
        if name not in self.database:
            self.database[name] = []

        # If database previously stored a single list vector, convert to multi-vector list
        if len(self.database[name]) > 0 and not isinstance(
            self.database[name][0], list
        ):
            self.database[name] = [self.database[name]]

        self.database[name].extend(normalized_embeddings)
        self.save_database()
        return True


def run_enrollment(db_path="data/database.json", cam_source=2, samples_needed=15):
    """
    Webcam-driven enrollment loop.

    Controls:
        SPACE = capture one sample of the current face
        s     = save enrollment and exit
        q     = cancel without saving
    """
    name = input("Enter person name to enroll: ").strip()
    if not name:
        print("No name given, aborting.")
        return

    camera = Camera(source=cam_source)
    detector = FaceDetector()
    landmarker = LandmarkDetector()
    aligner = FaceAligner()
    embedder = FaceEmbedder()
    enroller = FaceEnroller(db_path=db_path)

    samples = []
    print("SPACE = capture sample, s = save & finish, q = cancel")

    while True:
        frame = camera.get_frame()
        if frame is None:
            print("Camera returned no frame. Stopping.")
            break

        boxes = detector.detect_faces(frame)
        aligned = None

        if boxes:
            # largest box = closest/most prominent face
            box = max(boxes, key=lambda b: b[2] * b[3])
            pts = landmarker.extract_5pt(frame, bbox=box)
            aligned = aligner.align_face(frame, pts)
            x, y, w, h = box
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

        cv2.putText(
            frame,
            f"{name}: {len(samples)}/{samples_needed} samples",
            (20, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 0),
            2,
        )
        cv2.imshow("Enrollment", frame)
        if aligned is not None:
            cv2.imshow("Aligned (112x112)", aligned)

        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            print("Cancelled.")
            camera.release()
            return
        elif key == ord(" ") and aligned is not None:
            emb = embedder.extract_embedding(aligned)
            if emb is not None:
                samples.append(emb)
                print(f"Captured sample {len(samples)}")
        elif key == ord("s"):
            break

    camera.release()
    cv2.destroyAllWindows()

    if len(samples) < 3:
        print(f"Only {len(samples)} samples captured (need at least 3). Not saved.")
        return

    ok = enroller.enroll_user(name, samples)
    print(
        f"Enrolled '{name}' with {len(samples)} samples."
        if ok
        else "Enrollment failed."
    )


if __name__ == "__main__":
    run_enrollment()
