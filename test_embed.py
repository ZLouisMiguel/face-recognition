import cv2
import numpy as np
from src.camera import Camera
from src.detect import FaceDetector
from src.landmarks import LandmarkDetector
from src.align import FaceAligner
from src.embed import FaceEmbedder

cam = Camera(source=1)
detector = FaceDetector()
lm_detector = LandmarkDetector()
aligner = FaceAligner()
embedder = FaceEmbedder()

print("Testing Feature Embedding... Press 'q' to quit.")

while True:
    frame = cam.get_frame()
    if frame is None:
        break

    boxes = detector.detect_faces(frame)

    if len(boxes) > 0:
        box = boxes[0]
        pts = lm_detector.extract_5pt(frame, box)

        if pts is not None:
            aligned_face = aligner.align_face(frame, pts)

            if aligned_face is not None:
                # Extract face feature vector
                embedding = embedder.extract_embedding(aligned_face)

                if embedding is not None:
                    # Print stats about the feature vector on the frame
                    norm_val = np.linalg.norm(embedding)
                    text = f"Vector Dim: {len(embedding)} | Norm: {norm_val:.2f}"
                    cv2.putText(
                        frame,
                        text,
                        (20, 40),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.6,
                        (0, 255, 0),
                        2,
                    )

    cv2.imshow("Embedding Test", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cam.release()
