import cv2
import time
from src.camera import Camera
from src.detect import FaceDetector
from src.landmarks import LandmarkDetector
from src.align import FaceAligner
from src.embed import FaceEmbedder
from src.enroll import FaceEnroller

cam = Camera(source=1)
detector = FaceDetector()
lm_detector = LandmarkDetector()
aligner = FaceAligner()
embedder = FaceEmbedder()
enroller = FaceEnroller()

user_name = input("Enter name to enroll: ").strip()
if not user_name:
    user_name = "User_1"

collected_embeddings = []
max_samples = 30

print(f"\nEnrolling '{user_name}'. Look at the camera...")
print("Collecting samples... Press 'q' to cancel.")

while len(collected_embeddings) < max_samples:
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
                embedding = embedder.extract_embedding(aligned_face)

                if embedding is not None:
                    collected_embeddings.append(embedding)

                    # Visual feedback on progress
                    progress = len(collected_embeddings)
                    cv2.putText(
                        frame,
                        f"Enrolling {user_name}: {progress}/{max_samples}",
                        (20, 40),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.7,
                        (0, 255, 0),
                        2,
                    )

    cv2.imshow("Enrollment Test", frame)

    if cv2.waitKey(30) & 0xFF == ord("q"):
        break

cam.release()

if len(collected_embeddings) >= max_samples:
    enroller.enroll_user(user_name, collected_embeddings)
    print(f"\nSuccessfully enrolled '{user_name}'! Profile saved to data/database.json")
else:
    print("\nEnrollment canceled or incomplete.")
