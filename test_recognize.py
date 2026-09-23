import cv2
from src.camera import Camera
from src.detect import FaceDetector
from src.landmarks import LandmarkDetector
from src.align import FaceAligner
from src.embed import FaceEmbedder
from src.recognize import FaceRecognizer

cam = Camera(source=1)
detector = FaceDetector()
lm_detector = LandmarkDetector()
aligner = FaceAligner()
embedder = FaceEmbedder()
recognizer = FaceRecognizer(threshold=0.70)

print("Starting Real-Time Face Recognition... Press 'q' to quit.")

while True:
    frame = cam.get_frame()
    if frame is None:
        break

    boxes = detector.detect_faces(frame)

    for box in boxes:
        x, y, w, h = box
        pts = lm_detector.extract_5pt(frame, box)

        if pts is not None:
            aligned_face = aligner.align_face(frame, pts)

            if aligned_face is not None:
                embedding = embedder.extract_embedding(aligned_face)

                if embedding is not None:
                    name, confidence = recognizer.identify(embedding)

                    # Draw bounding box and label
                    color = (0, 255, 0) if name != "Unknown" else (0, 0, 255)
                    cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)

                    label = f"{name} ({confidence * 100:.1f}%)"
                    cv2.putText(
                        frame,
                        label,
                        (x, max(20, y - 10)),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.6,
                        color,
                        2,
                    )

    cv2.imshow("Face Recognition Pipeline", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cam.release()
