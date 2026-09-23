import cv2
from src.camera import Camera
from src.detect import FaceDetector
from src.landmarks import LandmarkDetector
from src.align import FaceAligner

cam = Camera(source=1)
detector = FaceDetector()
lm_detector = LandmarkDetector()
aligner = FaceAligner()

print("Testing Face Alignment... Press 'q' to quit.")

while True:
    frame = cam.get_frame()
    if frame is None:
        break

    boxes = detector.detect_faces(frame)

    if len(boxes) > 0:
        box = boxes[0]
        pts = lm_detector.extract_5pt(frame, box)

        if pts is not None:
            # Draw landmarks on camera frame
            for x, y in pts:
                cv2.circle(frame, (int(x), int(y)), 3, (0, 0, 255), -1)

            # Perform 5-point face alignment
            aligned_face = aligner.align_face(frame, pts)

            if aligned_face is not None:
                # Upscale aligned face display for easier inspection
                display_face = cv2.resize(aligned_face, (224, 224))
                cv2.imshow("Aligned Face (112x112)", display_face)

    cv2.imshow("Camera Feed", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cam.release()
