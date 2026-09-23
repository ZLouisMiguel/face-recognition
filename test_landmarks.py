import cv2
from src.camera import Camera
from src.detect import FaceDetector
from src.landmarks import LandmarkDetector

cam = Camera(source=1)
detector = FaceDetector()
lm_detector = LandmarkDetector()

print("Testing 5-Point Landmark Detection... Press 'q' to quit.")

while True:
    frame = cam.get_frame()
    if frame is None:
        break

    boxes = detector.detect_faces(frame)

    if len(boxes) > 0:
        # Use first face detected
        box = boxes[0]
        pts = lm_detector.extract_5pt(frame, box)

        if pts is not None:
            # Draw red points for 5 facial landmarks
            for pt_x, pt_y in pts:
                cv2.circle(frame, (int(pt_x), int(pt_y)), 5, (0, 0, 255), -1)

    cv2.imshow("5-Point Landmark Test", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cam.release()
