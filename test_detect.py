import cv2
from src.camera import Camera
from src.detect import FaceDetector

cam = Camera(source=1)
detector = FaceDetector()

print("Testing Face Detection... Press 'q' to quit.")

while True:
    frame = cam.get_frame()
    if frame is None:
        print("Failed to get frame.")
        break

    # Detect faces
    boxes = detector.detect_faces(frame)

    # Draw bounding boxes
    for x, y, w, h in boxes:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

    cv2.imshow("Face Detection Test", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cam.release()
