import cv2
import urllib.request
from pathlib import Path


class FaceDetector:
    def __init__(self):
        # We use OpenCV's built-in YuNet face detector model.
        # Resolve relative to this file (not cwd) so it always finds the
        # model already sitting at the project root.
        project_root = Path(__file__).resolve().parent.parent
        model_path = project_root / "face_detection_yunet_2023mar.onnx"
        model_url = (
            "https://github.com/opencv/opencv_zoo/raw/main/models/"
            f"face_detection_yunet/{model_path.name}"
        )

        if not model_path.exists():
            print("Downloading YuNet face detection model...")
            urllib.request.urlretrieve(model_url, str(model_path))

        # Initialize YuNet face detector with default frame size
        self.detector = cv2.FaceDetectorYN.create(
            model=str(model_path),
            config="",
            input_size=(640, 480),
            score_threshold=0.6,
            nms_threshold=0.3,
        )

    def detect_faces(self, frame):
        if frame is None:
            return []

        h, w, _ = frame.shape
        # Match detector input size to frame dimensions
        self.detector.setInputSize((w, h))

        _, faces = self.detector.detect(frame)

        boxes = []
        if faces is not None:
            for face in faces:
                # YuNet returns [x, y, w, h, ...]
                x, y, bw, bh = map(int, face[:4])
                x, y = max(0, x), max(0, y)
                boxes.append((x, y, bw, bh))

        return boxes
