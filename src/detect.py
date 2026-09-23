import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple

import cv2
import numpy as np


Box = Tuple[int, int, int, int]


@dataclass
class FaceDetection:
    box: Box
    landmarks: Optional[np.ndarray]
    score: float


class FaceDetector:
    def __init__(self):
        project_root = Path(__file__).resolve().parent.parent
        model_path = project_root / "face_detection_yunet_2023mar.onnx"
        model_url = (
            "https://github.com/opencv/opencv_zoo/raw/main/models/"
            f"face_detection_yunet/{model_path.name}"
        )

        if not model_path.exists():
            print("Downloading YuNet face detection model...")
            urllib.request.urlretrieve(model_url, str(model_path))

        self.detector = cv2.FaceDetectorYN.create(
            model=str(model_path),
            config="",
            input_size=(640, 480),
            score_threshold=0.6,
            nms_threshold=0.3,
        )

    @staticmethod
    def _extract_landmarks(face) -> Optional[np.ndarray]:
        # YuNet returns two eyes, nose, and two mouth corners after x/y/w/h.
        if len(face) < 14:
            return None
        points = np.asarray(face[4:14], dtype=np.float32).reshape(5, 2)
        if not np.isfinite(points).all():
            return None

        # Sort by image position so the result matches FaceAligner's
        # [left eye, right eye, nose, left mouth, right mouth] template.
        eyes = sorted((points[0], points[1]), key=lambda point: point[0])
        mouth = sorted((points[3], points[4]), key=lambda point: point[0])
        return np.asarray([eyes[0], eyes[1], points[2], mouth[0], mouth[1]], dtype=np.float32)

    def detect(self, frame) -> List[FaceDetection]:
        if frame is None:
            return []

        height, width = frame.shape[:2]
        self.detector.setInputSize((width, height))
        _status, faces = self.detector.detect(frame)

        detections = []
        if faces is None:
            return detections

        for face in faces:
            x, y, box_width, box_height = map(int, face[:4])
            x = max(0, x)
            y = max(0, y)
            box_width = max(0, min(box_width, width - x))
            box_height = max(0, min(box_height, height - y))
            detections.append(
                FaceDetection(
                    box=(x, y, box_width, box_height),
                    landmarks=self._extract_landmarks(face),
                    score=float(face[14]) if len(face) > 14 else 0.0,
                )
            )
        return detections

    def detect_faces(self, frame):
        """Compatibility wrapper used by the original manual scripts."""
        return [detection.box for detection in self.detect(frame)]
