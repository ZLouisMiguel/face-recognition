import cv2
import numpy as np


class LandmarkDetector:
    def __init__(self):
        # Use MediaPipe FaceMesh via task solution or legacy import
        try:
            import mediapipe.python.solutions.face_mesh as mp_face_mesh

            self.face_mesh = mp_face_mesh.FaceMesh(
                static_image_mode=False,
                max_num_faces=5,
                refine_landmarks=True,
                min_detection_confidence=0.5,
                min_tracking_confidence=0.5,
            )
            self.use_mediapipe = True
        except Exception:
            self.use_mediapipe = False

        # 5 Key Facial Points indices (Left Eye, Right Eye, Nose Tip, Left Mouth, Right Mouth)
        self.KEY_5_INDICES = [33, 263, 1, 61, 291]

    def extract_5pt(self, frame, bbox=None):
        if frame is None:
            return None

        h, w, _ = frame.shape

        if self.use_mediapipe:
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = self.face_mesh.process(rgb_frame)

            if results.multi_face_landmarks:
                face_landmarks = results.multi_face_landmarks[0]
                if bbox is not None and len(bbox) == 4:
                    x, y, bw, bh = bbox
                    target = np.array([x + bw / 2.0, y + bh / 2.0])

                    def face_center(face):
                        xs = [lm.x * w for lm in face.landmark]
                        ys = [lm.y * h for lm in face.landmark]
                        return np.array([np.mean(xs), np.mean(ys)])

                    face_landmarks = min(
                        results.multi_face_landmarks,
                        key=lambda face: np.linalg.norm(face_center(face) - target),
                    )
                pts = []
                for idx in self.KEY_5_INDICES:
                    lm = face_landmarks.landmark[idx]
                    pts.append([int(lm.x * w), int(lm.y * h)])
                return np.array(pts, dtype=np.float32)

        # Adjusted baseline backup proportions matching actual facial geometry
        if bbox is not None and len(bbox) == 4:
            x, y, bw, bh = bbox
            left_eye = [x + int(bw * 0.32), y + int(bh * 0.38)]
            right_eye = [x + int(bw * 0.68), y + int(bh * 0.38)]
            nose = [x + int(bw * 0.50), y + int(bh * 0.58)]
            left_mouth = [x + int(bw * 0.36), y + int(bh * 0.78)]
            right_mouth = [x + int(bw * 0.64), y + int(bh * 0.78)]
            return np.array(
                [left_eye, right_eye, nose, left_mouth, right_mouth], dtype=np.float32
            )

        return None
