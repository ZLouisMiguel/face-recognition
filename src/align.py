import cv2
import numpy as np


class FaceAligner:
    def __init__(self, output_size=(112, 112)):
        self.output_size = output_size

        # Target positions matching [Left Eye, Right Eye, Nose, Left Mouth, Right Mouth]
        # Coordinates configured for standard 112x112 output canvas
        self.target_pts = np.array(
            [
                [38.2946, 51.6963],  # Left Eye
                [73.5318, 51.5014],  # Right Eye
                [56.0252, 71.7366],  # Nose Tip
                [41.5493, 92.3655],  # Left Mouth Corner
                [70.7299, 92.2041],  # Right Mouth Corner
            ],
            dtype=np.float32,
        )

    def align_face(self, frame, landmarks):
        if frame is None or landmarks is None or len(landmarks) != 5:
            return None

        # Convert landmarks array explicitly to float32
        src_pts = np.array(landmarks, dtype=np.float32)

        # Calculate similarity transformation matrix (translation, rotation, scale)
        M, _ = cv2.estimateAffinePartial2D(src_pts, self.target_pts)

        if M is None:
            return None

        # Warp face into upright 112x112 layout
        aligned = cv2.warpAffine(frame, M, self.output_size, flags=cv2.INTER_CUBIC)

        return aligned
