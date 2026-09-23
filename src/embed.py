import os
from pathlib import Path

import cv2
import numpy as np
import onnxruntime as ort


class FaceEmbedder:
    """
    ArcFace embedder running the real ONNX model (CPU, onnxruntime).

    Model: models/embedder_arcface.onnx (tf2onnx-exported ResNet34 ArcFace).
    Input:  NHWC float32, 112x112x3 (confirmed by inspecting the model).
    Output: 512-d embedding, L2-normalized here before returning.
    """

    def __init__(self, model_path=None, input_size=(112, 112)):
        if model_path is None:
            # Resolve relative to this file, not the current working
            # directory, so it works no matter where you run python from.
            project_root = Path(__file__).resolve().parent.parent
            model_path = project_root / "models" / "embedder_arcface.onnx"

        if not os.path.exists(model_path):
            raise FileNotFoundError(
                f"ArcFace ONNX model not found at {model_path}. "
                f"Place embedder_arcface.onnx in the models/ folder."
            )

        self.input_size = input_size
        self.session = ort.InferenceSession(
            str(model_path), providers=["CPUExecutionProvider"]
        )
        self.input_name = self.session.get_inputs()[0].name
        self.output_name = self.session.get_outputs()[0].name

    def extract_embedding(self, aligned_face):
        if aligned_face is None:
            return None

        if aligned_face.shape[:2] != self.input_size:
            aligned_face = cv2.resize(aligned_face, self.input_size)

        # Model expects NHWC float32 scaled to [0, 1]. We keep OpenCV's
        # native BGR order (no RGB conversion) to match how this ResNet34
        # ArcFace export was trained/tested.
        face_img = aligned_face.astype(np.float32) / 255.0
        face_blob = np.expand_dims(face_img, axis=0)  # (1, 112, 112, 3)

        raw = self.session.run([self.output_name], {self.input_name: face_blob})[0]
        raw = raw.reshape(-1).astype(np.float32)

        norm = float(np.linalg.norm(raw))
        if norm < 1e-6:
            return None

        return raw / norm
