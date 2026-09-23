import cv2
import numpy as np


class TargetTracker:
    def __init__(self, threshold=0.4):
        self.locked_identity = None
        self.locked_embedding = None
        self.threshold = threshold
        self.target_box = None

    def lock_target(self, identity_name, embedding):
        self.locked_identity = identity_name
        self.locked_embedding = embedding

    def update_lock(self, current_embeddings, current_boxes):
        if self.locked_embedding is None:
            return None, False

        best_sim = -1.0
        best_idx = -1

        for idx, emb in enumerate(current_embeddings):
            sim = np.dot(self.locked_embedding, emb)
            if sim > best_sim:
                best_sim = sim
                best_idx = idx

        dist = 1.0 - best_sim
        if best_idx != -1 and dist <= self.threshold:
            self.target_box = current_boxes[best_idx]
            return self.target_box, True

        return None, False
