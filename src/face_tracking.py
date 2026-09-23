from dataclasses import dataclass
from typing import Iterable, Optional, Sequence, Tuple

import numpy as np


Box = Tuple[int, int, int, int]


@dataclass
class TrackCandidate:
    """One complete face observation; fields stay associated through the pipeline."""

    box: Box
    embedding: Optional[np.ndarray]
    identity: str = "Unknown"
    confidence: float = 0.0


@dataclass
class TrackResult:
    box: Optional[Box]
    is_locked: bool
    identity: Optional[str]
    state: str
    similarity: float = 0.0


class TargetTracker:
    """Stateful identity tracker with acquisition debounce and loss recovery."""

    SEARCHING = "SEARCHING"
    LOCKED = "LOCKED"
    LOST = "LOST"

    def __init__(
        self,
        threshold: float = 0.4,
        acquire_frames: int = 3,
        max_missing_frames: int = 10,
        match_threshold: Optional[float] = None,
    ):
        # match_threshold is accepted as a descriptive alias for new callers.
        self.threshold = threshold if match_threshold is None else match_threshold
        self.acquire_frames = max(1, int(acquire_frames))
        self.max_missing_frames = max(0, int(max_missing_frames))
        self.reset()

    def reset(self):
        self.locked_identity = None
        self.locked_embedding = None
        self.target_box = None
        self.state = self.SEARCHING
        self.missing_frames = 0
        self._pending_identity = None
        self._pending_embedding = None
        self._pending_frames = 0

    @staticmethod
    def _normalise(embedding):
        if embedding is None:
            return None
        vector = np.asarray(embedding, dtype=np.float32).reshape(-1)
        norm = np.linalg.norm(vector)
        if vector.size == 0 or not np.isfinite(norm) or norm < 1e-6:
            return None
        return vector / norm

    def _similarity(self, first, second):
        first = self._normalise(first)
        second = self._normalise(second)
        if first is None or second is None or first.shape != second.shape:
            return -1.0
        return float(np.dot(first, second))

    def lock_target(self, identity_name, embedding):
        vector = self._normalise(embedding)
        if not identity_name or vector is None:
            return False
        self.locked_identity = identity_name
        self.locked_embedding = vector
        self.state = self.LOCKED
        self.missing_frames = 0
        self._pending_identity = None
        self._pending_embedding = None
        self._pending_frames = 0
        return True

    def _result(self, box=None, similarity=0.0):
        identity = self.locked_identity if self.locked_identity is not None else None
        return TrackResult(
            box=box,
            is_locked=self.state == self.LOCKED,
            identity=identity,
            state=self.state,
            similarity=similarity,
        )

    def _lose_target(self):
        self.missing_frames += 1
        if self.missing_frames <= self.max_missing_frames:
            self.state = self.LOST
            return self._result()

        self.reset()
        return self._result()

    def _valid_candidates(self, candidates: Iterable[TrackCandidate]):
        valid = []
        for item in candidates:
            if item is None or item.embedding is None:
                continue
            vector = self._normalise(item.embedding)
            if vector is None:
                continue
            valid.append((item, vector))
        return valid

    def _acquire(self, valid):
        known = [(item, vector) for item, vector in valid if item.identity != "Unknown"]
        if not known:
            self.state = self.SEARCHING
            self._pending_identity = None
            self._pending_embedding = None
            self._pending_frames = 0
            return self._result()

        # Prefer the strongest recognized candidate when several faces are present.
        item, vector = max(known, key=lambda pair: pair[0].confidence)
        if item.identity == self._pending_identity:
            similarity = self._similarity(self._pending_embedding, vector)
            if similarity >= 1.0 - self.threshold:
                self._pending_frames += 1
            else:
                self._pending_embedding = vector
                self._pending_frames = 1
        else:
            self._pending_identity = item.identity
            self._pending_embedding = vector
            self._pending_frames = 1

        self.state = self.SEARCHING
        if self._pending_frames >= self.acquire_frames:
            self.lock_target(item.identity, vector)
            self.target_box = item.box
            return self._result(item.box, similarity=1.0)

        return self._result(item.box)

    def update(self, candidates: Sequence[TrackCandidate]) -> TrackResult:
        """Process one frame of complete face observations."""
        valid = self._valid_candidates(candidates)

        if self.locked_embedding is None:
            return self._acquire(valid)

        best = None
        for item, vector in valid:
            if item.identity not in ("Unknown", self.locked_identity):
                continue
            similarity = self._similarity(self.locked_embedding, vector)
            if best is None or similarity > best[0]:
                best = (similarity, item, vector)

        if best is None or 1.0 - best[0] > self.threshold:
            return self._lose_target()

        similarity, item, _vector = best
        self.target_box = item.box
        self.missing_frames = 0
        self.state = self.LOCKED
        return self._result(item.box, similarity=similarity)

    def update_lock(self, current_embeddings, current_boxes):
        """Compatibility adapter for the original parallel-list API."""
        candidates = []
        for embedding, box in zip(current_embeddings, current_boxes):
            candidates.append(
                TrackCandidate(
                    box=box,
                    embedding=embedding,
                    identity=self.locked_identity or "Unknown",
                )
            )
        result = self.update(candidates)
        return result.box, result.is_locked
