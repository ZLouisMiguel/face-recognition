import unittest
from dataclasses import dataclass

import numpy as np

from src.face_signals import OutputSignalManager
from src.face_tracking import TargetTracker

try:
    from src.face_tracking import TrackCandidate
except ImportError:
    @dataclass
    class TrackCandidate:
        box: tuple
        embedding: object
        identity: str
        confidence: float = 0.0


def candidate(name, embedding, box=(10, 20, 80, 80)):
    return TrackCandidate(box=box, embedding=np.asarray(embedding, dtype=np.float32), identity=name)


class TargetTrackerTests(unittest.TestCase):
    def test_requires_consistent_frames_before_locking(self):
        tracker = TargetTracker(match_threshold=0.2, acquire_frames=3)

        self.assertEqual(tracker.update([candidate("Alice", [1, 0])]).state, "SEARCHING")
        self.assertEqual(tracker.update([candidate("Alice", [1, 0])]).state, "SEARCHING")
        result = tracker.update([candidate("Alice", [1, 0])])

        self.assertEqual(result.state, "LOCKED")
        self.assertTrue(result.is_locked)
        self.assertEqual(result.identity, "Alice")

    def test_invalid_candidate_does_not_shift_box_association(self):
        tracker = TargetTracker(match_threshold=0.2, acquire_frames=1)
        result = tracker.update(
            [
                TrackCandidate(box=(1, 1, 1, 1), embedding=None, identity="Alice"),
                candidate("Alice", [1, 0], box=(20, 30, 40, 50)),
            ]
        )

        self.assertEqual(result.state, "LOCKED")
        self.assertEqual(result.box, (20, 30, 40, 50))

    def test_conflicting_known_identity_cannot_take_lock(self):
        tracker = TargetTracker(match_threshold=0.2, acquire_frames=1, max_missing_frames=1)
        tracker.update([candidate("Alice", [1, 0])])

        result = tracker.update([candidate("Bob", [1, 0], box=(90, 20, 80, 80))])

        self.assertEqual(result.state, "LOST")
        self.assertFalse(result.is_locked)
        self.assertEqual(result.identity, "Alice")

    def test_temporary_loss_can_reacquire_same_identity(self):
        tracker = TargetTracker(match_threshold=0.2, acquire_frames=1, max_missing_frames=2)
        tracker.update([candidate("Alice", [1, 0])])

        lost = tracker.update([])
        reacquired = tracker.update([candidate("Alice", [1, 0], box=(30, 30, 80, 80))])

        self.assertEqual(lost.state, "LOST")
        self.assertEqual(reacquired.state, "LOCKED")
        self.assertEqual(reacquired.box, (30, 30, 80, 80))

    def test_prolonged_loss_resets_lock_for_new_identity(self):
        tracker = TargetTracker(match_threshold=0.2, acquire_frames=1, max_missing_frames=1)
        tracker.update([candidate("Alice", [1, 0])])
        tracker.update([])
        reset = tracker.update([])

        self.assertEqual(reset.state, "SEARCHING")
        self.assertIsNone(reset.identity)

        new_lock = tracker.update([candidate("Bob", [0, 1])])
        self.assertEqual(new_lock.state, "LOCKED")
        self.assertEqual(new_lock.identity, "Bob")


class OutputSignalManagerTests(unittest.TestCase):
    def test_unlocked_status_is_red_and_locked_status_is_green(self):
        signals = OutputSignalManager()
        frame = np.zeros((120, 160, 3), dtype=np.uint8)

        signals.update_status(False)
        signals.draw_overlay(frame.copy())
        self.assertEqual(signals.color, (0, 0, 255))

        signals.update_status(True, "Alice")
        signals.draw_overlay(frame.copy(), (10, 10, 40, 40))
        self.assertEqual(signals.color, (0, 255, 0))


if __name__ == "__main__":
    unittest.main()
