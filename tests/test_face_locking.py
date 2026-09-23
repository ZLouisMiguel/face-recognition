import unittest
from dataclasses import dataclass
import json
import os
import tempfile

import numpy as np

from src.face_signals import OutputSignalManager
from src.face_tracking import TargetTracker
from src.detect import FaceDetector
from src.enroll import FaceEnroller
from src.recognize import FaceRecognizer
from src import recognize as recognize_module

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


class PipelineRobustnessTests(unittest.TestCase):
    def test_unknown_identity_is_displayed_as_stranger(self):
        self.assertEqual(recognize_module.display_identity("Unknown"), "Stranger")
        self.assertEqual(recognize_module.display_identity(""), "Stranger")
        self.assertEqual(recognize_module.display_identity("Alice"), "Alice")
        self.assertEqual(recognize_module.display_identity("Unknown", "Alice"), "Alice")

    def test_yunet_landmarks_are_sorted_for_alignment_template(self):
        # YuNet order is detector-specific; the aligner expects image-left to right.
        raw_face = [0, 0, 100, 100, 80, 20, 20, 20, 50, 50, 75, 80, 25, 80, 0.9]

        points = FaceDetector._extract_landmarks(raw_face)

        np.testing.assert_allclose(
            points,
            np.array([[20, 20], [80, 20], [50, 50], [25, 80], [75, 80]], dtype=np.float32),
        )

    def test_empty_identity_and_directoryless_database_are_safe(self):
        with tempfile.TemporaryDirectory() as directory:
            db_path = os.path.join(directory, "database.json")
            with open(db_path, "w") as stream:
                json.dump({"Nobody": []}, stream)

            recognizer = FaceRecognizer(db_path=db_path)
            self.assertEqual(recognizer.identify(np.array([1, 0], dtype=np.float32)), ("Unknown", 0.0))

            directoryless_path = os.path.join(directory, "nested", "database.json")
            second = FaceRecognizer(db_path=directoryless_path)
            second.add_identity("Alice", np.array([1, 0], dtype=np.float32))
            self.assertTrue(os.path.exists(directoryless_path))

            current_directory = os.getcwd()
            try:
                os.chdir(directory)
                enroller = FaceEnroller(db_path="enrollment.json")
                self.assertTrue(enroller.enroll_user("Alice", [np.array([1, 0])]))
                self.assertTrue(os.path.exists("enrollment.json"))
            finally:
                os.chdir(current_directory)

    def test_enrollment_preserves_each_expression_sample(self):
        with tempfile.TemporaryDirectory() as directory:
            path = os.path.join(directory, "database.json")
            enroller = FaceEnroller(db_path=path)
            samples = [
                np.array([1.0, 0.0]),
                np.array([0.0, 1.0]),
                np.array([1.0, 1.0]),
            ]

            self.assertTrue(enroller.enroll_user("Alice", samples))
            stored = enroller.database["Alice"]
            self.assertEqual(len(stored), len(samples))
            for vector in stored:
                self.assertAlmostEqual(float(np.linalg.norm(vector)), 1.0, places=5)


if __name__ == "__main__":
    unittest.main()
