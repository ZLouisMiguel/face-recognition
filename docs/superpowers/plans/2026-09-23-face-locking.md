# Face Tracking with Identity Lock Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Make the vision-only face-locking pipeline robust to multiple faces, skipped embeddings, temporary disappearance, and wrong-person candidates.

**Architecture:** YuNet produces per-face observations with associated five-point landmarks. The recognition loop converts each observation into one `TrackCandidate`, and a stateful `TargetTracker` owns acquisition, lock maintenance, loss grace, and reset. The UI renders the tracker's explicit state instead of inferring state from text.

**Tech Stack:** Python 3, OpenCV, NumPy, existing ArcFace ONNX embedder, standard-library `unittest`.

**Spec:** `docs/superpowers/specs/2026-09-23-face-locking-design.md`

## Global Constraints

- Keep the assignment vision-only; do not connect motor/servo control to the live loop.
- Preserve user working-tree camera source and local database changes.
- Prefer YuNet-associated landmarks and do not duplicate one global landmark result across faces.
- Keep dependencies unchanged; use standard-library tests.

## Review Focus

- A failed embedding must not shift a later face onto the wrong box — covered by observation pipeline and tracker tests.
- A similar-looking face with a different recognized name must not take the lock — covered by identity rejection test.
- A brief detector miss must not immediately erase the lock — covered by loss-grace test.
- A prolonged loss must clear state and allow a new identity — covered by reset/reacquisition test.
- `UNLOCKED` must render red, not green — covered by signal-state test.

### Task 1: Add failing tracker and signal tests

**Files:**
- Create: `tests/test_face_locking.py`
- Test: `src.face_tracking`, `src.face_signals`

- [ ] Write tests for the five review-focus behaviors plus stable acquisition after three observations.
- [ ] Run `python -m unittest discover -s tests -v` and verify the new tests fail for the current implementation.

### Task 2: Implement observation-aware detection and tracking

**Files:**
- Modify: `src/detect.py`
- Modify: `src/landmarks.py`
- Modify: `src/face_tracking.py`
- Modify: `src/recognize.py`

- [ ] Add YuNet five-point landmarks to each detection and keep `detect_faces()` as a box-only compatibility wrapper.
- [ ] Make MediaPipe select the closest detected face and support more than one face.
- [ ] Add `TrackCandidate`, explicit tracker states, acquisition debounce, identity gating, loss grace, and reset.
- [ ] Replace parallel recognition arrays with one candidate per detection.
- [ ] Run the focused test file after each behavior is implemented.

### Task 3: Fix persistence and status rendering

**Files:**
- Modify: `src/face_signals.py`
- Modify: `src/recognize.py`
- Modify: `src/enroll.py`

- [ ] Render colors from explicit state values.
- [ ] Make empty database entries and directoryless database paths safe.
- [ ] Run focused tests and compilation.

### Task 4: Document and verify the live workflow

**Files:**
- Modify: `README.md`

- [ ] Document enrollment, recognition, camera-index selection, controls, expected states, and evidence capture.
- [ ] Run the full automated suite, compileall, and diff checks.
- [ ] Report hardware-dependent camera checks separately from automated results.
