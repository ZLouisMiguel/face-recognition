# Face Recognition with Identity Lock

This repository implements the vision-only assessment stage: detect faces, recognize enrolled identities, and keep a stable lock on the selected person. Motor/servo correction is intentionally not connected yet.

## Requirements

- Windows camera accessible through OpenCV.
- Python environment in `.venv` with OpenCV, NumPy, MediaPipe, ONNX Runtime, and PySerial.
- `face_detection_yunet_2023mar.onnx` in the project root.
- `models/embedder_arcface.onnx` for ArcFace embeddings.

## Run automated tests

From this directory:

```powershell
.venv\Scripts\python.exe -m unittest discover -s tests -v
.venv\Scripts\python.exe -m compileall -q src
```

The tests cover acquisition debounce, skipped embeddings, wrong-identity rejection, temporary loss, reset/reacquisition, landmark ordering, database edge cases, and overlay colors.

## Enroll a person

```powershell
.venv\Scripts\python.exe -m src.enroll
```

Enter a name, face the camera, press `SPACE` to capture samples, then press `s` to save. Capture at least three clear samples with small changes in pose or expression. Press `q` to cancel.

For expression robustness, capture neutral, smiling, and slightly turned-face samples. Every normalized sample is now preserved instead of being averaged into one expression-specific vector. Re-enrolling an existing name appends the new samples to that identity profile.

The current working checkout is configured for camera index `2`. On the development machine, `camprobe.py` found index `1` working and index `2` unavailable, so run the probe first and change `cam_source` in `src/enroll.py` and `src/recognize.py` if needed.

## Run face recognition and identity lock

```powershell
.venv\Scripts\python.exe -m src.recognize
```

Expected live behavior:

1. `SEARCHING / UNLOCKED` in red while no known face is stable.
2. After three consistent recognized frames, `LOCKED: <name>` in green.
3. A short detector gap shows `TARGET LOST: <name>` in orange and can reacquire the same person.
4. After a longer loss, the system resets to searching and requires a new identity lock.

Every detected face is boxed. Known faces show their enrolled name; faces that are not in the database show `Stranger` in red.

Controls:

- `q`: quit.
- `+` or `=`: loosen the recognition distance threshold.
- `-`: tighten the recognition distance threshold.

For assessment evidence, record a short video showing the target moving, another person entering the frame, the green lock remaining on the enrolled identity, and the system returning to searching after the target leaves.
