import os
from pathlib import Path

# Define the project structure
STRUCTURE = {
    "data": ["db", "enroll", "debug_aligned"],
    "models": [],
    "src": [
        "__init__.py",
        "camera.py",
        "detect.py",
        "landmarks.py",
        "align.py",
        "embed.py",
        "enroll.py",
        "evaluate.py",
        "recognize.py",
        "haar_5pt.py",
    ],
}

# Additional files to create with minimal content
FILE_TEMPLATES = {
    "src/__init__.py": "",
    "src/camera.py": "import cv2\n\ndef main():\n    cap = cv2.VideoCapture(0)\n    if not cap.isOpened():\n        raise RuntimeError('Camera not opened.')\n    print('Press q to quit.')\n    while True:\n        ok, frame = cap.read()\n        if not ok:\n            break\n        cv2.imshow('Camera Test', frame)\n        if (cv2.waitKey(1) & 0xFF) == ord('q'):\n            break\n    cap.release()\n    cv2.destroyAllWindows()\n\nif __name__ == '__main__':\n    main()\n",
    "src/detect.py": "import cv2\n\ndef main():\n    cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'\n    face_cascade = cv2.CascadeClassifier(cascade_path)\n    if face_cascade.empty():\n        raise RuntimeError('Failed to load cascade.')\n    cap = cv2.VideoCapture(0)\n    if not cap.isOpened():\n        raise RuntimeError('Camera not opened.')\n    while True:\n        ok, frame = cap.read()\n        if not ok:\n            break\n        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)\n        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(60,60))\n        for (x,y,w,h) in faces:\n            cv2.rectangle(frame, (x,y), (x+w,y+h), (0,255,0), 2)\n        cv2.imshow('Face Detection', frame)\n        if (cv2.waitKey(1) & 0xFF) == ord('q'):\n            break\n    cap.release()\n    cv2.destroyAllWindows()\n\nif __name__ == '__main__':\n    main()\n",
    "src/landmarks.py": "# Placeholder for landmarks testing\n",
    "src/align.py": "# Placeholder for alignment testing\n",
    "src/embed.py": "# Placeholder for embedding testing\n",
    "src/enroll.py": "# Placeholder for enrollment\n",
    "src/evaluate.py": "# Placeholder for evaluation\n",
    "src/recognize.py": "# Placeholder for recognition\n",
    "src/haar_5pt.py": "# Placeholder for Haar + MediaPipe 5pt detector\n",
    "README.md": "# Face Recognition with ArcFace ONNX and 5-Point Alignment\n\nSee the book for full instructions.\n",
}

def create_structure():
    """Create directories and files if they don't already exist."""
    for folder, subdirs in STRUCTURE.items():
        for sub in subdirs:
            p = Path(folder) / sub
            p.mkdir(parents=True, exist_ok=True)

    for filepath, content in FILE_TEMPLATES.items():
        p = Path(filepath)
        if not p.exists():
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_text(content, encoding='utf-8')
            print(f"Created {filepath}")
        else:
            print(f"Skipped {filepath} (already exists)")

    # Create placeholder for the model file (user will download later)
    models_dir = Path("models")
    models_dir.mkdir(exist_ok=True)
    # Optionally create a .gitkeep to track empty folder
    (models_dir / ".gitkeep").touch(exist_ok=True)

    print("\nProject scaffold complete!")

if __name__ == "__main__":
    create_structure()