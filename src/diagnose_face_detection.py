# -*- coding: utf-8 -*-
r"""Deep-dive diagnostic for face detection failing on a REAL local photo.

Run:
    python diagnose_face_detection.py path\to\your\photo.jpg
"""
import sys
import cv2
import numpy as np

if len(sys.argv) < 2:
    print("Usage: python diagnose_face_detection.py path\\to\\photo.jpg")
    sys.exit(1)

path = sys.argv[1]

print("=" * 60)
print("1) Can OpenCV even load this file?")
print("=" * 60)
img = cv2.imread(path)
if img is None:
    print(f"cv2.imread returned None for: {path!r}")
    print("Common causes on Windows: non-ASCII characters in the path, "
          "or the file isn't actually a valid image.")
    # Try the unicode-safe way
    try:
        data = np.fromfile(path, dtype=np.uint8)
        img = cv2.imdecode(data, cv2.IMREAD_COLOR)
        if img is not None:
            print("BUT cv2.imdecode (unicode-safe) DID load it. "
                  "-> the bug is cv2.imread choking on the path, not the image itself.")
        else:
            print("cv2.imdecode also failed - the file itself may be corrupt.")
            sys.exit(1)
    except Exception as e:
        print("imdecode fallback also failed:", e)
        sys.exit(1)

print(f"image shape: {img.shape}  dtype: {img.dtype}")

print()
print("=" * 60)
print("2) Is the Haar cascade file itself valid?")
print("=" * 60)
cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
print("cascade path:", cascade_path)
import os
print("file exists:", os.path.exists(cascade_path))
cascade = cv2.CascadeClassifier(cascade_path)
print("cascade.empty() (True = broken):", cascade.empty())

print()
print("=" * 60)
print("3) Try detection with several parameter combinations")
print("=" * 60)
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
gray = cv2.equalizeHist(gray)  # improves contrast, sometimes helps a lot

configs = [
    dict(scaleFactor=1.1, minNeighbors=5, minSize=(30, 30)),   # our current default
    dict(scaleFactor=1.05, minNeighbors=3, minSize=(20, 20)),  # looser
    dict(scaleFactor=1.3, minNeighbors=3, minSize=(20, 20)),
    dict(scaleFactor=1.1, minNeighbors=3, minSize=(10, 10)),
    dict(scaleFactor=1.05, minNeighbors=2, minSize=(10, 10)),  # very loose
]
any_found = False
for cfg in configs:
    faces = cascade.detectMultiScale(gray, **cfg)
    print(f"  {cfg} -> {len(faces)} face(s) found" + (f"  boxes={list(faces)}" if len(faces) else ""))
    if len(faces):
        any_found = True

print()
if any_found:
    print("RESULT: Haar CAN find a face in this photo with looser parameters.")
    print("-> our shipped code's default parameters (1.1, 5, no minSize) are too strict for this photo.")
else:
    print("RESULT: Haar found NOTHING at any setting tried.")
    print("-> the issue is likely the photo itself (angle/lighting/resolution/occlusion), not the parameters.")
    print(f"   Try opening {path} and checking: is a face clearly visible, front-facing, well lit?")
