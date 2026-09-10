# -*- coding: utf-8 -*-
r"""Runs the REAL project vision pipeline (not just raw Haar) on a photo,
with no exception swallowed, to find what demo.py's blanket except is hiding.

Run:
    python diagnose_vision_pipeline.py path\to\photo.jpg
"""
import sys
import cv2
import numpy as np

if len(sys.argv) < 2:
    print("Usage: python diagnose_vision_pipeline.py path\\to\\photo.jpg")
    sys.exit(1)

path = sys.argv[1]
img = cv2.imread(path)
print("image shape:", None if img is None else img.shape)

print()
print("=" * 60)
print("Step 1: FaceDetector.detect() directly")
print("=" * 60)
from vision.face_detector import FaceDetector
d = FaceDetector()
print("backend:", d._backend)
face, conf = d.detect(img)
print("face found:", face is not None, "conf:", conf)
if face is not None:
    print("face crop shape:", face.shape)

print()
print("=" * 60)
print("Step 2: full VisionPipeline.infer() - same call demo.py makes")
print("=" * 60)
from vision.vision_pipeline import VisionPipeline
pipe = VisionPipeline()
try:
    v, a, c = pipe.infer(img)
    print(f"SUCCESS: v={v:+.2f} a={a:+.2f} c={c:.2f}")
except Exception as e:
    import traceback
    print("EXCEPTION (this is what demo.py's except Exception was hiding!):")
    traceback.print_exc()
