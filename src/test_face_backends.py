# -*- coding: utf-8 -*-
"""Standalone probe: tests mediapipe's FaceDetection init and OpenCV's Haar
cascade init directly, independent of face_detector.py, so we can see
exactly which one fails and why - even if face_detector.py hasn't been
updated yet.

Run:
    python test_face_backends.py
"""
print("=" * 60)
print("Testing mediapipe FaceDetection init...")
print("=" * 60)
try:
    import mediapipe as mp
    print("mediapipe version:", getattr(mp, "__version__", "unknown"))
    det = mp.solutions.face_detection.FaceDetection(
        model_selection=0, min_detection_confidence=0.4
    )
    print("RESULT: mediapipe FaceDetection() OK")
except Exception as e:
    print("RESULT: mediapipe FAILED ->", repr(e))

print()
print("=" * 60)
print("Testing OpenCV Haar cascade init...")
print("=" * 60)
try:
    import cv2
    print("opencv version:", cv2.__version__)
    print("cv2.data.haarcascades path:", cv2.data.haarcascades)
    cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    print("full cascade path:", cascade_path)
    import os
    print("does that file exist on disk?", os.path.exists(cascade_path))
    cascade = cv2.CascadeClassifier(cascade_path)
    print("cascade.empty() (True = failed to load):", cascade.empty())
    if cascade.empty():
        print("RESULT: OpenCV Haar FAILED -> cascade file did not load")
    else:
        print("RESULT: OpenCV Haar OK")
except Exception as e:
    print("RESULT: OpenCV Haar FAILED with exception ->", repr(e))
