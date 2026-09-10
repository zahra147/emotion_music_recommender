# -*- coding: utf-8 -*-
"""Face detection wrapper.

Uses MediaPipe if available (light + real-time). Falls back to OpenCV's
Haar cascade. Returns the largest detected face crop plus a confidence score.
"""
from __future__ import annotations
from typing import Optional, Tuple

import numpy as np


class FaceDetector:
    def __init__(self):
        self._backend = None
        self._mp = None
        self._cascade = None
        self._mediapipe_error = None  # set if mediapipe was tried and failed
        try:
            import mediapipe as mp  # type: ignore
            self._mp = mp
            self._detector = mp.solutions.face_detection.FaceDetection(
                model_selection=0, min_detection_confidence=0.4
            )
            self._backend = "mediapipe"
        except Exception as e:
            # Record *why* mediapipe wasn't used (e.g. an incompatible
            # installed version) instead of silently hiding it - callers
            # (walkthrough.py, the Streamlit demo) surface this so a
            # weaker fallback backend doesn't go unnoticed during a demo.
            self._mediapipe_error = repr(e)
            self._backend = "haar" if self._ensure_haar() else None

    def _ensure_haar(self):
        """Lazily init the Haar cascade. Used both as the primary fallback
        at __init__ time and as a rescue path if mediapipe blows up later
        (see detect())."""
        if getattr(self, "_cascade", None) is not None:
            return True
        try:
            import cv2  # type: ignore
            self._cv2 = cv2
            self._cascade = cv2.CascadeClassifier(
                cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
            )
            return True
        except Exception as e2:
            self._mediapipe_error = f"{self._mediapipe_error} | haar also unavailable: {e2!r}"
            return False

    def detect(self, image_bgr: np.ndarray) -> Tuple[Optional[np.ndarray], float]:
        """Return (face_crop, confidence). face_crop is None if no face found."""
        if self._backend == "mediapipe":
            try:
                return self._detect_mediapipe(image_bgr)
            except Exception as e:
                # mediapipe can initialize fine but still blow up on the
                # first real .process() call (e.g. a protobuf version
                # installed for another package, like TensorFlow, is
                # incompatible with mediapipe's compiled proto messages).
                # Don't crash the whole pipeline for that - permanently
                # drop to Haar for the rest of this session and retry.
                self._mediapipe_error = (
                    f"mediapipe failed at detect-time (not at init): {e!r}"
                )
                if self._ensure_haar():
                    self._backend = "haar"
                    return self._detect_haar(image_bgr)
                self._backend = None
                return None, 0.0
        if self._backend == "haar":
            return self._detect_haar(image_bgr)
        # No backend installed -> signal "no face" so fusion leans on text.
        return None, 0.0

    def _detect_mediapipe(self, image_bgr):
        h, w = image_bgr.shape[:2]
        rgb = image_bgr[:, :, ::-1]
        res = self._detector.process(rgb)
        if not res.detections:
            return None, 0.0
        det = max(res.detections, key=lambda d: d.score[0])
        box = det.location_data.relative_bounding_box
        x, y = int(box.xmin * w), int(box.ymin * h)
        bw, bh = int(box.width * w), int(box.height * h)
        x, y = max(0, x), max(0, y)
        crop = image_bgr[y:y + bh, x:x + bw]
        if crop.size == 0:
            return None, 0.0
        return crop, float(det.score[0])

    def _detect_haar(self, image_bgr):
        gray = self._cv2.cvtColor(image_bgr, self._cv2.COLOR_BGR2GRAY)
        faces = self._cascade.detectMultiScale(gray, 1.1, 5)
        if len(faces) == 0:
            return None, 0.0
        x, y, w, h = max(faces, key=lambda f: f[2] * f[3])
        # Haar has no probability; use a fixed moderate confidence.
        return image_bgr[y:y + h, x:x + w], 0.6
