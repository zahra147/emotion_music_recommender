# -*- coding: utf-8 -*-
"""Vision branch: image -> (valence, arousal, confidence)."""
from __future__ import annotations
from typing import Tuple

import numpy as np

from config import FER_INPUT_SIZE
from emotion_space import probs_to_va
from vision.face_detector import FaceDetector
from vision.fer_model import FERModel


class VisionPipeline:
    def __init__(self):
        self.detector = FaceDetector()
        self.fer = FERModel()

    def _preprocess(self, face_bgr: np.ndarray) -> np.ndarray:
        """BGR crop -> normalized 48x48 grayscale."""
        try:
            import cv2
            gray = cv2.cvtColor(face_bgr, cv2.COLOR_BGR2GRAY)
            gray = cv2.resize(gray, FER_INPUT_SIZE)
        except Exception:
            # Minimal fallback without OpenCV: average channels + crude resize.
            gray = face_bgr.mean(axis=2)
            gray = np.array(gray)
        return gray.astype("float32") / 255.0

    def infer(self, image_bgr: np.ndarray) -> Tuple[float, float, float]:
        """Return (valence, arousal, confidence). confidence=0 => no face."""
        face, conf = self.detector.detect(image_bgr)
        if face is None:
            return 0.0, 0.0, 0.0
        probs = self.fer.predict(self._preprocess(face))
        v, a = probs_to_va(probs)
        return v, a, conf
