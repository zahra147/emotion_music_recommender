# -*- coding: utf-8 -*-
"""Personal calibration: an alternative to the trained CNN that compares a
live photo against a handful of the user's OWN reference photos (one per
emotion), using face *geometry* (landmarks) instead of learned pixel
features.

This is a deliberately different, more classical approach than the CNN:
- CNN branch: learns pixel-level features from thousands of OTHER people's
  faces (generalizes across people, but is a black box).
- Personal branch: 468 3D face-mesh landmarks (mediapipe Face Mesh),
  normalized for position/scale, compared to the user's own reference
  photos via a simple distance-based nearest-neighbor match. Nothing is
  learned here - it's geometric feature engineering - but it's calibrated
  to exactly one person's face, which a generic model never is.

The two estimates get blended in demo.py, alongside the text branch, using
the same confidence-weighted fusion idea already used in fusion/fusion.py.
"""
from __future__ import annotations
from typing import Dict, List, Optional

import numpy as np

from config import EMOTIONS


class LandmarkExtractor:
    def __init__(self):
        self._mesh = None
        self._error: Optional[str] = None
        try:
            import mediapipe as mp
            self._mesh = mp.solutions.face_mesh.FaceMesh(
                static_image_mode=True, max_num_faces=1,
                refine_landmarks=False, min_detection_confidence=0.4,
            )
        except Exception as e:
            self._error = repr(e)

    def extract(self, image_bgr: np.ndarray) -> Optional[np.ndarray]:
        """Returns a flattened, normalized landmark vector, or None if no
        face was found / mediapipe isn't available. Normalization: centered
        on the nose tip (landmark 1), scaled by inter-ocular distance
        (landmarks 33 and 263) so the vector is roughly invariant to the
        face's position and size in the photo."""
        if self._mesh is None:
            return None
        try:
            rgb = image_bgr[:, :, ::-1]
            res = self._mesh.process(rgb)
        except Exception as e:
            # Same class of failure as FaceDetector: mediapipe can init fine
            # but still blow up on the first real .process() call (e.g. a
            # protobuf version conflict with another installed package).
            self._error = f"failed at process()-time: {e!r}"
            return None
        if not res.multi_face_landmarks:
            return None
        lm = res.multi_face_landmarks[0].landmark
        pts = np.array([[p.x, p.y, p.z] for p in lm], dtype="float32")
        pts = pts - pts[1]  # center on nose tip
        eye_dist = float(np.linalg.norm(pts[33] - pts[263]))
        if eye_dist < 1e-6:
            eye_dist = 1.0
        return (pts / eye_dist).flatten()


def match_personal_gallery(
    query_vec: np.ndarray, gallery: Dict[str, np.ndarray], temperature: float = 1.0
) -> Optional[List[float]]:
    """Compare `query_vec` against each reference in `gallery`
    (emotion_name -> landmark_vector) and return a probability vector over
    the full EMOTIONS list via a softmax over negative distances. Emotions
    with no reference photo get probability 0. Returns None if the gallery
    is empty.

    This reuses the exact same "probability vector over 7 classes" shape
    the CNN and text branches produce, so it can be converted to a (v, a)
    point with the same emotion_space.probs_to_va() function everywhere
    else in the project uses.
    """
    if not gallery:
        return None
    emos = list(gallery.keys())
    dists = np.array([np.linalg.norm(query_vec - gallery[e]) for e in emos])
    logits = -dists / max(temperature, 1e-6)
    logits -= logits.max()
    weights = np.exp(logits)
    weights /= weights.sum()
    probs = [0.0] * len(EMOTIONS)
    for e, w in zip(emos, weights):
        probs[EMOTIONS.index(e)] = float(w)
    return probs
