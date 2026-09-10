# -*- coding: utf-8 -*-
"""Confidence-weighted late fusion in the V-A space.

Combines the vision (v_f, a_f, c_f) and text (v_t, a_t, c_t) estimates into a
single (valence, arousal) point. Weights are the per-branch confidences, so a
missing / low-confidence modality is automatically down-weighted (graceful
degradation). If both confidences are ~0 we return the neutral origin.
"""
from __future__ import annotations
from typing import Tuple

from config import MIN_FACE_CONFIDENCE
from emotion_space import va_to_nearest_label


def fuse(
    vision: Tuple[float, float, float],
    text: Tuple[float, float, float],
) -> dict:
    v_f, a_f, c_f = vision
    v_t, a_t, c_t = text

    # Discard vision if the face confidence is below the trust threshold.
    if c_f < MIN_FACE_CONFIDENCE:
        c_f = 0.0

    w_sum = c_f + c_t
    if w_sum <= 1e-8:
        v, a = 0.0, 0.0
    else:
        v = (c_f * v_f + c_t * v_t) / w_sum
        a = (c_f * a_f + c_t * a_t) / w_sum

    return {
        "valence": v,
        "arousal": a,
        "label": va_to_nearest_label(v, a),
        "weights": {"vision": c_f, "text": c_t},
    }
