# -*- coding: utf-8 -*-
"""Shared Valence-Arousal (V-A) emotion space.

Both the vision branch and the text branch output a probability vector over
the 7 EMOTIONS. We map that vector into a single (valence, arousal) point,
which is the common representation used for fusion and for music matching.

Coordinates follow Russell's circumplex model, roughly in [-1, 1].
"""
from __future__ import annotations
from typing import Sequence, Tuple

from config import EMOTIONS

# Approximate (valence, arousal) coordinates for each discrete emotion.
EMOTION_TO_VA = {
    "angry":    (-0.7,  0.7),
    "disgust":  (-0.6,  0.3),
    "fear":     (-0.6,  0.8),
    "happy":    ( 0.8,  0.6),
    "sad":      (-0.7, -0.5),
    "surprise": ( 0.3,  0.8),
    "neutral":  ( 0.0,  0.0),
}


def probs_to_va(probs: Sequence[float]) -> Tuple[float, float]:
    """Convert a probability vector over EMOTIONS to a (valence, arousal) point.

    We take the probability-weighted average of each emotion's V-A coordinate,
    which yields a smooth point inside the circumplex instead of a hard label.
    """
    if len(probs) != len(EMOTIONS):
        raise ValueError(f"expected {len(EMOTIONS)} probabilities, got {len(probs)}")
    v = sum(p * EMOTION_TO_VA[e][0] for p, e in zip(probs, EMOTIONS))
    a = sum(p * EMOTION_TO_VA[e][1] for p, e in zip(probs, EMOTIONS))
    return (float(v), float(a))


def va_to_nearest_label(v: float, a: float) -> str:
    """Return the discrete emotion whose V-A coordinate is closest to (v, a)."""
    best, best_d = "neutral", float("inf")
    for emo, (ev, ea) in EMOTION_TO_VA.items():
        d = (ev - v) ** 2 + (ea - a) ** 2
        if d < best_d:
            best, best_d = emo, d
    return best
