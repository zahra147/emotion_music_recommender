# -*- coding: utf-8 -*-
"""End-to-end orchestration: text (+ optional image) -> emotion -> music.

This is the single entry point both the unit tests and the Streamlit demo
call. It wires together the branches that already exist elsewhere in the
project without duplicating any of their logic:

    text.text_emotion.TextEmotionModel   -> (v, a, confidence)
    vision.vision_pipeline.VisionPipeline -> (v, a, confidence)  [generic CNN]
    vision.face_landmarks (optional)      -> (v, a, confidence)  [personal]
    fusion.fusion.fuse                    -> combined (v, a) + label
    recommender.recommend.recommend       -> top-k closest songs

If no image is given, or the vision stack isn't available / finds no face,
the vision branch is treated as zero-confidence and fusion falls back to
text only (graceful degradation, same behaviour as walkthrough.py).

If a `personal_estimate` (v, a, confidence) is supplied - typically produced
by matching live webcam landmarks against the user's own reference photos,
see vision/face_landmarks.py - it's blended with the generic CNN's vision
estimate (confidence-weighted, same idea as the main fusion step) before
that combined vision branch is fused with text.
"""
from __future__ import annotations
from typing import Optional, Tuple

import numpy as np

from config import TOP_K
from fusion.fusion import fuse
from recommender.recommend import recommend
from text.text_emotion import TextEmotionModel

NEUTRAL = (0.0, 0.0, 0.0)


def _run_vision(image_bgr: Optional[np.ndarray]):
    """Run the vision branch. Never raises: any failure -> zero confidence,
    but the real exception is printed to the console and returned alongside
    the result (see run()'s "vision_error" key) instead of being silently
    swallowed - a missing dependency (e.g. tensorflow not installed) should
    never look identical to "no face was found in this photo"."""
    if image_bgr is None:
        return NEUTRAL, None
    try:
        from vision.vision_pipeline import VisionPipeline
        pipe = VisionPipeline()
        return pipe.infer(image_bgr), None
    except Exception as e:
        import sys
        print(f"[demo._run_vision] vision pipeline failed: {e!r}", file=sys.stderr)
        return NEUTRAL, repr(e)


def _blend(a: Tuple[float, float, float], b: Tuple[float, float, float]):
    """Confidence-weighted average of two (v, a, confidence) estimates from
    the SAME modality (here: two different vision estimators). Kept
    separate from fusion.fuse(), which is specifically the documented
    face+text fusion formula for the report - this is an internal detail of
    how the vision branch itself is produced."""
    va, aa, ca = a
    vb, ab, cb = b
    total = ca + cb
    if total <= 1e-8:
        return 0.0, 0.0, 0.0
    return (ca * va + cb * vb) / total, (ca * aa + cb * ab) / total, max(ca, cb)


def run(text: str = "", image_bgr: Optional[np.ndarray] = None,
        personal_estimate: Optional[Tuple[float, float, float]] = None,
        top_k: int = TOP_K) -> dict:
    """Run the full pipeline once and return every intermediate value.

    Args:
        text: user-written sentence describing how they feel. May be "".
        image_bgr: OpenCV-style BGR image array (e.g. from cv2.imread or a
            Streamlit upload converted to BGR), or None if no image.
        personal_estimate: optional (valence, arousal, confidence) from
            matching this image against the user's own calibration photos
            (see vision/face_landmarks.py). Blended with the CNN's own
            estimate before fusion with text.
        top_k: how many songs to return.

    Returns:
        {
          "text_branch":   {"valence", "arousal", "confidence"},
          "cnn_branch":    {"valence", "arousal", "confidence"},
          "personal_branch": {"valence", "arousal", "confidence"} or None,
          "vision_branch": {"valence", "arousal", "confidence"},  (blended)
          "fused":         {"valence", "arousal", "label", "weights"},
          "recommendations": [ {title, artist, valence, arousal, distance}, ... ],
        }
    """
    text_model = TextEmotionModel()
    v_t, a_t, c_t = text_model.infer(text or "")
    cnn_vision, vision_error = _run_vision(image_bgr)

    if personal_estimate is not None:
        v_f, a_f, c_f = _blend(cnn_vision, personal_estimate)
    else:
        v_f, a_f, c_f = cnn_vision

    fused = fuse((v_f, a_f, c_f), (v_t, a_t, c_t))
    recommendations = recommend(fused["valence"], fused["arousal"], top_k=top_k)

    return {
        "text_branch": {"valence": v_t, "arousal": a_t, "confidence": c_t},
        "cnn_branch": {"valence": cnn_vision[0], "arousal": cnn_vision[1], "confidence": cnn_vision[2]},
        "vision_error": vision_error,  # None unless the vision pipeline raised (e.g. missing dependency)
        "personal_branch": (
            {"valence": personal_estimate[0], "arousal": personal_estimate[1], "confidence": personal_estimate[2]}
            if personal_estimate is not None else None
        ),
        "vision_branch": {"valence": v_f, "arousal": a_f, "confidence": c_f},
        "fused": fused,
        "recommendations": recommendations,
    }


if __name__ == "__main__":
    import json
    out = run("I feel happy and full of energy")
    print(json.dumps(out, indent=2, ensure_ascii=False))
