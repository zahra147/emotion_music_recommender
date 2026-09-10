# -*- coding: utf-8 -*-
"""Recommender: nearest songs to the user's (valence, arousal) point."""
from __future__ import annotations
import math
import random
from typing import Dict, List

from config import TOP_K, RECO_JITTER
from recommender.music_library import load_library


def recommend(valence: float, arousal: float,
              top_k: int = TOP_K, jitter: float = RECO_JITTER,
              library: List[Dict] | None = None) -> List[Dict]:
    """Return the top_k closest songs in V-A space (with optional jitter)."""
    songs = library if library is not None else load_library()
    qv = valence + random.uniform(-jitter, jitter)
    qa = arousal + random.uniform(-jitter, jitter)

    scored = []
    for s in songs:
        dist = math.hypot(s["valence"] - qv, s["arousal"] - qa)
        item = dict(s)
        item["distance"] = round(dist, 4)
        scored.append(item)

    scored.sort(key=lambda x: x["distance"])
    return scored[:top_k]
