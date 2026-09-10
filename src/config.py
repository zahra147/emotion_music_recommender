# -*- coding: utf-8 -*-
"""Central configuration for the emotion-based music recommender."""
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
WEIGHTS_DIR = ROOT / "weights"

# 7 emotion classes shared by both vision and text branches (FER2013 order).
EMOTIONS = ["angry", "disgust", "fear", "happy", "sad", "surprise", "neutral"]

# Face model input size (48x48 grayscale for a light FER-CNN).
FER_INPUT_SIZE = (48, 48)

# Fusion: minimum face-detection confidence below which we trust text more.
MIN_FACE_CONFIDENCE = 0.35

# Recommender: how many songs to return.
TOP_K = 5
# Small random jitter added to the query point so results are not identical
# every time (diversity). Set to 0.0 for deterministic behaviour.
RECO_JITTER = 0.05
