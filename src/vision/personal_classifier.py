# -*- coding: utf-8 -*-
"""A genuine learning layer, personal to the user - transfer learning on
top of the already-trained FER-CNN.

Why this instead of (or in addition to) raw landmark-distance matching
(face_landmarks.py): that approach never learns anything - it just compares
raw geometry with a fixed distance formula. This module instead:

  1. Reuses the trained CNN (vision/fer_model.py) as a FROZEN feature
     extractor, reading off its 64-dim internal representation (the Dense
     layer right before the final softmax) instead of its final
     prediction. That 64-dim space already encodes general,
     expression-relevant structure learned from thousands of other
     people's faces.
  2. Trains a small classifier (logistic regression) ONLY on the user's
     own calibration photos, in that 64-dim embedding space. With a good
     frozen feature extractor, a handful of examples per class (3-5) is
     often enough for a linear classifier to find a reasonable person-
     specific decision boundary - this is the standard few-shot transfer
     learning pattern: freeze the backbone, fit a tiny head on new data.

This is a real, if small, training step - unlike the nearest-neighbor
landmark approach, this module actually fits parameters to data.
"""
from __future__ import annotations
from typing import Dict, List, Optional

import numpy as np

from config import EMOTIONS, FER_INPUT_SIZE


class EmbeddingExtractor:
    """Wraps the trained FER-CNN, exposing its penultimate Dense(64) layer
    as a frozen feature extractor instead of its final 7-class prediction."""

    def __init__(self):
        self._embed_model = None
        self.error: Optional[str] = None
        try:
            from vision.fer_model import FERModel
            fer = FERModel()
            if fer._model is None:
                self.error = "FER-CNN weights not found - train/ship fer_cnn.weights.h5 first"
                return
            import tensorflow as tf
            # A model loaded via load_weights() (rather than a full saved
            # model) has no defined .input until it's actually been called
            # once - build it with a throwaway forward pass first.
            dummy = np.zeros((1, *FER_INPUT_SIZE, 1), dtype="float32")
            fer._model.predict(dummy, verbose=0)
            # layers[-2] = the Dense(64, relu) right before the final
            # Dense(7, softmax) - see vision/fer_model.py's build_fer_cnn().
            penultimate = fer._model.layers[-2].output
            self._embed_model = tf.keras.Model(inputs=fer._model.inputs[0], outputs=penultimate)
        except Exception as e:
            self.error = repr(e)

    def extract(self, face_gray_normalized: np.ndarray) -> Optional[np.ndarray]:
        """face_gray_normalized: (48,48) grayscale in [0,1], same
        preprocessing as VisionPipeline._preprocess. Returns a 64-dim
        embedding, or None if the extractor isn't available."""
        if self._embed_model is None:
            return None
        x = face_gray_normalized.reshape(1, *FER_INPUT_SIZE, 1).astype("float32")
        return self._embed_model.predict(x, verbose=0)[0]


class PersonalClassifier:
    """Small classifier trained only on this user's own calibration-photo
    embeddings. Needs at least 2 different calibrated emotions to fit."""

    def __init__(self):
        self._clf = None
        self._classes: List[str] = []

    @property
    def is_fitted(self) -> bool:
        return self._clf is not None

    def fit(self, samples: Dict[str, List[np.ndarray]]) -> bool:
        """samples: {emotion_name: [embedding, embedding, ...]}.
        Returns False (and leaves the classifier unfit) if fewer than 2
        distinct emotions have samples."""
        X, y = [], []
        for emo, vecs in samples.items():
            for v in vecs:
                X.append(v)
                y.append(emo)
        if len(set(y)) < 2:
            return False
        from sklearn.linear_model import LogisticRegression
        clf = LogisticRegression(max_iter=1000, C=1.0)
        clf.fit(np.array(X), y)
        self._clf = clf
        self._classes = list(clf.classes_)
        return True

    def predict_probs(self, embedding: np.ndarray) -> Optional[List[float]]:
        """Returns a probability vector over the full EMOTIONS list
        (0 for any emotion the user never calibrated)."""
        if self._clf is None:
            return None
        raw = self._clf.predict_proba(embedding.reshape(1, -1))[0]
        probs = [0.0] * len(EMOTIONS)
        for cls, p in zip(self._classes, raw):
            probs[EMOTIONS.index(cls)] = float(p)
        return probs
