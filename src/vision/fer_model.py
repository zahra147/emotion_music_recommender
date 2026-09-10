# -*- coding: utf-8 -*-
"""Facial Emotion Recognition (FER) CNN — Keras.

`build_fer_cnn` defines a light CNN suitable for training on FER2013.
`FERModel` loads trained weights and predicts a probability vector.
If no weights are available it returns a neutral distribution so the rest
of the pipeline still runs end-to-end during development.
"""
from __future__ import annotations
from pathlib import Path
from typing import List

import numpy as np

from config import EMOTIONS, FER_INPUT_SIZE, WEIGHTS_DIR


def build_fer_cnn():
    """CNN for 48x48 grayscale face crops, trained in train_fer_cnn.py.

    Note: an earlier, larger version of this function (32/64/128 filters,
    Flatten -> Dense(128), ~684k params) is what the design doc originally
    proposed. On the ~3k real training images actually available (see
    train_fer_cnn.py's docstring - a real face+emotion dataset, not
    FER2013, since FER2013 needs Kaggle access this environment doesn't
    have), that larger network memorized the majority classes within a
    couple of epochs and collapsed to predicting only "happy"/"neutral"
    regardless of input. This smaller, more regularized architecture
    (16/32/64 filters, GlobalAveragePooling instead of Flatten, L2 + heavier
    dropout) generalizes much better on a dataset this small - this is the
    architecture the shipped weights (weights/fer_cnn.weights.h5) match.
    """
    from tensorflow.keras import layers, models, regularizers  # local import (heavy dep)

    reg = regularizers.l2(1e-4)
    m = models.Sequential([
        layers.Input(shape=(*FER_INPUT_SIZE, 1)),
        layers.Conv2D(16, 3, activation="relu", padding="same", kernel_regularizer=reg),
        layers.BatchNormalization(),
        layers.MaxPooling2D(),
        layers.Conv2D(32, 3, activation="relu", padding="same", kernel_regularizer=reg),
        layers.BatchNormalization(),
        layers.MaxPooling2D(),
        layers.Conv2D(64, 3, activation="relu", padding="same", kernel_regularizer=reg),
        layers.BatchNormalization(),
        layers.MaxPooling2D(),
        layers.GlobalAveragePooling2D(),
        layers.Dropout(0.5),
        layers.Dense(64, activation="relu", kernel_regularizer=reg),
        layers.Dropout(0.3),
        layers.Dense(len(EMOTIONS), activation="softmax"),
    ])
    m.compile(optimizer="adam",
              loss="categorical_crossentropy",
              metrics=["accuracy"])
    return m


class FERModel:
    def __init__(self, weights_path: Path | None = None):
        self.weights_path = weights_path or (WEIGHTS_DIR / "fer_cnn.weights.h5")
        self._model = None
        if self.weights_path.exists():
            self._model = build_fer_cnn()
            self._model.load_weights(str(self.weights_path))

    def predict(self, face_gray_48: np.ndarray) -> List[float]:
        """face_gray_48: (48,48) grayscale in [0,1]. Returns 7-class probs."""
        if self._model is None:
            # No trained weights yet -> neutral so the demo still works.
            probs = [0.0] * len(EMOTIONS)
            probs[EMOTIONS.index("neutral")] = 1.0
            return probs
        x = face_gray_48.reshape(1, *FER_INPUT_SIZE, 1).astype("float32")
        return self._model.predict(x, verbose=0)[0].tolist()
