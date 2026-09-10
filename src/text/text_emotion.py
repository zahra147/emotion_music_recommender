# -*- coding: utf-8 -*-
"""Text branch: user text -> (valence, arousal, confidence).

Two layers, in priority order:

1. A trained BiLSTM classifier (Embedding -> BiLSTM -> Dense(softmax)) over
   the 7 EMOTIONS, trained on GoEmotions (Ekman taxonomy) - see
   train_text_emotion.py. Used whenever its weights are present AND the
   input text has at least one word the model actually recognizes.

2. A transparent bilingual keyword lexicon, used as a fallback when:
     - the trained weights aren't on disk (keeps the pipeline runnable
       before/without training, same graceful-degradation philosophy as
       the rest of this project), or
     - the model doesn't recognize *any* word in the input - which in
       practice means either the text is Persian (the training data was
       English-only Reddit comments, so Persian is entirely out-of-
       vocabulary) or it's gibberish, e.g. "zzzz qqqq".

This keeps a single, small, auditable rule-based path for the language the
trained model was never exposed to, while letting real deep learning handle
the common case (English sentences with recognizable vocabulary).
"""
from __future__ import annotations
import json
from pathlib import Path
from typing import List, Optional, Tuple

from config import EMOTIONS, WEIGHTS_DIR
from emotion_space import probs_to_va

# Minimal bilingual keyword lexicon - the fallback path (see module docstring).
KEYWORDS = {
    "happy":    ["happy", "great", "joy", "خوشحال", "شاد", "عالی"],
    "sad":      ["sad", "down", "unhappy", "غمگین", "ناراحت", "دلگیر"],
    "angry":    ["angry", "mad", "furious", "عصبانی", "خشمگین"],
    "fear":     ["afraid", "scared", "anxious", "ترس", "نگران", "مضطرب"],
    "surprise": ["surprised", "wow", "shocked", "شگفت", "تعجب"],
    "disgust":  ["disgust", "gross", "متنفر", "چندش"],
    "neutral":  ["ok", "fine", "معمولی", "خوبم"],
}


class TextEmotionModel:
    def __init__(self, weights_path: Path | None = None):
        self.weights_path = weights_path or (WEIGHTS_DIR / "text_emotion.weights.h5")
        self._model = None
        self._tokenizer = None
        self._max_len = None
        self._oov_index = None
        self._load_trained_model()

    def _load_trained_model(self) -> None:
        """Best-effort load of the trained BiLSTM. Any failure (weights
        missing, TensorFlow not installed, files from an older run) just
        leaves self._model as None and the class quietly falls back to the
        keyword lexicon - same graceful-degradation pattern the vision
        branch uses for missing FER weights."""
        tokenizer_path = WEIGHTS_DIR / "text_tokenizer.json"
        config_path = WEIGHTS_DIR / "text_model_config.json"
        if not (self.weights_path.exists() and tokenizer_path.exists()
                and config_path.exists()):
            return
        try:
            import tensorflow as tf
            from tensorflow.keras import layers, models
            from tensorflow.keras.preprocessing.text import tokenizer_from_json

            with open(config_path, encoding="utf-8") as f:
                cfg = json.load(f)
            with open(tokenizer_path, encoding="utf-8") as f:
                self._tokenizer = tokenizer_from_json(f.read())

            self._max_len = cfg["max_len"]
            model = models.Sequential([
                layers.Input(shape=(self._max_len,)),
                layers.Embedding(cfg["vocab_size"], cfg["embed_dim"], mask_zero=True),
                layers.Bidirectional(layers.LSTM(64)),
                layers.Dropout(0.4),
                layers.Dense(64, activation="relu"),
                layers.Dense(len(EMOTIONS), activation="softmax"),
            ])
            model.load_weights(str(self.weights_path))
            self._model = model
            self._oov_index = self._tokenizer.word_index.get(
                self._tokenizer.oov_token) if self._tokenizer.oov_token else None
        except Exception:
            self._model = None
            self._tokenizer = None

    def _predict_with_model(self, text: str) -> Optional[List[float]]:
        """Returns None (signal to fall back) if the model isn't loaded or
        every token in `text` is out-of-vocabulary - the trained model was
        never taught anything about those words, so a confident guess would
        be worse than the transparent keyword fallback."""
        if self._model is None:
            return None
        seq = self._tokenizer.texts_to_sequences([text])[0]
        recognized = [t for t in seq if t != self._oov_index]
        if not recognized:
            return None
        from tensorflow.keras.preprocessing.sequence import pad_sequences
        padded = pad_sequences([seq], maxlen=self._max_len,
                                padding="post", truncating="post")
        probs = self._model.predict(padded, verbose=0)[0]
        return probs.tolist()

    def _predict_probs(self, text: str) -> List[float]:
        t = text.lower()

        model_probs = self._predict_with_model(t)
        if model_probs is not None:
            return model_probs

        # --- keyword fallback (Persian, OOV/gibberish, or no trained model) ---
        scores = {e: 0.0 for e in EMOTIONS}
        for emo, words in KEYWORDS.items():
            scores[emo] += sum(1.0 for w in words if w in t)
        total = sum(scores.values())
        if total == 0:
            probs = [0.0] * len(EMOTIONS)
            probs[EMOTIONS.index("neutral")] = 1.0
            return probs
        return [scores[e] / total for e in EMOTIONS]

    def infer(self, text: str) -> Tuple[float, float, float]:
        """Return (valence, arousal, confidence)."""
        if not text or not text.strip():
            return 0.0, 0.0, 0.0
        probs = self._predict_probs(text)
        v, a = probs_to_va(probs)
        # Confidence ~ how peaked the distribution is (0..1).
        confidence = max(probs)
        return v, a, float(confidence)
