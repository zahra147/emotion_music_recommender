# -*- coding: utf-8 -*-
"""Diagnose exactly why TextEmotionModel isn't loading the trained weights."""
from pathlib import Path
from config import WEIGHTS_DIR

print("WEIGHTS_DIR resolves to:", WEIGHTS_DIR.resolve())
print()

weights_path = WEIGHTS_DIR / "text_emotion.weights.h5"
tokenizer_path = WEIGHTS_DIR / "text_tokenizer.json"
config_path = WEIGHTS_DIR / "text_model_config.json"

for label, p in [("weights", weights_path), ("tokenizer", tokenizer_path), ("config", config_path)]:
    exists = p.exists()
    size = p.stat().st_size if exists else 0
    print(f"{label:10s} exists={exists!s:5s} size={size:>10,} bytes  path={p}")

print()
print("Now trying the actual load, with the exception NOT swallowed...")
import json
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.preprocessing.text import tokenizer_from_json
from config import EMOTIONS

with open(config_path, encoding="utf-8") as f:
    cfg = json.load(f)
print("config contents:", cfg)

with open(tokenizer_path, encoding="utf-8") as f:
    tok = tokenizer_from_json(f.read())
print("tokenizer loaded OK, vocab size:", len(tok.word_index))

max_len = cfg["max_len"]
model = models.Sequential([
    layers.Input(shape=(max_len,)),
    layers.Embedding(cfg["vocab_size"], cfg["embed_dim"], mask_zero=True),
    layers.Bidirectional(layers.LSTM(64)),
    layers.Dropout(0.4),
    layers.Dense(64, activation="relu"),
    layers.Dense(len(EMOTIONS), activation="softmax"),
])
model.load_weights(str(weights_path))
print("MODEL LOADED SUCCESSFULLY - the bug is being swallowed somewhere in _load_trained_model's try/except.")
