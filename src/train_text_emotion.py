# -*- coding: utf-8 -*-
"""Train the text branch: Embedding -> BiLSTM -> Dense(softmax) over the 7
project EMOTIONS classes (config.EMOTIONS), exactly as specified in the
design doc (section 4.2).

Dataset: GoEmotions (Demszky et al., 2020), re-mapped by the authors of
monologg/GoEmotions-pytorch to Ekman's 6 basic emotions + neutral - which is
precisely the 7-class taxonomy this project already uses. Source:
https://github.com/monologg/GoEmotions-pytorch (data/ekman/*.tsv)

Only single-label rows are used (a handful of GoEmotions rows carry more
than one label; those are dropped to keep this a clean multi-class task).

Run from `src`:
    python train_text_emotion.py
Produces:
    weights/text_emotion.h5
    weights/text_tokenizer.json
"""
from __future__ import annotations
import json
from pathlib import Path

import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

from config import EMOTIONS, WEIGHTS_DIR

DATA_DIR = Path(__file__).resolve().parent.parent / "dataset_text"
# GoEmotions-Ekman label order (from monologg/GoEmotions-pytorch labels.txt),
# mapped onto this project's EMOTIONS names.
GOEMOTIONS_EKMAN_LABELS = ["angry", "disgust", "fear", "happy", "neutral", "sad", "surprise"]

MAX_VOCAB = 20000
MAX_LEN = 30
EMBED_DIM = 100
BATCH_SIZE = 128
EPOCHS = 6


def load_split(name: str):
    texts, labels = [], []
    with open(DATA_DIR / name, encoding="utf-8") as f:
        for line in f:
            parts = line.rstrip("\n").split("\t")
            if len(parts) != 3:
                continue
            text, label_field, _id = parts
            if "," in label_field:  # drop multi-label rows
                continue
            label_name = GOEMOTIONS_EKMAN_LABELS[int(label_field)]
            texts.append(text)
            labels.append(EMOTIONS.index(label_name))
    return texts, labels


def build_model(vocab_size: int) -> tf.keras.Model:
    m = models.Sequential([
        layers.Input(shape=(MAX_LEN,)),
        layers.Embedding(vocab_size, EMBED_DIM, mask_zero=True),
        layers.Bidirectional(layers.LSTM(64)),
        layers.Dropout(0.4),
        layers.Dense(64, activation="relu"),
        layers.Dense(len(EMOTIONS), activation="softmax"),
    ])
    m.compile(optimizer="adam",
              loss="sparse_categorical_crossentropy",
              metrics=["accuracy"])
    return m


def main():
    print("Loading GoEmotions (Ekman taxonomy) splits...")
    train_texts, train_labels = load_split("train.tsv")
    dev_texts, dev_labels = load_split("dev.tsv")
    test_texts, test_labels = load_split("test.tsv")
    print(f"  train={len(train_texts)}  dev={len(dev_texts)}  test={len(test_texts)}")

    print("Class distribution (train):")
    for i, e in enumerate(EMOTIONS):
        print(f"  {e:9s} {train_labels.count(i)}")

    tok = Tokenizer(num_words=MAX_VOCAB, oov_token="<OOV>")
    tok.fit_on_texts(train_texts)

    def to_padded(texts):
        seqs = tok.texts_to_sequences(texts)
        return pad_sequences(seqs, maxlen=MAX_LEN, padding="post", truncating="post")

    X_train, y_train = to_padded(train_texts), np.array(train_labels)
    X_dev, y_dev = to_padded(dev_texts), np.array(dev_labels)
    X_test, y_test = to_padded(test_texts), np.array(test_labels)

    vocab_size = min(MAX_VOCAB, len(tok.word_index) + 1)
    model = build_model(vocab_size)
    model.summary()

    # Class weights: GoEmotions is imbalanced (lots of "neutral"/"joy",
    # very little "fear"/"surprise") - weight the loss so rare classes
    # aren't just ignored.
    counts = np.bincount(y_train, minlength=len(EMOTIONS)).astype("float32")
    class_weight = {i: float(len(y_train) / (len(EMOTIONS) * max(c, 1)))
                    for i, c in enumerate(counts)}
    print("class_weight:", class_weight)

    model.fit(
        X_train, y_train,
        validation_data=(X_dev, y_dev),
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        class_weight=class_weight,
        verbose=2,
    )

    print("\nTest set evaluation:")
    loss, acc = model.evaluate(X_test, y_test, verbose=0)
    print(f"  loss={loss:.3f}  accuracy={acc:.3f}")

    from sklearn.metrics import classification_report
    y_pred = model.predict(X_test, verbose=0).argmax(axis=1)
    print(classification_report(y_test, y_pred, target_names=EMOTIONS, digits=2))

    WEIGHTS_DIR.mkdir(parents=True, exist_ok=True)
    model.save_weights(str(WEIGHTS_DIR / "text_emotion.weights.h5"))
    with open(WEIGHTS_DIR / "text_tokenizer.json", "w", encoding="utf-8") as f:
        f.write(tok.to_json())
    # Also persist the architecture params needed to rebuild the model.
    with open(WEIGHTS_DIR / "text_model_config.json", "w", encoding="utf-8") as f:
        json.dump({"vocab_size": vocab_size, "max_len": MAX_LEN,
                    "embed_dim": EMBED_DIM}, f)
    print(f"\nSaved weights to {WEIGHTS_DIR}")


if __name__ == "__main__":
    main()
