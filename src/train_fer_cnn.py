# -*- coding: utf-8 -*-
"""Train the vision branch: a CNN over 48x48 grayscale face crops.

Dataset: "facial_expressions" (muxspace/facial_expressions on GitHub) - real
photographs (mostly from LFW) hand-labeled with an emotion per image. It is
NOT FER2013 (which needs Kaggle access we don't have here), but it maps
cleanly onto the same 7 classes and is genuine face + emotion data, not
synthetic.

For every image we run this project's OWN face detector (vision.face_detector)
to crop just the face, exactly like production inference does - so the model
trains on the same kind of input it will see at inference time. That pass is
slow (~60ms/image) so the result is cached to dataset_vision_cache.npz next
to this script's project root; delete that file to force a full rebuild.

The dataset is heavily imbalanced (only ~21 "fear" examples vs. ~1200
"happy"/"neutral"). A first attempt with the exact architecture proposed in
the design doc (build_fer_cnn in vision/fer_model.py, ~684k params) collapsed
to predicting only "happy"/"neutral" regardless of input. This script instead
uses build_light_fer_cnn() (much smaller + L2 + heavier dropout) and
oversamples+augments (flip/brightness/shift) the tiniest classes so the loss
actually sees enough of them to learn something.

Run from `src`:
    python train_fer_cnn.py
Produces:
    weights/fer_cnn.weights.h5
"""
from __future__ import annotations
import csv
from pathlib import Path

import cv2
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.utils.class_weight import compute_class_weight

from config import EMOTIONS, FER_INPUT_SIZE, WEIGHTS_DIR
from vision.face_detector import FaceDetector
from vision.fer_model import build_fer_cnn

ROOT = Path(__file__).resolve().parent.parent
DATASET_DIR = ROOT / "facial_expressions"
IMAGES_DIR = DATASET_DIR / "images"
LEGEND_CSV = DATASET_DIR / "data" / "legend.csv"
CACHE_PATH = ROOT / "dataset_vision_cache.npz"

# facial_expressions label -> this project's EMOTIONS name. Rows with a
# label not in this map (e.g. "contempt") are skipped.
LABEL_MAP = {
    "anger": "angry", "ANGER": "angry",
    "disgust": "disgust", "DISGUST": "disgust",
    "fear": "fear", "FEAR": "fear",
    "happiness": "happy", "HAPPINESS": "happy",
    "sadness": "sad", "SADNESS": "sad",
    "surprise": "surprise", "SURPRISE": "surprise",
    "neutral": "neutral", "NEUTRAL": "neutral",
}

MAX_PER_CLASS = 1200          # cap the huge "neutral"/"happy" classes
MIN_PER_CLASS_AUGMENTED = 250  # oversample tiny classes up to this (train split only)
EPOCHS = 25
BATCH_SIZE = 32


def load_rows():
    rows = []
    with open(LEGEND_CSV, encoding="utf-8") as f:
        for r in csv.DictReader(f):
            emo = LABEL_MAP.get(r["emotion"])
            if emo:
                rows.append((r["image"], emo))
    return rows


def cap_per_class(rows):
    by_class = {}
    for img, emo in rows:
        by_class.setdefault(emo, []).append(img)
    kept = []
    for emo, imgs in by_class.items():
        if len(imgs) > MAX_PER_CLASS:
            rng = np.random.default_rng(0)
            imgs = list(rng.choice(imgs, size=MAX_PER_CLASS, replace=False))
        kept += [(img, emo) for img in imgs]
        print(f"  {emo:9s} using {len(imgs)} / {len(by_class[emo])} available")
    return kept


def build_dataset(rows):
    detector = FaceDetector()
    print("face detector backend:", detector._backend)
    X, y = [], []
    n_no_face = 0
    for i, (fname, emo) in enumerate(rows):
        path = IMAGES_DIR / fname
        img = cv2.imread(str(path))
        if img is None:
            continue
        face, conf = detector.detect(img)
        if face is None:
            n_no_face += 1
            continue
        gray = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)
        gray = cv2.resize(gray, FER_INPUT_SIZE)
        X.append(gray.astype("float32") / 255.0)
        y.append(EMOTIONS.index(emo))
        if (i + 1) % 500 == 0:
            print(f"  processed {i + 1}/{len(rows)}  (no-face so far: {n_no_face})")
    print(f"done. {len(X)} usable faces, {n_no_face} images with no detected face.")
    X = np.stack(X)[..., np.newaxis]
    y = np.array(y)
    return X, y


def get_dataset():
    if CACHE_PATH.exists():
        print(f"Loading cached preprocessed dataset from {CACHE_PATH}")
        data = np.load(CACHE_PATH)
        return data["X"], data["y"]
    print("Loading legend.csv...")
    rows = load_rows()
    print(f"  {len(rows)} labeled rows across {len(set(e for _, e in rows))} classes")
    rows = cap_per_class(rows)
    print("\nRunning face detection + preprocessing on every image "
          "(this is slow - one detection call per image)...")
    X, y = build_dataset(rows)
    np.savez_compressed(CACHE_PATH, X=X, y=y)
    print(f"Cached to {CACHE_PATH}")
    return X, y


def _rotate_zoom(img2d: np.ndarray, angle_deg: float, zoom: float) -> np.ndarray:
    h, w = img2d.shape
    M = cv2.getRotationMatrix2D((w / 2, h / 2), angle_deg, zoom)
    return cv2.warpAffine(img2d, M, (w, h), borderMode=cv2.BORDER_REFLECT101)


def augment_batch(imgs: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Cheap augmentations for 48x48 grayscale face crops: horizontal flip,
    small brightness jitter, small translation. Used only to oversample the
    classes with very few real examples (fear, disgust, surprise, sad).

    Note: rotation+zoom were tried too and made results *worse* here - with
    only ~14-23 real images for the tiniest classes, stronger augmentation
    just perturbs noise into copies of the same handful of images rather
    than adding real information, and dilutes the signal for other classes.
    """
    out = imgs.copy()
    if rng.random() < 0.5:
        out = out[:, :, ::-1, :]
    out = np.clip(out + rng.uniform(-0.08, 0.08), 0.0, 1.0)
    sx, sy = rng.integers(-3, 4, size=2)
    out = np.roll(out, shift=(sx, sy), axis=(1, 2))
    return out


def oversample_minority_classes(X, y):
    rng = np.random.default_rng(0)
    Xs, ys = [X], [y]
    for cls in range(len(EMOTIONS)):
        idx = np.where(y == cls)[0]
        n = len(idx)
        if 0 < n < MIN_PER_CLASS_AUGMENTED:
            n_needed = MIN_PER_CLASS_AUGMENTED - n
            picks = rng.choice(idx, size=n_needed, replace=True)
            aug = augment_batch(X[picks], rng)
            Xs.append(aug)
            ys.append(np.full(n_needed, cls))
            print(f"  oversampled {EMOTIONS[cls]:9s} {n} -> {n + n_needed} "
                  f"(+{n_needed} augmented copies)")
    return np.concatenate(Xs), np.concatenate(ys)


def main():
    tf.random.set_seed(0)
    np.random.seed(0)
    X, y = get_dataset()
    print("\nClass counts (full set, before train/test split):")
    for i, e in enumerate(EMOTIONS):
        print(f"  {e:9s} {(y == i).sum()}")

    X_trainval, X_test, y_trainval, y_test = train_test_split(
        X, y, test_size=0.15, random_state=0, stratify=y
    )
    # Carve the validation set out of REAL images only, before any
    # oversampling/augmentation is added - otherwise Keras's tail-based
    # validation_split would end up validating almost entirely on
    # synthetic duplicates of the rare classes (which is what happened
    # in the first attempt: val_accuracy stuck at 0%).
    X_train, X_val, y_train, y_val = train_test_split(
        X_trainval, y_trainval, test_size=0.15, random_state=0, stratify=y_trainval
    )
    X_train, y_train = oversample_minority_classes(X_train, y_train)
    # Shuffle so batches mix real + synthetic examples of every class.
    perm = np.random.default_rng(0).permutation(len(X_train))
    X_train, y_train = X_train[perm], y_train[perm]

    class_weights_arr = compute_class_weight(
        class_weight="balanced", classes=np.arange(len(EMOTIONS)), y=y_train
    )
    class_weight = {i: w for i, w in enumerate(class_weights_arr)}
    print("class_weight:", class_weight)

    model = build_fer_cnn()
    model.summary()

    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss", patience=6, restore_best_weights=True),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss", factor=0.5, patience=3, min_lr=1e-5),
    ]

    model.fit(
        X_train, tf.keras.utils.to_categorical(y_train, len(EMOTIONS)),
        validation_data=(X_val, tf.keras.utils.to_categorical(y_val, len(EMOTIONS))),
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        class_weight=class_weight,
        callbacks=callbacks,
        verbose=2,
    )

    print("\nTest set evaluation:")
    y_pred = model.predict(X_test, verbose=0).argmax(axis=1)
    print(classification_report(y_test, y_pred, target_names=EMOTIONS,
                                 digits=2, zero_division=0))

    from sklearn.metrics import confusion_matrix
    cm = confusion_matrix(y_test, y_pred, labels=range(len(EMOTIONS)))
    print("Confusion matrix (rows=true, cols=predicted):")
    print("           " + " ".join(f"{e[:4]:>5s}" for e in EMOTIONS))
    for i, row in enumerate(cm):
        print(f"  {EMOTIONS[i]:9s} " + " ".join(f"{v:5d}" for v in row))

    WEIGHTS_DIR.mkdir(parents=True, exist_ok=True)
    out_path = WEIGHTS_DIR / "fer_cnn.weights.h5"
    model.save_weights(str(out_path))
    print(f"\nSaved weights to {out_path}")


if __name__ == "__main__":
    main()
