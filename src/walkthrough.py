# -*- coding: utf-8 -*-
"""Zero-to-hundred trace of ONE input through the whole pipeline.

هر مرحله را با اعداد واقعی نشان می‌دهد — مناسب اجرای زنده سر ارائه.

    cd src
    python walkthrough.py
    python walkthrough.py --text "I am so angry right now"
    python walkthrough.py --text "..." --image face.jpg
"""
from __future__ import annotations
import argparse
import sys

from config import EMOTIONS, MIN_FACE_CONFIDENCE, TOP_K
from emotion_space import EMOTION_TO_VA, probs_to_va, va_to_nearest_label
from fusion.fusion import fuse
from recommender.music_library import load_library
from recommender.recommend import recommend
from text.text_emotion import KEYWORDS, TextEmotionModel

try:  # keep non-ASCII safe on every console
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

NEUTRAL = (0.0, 0.0, 0.0)


def step(n: int, title: str) -> None:
    print("\n" + "-" * 66)
    print(f"STEP {n}: {title}")
    print("-" * 66)


def show_probs(probs) -> None:
    for e, p in zip(EMOTIONS, probs):
        if p > 0:
            bar = "#" * int(round(p * 30))
            print(f"    {e:9s} {p:5.2f}  {bar}")


def trace_text(text: str):
    step(1, "Text input -> matched keywords")
    print(f"  raw text     : {text!r}")
    print(f"  lowercased   : {text.lower()!r}")
    hits = {e: [w for w in ws if w in text.lower()] for e, ws in KEYWORDS.items()}
    hits = {e: ws for e, ws in hits.items() if ws}
    print(f"  keyword hits : {hits if hits else 'NONE -> falls back to neutral'}")

    step(2, "Text -> probability vector over the 7 classes")
    model = TextEmotionModel()
    probs = model._predict_probs(text) if text.strip() else [0.0] * len(EMOTIONS)
    show_probs(probs)
    v_t, a_t, c_t = model.infer(text)
    print(f"  confidence (max prob) = {c_t:.2f}")

    step(3, "Probability vector -> point in Valence-Arousal space")
    print("  formula: v = SUM(p_i * valence_i),  a = SUM(p_i * arousal_i)")
    for e, p in zip(EMOTIONS, probs):
        if p > 0:
            ev, ea = EMOTION_TO_VA[e]
            print(f"    {p:.2f} * {e:9s}({ev:+.2f}, {ea:+.2f})")
    print(f"  => TEXT branch = (v={v_t:+.2f}, a={a_t:+.2f}, conf={c_t:.2f})")
    return (v_t, a_t, c_t)


def trace_vision(image_path: str | None):
    step(4, "Image -> face -> emotion -> Valence-Arousal")
    if not image_path:
        print("  no --image given, so this branch reports zero confidence:")
        print(f"  => VISION branch = {NEUTRAL}  (fusion will ignore it)")
        return NEUTRAL
    try:
        import cv2
        from vision.vision_pipeline import VisionPipeline
        img = cv2.imread(image_path)
        if img is None:
            print(f"  could not read {image_path!r}; treating as no face.")
            return NEUTRAL
        print(f"  image loaded : shape={img.shape}")
        pipe = VisionPipeline()
        print(f"  detector     : backend={pipe.detector._backend}")
        face, conf = pipe.detector.detect(img)
        if face is None:
            print("  no face found -> zero confidence")
            return NEUTRAL
        print(f"  face crop    : shape={face.shape}, detector conf={conf:.2f}")
        probs = pipe.fer.predict(pipe._preprocess(face))
        weights_loaded = pipe.fer._model is not None
        print(f"  FER weights  : {'loaded' if weights_loaded else 'MISSING -> neutral'}")
        show_probs(probs)
        v_f, a_f = probs_to_va(probs)
        print(f"  => VISION branch = (v={v_f:+.2f}, a={a_f:+.2f}, conf={conf:.2f})")
        return (v_f, a_f, conf)
    except Exception as e:
        print(f"  vision unavailable ({e}) -> zero confidence")
        return NEUTRAL


def trace_fusion(vision, text):
    step(5, "Fusion: two estimates -> one point")
    v_f, a_f, c_f = vision
    v_t, a_t, c_t = text
    kept = c_f if c_f >= MIN_FACE_CONFIDENCE else 0.0
    print(f"  vision (v={v_f:+.2f}, a={a_f:+.2f}) conf={c_f:.2f} "
          f"-> weight {kept:.2f}"
          f"{'  (below threshold, discarded)' if kept == 0.0 and c_f > 0 else ''}")
    print(f"  text   (v={v_t:+.2f}, a={a_t:+.2f}) conf={c_t:.2f} -> weight {c_t:.2f}")
    total = kept + c_t
    if total > 0:
        print(f"  v = ({kept:.2f}*{v_f:+.2f} + {c_t:.2f}*{v_t:+.2f}) / {total:.2f}")
        print(f"  a = ({kept:.2f}*{a_f:+.2f} + {c_t:.2f}*{a_t:+.2f}) / {total:.2f}")
    else:
        print("  both weights are zero -> fall back to the neutral origin (0, 0)")
    fused = fuse(vision, text)
    print(f"  => FUSED = (v={fused['valence']:+.2f}, a={fused['arousal']:+.2f})")

    step(6, "Interpreting the point as a human-readable emotion")
    print("  nearest emotion coordinate in the circumplex:")
    print(f"  => label = {fused['label']}")
    return fused


def trace_reco(fused):
    step(7, "Point -> music: euclidean distance to every song")
    lib = load_library()
    qv, qa = fused["valence"], fused["arousal"]
    rows = sorted(
        ((s, ((s["valence"] - qv) ** 2 + (s["arousal"] - qa) ** 2) ** 0.5)
         for s in lib), key=lambda r: r[1])
    print(f"  library = {len(lib)} songs; query = (v={qv:+.2f}, a={qa:+.2f})\n")
    for i, (s, d) in enumerate(rows, 1):
        mark = "<= picked" if i <= TOP_K else ""
        print(f"   {i:2d}. {s['title']:12s} (v={s['valence']:+.2f}, "
              f"a={s['arousal']:+.2f})  d={d:.3f} {mark}")

    step(8, "Final output shown to the user")
    for i, s in enumerate(recommend(qv, qa), 1):
        print(f"   {i}. {s['title']} - {s['artist']}  (d={s['distance']})")
    print("\n  note: a small random jitter is added, so distances shift slightly.")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--text", default="I feel happy and full of energy")
    ap.add_argument("--image", default=None)
    args = ap.parse_args()

    print("=" * 66)
    print("STEP 0: what the user gives the system")
    print("=" * 66)
    print(f"  text  : {args.text!r}")
    print(f"  image : {args.image or 'none'}")

    text_va = trace_text(args.text)
    vision_va = trace_vision(args.image)
    fused = trace_fusion(vision_va, text_va)
    trace_reco(fused)
    print("\nDone.\n")


if __name__ == "__main__":
    main()
