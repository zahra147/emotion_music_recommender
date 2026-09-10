# -*- coding: utf-8 -*-
"""Live demo UI for the emotion-based music recommender.

Run from inside `src`:
    streamlit run streamlit_app.py

Design concept: a four-channel "mixing console". The app fuses two input
signals (face, text) into one channel (fusion) that drives an output
(music) - so each of the four pipeline stages gets its own accent color,
and those four colors are literally the four quadrants of the
Valence-Arousal plane the app computes with (amber = high valence/high
arousal, teal = high valence/low arousal, indigo = low valence/low arousal,
coral = low valence/high arousal). The palette isn't decoration - it's the
app's own mental model made visible.

Uses the exact same code the unit tests and walkthrough.py use (demo.run,
config, emotion_space, fusion, recommender) - nothing here duplicates
pipeline logic, it only visualizes it.
"""
from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image, ImageOps

from config import EMOTIONS, MIN_FACE_CONFIDENCE
from demo import run
from emotion_space import EMOTION_TO_VA, probs_to_va
from recommender.music_library import load_library
from text.text_emotion import KEYWORDS, TextEmotionModel

# ---------------------------------------------------------------- palette --
BG = "#14121F"
PANEL = "#1F1B31"
PANEL_LINE = "#2E2A45"
TEXT = "#F3EFE7"
MUTED = "#9C97B3"
AMBER = "#F2A65A"   # channel 1 - text - high valence / high arousal
TEAL = "#4FB6A8"    # channel 2 - vision - high valence / low arousal
INDIGO = "#6C7BEA"  # channel 3 - fusion - low valence / low arousal
CORAL = "#E1685A"   # channel 4 - music - low valence / high arousal

st.set_page_config(page_title="Emotion-Based Music Recommender", page_icon="🎛️", layout="wide")

# ------------------------------------------------------------------- CSS --
st.markdown(f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=Fraunces:opsz,wght@9..144,500..700&display=swap');

html, body, [class*="css"] {{
    font-family: 'Inter', sans-serif;
}}
.stApp {{
    background:
        radial-gradient(circle at 15% 0%, rgba(242,166,90,0.06), transparent 40%),
        radial-gradient(circle at 85% 100%, rgba(225,104,90,0.06), transparent 40%),
        {BG};
}}
[data-testid="stSidebar"] {{
    background: {PANEL};
    border-right: 1px solid {PANEL_LINE};
}}
[data-testid="stSidebar"] * {{
    color: {TEXT};
}}
[data-testid="stSidebar"] textarea, [data-testid="stSidebar"] [data-testid="stFileUploaderDropzone"] {{
    background: {BG};
    border: 1px solid {PANEL_LINE};
    border-radius: 10px;
    color: {TEXT};
}}

/* ---- hero (kept compact - this was too large before) ---- */
.hero-title {{
    font-family: 'Fraunces', serif;
    font-size: 1.7rem;
    font-weight: 600;
    color: {TEXT};
    margin: 0 0 0.25rem 0;
    letter-spacing: -0.01em;
}}
.hero-sub {{
    color: {MUTED};
    font-size: 0.92rem;
    line-height: 1.6;
    max-width: 640px;
    margin-bottom: 0.8rem;
}}
.hero-bar {{
    display: flex;
    height: 4px;
    border-radius: 999px;
    overflow: hidden;
    margin-bottom: 1.6rem;
}}
.hero-bar div {{ flex: 1; }}

/* ---- native bordered container (st.container(border=True)) ---- */
div[data-testid="stVerticalBlockBorderWrapper"] {{
    background: {PANEL};
    border: 1px solid {PANEL_LINE} !important;
    border-radius: 14px !important;
    padding: 0.3rem 0.5rem;
    margin-bottom: 1.4rem;
}}
.channel {{
    display: flex;
    align-items: center;
    gap: 0.6rem;
    margin-bottom: 0.5rem;
}}
.channel-badge {{
    width: 32px; height: 32px;
    border-radius: 9px;
    display: flex; align-items: center; justify-content: center;
    font-family: 'Fraunces', serif;
    font-weight: 700;
    font-size: 1rem;
    color: {BG};
    flex-shrink: 0;
}}
.channel-title {{
    font-size: 1.1rem;
    font-weight: 600;
    color: {TEXT};
}}

.stat-row {{ display: flex; gap: 0.7rem; margin: 0.2rem 0 1rem 0; flex-wrap: wrap; }}
.stat-chip {{
    background: {BG};
    border: 1px solid {PANEL_LINE};
    border-left: 3px solid var(--accent);
    border-radius: 8px;
    padding: 0.55rem 0.9rem;
    min-width: 110px;
}}
.stat-chip .label {{ color: {MUTED}; font-size: 0.74rem; }}
.stat-chip .value {{
    font-family: 'Fraunces', serif;
    color: {TEXT};
    font-size: 1.3rem;
    font-weight: 600;
}}

[data-testid="stButton"] button {{
    border-radius: 999px;
    background: linear-gradient(135deg, {AMBER}, {CORAL});
    color: {BG};
    font-weight: 700;
    border: none;
    box-shadow: 0 4px 18px rgba(242,166,90,0.25);
}}
[data-testid="stButton"] button:hover {{
    filter: brightness(1.08);
    box-shadow: 0 6px 22px rgba(242,166,90,0.4);
}}

[data-testid="stVideo"] {{
    padding: 0.9rem;
    background: {PANEL};
    border: 1px solid {PANEL_LINE};
    border-left: 3px solid {CORAL};
    border-radius: 12px;
}}
[data-testid="stImage"] img {{
    border-radius: 12px;
    border: 1px solid {PANEL_LINE};
}}
[data-testid="stDataFrame"] {{
    border: 1px solid {PANEL_LINE};
    border-radius: 10px;
    overflow: hidden;
}}
hr {{ border-color: {PANEL_LINE}; }}
</style>
""", unsafe_allow_html=True)


def channel_header(number: str, title: str, color: str) -> None:
    st.markdown(f"""
    <div class="channel">
      <div class="channel-badge" style="background:{color};">{number}</div>
      <div class="channel-title">{title}</div>
    </div>
    <div style="height:3px;border-radius:3px;background:{color};opacity:0.5;margin-bottom:0.9rem;"></div>
    """, unsafe_allow_html=True)


def stat_chips(items: list[tuple[str, str, str]]) -> None:
    """items: list of (label, value, accent_hex)."""
    chips = "".join(
        f'<div class="stat-chip" style="--accent:{color};">'
        f'<div class="label">{label}</div><div class="value">{value}</div></div>'
        for label, value, color in items
    )
    st.markdown(f'<div class="stat-row">{chips}</div>', unsafe_allow_html=True)


# ------------------------------------------------------------------ hero --
st.markdown('<div class="hero-title">🎛️ Emotion → Music Mixer</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="hero-sub">Two input channels (face + text) are combined on a shared mixing '
    'plane (Valence–Arousal space) to pick the closest matching song.</div>',
    unsafe_allow_html=True,
)
st.markdown(
    f'<div class="hero-bar">'
    f'<div style="background:{AMBER}"></div><div style="background:{TEAL}"></div>'
    f'<div style="background:{INDIGO}"></div><div style="background:{CORAL}"></div>'
    f'</div>',
    unsafe_allow_html=True,
)

# ---------------------------------------------------------------- sidebar --
if "personal_samples" not in st.session_state:
    st.session_state.personal_samples = {}  # emotion -> list of 64-dim CNN embeddings
if "personal_classifier" not in st.session_state:
    st.session_state.personal_classifier = None  # trained PersonalClassifier, or None

with st.sidebar:
    st.markdown("### Input")
    text = st.text_area(
        "How are you feeling? (English works best)",
        value="I feel happy and full of energy",
        height=100,
    )

    source = st.radio("Face photo source", ["Upload", "Webcam"], horizontal=True)
    if source == "Upload":
        uploaded = st.file_uploader("Face photo (optional)", type=["jpg", "jpeg", "png"])
    else:
        uploaded = st.camera_input("Take a photo")
    st.caption(f"Face-confidence threshold: {MIN_FACE_CONFIDENCE}  (see config.py)")

    with st.expander("🧠 Personal calibration (transfer learning)"):
        st.caption(
            "Layer 2: freeze the trained CNN and read off its internal "
            "64-dim representation (the layer right before its final "
            "prediction) for a few of YOUR OWN photos, then fit a small "
            "classifier on just those - a real, if tiny, training step, "
            "specific to your face. Take 3-5 photos per emotion for it to "
            "work well."
        )
        from vision.face_detector import FaceDetector
        from vision.vision_pipeline import VisionPipeline
        from vision.personal_classifier import EmbeddingExtractor, PersonalClassifier

        cal_emotion = st.selectbox("Emotion to add a sample for", EMOTIONS)
        cal_photo = st.camera_input(f"Photo of you looking '{cal_emotion}'", key=f"cal_{cal_emotion}")
        if cal_photo is not None:
            cal_pil = ImageOps.exif_transpose(Image.open(cal_photo)).convert("RGB")
            cal_bgr = np.array(cal_pil)[:, :, ::-1].copy()
            pipe = VisionPipeline()
            face, conf = pipe.detector.detect(cal_bgr)
            if face is None:
                st.error("No face detected in this photo - try again with better lighting/framing.")
            else:
                gray = pipe._preprocess(face)
                extractor = EmbeddingExtractor()
                vec = extractor.extract(gray)
                if vec is None:
                    st.error(f"Could not extract embedding: {extractor.error}")
                else:
                    st.session_state.personal_samples.setdefault(cal_emotion, []).append(vec)
                    st.success(
                        f"Saved sample #{len(st.session_state.personal_samples[cal_emotion])} "
                        f"for '{cal_emotion}'."
                    )

        counts = {e: len(v) for e, v in st.session_state.personal_samples.items()}
        if counts:
            st.write("**Samples collected:** " + ", ".join(f"{e} ({n})" for e, n in counts.items()))
        n_emotions = sum(1 for n in counts.values() if n > 0)
        c1, c2 = st.columns(2)
        with c1:
            if st.button("🎓 Train", disabled=n_emotions < 2, use_container_width=True):
                clf = PersonalClassifier()
                ok = clf.fit(st.session_state.personal_samples)
                if ok:
                    st.session_state.personal_classifier = clf
                    st.success("Personal classifier trained.")
                else:
                    st.error("Need at least 2 different calibrated emotions.")
        with c2:
            if st.button("🗑️ Clear", use_container_width=True):
                st.session_state.personal_samples = {}
                st.session_state.personal_classifier = None
                st.rerun()

        if n_emotions < 2:
            st.caption("Collect samples for at least 2 emotions, then Train.")
        if st.session_state.personal_classifier is not None:
            st.caption("✅ Trained and active - will be blended with the generic CNN below.")

    go = st.button("▶ Run", type="primary", use_container_width=True)

if not go:
    st.info("Write a sentence (and optionally upload/capture a photo), then click Run.")
    st.stop()

# Convert an uploaded image (RGB, PIL) to the BGR array our pipeline expects.
image_bgr = None
if uploaded is not None:
    pil_img = ImageOps.exif_transpose(Image.open(uploaded)).convert("RGB")
    image_bgr = np.array(pil_img)[:, :, ::-1].copy()

# Layer 2: predict with the personal classifier trained above, if any.
personal_estimate = None
if image_bgr is not None and st.session_state.personal_classifier is not None:
    pipe = VisionPipeline()
    face, conf = pipe.detector.detect(image_bgr)
    if face is not None:
        gray = pipe._preprocess(face)
        query_vec = EmbeddingExtractor().extract(gray)
        if query_vec is not None:
            probs = st.session_state.personal_classifier.predict_probs(query_vec)
            if probs is not None:
                pv, pa = probs_to_va(probs)
                personal_estimate = (pv, pa, max(probs))

result = run(text, image_bgr=image_bgr, personal_estimate=personal_estimate)

# ------------------------------------------------------------- 1. text ----
with st.container(border=True):
    channel_header("1", "Text Channel", AMBER)
    c1, c2 = st.columns([1, 1])

    with c1:
        hits = {e: [w for w in ws if w in text.lower()] for e, ws in KEYWORDS.items()}
        hits = {e: ws for e, ws in hits.items() if ws}
        st.write("**Matched keywords:**")
        st.json(hits if hits else "none — treated as neutral")

        tb = result["text_branch"]
        stat_chips([
            ("Valence", f"{tb['valence']:+.2f}", AMBER),
            ("Arousal", f"{tb['arousal']:+.2f}", AMBER),
            ("Confidence", f"{tb['confidence']:.2f}", AMBER),
        ])

    with c2:
        model = TextEmotionModel()
        probs = model._predict_probs(text) if text.strip() else [0.0] * len(EMOTIONS)
        st.write("**Probability vector over 7 classes:**")
        st.bar_chart({"probability": dict(zip(EMOTIONS, probs))}, color=AMBER)

# ----------------------------------------------------------- 2. vision ----
with st.container(border=True):
    channel_header("2", "Vision Channel", TEAL)
    if image_bgr is None:
        st.warning("No photo given → this channel gets zero confidence and is ignored.")
    else:
        vb = result["vision_branch"]
        vc1, vc2 = st.columns([1, 2])
        with vc1:
            st.image(uploaded, caption="Input photo", use_container_width=True)
        with vc2:
            # Surface which face-detection backend actually ran, so a silent
            # fallback (e.g. mediapipe -> haar) doesn't go unnoticed live.
            try:
                from vision.face_detector import FaceDetector
                probe = FaceDetector()
                backend = probe._backend
                mp_error = probe._mediapipe_error
            except Exception:
                backend, mp_error = None, None

            if backend == "mediapipe":
                st.caption("✅ Face detector backend: mediapipe")
            elif backend == "haar":
                st.caption("⚠️ Face detector backend: OpenCV Haar (fallback, less accurate than mediapipe)")
                if mp_error:
                    st.caption(f"Why mediapipe wasn't used: `{mp_error}`")
            else:
                st.caption("❌ No face-detection backend available (neither mediapipe nor opencv installed)")

            if vb["confidence"] == 0.0:
                st.warning(
                    "No face was found in this photo → confidence 0, this channel gets "
                    "no weight in fusion and the result comes entirely from text. Try a "
                    "well-lit, front-facing photo."
                )

            cnn_b = result["cnn_branch"]
            st.markdown("**Layer 1 — generic CNN** (trained on dataset faces)")
            stat_chips([
                ("Valence", f"{cnn_b['valence']:+.2f}", TEAL),
                ("Arousal", f"{cnn_b['arousal']:+.2f}", TEAL),
                ("Confidence", f"{cnn_b['confidence']:.2f}", TEAL),
            ])

            pb = result["personal_branch"]
            if pb is not None:
                st.markdown("**Layer 2 — your personal classifier** (transfer learning on frozen CNN embeddings)")
                stat_chips([
                    ("Valence", f"{pb['valence']:+.2f}", INDIGO),
                    ("Arousal", f"{pb['arousal']:+.2f}", INDIGO),
                    ("Confidence", f"{pb['confidence']:.2f}", INDIGO),
                ])
            else:
                st.caption("Layer 2 (personal classifier) not used — train it in the sidebar first, "
                           "or no face was found in this photo.")

            st.markdown("**Blended vision estimate** (Layer 1 + Layer 2, confidence-weighted)")
            stat_chips([
                ("Valence", f"{vb['valence']:+.2f}", TEAL),
                ("Arousal", f"{vb['arousal']:+.2f}", TEAL),
                ("Confidence", f"{vb['confidence']:.2f}", TEAL),
            ])

# ------------------------------------------------------------ 3. fusion ---
with st.container(border=True):
    channel_header("3", "Fusion in V–A Space", INDIGO)
    fused = result["fused"]
    w = fused["weights"]
    st.latex(
        r"v = \dfrac{w_{face} \cdot v_{face} + w_{text} \cdot v_{text}}"
        r"{w_{face} + w_{text}}"
    )
    stat_chips([
        ("Face weight", f"{w['vision']:.2f}", INDIGO),
        ("Text weight", f"{w['text']:.2f}", INDIGO),
        ("Final label", fused["label"], INDIGO),
    ])
    st.success(f"Final emotion point: (v = {fused['valence']:+.2f}, a = {fused['arousal']:+.2f})")

# --------------------------------------------------------- 4. recommend ---
with st.container(border=True):
    channel_header("4", "Music Recommendation", CORAL)
    lib = load_library()
    recs = result["recommendations"]
    rec_titles = {r["title"] for r in recs}

    plt.rcParams.update({
        "text.color": TEXT, "axes.labelcolor": TEXT,
        "xtick.color": MUTED, "ytick.color": MUTED,
    })
    fig, ax = plt.subplots(figsize=(5.5, 5.5))
    fig.patch.set_facecolor(BG)
    ax.set_facecolor(PANEL)

    # Quadrant tints - the same 4 channel colors, now shown where they actually
    # live in the V-A plane.
    ax.fill_between([0, 1.1], 0, 1.1, color=AMBER, alpha=0.10, zorder=0)
    ax.fill_between([-1.1, 0], 0, 1.1, color=TEAL, alpha=0.10, zorder=0)
    ax.fill_between([-1.1, 0], -1.1, 0, color=INDIGO, alpha=0.12, zorder=0)
    ax.fill_between([0, 1.1], -1.1, 0, color=CORAL, alpha=0.10, zorder=0)

    for s in lib:
        picked = s["title"] in rec_titles
        ax.scatter(
            s["valence"], s["arousal"],
            s=170 if picked else 60,
            c=AMBER if picked else MUTED,
            edgecolors=BG if picked else "none",
            linewidths=1.2,
            zorder=3 if picked else 2,
        )
        ax.annotate(s["title"], (s["valence"], s["arousal"]),
                    fontsize=8, color=TEXT if picked else MUTED,
                    xytext=(4, 4), textcoords="offset points")

    for emo, (ev, ea) in EMOTION_TO_VA.items():
        ax.annotate(emo, (ev, ea), fontsize=7, color=MUTED, ha="center", va="center")

    ax.scatter(fused["valence"], fused["arousal"], s=280, marker="*",
               c=CORAL, edgecolors=TEXT, linewidths=1, zorder=4, label="you")
    ax.axhline(0, color=PANEL_LINE, lw=1)
    ax.axvline(0, color=PANEL_LINE, lw=1)
    ax.set_xlim(-1.1, 1.1)
    ax.set_ylim(-1.1, 1.1)
    ax.set_xlabel("Valence")
    ax.set_ylabel("Arousal")
    legend = ax.legend(loc="upper left", facecolor=PANEL, edgecolor=PANEL_LINE)
    for t in legend.get_texts():
        t.set_color(TEXT)
    ax.set_title("V–A space: music library + your emotion", color=TEXT)
    for spine in ax.spines.values():
        spine.set_color(PANEL_LINE)

    gcol1, gcol2 = st.columns([1, 1])
    with gcol1:
        st.pyplot(fig, use_container_width=True)
    with gcol2:
        st.write("**Closest songs:**")
        table = pd.DataFrame([
            {
                "#": i,
                "Title": r["title"],
                "Artist": r["artist"],
                "Valence": round(r["valence"], 2),
                "Arousal": round(r["arousal"], 2),
                "Distance": r["distance"],
            }
            for i, r in enumerate(recs, 1)
        ]).set_index("#")
        st.dataframe(table, use_container_width=True)

    st.caption(
        "Note: a small random jitter is added to the query point, "
        "so distances shift slightly between runs."
    )

    st.markdown('<div class="channel-title" style="margin-top:0.8rem;">🎵 Now Playing</div>',
                unsafe_allow_html=True)
    top = recs[0]
    if top.get("youtube_url"):
        st.write(f"**{top['title']}** — {top['artist']}")
        st.video(top["youtube_url"])
        st.caption(
            f"Music by {top['artist']} (incompetech.com) — free under the "
            "Creative Commons Attribution 4.0 license"
        )
    else:
        st.info("No playback link available for this track.")
