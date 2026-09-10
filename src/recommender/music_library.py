# -*- coding: utf-8 -*-
"""Music library: each track is a point in the (valence, arousal) space.

Loads tracks from data/music_library.csv with columns:
    title, artist, valence, arousal, youtube_url
If the file is missing, a built-in sample library of real, freely-licensed
(Creative Commons BY) tracks by Kevin MacLeod is used, each with a
youtube_url so the Streamlit app can actually play them - see _SAMPLE below.
"""
from __future__ import annotations
import csv
from pathlib import Path
from typing import List, Dict

from config import DATA_DIR

# Real, freely-licensed tracks (Creative Commons BY) by Kevin MacLeod
# (incompetech.com), hand-placed in Valence-Arousal space based on his own
# mood tags ("feel") from the official catalog
# (https://incompetech.com/music/royalty-free/pieces.json). youtube_url
# points at his official channel uploads, so playback needs no downloaded
# audio file - just embed the video/audio player.
#
# Attribution (Creative Commons BY 4.0): "Kevin MacLeod (incompetech.com)"
_SAMPLE = [
    {"title": "Flying Kerfuffle", "artist": "Kevin MacLeod", "valence": 0.8, "arousal": 0.7,
     "youtube_url": "https://www.youtube.com/watch?v=P4P2H_qPIew"},
    {"title": "Paradise Found", "artist": "Kevin MacLeod", "valence": 0.8, "arousal": 0.5,
     "youtube_url": "https://www.youtube.com/watch?v=trcowwPXeTk"},
    {"title": "Neon Laser Horizon", "artist": "Kevin MacLeod", "valence": 0.5, "arousal": 0.7,
     "youtube_url": "https://www.youtube.com/watch?v=EYcAo2DGoKU"},
    {"title": "Moonlight Beach", "artist": "Kevin MacLeod", "valence": 0.5, "arousal": -0.3,
     "youtube_url": "https://www.youtube.com/watch?v=ywnnzesBYcY"},
    {"title": "Guzheng City", "artist": "Kevin MacLeod", "valence": 0.2, "arousal": -0.1,
     "youtube_url": "https://www.youtube.com/watch?v=u7d6hX7jCr4"},
    {"title": "Deep Relaxation", "artist": "Kevin MacLeod", "valence": 0.2, "arousal": -0.7,
     "youtube_url": "https://www.youtube.com/watch?v=fgauFOtjRTE"},
    {"title": "Past Sadness", "artist": "Kevin MacLeod", "valence": -0.6, "arousal": -0.4,
     "youtube_url": "https://www.youtube.com/watch?v=TCpAjWUn0qg"},
    {"title": "Night Vigil", "artist": "Kevin MacLeod", "valence": -0.7, "arousal": -0.6,
     "youtube_url": "https://www.youtube.com/watch?v=Jhw0F4gvYfM"},
    {"title": "Circus of Freaks", "artist": "Kevin MacLeod", "valence": -0.5, "arousal": 0.6,
     "youtube_url": "https://www.youtube.com/watch?v=VbXF0maHWrI"},
    {"title": "Symmetry", "artist": "Kevin MacLeod", "valence": -0.6, "arousal": 0.7,
     "youtube_url": "https://www.youtube.com/watch?v=TL5Wb3rQfn4"},
]


def load_library(path: Path | None = None) -> List[Dict]:
    path = path or (DATA_DIR / "music_library.csv")
    if not path.exists():
        return [dict(t) for t in _SAMPLE]
    songs: List[Dict] = []
    with open(path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            songs.append({
                "title": row["title"],
                "artist": row.get("artist", ""),
                "valence": float(row["valence"]),
                "arousal": float(row["arousal"]),
                "youtube_url": row.get("youtube_url", ""),
            })
    return songs
