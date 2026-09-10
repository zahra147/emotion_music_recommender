# -*- coding: utf-8 -*-
"""Unit tests for the emotion -> music pipeline.

Run from the `src` directory:
    python -m unittest discover -s tests -t .
"""
import unittest

from config import EMOTIONS, MIN_FACE_CONFIDENCE
from emotion_space import EMOTION_TO_VA, probs_to_va, va_to_nearest_label
from fusion.fusion import fuse
from recommender.music_library import load_library
from recommender.recommend import recommend
from text.text_emotion import TextEmotionModel


def one_hot(emotion):
    probs = [0.0] * len(EMOTIONS)
    probs[EMOTIONS.index(emotion)] = 1.0
    return probs


class TestEmotionSpace(unittest.TestCase):
    def test_happy_is_positive_and_energetic(self):
        v, a = probs_to_va(one_hot("happy"))
        self.assertGreater(v, 0.5)
        self.assertGreater(a, 0.3)

    def test_sad_is_negative_and_low_energy(self):
        v, a = probs_to_va(one_hot("sad"))
        self.assertLess(v, -0.3)
        self.assertLess(a, 0.0)

    def test_angry_is_negative_but_energetic(self):
        """Angry and sad share low valence; arousal is what separates them."""
        v, a = probs_to_va(one_hot("angry"))
        self.assertLess(v, 0.0)
        self.assertGreater(a, 0.3)

    def test_neutral_is_origin(self):
        self.assertEqual(probs_to_va(one_hot("neutral")), (0.0, 0.0))

    def test_wrong_length_raises(self):
        with self.assertRaises(ValueError):
            probs_to_va([1.0, 0.0])

    def test_label_round_trip(self):
        for emo, (v, a) in EMOTION_TO_VA.items():
            self.assertEqual(va_to_nearest_label(v, a), emo)


class TestTextBranch(unittest.TestCase):
    def setUp(self):
        self.model = TextEmotionModel()

    def test_empty_text_has_zero_confidence(self):
        self.assertEqual(self.model.infer(""), (0.0, 0.0, 0.0))
        self.assertEqual(self.model.infer("   "), (0.0, 0.0, 0.0))

    def test_english_happy(self):
        v, a, c = self.model.infer("I feel happy and great today")
        self.assertGreater(v, 0.0)
        self.assertGreater(c, 0.0)

    def test_persian_sad(self):
        v, a, c = self.model.infer("خیلی ناراحتم")
        self.assertLess(v, 0.0)

    def test_unknown_text_falls_back_to_neutral(self):
        v, a, c = self.model.infer("zzzz qqqq")
        self.assertEqual((v, a), (0.0, 0.0))


class TestFusion(unittest.TestCase):
    def test_text_only_uses_text_point(self):
        r = fuse((0.0, 0.0, 0.0), (0.8, 0.6, 0.9))
        self.assertAlmostEqual(r["valence"], 0.8)
        self.assertAlmostEqual(r["arousal"], 0.6)
        self.assertEqual(r["weights"]["vision"], 0.0)

    def test_vision_only_uses_vision_point(self):
        r = fuse((-0.7, -0.5, 0.9), (0.0, 0.0, 0.0))
        self.assertAlmostEqual(r["valence"], -0.7)
        self.assertEqual(r["label"], "sad")

    def test_low_confidence_vision_is_discarded(self):
        low = MIN_FACE_CONFIDENCE / 2
        r = fuse((0.8, 0.6, low), (-0.7, -0.5, 0.9))
        self.assertAlmostEqual(r["valence"], -0.7)
        self.assertEqual(r["weights"]["vision"], 0.0)

    def test_equal_confidence_gives_midpoint(self):
        r = fuse((1.0, 1.0, 0.8), (0.0, 0.0, 0.8))
        self.assertAlmostEqual(r["valence"], 0.5)
        self.assertAlmostEqual(r["arousal"], 0.5)

    def test_no_signal_returns_neutral_origin(self):
        r = fuse((0.0, 0.0, 0.0), (0.0, 0.0, 0.0))
        self.assertEqual((r["valence"], r["arousal"]), (0.0, 0.0))
        self.assertEqual(r["label"], "neutral")


class TestRecommender(unittest.TestCase):
    def test_library_is_usable(self):
        lib = load_library()
        self.assertGreater(len(lib), 0)
        for key in ("title", "artist", "valence", "arousal"):
            self.assertIn(key, lib[0])

    def test_happy_query_returns_happy_song(self):
        top = recommend(0.8, 0.6, top_k=1, jitter=0.0)
        self.assertEqual(top[0]["title"], "Flying Kerfuffle")

    def test_sad_query_returns_sad_song(self):
        top = recommend(-0.7, -0.5, top_k=1, jitter=0.0)
        self.assertEqual(top[0]["title"], "Night Vigil")

    def test_results_are_sorted_and_limited(self):
        top = recommend(0.0, 0.0, top_k=3, jitter=0.0)
        self.assertEqual(len(top), 3)
        dists = [s["distance"] for s in top]
        self.assertEqual(dists, sorted(dists))


class TestEndToEnd(unittest.TestCase):
    def test_text_only_run(self):
        from demo import run
        out = run("I feel happy and full of energy")
        self.assertIn("fused", out)
        self.assertIn("recommendations", out)
        self.assertEqual(out["fused"]["label"], "happy")
        self.assertGreater(len(out["recommendations"]), 0)


if __name__ == "__main__":
    unittest.main(verbosity=2)
