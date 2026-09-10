# -*- coding: utf-8 -*-
"""Run this with the exact same command you use to launch Streamlit's Python,
e.g.:
    python diagnose_env.py

It tells you:
  - which python interpreter is actually running
  - where pip installs packages for that interpreter
  - whether opencv / mediapipe / streamlit are importable from HERE
"""
import subprocess
import sys

print("=" * 60)
print("1) Which Python is this?")
print("=" * 60)
print("sys.executable:", sys.executable)
print("sys.version   :", sys.version)

print()
print("=" * 60)
print("2) Where would `pip install` put packages for THIS python?")
print("=" * 60)
subprocess.run([sys.executable, "-m", "pip", "--version"])

print()
print("=" * 60)
print("3) Can this python import opencv / mediapipe / streamlit?")
print("=" * 60)
for pkg in ["cv2", "mediapipe", "streamlit", "numpy", "PIL"]:
    try:
        mod = __import__(pkg)
        path = getattr(mod, "__file__", "built-in")
        print(f"  OK   {pkg:12s} -> {path}")
    except Exception as e:
        print(f"  FAIL {pkg:12s} -> {e!r}")

print()
print("=" * 60)
print("4) What python does the `streamlit` COMMAND actually use?")
print("=" * 60)
locator = "where" if sys.platform.startswith("win") else "which"
try:
    out = subprocess.run([locator, "streamlit"], capture_output=True, text=True)
    print(f"{locator} streamlit ->", out.stdout.strip() or out.stderr.strip())
except Exception as e:
    print(f"could not run `{locator} streamlit`:", e)

print()
print("If step 3 shows FAIL for cv2/mediapipe here, but you already ran")
print("`pip install opencv-python-headless mediapipe`, it means that pip")
print("command installed into a DIFFERENT python than the one running this")
print("script (or running streamlit). Fix: run")
print(f'    "{sys.executable}" -m pip install opencv-python-headless mediapipe')
print("using the exact sys.executable path printed in step 1 above - that")
print("guarantees the install lands in the same interpreter Streamlit uses.")
