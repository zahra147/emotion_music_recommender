# emotion_music_recommender
Emotion-based music recommendation system using Python and computer vision
README.md (English Version — Ready for GitHub)
🎵 Emotion-Based Music Recommender
Multimodal (Vision + Text) • Deep Learning • Valence–Arousal Space
This project is a multimodal music recommendation system that predicts the user’s emotional state from facial expressions and text, maps both signals into the Valence–Arousal affective space, fuses them based on confidence, and recommends the most suitable music tracks.

The entire pipeline is modular, testable, and fully runnable end‑to‑end.

⭐ Key Features
Facial emotion recognition using a CNN

Text emotion recognition using a BiLSTM

Mapping emotions to Valence–Arousal coordinates

Weighted fusion of Vision + Text channels

Music recommendation using KNN in V‑A space

Unit tests, demo scripts, and clean modular architecture

🧠 System Architecture
Code
Image / Text
     ↓
Vision CNN / Text BiLSTM
     ↓
Valence–Arousal Mapping
     ↓
Weighted Fusion (confidence-based)
     ↓
Music Recommender (KNN)
     ↓
Top‑k Songs
📁 Project Structure
Code
src/
  config.py               Global settings (thresholds, top_k)
  emotion_space.py        Mapping classes ⇄ Valence–Arousal
  vision/
    face_detector.py      Face detection (MediaPipe / Haar)
    fer_model.py          CNN model for facial emotion recognition
    vision_pipeline.py    Image → (v, a, confidence)
  text/
    text_emotion.py       Text → (v, a, confidence)
  fusion/
    fusion.py             Weighted fusion of both channels
  recommender/
    music_library.py      Load music library in V‑A space
    recommend.py          KNN + diversity for final recommendations
  demo.py                 End‑to‑end demo
  walkthrough.py          Step‑by‑step trace of the full pipeline
  tests/                  Unit tests
🔬 Deep Learning Components
🎯 Vision Channel (CNN)
The CNN architecture includes:

3× Convolution layers

Batch Normalization

Global Average Pooling

Dense output layer for 7 emotions

🔥 Model Improvement
Initial accuracy: 52%  
Problem: extremely low samples for fear and disgust in the original dataset.

Failed attempts:

Strong augmentation

Focal loss

Larger CNN

Real solution: adding CK+ dataset (balanced, posed expressions)

Results:

Accuracy: 52% → 67%

F1 (fear): ≈0 → 0.87

F1 (disgust): ≈0 → 0.89

Conclusion:  
Better data > more complex models.

📝 Text Channel (BiLSTM)
Initial accuracy: 53% on GoEmotions.

Problem: noisy, imbalanced Reddit-style text.

Solution: add ISEAR dataset (clean, balanced academic survey)

Results:

Accuracy: 53% → 57%

F1 improved for most classes

Slight drop in disgust due to domain mismatch

🔗 Fusion Layer
Weighted fusion based on confidence:

Code
v_final = w1 * v_vision + w2 * v_text
a_final = w1 * a_vision + w2 * a_text
🎧 Music Recommendation
Songs are embedded in Valence–Arousal space

KNN retrieves nearest tracks

Diversity added to avoid repetitive outputs

▶ Demo
bash
cd src
python demo.py --text "I feel happy and full of energy"
python demo.py --image face.jpg --text "I am tired and sad"
🧪 Unit Tests
bash
cd src
python -m unittest discover -s tests -t .
📌 Scientific Takeaways
Data quality and balance matter more than model complexity

CK+ and ISEAR significantly improved both channels

Multimodal fusion outperforms single‑channel emotion recognition

The project demonstrates a real deep‑learning workflow:
problem → failed attempts → correct hypothesis → data fix → measurable improvement

📚 TODO
Replace text model with ParsBERT

Expand music library

Feature‑level fusion

CNN‑LSTM for video sequences

👤 Developer
Zahra — ML & IoT Developer
Deep Learning • Computer Vision • NLP

