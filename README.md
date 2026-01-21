# Comparative Evaluation of Hand Gesture Recognition Methods for Real-Time Game Control

## Overview
This project investigates real-time hand gesture recognition for controlling the **Dino offline game**. Three approaches are implemented and compared:

1. **OpenCV Baseline** – Contour-based finger counting.
2. **MediaPipe Hands** – Robust hand landmark detection and thresholding.
3. **MediaPipe + SVM** – Supervised classifier trained on hand landmarks to detect gestures.

The goal is to demonstrate the performance differences in gesture recognition reliability and responsiveness.

---

## Features
- Real-time hand gesture control of the offline Dino game.
- Camera overlay showing hand detection and visual feedback.
- Jump triggered by specific gestures (index finger up for jump).
- Comparison of three approaches:
  - OpenCV heuristic: sensitive to lighting, false positives.
  - MediaPipe: accurate and robust.
  - MediaPipe + SVM: frame-by-frame classifier (needs sufficient training data).

---

## Setup

### Requirements
- Python 3.10+
- OpenCV: `pip install opencv-python`
- Mediapipe: `pip install mediapipe`
- PyAutoGUI: `pip install pyautogui`
- Joblib (for SVM model): `pip install joblib`
- Selenium (optional for online Dino): `pip install selenium`
- Chrome browser (or Edge for webbrowser mode)

### Folder Structure
```
hand-gesture-dino/
│
├─ src/
│   ├─ collect_hand_data.py       # Dataset collection
│   ├─ mediapipe_control.py       # MediaPipe live demo
│   ├─ opencv_baseline.py         # OpenCV baseline demo
│   ├─ train_hand_svm.py          # SVM training
│   └─ realtime_hand_svm.py       # MediaPipe + SVM demo
│
├─ dataset/                       # Hand landmark dataset (ignored in Git)
│   ├─ X.npy
│   ├─ y.npy
│
├─ models/                        # Trained SVM model
│   └─ hand_svm_model.joblib
│
├─ dino_game/                     # Offline Dino clone
│   └─ dino.html
│
├─ results/                       # Optional experimental outputs, metrics
│
├─ venv/
│
├─ .gitignore                     
├─ README.md
└─ requirements.txt
```

---

## Usage

### 1. Collect Hand Data
Run:
```
python src/collect_hand_data.py
```
- Press `i` for idle gesture, `j` for jump gesture.
- At least **50–100 samples per class** recommended.
- Saved in `dataset/` folder (ignored in Git).

### 2. Train SVM
Run:
```
python src/train_hand_svm.py
```
- Loads collected dataset from `dataset/`.
- Trains a simple linear SVM.
- Saves model in `models/hand_svm_model.joblib`.
- Saves training metrics (`dataset/metrics.json`) for accuracy, precision, etc.

### 3. Play Dino Game
#### MediaPipe (Robust)
```
python src/mediapipe_control.py
```

#### OpenCV Baseline
```
python src/opencv_baseline.py
```

#### MediaPipe + SVM
```
python src/realtime_hand_svm.py
```

**Notes:**
- Offline Dino opens in browser automatically.
- Camera overlay shows live hand detection.
- Jump is triggered by index finger up (MediaPipe) or predicted gesture (SVM).
- Press `q` to quit.

---

## Comparison of Methods

| Method               | Accuracy | Robustness | Notes |
|----------------------|---------|------------|-------|
| OpenCV Baseline       | Low     | Poor       | Sensitive to lighting, false positives, often detects face as fingers. |
| MediaPipe Hands       | High    | Strong     | Accurate hand tracking, minimal false positives, responsive. |
| MediaPipe + SVM       | Medium  | Medium     | Needs sufficient dataset, frame-by-frame predictions; can miss gestures if dataset is small. |

---

## Future Work
- Increase dataset with multiple users.
- Add more gestures (duck, run, pause).
- Temporal smoothing or sequence models (RNN/LSTM) for more reliable jump detection.
- Automated game metrics collection (score, jumps, failures) via browser integration.

---

## Screenshots / GIF
```
![Demo](assets/demo.gif)
```
Will do

---

## References
- [MediaPipe Hands Documentation](https://developers.google.com/mediapipe/solutions/vision/hand_landmarker)
- [OpenCV Contour Analysis](https://docs.opencv.org/master/d9/d8b/tutorial_py_contours_hierarchy.html)
- Chrome Dino Game: Offline clone


