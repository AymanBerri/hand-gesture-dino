# Hand Gesture Dino

Control the Dino game using hand gestures! This project demonstrates the difference between a **naive OpenCV baseline** and **MediaPipe hand tracking**, showing how computer vision models improve accuracy and reliability.

---

## 🚀 Features

- **OpenCV Baseline**: Contour-based finger detection.
  - Simple and fast, but sensitive to background, lighting, and false positives.
- **MediaPipe Version**: ML-based hand tracking.
  - Robust, ignores face and background noise, stable jump detection.
- **Automatic Game Launch**: Uses Selenium to open an online Dino clone.
- **Gesture Control**:
  - 1 finger up → jump
  - Can be extended for ducking or other gestures.

---

## 📸 Demo

Place your hand in front of the camera:

![Demo GIF](./demo.gif)
Working on it, sorry :/

> The above GIF shows the OpenCV baseline (left) vs MediaPipe (right). Notice how MediaPipe avoids false jumps and tracks the hand reliably.

---

## 🛠 Installation

1. Clone the repository:
`git clone https://github.com/YOUR_USERNAME/hand-gesture-dino.git`
`cd hand-gesture-dino`

2. Create a virtual environment:
`python -m venv venv`
`source venv/bin/activate`  # Windows: `venv\Scripts\activate`

3. Install dependencies:
`pip install -r requirements.txt`

---

## 🎮 Usage

Run either version:

- **OpenCV baseline**:
`python src/opencv_baseline.py`

- **MediaPipe version**:
`python src/mediapipe_version.py`

Steps:

1. The Dino game will open automatically in your browser.
2. Place your hand in front of the camera.
3. Raise one finger to jump.
4. Press `q` to quit.

---

## 📝 Observations

| Feature                 | OpenCV Baseline | MediaPipe Version |
|-------------------------|----------------|-----------------|
| False positives         | High           | Minimal         |
| Background sensitivity  | High           | Low             |
| Jump reliability        | Medium         | High            |
| Ease of use             | Medium         | High            |

The OpenCV baseline works but struggles with lighting and unintended objects in view. MediaPipe provides accurate hand detection and much more stable control.

---

## 💡 Extensions

- Detect 2 fingers for ducking → DOWN arrow in game
- Dynamic thresholds to adapt to lighting changes
- Real-time game score display using Selenium
- Side-by-side comparison video for demonstration

---

## 🔗 References

- [MediaPipe Hands Documentation](https://developers.google.com/mediapipe/solutions/vision/hand_tracking)
- [Chrome Dino Game Online Clone](https://dinogame.im)


