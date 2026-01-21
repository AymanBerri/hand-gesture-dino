import cv2
import mediapipe as mp
import numpy as np
import pyautogui
import joblib
import time
import os
import subprocess

# -------------------------------
# 1) Launch offline Dino game
# -------------------------------
dino_path = os.path.abspath("dino_game/dino.html")  # local HTML file
chrome_path = r"C:\Program Files\Google\Chrome\Application\chrome.exe"

print("Launching offline Dino game...")
subprocess.Popen([chrome_path, "--start-fullscreen", dino_path])
time.sleep(3)  # wait for browser to open
pyautogui.click(500, 500)  # focus game window
pyautogui.press("space")    # start game
print("Game started. MediaPipe + SVM control active!")

# -------------------------------
# 2) Load trained SVM model
# -------------------------------
svm = joblib.load("models/hand_svm_model.joblib")  # saved SVM model
print("SVM model loaded!")

# -------------------------------
# 3) Setup MediaPipe Hands
# -------------------------------
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7
)
mp_draw = mp.solutions.drawing_utils

# -------------------------------
# 4) Setup webcam
# -------------------------------
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

cv2.namedWindow("MediaPipe + SVM Dino Control", cv2.WINDOW_NORMAL)
cv2.setWindowProperty("MediaPipe + SVM Dino Control", cv2.WND_PROP_TOPMOST, 1)
cv2.resizeWindow("MediaPipe + SVM Dino Control", 320, 240)
cv2.moveWindow("MediaPipe + SVM Dino Control", 1000, 50)  # top-right corner

# -------------------------------
# 5) Real-time gesture loop
# -------------------------------
gesture_frames = 0
frames_needed = 3  # stable detection over frames

while True:
    ret, frame = cap.read()
    if not ret:
        continue

    frame = cv2.flip(frame, 1)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = hands.process(rgb_frame)

    jump = False
    if result.multi_hand_landmarks:
        hand_landmarks = result.multi_hand_landmarks[0]
        mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

        # Flatten landmarks (21 x 2 -> 42-dim)
        features = np.array([lm.x for lm in hand_landmarks.landmark] +
                            [lm.y for lm in hand_landmarks.landmark]).reshape(1, -1)

        # Predict jump (1) / idle (0)
        pred = svm.predict(features)
        if pred[0] == 1:
            gesture_frames += 1
        else:
            gesture_frames = 0

        if gesture_frames >= frames_needed:
            jump = True
            gesture_frames = frames_needed  # hold value

    # -------------------------------
    # 6) Control Dino game
    # -------------------------------
    if jump:
        pyautogui.keyDown("space")
        cv2.putText(frame, "JUMP!", (40, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 3)
        cv2.rectangle(frame, (30, 30), (220, 100), (0, 0, 255), 3)
    else:
        pyautogui.keyUp("space")
        cv2.putText(frame, "Idle", (40, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    cv2.imshow("MediaPipe + SVM Dino Control", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

# -------------------------------
# 7) Cleanup
# -------------------------------
cap.release()
cv2.destroyAllWindows()
hands.close()
