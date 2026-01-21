import time
import os
import cv2
import pyautogui
import mediapipe as mp
import webbrowser

# -------------------------------
# 1) Open OFFLINE Dino Game
# -------------------------------
print("Opening offline Dino game...")

dino_path = os.path.abspath("dino_game/dino.html")
webbrowser.open(f"file:///{dino_path}")

time.sleep(3)  # give browser time to open
pyautogui.press("space")  # start the game

print("Game started! Gesture control active.")

# -------------------------------
# 2) MediaPipe Hands setup
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
# 3) Webcam setup
# -------------------------------
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

cv2.namedWindow("Hand Gesture Dino Control", cv2.WINDOW_NORMAL)
cv2.setWindowProperty("Hand Gesture Dino Control", cv2.WND_PROP_TOPMOST, 1)
cv2.resizeWindow("Hand Gesture Dino Control", 320, 240)
cv2.moveWindow("Hand Gesture Dino Control", 1000, 50)  # top-right corner

# -------------------------------
# 4) Gesture detection variables
# -------------------------------
hand_up_threshold = 0.45  # lower = higher hand needed

print("Press 'q' to quit.")

# -------------------------------
# 5) Real-time loop (no cooldown)
# -------------------------------
while True:
    ret, frame = cap.read()
    if not ret:
        continue

    frame = cv2.flip(frame, 1)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = hands.process(rgb)

    jump_triggered = False

    if result.multi_hand_landmarks:
        handLms = result.multi_hand_landmarks[0]

        # Draw landmarks
        mp_draw.draw_landmarks(frame, handLms, mp_hands.HAND_CONNECTIONS)

        # Index finger tip
        index_y = handLms.landmark[8].y

        if index_y < hand_up_threshold:
            pyautogui.keyDown("space")  # hold space while finger is up
            jump_triggered = True
        else:
            pyautogui.keyUp("space")    # release space if finger goes down

    else:
        pyautogui.keyUp("space")        # no hand detected

    # Visual feedback
    if jump_triggered:
        cv2.putText(frame, "JUMP!", (40, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 3)
        cv2.rectangle(frame, (50, 50), (220, 150), (0, 255, 0), 3)
    else:
        cv2.putText(frame, "Gesture Ready", (40, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)

    cv2.imshow("Hand Gesture Dino Control", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

# -------------------------------
# 6) Cleanup
# -------------------------------
pyautogui.keyUp("space")  # make sure space is released
cap.release()
cv2.destroyAllWindows()
hands.close()
