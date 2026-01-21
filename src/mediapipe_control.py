import time
from selenium import webdriver
from selenium.webdriver.common.keys import Keys

import cv2
import mediapipe as mp

# ----- 1) Launch browser and open a publicly hosted Dino game -----
print("Launching browser...")

# Create a Chrome WebDriver instance
driver = webdriver.Chrome()

# Go to an online Dino game clone
dino_url = "https://dinogame.im"
driver.get(dino_url)

# Allow time for page to load
time.sleep(3)

# Focus page and start the game by sending SPACE
body_elem = driver.find_element("tag name", "body")
body_elem.send_keys(Keys.SPACE)

print("Game started! Gesture control is active.")

# ----- 2) Set up MediaPipe Hands for gesture detection -----
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7,
)

mp_draw = mp.solutions.drawing_utils

cap = cv2.VideoCapture(0)

last_jump = 0
jump_cooldown = 0.7
hand_up_threshold = 0.5
prev_index_y = None

print("Starting gesture detection (press 'q' to quit)...")

# ----- 3) Real‑time gesture loop -----
while True:
    ret, frame = cap.read()
    if not ret:
        continue

    frame = cv2.flip(frame, 1)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    result = hands.process(rgb_frame)

    if result.multi_hand_landmarks:
        for handLms in result.multi_hand_landmarks:
            mp_draw.draw_landmarks(frame, handLms, mp_hands.HAND_CONNECTIONS)

            index_y = handLms.landmark[8].y

            if prev_index_y is not None:
                if (
                    index_y < hand_up_threshold
                    and index_y < prev_index_y
                    and (time.time() - last_jump) > jump_cooldown
                ):
                    # Send SPACE to the game
                    body_elem.send_keys(Keys.SPACE)
                    last_jump = time.time()
                    cv2.putText(
                        frame,
                        "JUMP!",
                        (50, 50),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        1,
                        (0, 0, 255),
                        2,
                    )

            prev_index_y = index_y

    cv2.imshow("Hand Gesture Dino", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()

# Quit browser
driver.quit()
