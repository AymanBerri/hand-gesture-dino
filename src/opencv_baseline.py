import cv2
import numpy as np
import time
import os
import subprocess
import pyautogui
from selenium import webdriver
from selenium.webdriver.common.keys import Keys

# -------------------------------
# 1) Launch offline Dino game in Chrome
# -------------------------------
print("Launching offline Dino game...")

dino_path = os.path.abspath("dino_game/dino.html")  # your local HTML
chrome_path = r"C:\Program Files\Google\Chrome\Application\chrome.exe"

subprocess.Popen([chrome_path, "--start-fullscreen", dino_path])
time.sleep(3)  # wait for Chrome
# focus game window
pyautogui.click(500, 500)
time.sleep(0.3)
pyautogui.press("space")  # start game

print("Game started. Gesture control active.")

# -------------------------------
# 2) OpenCV baseline finger detection
# -------------------------------
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

cv2.namedWindow("Camera", cv2.WINDOW_NORMAL)
cv2.setWindowProperty("Camera", cv2.WND_PROP_TOPMOST, 1)
cv2.resizeWindow("Camera", 320, 240)
cv2.moveWindow("Camera", 1000, 50)  # top-right

last_jump = 0
jump_cooldown = 0.7
gesture_frames = 0
frames_needed = 3
min_area = 8000  # filter out face / big blobs

print("OpenCV Finger Count Control - Press q to quit...")

while True:
    ret, frame = cap.read()
    if not ret:
        continue

    frame = cv2.flip(frame, 1)

    # Crop ROI to avoid face
    roi = frame[100:400, 300:600]  # adjust for your hand

    # Preprocess
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (7, 7), 0)
    _, thresh = cv2.threshold(blur, 127, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    finger_count = 0

    if contours:
        largest = max(contours, key=cv2.contourArea)
        area = cv2.contourArea(largest)

        # Only hand-sized contour
        if min_area < area < 20000:
            hull = cv2.convexHull(largest, returnPoints=False)
            if hull is not None and len(hull) > 3:
                defects = cv2.convexityDefects(largest, hull)
                if defects is not None:
                    for i in range(defects.shape[0]):
                        s, e, f, d = defects[i, 0]
                        start = tuple(largest[s][0])
                        end = tuple(largest[e][0])
                        far = tuple(largest[f][0])

                        a = np.linalg.norm(np.array(start) - np.array(end))
                        b = np.linalg.norm(np.array(start) - np.array(far))
                        c = np.linalg.norm(np.array(end) - np.array(far))
                        angle = np.arccos((b**2 + c**2 - a**2) / (2 * b * c))
                        if angle <= np.pi / 2:
                            finger_count += 1

                    finger_count += 1  # total fingers

                    # Draw contour + defects
                    cv2.drawContours(roi, [largest], -1, (255, 0, 0), 2)
                    cv2.putText(frame, f"Fingers: {finger_count}", (20, 50),
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

                    # Trigger jump if exactly 1 finger
                    if finger_count == 1:
                        gesture_frames += 1
                    else:
                        gesture_frames = 0

                    if gesture_frames >= frames_needed and (time.time() - last_jump) > jump_cooldown:
                        pyautogui.press("space")
                        last_jump = time.time()
                        gesture_frames = 0
                        cv2.putText(frame, "JUMP!", (20, 100),
                                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    cv2.imshow("Camera", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
