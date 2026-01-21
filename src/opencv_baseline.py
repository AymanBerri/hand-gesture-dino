import cv2
import numpy as np
import time
from selenium import webdriver
from selenium.webdriver.common.keys import Keys

# -------------------------------
# 1) Launch browser and open Dino game
# -------------------------------
print("Launching browser...")

driver = webdriver.Chrome()  # Selenium Manager auto-downloads driver
dino_url = "https://dinogame.im"  # public Dino game clone
driver.get(dino_url)

time.sleep(3)  # wait for page to load

# Focus page and start game
body_elem = driver.find_element("tag name", "body")
body_elem.send_keys(Keys.SPACE)

print("Game started! Gesture control is active.")

# -------------------------------
# 2) OpenCV baseline finger detection
# -------------------------------
cap = cv2.VideoCapture(0)

last_jump = 0
jump_cooldown = 0.7  # seconds
min_area = 8000       # bigger area to ignore face
gesture_frames = 0
frames_needed = 3     # consecutive frames for stable detection

print("OpenCV Finger Count Control - Press q to quit...")

while True:
    ret, frame = cap.read()
    if not ret:
        continue

    frame = cv2.flip(frame, 1)

    # Crop to ROI to avoid detecting face
    roi = frame[100:400, 300:600]  # adjust coordinates for your hand
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (7, 7), 0)

    # Thresholding
    _, thresh = cv2.threshold(blur, 127, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    # Find contours
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if contours:
        largest = max(contours, key=cv2.contourArea)
        if cv2.contourArea(largest) > min_area:
            hull = cv2.convexHull(largest, returnPoints=False)
            if hull is not None and len(hull) > 3:
                defects = cv2.convexityDefects(largest, hull)
                if defects is not None:
                    finger_count = 0
                    for i in range(defects.shape[0]):
                        s, e, f, d = defects[i, 0]
                        start = tuple(largest[s][0])
                        end = tuple(largest[e][0])
                        far = tuple(largest[f][0])

                        # Angle between fingers
                        a = np.linalg.norm(np.array(start) - np.array(end))
                        b = np.linalg.norm(np.array(start) - np.array(far))
                        c = np.linalg.norm(np.array(end) - np.array(far))
                        angle = np.arccos((b**2 + c**2 - a**2) / (2 * b * c))

                        if angle <= np.pi / 2:
                            finger_count += 1

                    finger_count = finger_count + 1  # total fingers
                    cv2.putText(frame, f"Fingers: {finger_count}", (20, 50),
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

                    # Trigger jump if exactly 1 finger detected
                    if finger_count == 1:
                        gesture_frames += 1
                    else:
                        gesture_frames = 0

                    if gesture_frames >= frames_needed and (time.time() - last_jump) > jump_cooldown:
                        # Send SPACE to the Dino game
                        body_elem.send_keys(Keys.SPACE)
                        last_jump = time.time()
                        gesture_frames = 0
                        cv2.putText(frame, "JUMP!", (20, 100),
                                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

            # Draw contour in ROI
            cv2.drawContours(roi, [largest], -1, (255, 0, 0), 2)

    # Show webcam
    cv2.imshow("Finger Count Control (Baseline)", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
driver.quit()
