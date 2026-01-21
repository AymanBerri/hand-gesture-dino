import cv2
import mediapipe as mp
import numpy as np
import os

# --- Setup MediaPipe ---
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7
)

# --- Webcam ---
cap = cv2.VideoCapture(0)

X = []  # features
y = []  # labels

classes = {"idle": 0, "jump": 1}

# --- Create dataset folder ---
dataset_folder = "dataset"
os.makedirs(dataset_folder, exist_ok=True)

print("Data collection for hand gestures!")
print("Press 'i' for idle, 'j' for jump, 'q' to quit.")

while True:
    ret, frame = cap.read()
    if not ret:
        continue

    frame = cv2.flip(frame, 1)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = hands.process(rgb_frame)

    if result.multi_hand_landmarks:
        hand_landmarks = result.multi_hand_landmarks[0]

        # Flatten 21 landmarks (x, y) → 42-dim feature vector
        features = []
        for lm in hand_landmarks.landmark:
            features.append(lm.x)
            features.append(lm.y)

        cv2.putText(frame, "Hand detected", (20, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    else:
        features = None
        cv2.putText(frame, "No hand detected", (20, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    cv2.imshow("Collecting Hand Data", frame)
    key = cv2.waitKey(1) & 0xFF

    if key == ord("q"):
        break
    elif key == ord("i") and features is not None:
        X.append(features)
        y.append(classes["idle"])
        print(f"Idle sample collected: {len(y)}")
        # Optional: save frame as image
        cv2.imwrite(os.path.join(dataset_folder, f"idle_{len(y)}.png"), frame)
    elif key == ord("j") and features is not None:
        X.append(features)
        y.append(classes["jump"])
        print(f"Jump sample collected: {len(y)}")
        cv2.imwrite(os.path.join(dataset_folder, f"jump_{len(y)}.png"), frame)

cap.release()
cv2.destroyAllWindows()

# Save dataset in dataset folder
X = np.array(X)
y = np.array(y)
np.save(os.path.join(dataset_folder, "X.npy"), X)
np.save(os.path.join(dataset_folder, "y.npy"), y)

print(f"Dataset saved in '{dataset_folder}'! Total samples: {len(y)}")
