import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, confusion_matrix
import joblib
import os
import json

# -------------------------------
# Paths
# -------------------------------
dataset_folder = "dataset"
model_folder = "models"

os.makedirs(model_folder, exist_ok=True)

dataset_X_path = os.path.join(dataset_folder, "X.npy")
dataset_y_path = os.path.join(dataset_folder, "y.npy")
model_path = os.path.join(model_folder, "hand_svm_model.joblib")
metrics_path = os.path.join(dataset_folder, "metrics.json")

# -------------------------------
# Load dataset
# -------------------------------
X = np.load(dataset_X_path)
y = np.load(dataset_y_path)

print(f"Loaded {X.shape[0]} samples. Feature shape: {X.shape[1]}")

# -------------------------------
# Train/Test split
# -------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# -------------------------------
# Train SVM
# -------------------------------
clf = SVC(kernel="linear", probability=True)
clf.fit(X_train, y_train)

# -------------------------------
# Evaluate
# -------------------------------
y_pred = clf.predict(X_test)
acc = accuracy_score(y_test, y_pred)
cm = confusion_matrix(y_test, y_pred)

print(f"Test Accuracy: {acc*100:.2f}%")
print("Confusion Matrix:")
print(cm)

# -------------------------------
# Save trained model
# -------------------------------
joblib.dump(clf, model_path)
print(f"Trained SVM model saved at {model_path}")

# -------------------------------
# Save metrics for reporting
# -------------------------------
metrics = {
    "accuracy": float(acc),
    "confusion_matrix": cm.tolist(),
    "num_samples": int(X.shape[0]),
    "num_train": int(X_train.shape[0]),
    "num_test": int(X_test.shape[0]),
}

with open(metrics_path, "w") as f:
    json.dump(metrics, f, indent=4)

print(f"Metrics saved at {metrics_path}")
