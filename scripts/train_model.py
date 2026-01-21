import os
import cv2
import numpy as np
import torch
import joblib
import matplotlib.pyplot as plt  # type: ignore
from facenet_pytorch import MTCNN, InceptionResnetV1
from sklearn.svm import SVC
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay, classification_report
from sklearn.model_selection import train_test_split

# ─────────────────────────────
# INITIAL SETUP
# ─────────────────────────────
print("Loading FaceNet model...")
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
mtcnn = MTCNN(image_size=160, margin=0, keep_all=False, device=device)
resnet = InceptionResnetV1(pretrained='vggface2').eval().to(device)

DATASET_DIR = "dataset"
MODEL_DIR = "models"
os.makedirs(MODEL_DIR, exist_ok=True)

embeddings, labels = [], []

# ─────────────────────────────
# DATASET LOOP
# ─────────────────────────────
print("Processing dataset...")

for student_name in os.listdir(DATASET_DIR):
    student_folder = os.path.join(DATASET_DIR, student_name)
    if not os.path.isdir(student_folder):
        continue

    for img_name in os.listdir(student_folder):
        img_path = os.path.join(student_folder, img_name)
        img = cv2.imread(img_path)
        if img is None:
            continue

        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        face = mtcnn(img_rgb)
        if face is not None:
            if face.ndim == 3:
                face = face.unsqueeze(0)

            with torch.no_grad():
                emb = resnet(face.to(device)).detach().cpu().numpy()

            embeddings.append(emb.flatten())
            labels.append(student_name)

print(f" Collected {len(embeddings)} embeddings from {len(set(labels))} students.")

if len(embeddings) == 0:
    print("No faces detected. Make sure dataset images are clear.")
    exit()

# ─────────────────────────────
# TRAIN + EVALUATE MODEL
# ─────────────────────────────
print("Training SVM model...")
X = np.array(embeddings)
y = np.array(labels)

label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)

# Split dataset into train/test sets (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(
    X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
)

# Train SVM
svm_model = SVC(kernel='linear', probability=True)
svm_model.fit(X_train, y_train)

# ─────────────────────────────
# EVALUATION METRICS
# ─────────────────────────────
y_pred = svm_model.predict(X_test)
test_accuracy = accuracy_score(y_test, y_pred) * 100
print(f"Test Accuracy (FaceNet + SVM): {test_accuracy:.2f}%")

# Classification Report
print("\n Classification Report:")
print(classification_report(y_test, y_pred, target_names=label_encoder.classes_))

# ─────────────────────────────
# CONFUSION MATRIX
# ─────────────────────────────
cm = confusion_matrix(y_test, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=label_encoder.classes_)

fig, ax = plt.subplots(figsize=(6,6))
disp.plot(ax=ax, cmap="Blues", colorbar=True)
plt.title("Confusion Matrix - FaceTrack (SVM)")
plt.xticks(rotation=45)
plt.tight_layout()

# Save confusion matrix image
cm_path = os.path.join(MODEL_DIR, "confusion_matrix.png")
plt.savefig(cm_path)
plt.show()

print(f"Confusion matrix saved at: {cm_path}")

# ─────────────────────────────
# SAVE MODELS
# ─────────────────────────────
joblib.dump(svm_model, os.path.join(MODEL_DIR, "svm_model.pkl"))
joblib.dump(label_encoder, os.path.join(MODEL_DIR, "label_encoder.pkl"))
print("Model and label encoder saved in 'models/' folder.")
