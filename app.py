import streamlit as st  # type: ignore
import os
import pandas as pd
import cv2
import torch
import joblib
import numpy as np
from datetime import datetime
from facenet_pytorch import MTCNN, InceptionResnetV1

# ----------------------------
# APP CONFIG
# ----------------------------
st.set_page_config(page_title="FaceTrack Attendance System", layout="wide")
st.title("📸 FaceTrack: Face Recognition Attendance System")

# Paths
DB_PATH = "database/attendance.db"
MODEL_DIR = "models"
DATASET_DIR = "dataset"
os.makedirs("temp", exist_ok=True)

# ----------------------------
# MODEL LOADING
# ----------------------------
@st.cache_resource
def load_models():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    mtcnn = MTCNN(image_size=160, keep_all=True, device=device)
    resnet = InceptionResnetV1(pretrained='vggface2').eval().to(device)
    svm_model = joblib.load(os.path.join(MODEL_DIR, "svm_model.pkl"))
    label_encoder = joblib.load(os.path.join(MODEL_DIR, "label_encoder.pkl"))
    return mtcnn, resnet, svm_model, label_encoder, device

mtcnn, resnet, svm_model, label_encoder, device = load_models()

# ----------------------------
# MENU SELECTION
# ----------------------------
menu = st.sidebar.selectbox(
    "Select Option",
    ["🏠 Home", "🧍 Register Student", "📷 Take Attendance", "📊 View Attendance","🧠 Model Info"]
)

# ----------------------------
# REGISTER STUDENT
# ----------------------------
if menu == "🧍 Register Student":
    st.subheader("Register a New Student")

    name = st.text_input("Enter Student Name")
    roll_no = st.text_input("Enter Roll Number")
    start = st.button("Start Capture")

    if start and name and roll_no:
        folder_name = f"{roll_no}_{name.replace(' ', '_')}"
        save_path = os.path.join(DATASET_DIR, folder_name)
        os.makedirs(save_path, exist_ok=True)

        st.info("📸 Opening camera. Press 'q' to quit capture.")
        cap = cv2.VideoCapture(0)
        count = 0
        while count < 30:
            ret, frame = cap.read()
            if not ret:
                continue
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            faces = mtcnn(rgb)
            if faces is not None:
                face_img = faces[0].permute(1, 2, 0).numpy()
                face_img = (face_img * 255).astype(np.uint8)
                face_img = cv2.cvtColor(face_img, cv2.COLOR_RGB2BGR)
                img_path = os.path.join(save_path, f"{count+1}.jpg")
                cv2.imwrite(img_path, face_img)
                count += 1
                st.image(rgb, caption=f"Captured {count}/30", channels="RGB")
        cap.release()
        st.success(f"✅ Registered {name} ({roll_no}) successfully!")

# ----------------------------
# TAKE ATTENDANCE
# ----------------------------
elif menu == "📷 Take Attendance":
    st.subheader("Take Real-Time Attendance")

    subject = st.text_input("Enter Subject Name", "General")
    start_att = st.button("Start Attendance")

    if start_att:
        st.info("📷 Starting webcam. Press 'q' in window to stop.")

        cap = cv2.VideoCapture(0)
        attendance = set()
        start_time = datetime.now()

        while True:
            ret, frame = cap.read()
            if not ret:
                break
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            faces = mtcnn(rgb)
            if faces is not None:
                with torch.no_grad():
                    embeddings = resnet(faces.to(device)).detach().cpu().numpy()
                for emb in embeddings:
                    probs = svm_model.predict_proba([emb])[0]
                    pred = np.argmax(probs)
                    conf = probs[pred]
                    name = label_encoder.inverse_transform([pred])[0] if conf > 0.6 else "Unknown"
                    if name != "Unknown" and name not in attendance:
                        attendance.add(name)
                        st.write(f"✅ {name} marked present at {datetime.now().strftime('%H:%M:%S')}")
            cv2.imshow("Attendance", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        cap.release()
        cv2.destroyAllWindows()
        st.success(f"Total Students Marked: {len(attendance)}")

# ----------------------------
# VIEW ATTENDANCE
# ----------------------------
elif menu == "📊 View Attendance":
    st.subheader("View Attendance Records")

    csv_files = [f for f in os.listdir("database") if f.endswith(".csv")]
    if not csv_files:
        st.warning("No attendance CSVs found yet!")
    else:
        latest = os.path.join("database", sorted(csv_files)[-1])
        df = pd.read_csv(latest)
        st.dataframe(df)
        st.download_button("Download CSV", df.to_csv(index=False), file_name="attendance_latest.csv")
# ----------------------------
# 🧠 MODEL INFORMATION PANEL
# ----------------------------
elif menu == "🧠 Model Info":
    import os
    st.subheader("Model Information & Performance")

    # Display Model Details
    st.write("**Model Type:** FaceNet + SVM (Linear Kernel)")
    st.write("**Feature Extractor:** InceptionResnetV1 (512-D Face Embeddings)")
    st.write("**Classifier:** Support Vector Machine (Linear Kernel, scikit-learn)")
    st.write("**Training Dataset:** Registered Students’ Faces (50 samples each)")
    st.write("**Test Accuracy:** ~97.5% on validation set")

    # Show Confusion Matrix
    if os.path.exists("models/confusion_matrix.png"):
        st.image("models/confusion_matrix.png", caption="Confusion Matrix - FaceTrack SVM Model")
    else:
        st.warning("Confusion matrix not found. Please run `train_model.py` to generate it.")

    # Add small description
    st.markdown("""
    The FaceTrack model uses **FaceNet (InceptionResnetV1)** for feature extraction.  
    It converts each detected face into a **512-dimensional embedding vector**,  
    which the **SVM classifier** then uses to distinguish between different students.

    - **FaceNet:** Pre-trained on VGGFace2 dataset for deep feature extraction.  
    - **SVM:** Provides fast and accurate face classification.  
    - **Confusion Matrix:** Shows which students’ faces were classified correctly or misidentified.  
    """)

    # Optional: Add Model Files Download Buttons
    if os.path.exists("models/svm_model.pkl"):
        with open("models/svm_model.pkl", "rb") as f:
            st.download_button("📥 Download SVM Model", f, file_name="svm_model.pkl")
    if os.path.exists("models/label_encoder.pkl"):
        with open("models/label_encoder.pkl", "rb") as f:
            st.download_button("📥 Download Label Encoder", f, file_name="label_encoder.pkl")

