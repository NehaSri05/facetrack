import os
import cv2
import time
import torch
import joblib
import csv
import numpy as np
import sqlite3
import threading
from datetime import datetime
import pygame
from facenet_pytorch import MTCNN, InceptionResnetV1

from create_timetable import get_current_subject
3
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.abspath(os.path.join(BASE_DIR, "..", "models"))
DB_DIR = os.path.abspath(os.path.join(BASE_DIR, "..", "database"))
DB_PATH = os.path.join(DB_DIR, "attendance.db")
SOUND_PATH = os.path.abspath(os.path.join(BASE_DIR, "..", "censor-beep-2-372461.mp3"))

os.makedirs(DB_DIR, exist_ok=True)
print(f"[INFO] Using database at: {DB_PATH}")

def play_attendance_sound():
    """Play beep sound asynchronously when attendance is marked (pygame)."""
    def _play():
        try:
            if not os.path.exists(SOUND_PATH):
                print(f"[⚠️] Sound file not found: {SOUND_PATH}")
                return
            try:
                if not pygame.mixer.get_init():
                    pygame.mixer.init()
            except Exception:
                pygame.mixer.quit()
                pygame.mixer.init()
            pygame.mixer.music.load(SOUND_PATH)
            pygame.mixer.music.play()
            pygame.time.wait(1000)
        except Exception as e:
            print(f"[Sound Error] {e}")
    threading.Thread(target=_play, daemon=True).start()

def ensure_csv_exists():
    date_str = datetime.now().strftime("%Y-%m-%d")
    csv_path = os.path.join(DB_DIR, f"attendance_{date_str}.csv")
    if not os.path.exists(csv_path):
        with open(csv_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["Roll No", "Name", "Subject", "Date", "Time", "Status"])
        print(f"Created new CSV file for today: {csv_path}")
    return csv_path

def mark_attendance(name, subject):
    try:
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS attendance(
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                roll_no TEXT,
                name TEXT,
                subject TEXT,
                date TEXT,
                time TEXT,
                status TEXT
            )
        """)
        now = datetime.now()
        date_str = now.strftime("%Y-%m-%d")
        time_str = now.strftime("%H:%M:%S")
        cursor.execute("SELECT 1 FROM attendance WHERE name=? AND subject=? AND date=?",
                       (name, subject, date_str))
        exists = cursor.fetchone()
        if not exists:
            roll_no = name.split("_")[0] if "_" in name else "Unknown"
            cursor.execute("""
                INSERT INTO attendance (roll_no, name, subject, date, time, status)
                VALUES (?, ?, ?, ?, ?, ?)
            """, (roll_no, name, subject, date_str, time_str, "Present"))
            conn.commit()
            csv_path = ensure_csv_exists()
            with open(csv_path, "a", newline="") as f:
                writer = csv.writer(f)
                writer.writerow([roll_no, name, subject, date_str, time_str, "Present"])
            print(f"[MARKED] {name} | Subject: {subject} | Time: {time_str}")
            play_attendance_sound()
        else:
            print(f"[INFO] Already marked today for {name}")
    except Exception as e:
        print(f"[DATABASE ERROR] {e}")
    finally:
        conn.close()

print("Loading models...")
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
mtcnn = MTCNN(image_size=160, keep_all=True, device=device)
resnet = InceptionResnetV1(pretrained='vggface2').eval().to(device)

svm_model_path = os.path.join(MODEL_DIR, "svm_model.pkl")
label_encoder_path = os.path.join(MODEL_DIR, "label_encoder.pkl")

if not os.path.exists(svm_model_path) or not os.path.exists(label_encoder_path):
    print("Model files not found in:", MODEL_DIR)
    print("Please run your training script first (train_model.py).")
    exit()

svm_model = joblib.load(svm_model_path)
label_encoder = joblib.load(label_encoder_path)
print("Models loaded successfully.\n")

print("Choose attendance mode:")
print("1️.Normal mode")
print("2️.Timetable-based mode")
choice = input("Enter your choice (1 or 2): ").strip()

if choice == "2":
    subject_name = get_current_subject()
    if subject_name and subject_name != "No Class":
        print(f"Auto-detected subject from timetable: {subject_name}")
    else:
        print("No class right now. Switching to manual subject entry.")
        subject_name = input("Enter subject name manually: ").strip() or "General"
else:
    subject_name = input("Enter subject name: ").strip() or "General"

print(f"\n Starting attendance for: {subject_name}")
print("Duration: 10 minutes — press 'q' to quit manually.\n")

cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Error: Cannot open webcam.")
    exit()

attendance = set()
start_time = time.time()
TIME_LIMIT = 10 * 60
ensure_csv_exists()

while True:
    ret, frame = cap.read()
    if not ret:
        break
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    faces = mtcnn(rgb)
    if faces is not None:
        if hasattr(faces, "ndimension") and faces.ndimension() == 3:
            faces = faces.unsqueeze(0)
        with torch.no_grad():
            embeddings = resnet(faces.to(device)).detach().cpu().numpy()
        for emb in embeddings:
            probs = svm_model.predict_proba([emb])[0]
            pred = np.argmax(probs)
            confidence = probs[pred]
            name = label_encoder.inverse_transform([pred])[0] if confidence > 0.6 else "Unknown"
            if name != "Unknown":
                cv2.putText(frame, f"{name} ({confidence:.2f})", (20, 80),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
                if name not in attendance:
                    attendance.add(name)
                    mark_attendance(name, subject_name)
            else:
                cv2.putText(frame, "Unknown", (20, 80),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
    elapsed = time.time() - start_time
    remaining = max(0, int(TIME_LIMIT - elapsed))
    mins, secs = divmod(remaining, 60)
    cv2.rectangle(frame, (0, 0), (430, 70), (0, 0, 0), -1)
    cv2.putText(frame, f"Time left: {mins:02}:{secs:02}", (10, 25),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    cv2.putText(frame, f"Marked: {len(attendance)}", (10, 55),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
    cv2.imshow("Attendance System", frame)
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        print("\n Attendance ended manually.")
        break
    if remaining <= 0:
        print("\n Time limit reached (10 minutes).")
        break

cap.release()
cv2.destroyAllWindows()

print(f"\n Total students marked present: {len(attendance)}")
print("Attendance saved in:", DB_PATH)
print("Daily CSV available at:", os.path.join(DB_DIR, f"attendance_{datetime.now().strftime('%Y-%m-%d')}.csv"))
