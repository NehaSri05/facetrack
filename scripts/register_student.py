import cv2
import os
import sqlite3
import torch
import numpy as np
from facenet_pytorch import MTCNN

# ─────────────────────────────
# CONFIGURATION
# ─────────────────────────────
DATASET_DIR = "dataset"
DB_PATH = "database/attendance.db"
NUM_IMAGES = 50  # capture 50 images per student
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
mtcnn = MTCNN(keep_all=False, device=DEVICE)

# ─────────────────────────────
# REGISTER STUDENT FUNCTION
# ─────────────────────────────
def register_student():
    name = input("Enter student name: ").strip()
    roll = input("Enter roll number: ").strip()

    # ensure dataset folder exists
    folder_name = f"{roll}_{name.replace(' ', '_')}"
    save_path = os.path.join(DATASET_DIR, folder_name)
    os.makedirs(save_path, exist_ok=True)

    # save student to database
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS students(
            roll_no TEXT PRIMARY KEY,
            name TEXT
        )
    """)
    cursor.execute("INSERT OR REPLACE INTO students (roll_no, name) VALUES (?, ?)", (roll, name))
    conn.commit()
    conn.close()
    print(f"✅ Student '{name}' ({roll}) added to database.")

    # start webcam capture
    cap = cv2.VideoCapture(0)
    count = 0
    print(f"📸 Capturing {NUM_IMAGES} images... Press 'q' to stop early.")

    while count < NUM_IMAGES:
        ret, frame = cap.read()
        if not ret:
            continue

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        face = mtcnn(rgb)

        if face is not None:
            # Convert tensor to uint8 NumPy image
            face_img = face.permute(1, 2, 0).cpu().numpy()
            face_img = (face_img * 255).astype(np.uint8)  # Convert float32 -> uint8
            face_img = cv2.cvtColor(face_img, cv2.COLOR_RGB2BGR)
            img_path = os.path.join(save_path, f"{count+1}.jpg")
            cv2.imwrite(img_path, face_img)
            count += 1
            cv2.imshow("Captured Face", face_img)
            print(f"Captured {count}/{NUM_IMAGES}", end='\r')

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    print(f"\n✅ {count} images saved to: {save_path}")

# ─────────────────────────────
# MAIN
# ─────────────────────────────
if __name__ == "__main__":
    register_student()
