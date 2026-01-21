import os
import sqlite3
import pandas as pd
from datetime import datetime
from playsound import playsound
import threading

# ─────────────────────────────
# CONFIGURATION
# ─────────────────────────────
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DB_DIR = os.path.abspath(os.path.join(BASE_DIR, "..", "database"))
DB_PATH = os.path.join(DB_DIR, "attendance.db")
EXCEL_DIR = os.path.join(DB_DIR, "attendance_excels")
SOUND_PATH = os.path.abspath(os.path.join(BASE_DIR, "..", "censor-beep-2-372461.mp3"))

os.makedirs(EXCEL_DIR, exist_ok=True)

# ─────────────────────────────
# SOUND FUNCTION
# ─────────────────────────────
def play_done_sound():
    try:
        if os.path.exists(SOUND_PATH):
            threading.Thread(target=playsound, args=(SOUND_PATH,), daemon=True).start()
        else:
            print(f"[⚠️] Sound file not found: {SOUND_PATH}")
    except Exception as e:
        print(f"[⚠️ Sound Error] {e}")

# ─────────────────────────────
# EXPORT FUNCTION
# ─────────────────────────────
def export_daily_excels():
    if not os.path.exists(DB_PATH):
        print("❌ Database not found! Please run attendance system first.")
        return

    conn = sqlite3.connect(DB_PATH)
    query = "SELECT roll_no, name, subject, date, time, status FROM attendance"
    df = pd.read_sql_query(query, conn)
    conn.close()

    if df.empty:
        print("⚠️ No attendance records found.")
        return

    # Remove duplicate entries for same student+subject+date
    df.drop_duplicates(subset=["roll_no", "name", "subject", "date"], inplace=True)

    # Group by date
    for date, group in df.groupby("date"):
        file_path = os.path.join(EXCEL_DIR, f"attendance_{date}.xlsx")
        group.to_excel(file_path, index=False)
        print(f"[📅] Saved: {file_path}")

    play_done_sound()
    print(f"\n✅ All daily attendance exported successfully to: {EXCEL_DIR}")

# ─────────────────────────────
# MAIN
# ─────────────────────────────
if __name__ == "__main__":
    print("📊 Exporting attendance data to daily Excel sheets...\n")
    export_daily_excels()
