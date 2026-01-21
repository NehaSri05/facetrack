import sqlite3
import os
import pandas as pd

DB_PATH = "database/attendance.db"

def view_attendance():
    if not os.path.exists(DB_PATH):
        print("❌ Database not found. Run setup_database.py first.")
        input("\nPress Enter to return to menu...")
        return

    conn = sqlite3.connect(DB_PATH)
    df = pd.read_sql_query("""
        SELECT name AS "Student Name", roll_no AS "Roll No", subject AS "Subject",
               date AS "Date", time AS "Time", status AS "Status"
        FROM attendance
        ORDER BY date DESC, time DESC
    """, conn)
    conn.close()

    if df.empty:
        print("⚠️ No attendance records found.")
    else:
        print("\n========== 📋 Attendance Records ==========\n")
        print(df.to_string(index=False))
        print("\n===========================================\n")

    input("Press Enter to return to menu...")

if __name__ == "__main__":
    view_attendance()
