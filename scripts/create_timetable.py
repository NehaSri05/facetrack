import sqlite3
import os
from datetime import datetime

# Ensure the database folder exists
os.makedirs("database", exist_ok=True)

# Connect to your existing attendance database
conn = sqlite3.connect("database/attendance.db")
cursor = conn.cursor()

# Create timetable table if it doesn't exist already
cursor.execute("""
CREATE TABLE IF NOT EXISTS timetable (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    section TEXT,
    day TEXT,
    start_time TEXT,
    end_time TEXT,
    subject TEXT
)
""")

# Real timetable for section AI (7 periods/day, 40-min lunch break)
# 50-minute periods, no gaps between classes, realistic full-day schedule
data = [
    # Monday
    ("AI", "Monday",    "09:00", "09:50", "NLP"),
    ("AI", "Monday",    "09:50", "10:40", "DL"),
    ("AI", "Monday",    "10:40", "11:30", "CC"),
    ("AI", "Monday",    "11:30", "12:20", "BD"),
    # Lunch break 12:20 – 13:00
    ("AI", "Monday",    "13:00", "13:50", "CS"),
    ("AI", "Monday",    "13:50", "14:40", "DL"),
    ("AI", "Monday",    "14:40", "15:30", "NLP"),

    # Tuesday
    ("AI", "Tuesday",   "09:00", "09:50", "CC"),
    ("AI", "Tuesday",   "09:50", "10:40", "BD"),
    ("AI", "Tuesday",   "10:40", "11:30", "DL"),
    ("AI", "Tuesday",   "11:30", "12:20", "NLP"),
    ("AI", "Tuesday",   "13:00", "13:50", "CS"),
    ("AI", "Tuesday",   "13:50", "14:40", "CC"),
    ("AI", "Tuesday",   "14:40", "15:30", "BD"),

    # Wednesday
    ("AI", "Wednesday", "09:00", "09:50", "CS"),
    ("AI", "Wednesday", "09:50", "10:40", "NLP"),
    ("AI", "Wednesday", "10:40", "11:30", "DL"),
    ("AI", "Wednesday", "11:30", "12:20", "CC"),
    ("AI", "Wednesday", "13:00", "13:50", "BD"),
    ("AI", "Wednesday", "13:50", "14:40", "DL"),
    ("AI", "Wednesday", "14:40", "15:30", "NLP"),

    # Thursday
    ("AI", "Thursday",  "09:00", "09:50", "BD"),
    ("AI", "Thursday",  "09:50", "10:40", "CC"),
    ("AI", "Thursday",  "10:40", "11:30", "NLP"),
    ("AI", "Thursday",  "11:30", "12:20", "DL"),
    ("AI", "Thursday",  "13:00", "13:50", "CS"),
    ("AI", "Thursday",  "13:50", "14:40", "NLP"),
    ("AI", "Thursday",  "14:40", "15:30", "CC"),

    # Friday
    ("AI", "Friday",    "09:00", "09:50", "DL"),
    ("AI", "Friday",    "09:50", "10:40", "NLP"),
    ("AI", "Friday",    "10:40", "11:30", "BD"),
    ("AI", "Friday",    "11:30", "12:20", "CS"),
    ("AI", "Friday",    "13:00", "13:50", "CC"),
    ("AI", "Friday",    "13:50", "14:40", "BD"),
    ("AI", "Friday",    "14:40", "15:30", "DL"),
]

# Clear existing timetable for this section (avoid duplicates)
cursor.execute("DELETE FROM timetable WHERE section = 'AI'")

# Insert the updated data
cursor.executemany("""
INSERT INTO timetable (section, day, start_time, end_time, subject)
VALUES (?, ?, ?, ?, ?)
""", data)

# Commit and close
conn.commit()
conn.close()

print("Real timetable created successfully for section AI (7 periods/day with 40-min lunch break).")


# -------------------------------------------------------------
# Function to get the current subject based on system time
# -------------------------------------------------------------
def get_current_subject(section="AI"):
    """Return the current subject based on timetable and current system time."""
    conn = sqlite3.connect("database/attendance.db")
    cursor = conn.cursor()

    now = datetime.now()
    current_day = now.strftime("%A")
    current_time = now.strftime("%H:%M")

    cursor.execute("""
        SELECT subject FROM timetable
        WHERE section = ?
        AND day = ?
        AND start_time <= ?
        AND end_time >= ?
    """, (section, current_day, current_time, current_time))

    result = cursor.fetchone()
    conn.close()

    return result[0] if result else "No Class"
