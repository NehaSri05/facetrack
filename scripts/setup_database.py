import sqlite3, os

os.makedirs("database", exist_ok=True)
conn = sqlite3.connect("database/attendance.db")
cursor = conn.cursor()

# Students table
cursor.execute("""
CREATE TABLE IF NOT EXISTS students(
    roll_no TEXT PRIMARY KEY,
    name TEXT,
    email TEXT
)
""")

# Attendance table
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

conn.commit()
conn.close()
print("✅ Database setup complete at: database/attendance.db")
