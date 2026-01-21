import os
import subprocess
import sys
import time

SCRIPTS_DIR = "scripts"

def clear_screen():
    os.system('cls' if os.name == 'nt' else 'clear')

def show_menu():
    clear_screen()
    print("""
========== 🎓 Face Recognition Attendance System ==========

1️⃣  Register new student

─── TRAIN MODELS ─────────────────────────────
2️⃣  Train FaceNet (General) model

─── TAKE ATTENDANCE ──────────────────────────
3️⃣  Take attendance (FaceNet / General model)

─── OTHER OPTIONS ────────────────────────────
4️⃣  View attendance records
5️⃣  Export daily attendance to Excel
6️⃣  Exit
===========================================================
""")

def run_script(script_name):
    script_path = os.path.join(SCRIPTS_DIR, script_name)
    if not os.path.exists(script_path):
        print(f"❌ Script not found: {script_path}")
        input("\nPress Enter to return to menu...")
        return

    print(f"\n▶ Running: {script_path}\n(Press Ctrl+C to stop)\n")
    try:
        proc = subprocess.run([sys.executable, script_path], check=False)
        if proc.returncode != 0:
            print(f"\n⚠️ Script exited with code: {proc.returncode}")
    except KeyboardInterrupt:
        print("\n❌ Script interrupted by user (Ctrl+C).")
    except Exception as e:
        print(f"\n❌ Error while running script: {e}")

    input("\n✅ Press Enter to return to the main menu...")

def main():
    while True:
        show_menu()
        choice = input("Enter your choice (1–6): ").strip()

        if choice == "1":
            run_script("register_student.py")

        # ─── TRAINING ───────────────────────────────
        elif choice == "2":
            run_script("train_model.py")          # FaceNet/SVM trainer

        # ─── ATTENDANCE ─────────────────────────────
        elif choice == "3":
            run_script("attendance_system.py")    # FaceNet attendance

        # ─── OTHER ─────────────────────────────────
        elif choice == "4":
            run_script("view_attendance.py")
        elif choice == "5":
            run_script("export_attendance_excel.py")
        elif choice == "6":
            print("👋 Exiting system. Goodbye!")
            break

        else:
            print("❌ Invalid choice. Try again.")
            time.sleep(1)

if __name__ == "__main__":
    main()
