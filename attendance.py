import csv
import os
from datetime import datetime

ATTENDANCE_FOLDER = "Attendance"

# Create folder if it doesn't exist
os.makedirs(ATTENDANCE_FOLDER, exist_ok=True)


def mark_attendance(name):
    today = datetime.now().strftime("%d-%m-%Y")

    file_path = os.path.join(
        ATTENDANCE_FOLDER,
        f"attendance_{today}.csv"
    )

    file_exists = os.path.isfile(file_path)

    with open(file_path, "a", newline="") as file:
        writer = csv.writer(file)

        # Write header only once
        if not file_exists:
            writer.writerow(["Name", "Date", "Time", "Status"])

        writer.writerow([
            name,
            today,
            datetime.now().strftime("%H:%M:%S"),
            "Present"
        ])

    print(f"{name} attendance saved.")