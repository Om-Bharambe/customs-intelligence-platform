import csv
import os
from datetime import datetime

FEEDBACK_FILE = "feedback/feedback_log.csv"

def save_feedback(image_name, detections, feedback_status, issue_type="", note=""):
    os.makedirs("feedback", exist_ok=True)

    file_exists = os.path.exists(FEEDBACK_FILE)

    with open(FEEDBACK_FILE, "a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)

        if not file_exists:
            writer.writerow([
                "timestamp",
                "image_name",
                "detections",
                "feedback_status",
                "issue_type",
                "note"
            ])

        writer.writerow([
            datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            image_name,
            str(detections),
            feedback_status,
            issue_type,
            note
        ])