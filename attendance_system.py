import cv2
import pickle
import csv
import os
import time
import numpy as np
import pyttsx3
import threading

from datetime import datetime
from deepface import DeepFace
from scipy.spatial.distance import cosine
from anti_spoofing.src.anti_spoof_predict import AntiSpoofPredict
from anti_spoofing.src.generate_patches import CropImage

# =====================================================
# VOICE ENGINE
# =====================================================

def speak(text):

    try:

        engine = pyttsx3.init()

        engine.setProperty("rate", 170)
        engine.setProperty("volume", 1)

        voices = engine.getProperty("voices")

        if len(voices) > 1:
            engine.setProperty("voice", voices[1].id)

        print("SPEAK:", text)

        engine.say(text)
        engine.runAndWait()

        engine.stop()

    except Exception as e:

        print("Speech Error:", e)
        

# =====================================================
# LOAD FACE DATABASE
# =====================================================

with open("face_db.pkl", "rb") as f:
    db = pickle.load(f)


# =====================================================
# ATTENDANCE FOLDER
# =====================================================

ATTENDANCE_FOLDER = "Attendance"

os.makedirs(ATTENDANCE_FOLDER, exist_ok=True)

CSV_FILE = os.path.join(
    ATTENDANCE_FOLDER,
    "attendance.csv"
)

# =====================================================
# CREATE CSV IF NOT EXISTS
# =====================================================

if not os.path.exists(CSV_FILE):

    with open(CSV_FILE, "w", newline="") as file:

        writer = csv.writer(file)

        writer.writerow([
            "Student Name",
            "Date",
            "Time",
            "Status"
        ])

# =====================================================
# LOAD TODAY'S ATTENDANCE
# =====================================================

marked_students = set()

today = datetime.now().strftime("%d-%m-%Y")

if os.path.exists(CSV_FILE):

    with open(CSV_FILE, "r") as file:

        reader = csv.reader(file)

        next(reader, None)

        for row in reader:

            if len(row) >= 4 and row[1] == today:

                marked_students.add(row[0])

attendance_count = len(marked_students)

print("--------------------------------")
print("Already Marked Today")
print(marked_students)
print("--------------------------------")

# =====================================================
# LOAD HAAR CASCADE
# =====================================================

face_detector = cv2.CascadeClassifier(
    cv2.data.haarcascades +
    "haarcascade_frontalface_default.xml"
)

# =====================================================
# OPEN CAMERA
# =====================================================

cap = cv2.VideoCapture(0)

cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)

print("Width:", cap.get(cv2.CAP_PROP_FRAME_WIDTH))
print("Height:", cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

if not cap.isOpened():

    print("Unable to open webcam.")
    exit()

# =====================================================
# FULL SCREEN
# =====================================================

cv2.namedWindow(
    "Attendance System",
    cv2.WINDOW_NORMAL
)

cv2.setWindowProperty(
    "Attendance System",
    cv2.WND_PROP_FULLSCREEN,
    cv2.WINDOW_FULLSCREEN
)

# =====================================================
# LOAD FACENET ONCE
# =====================================================

print("Loading FaceNet...")

DeepFace.represent(
    img_path=np.zeros((160,160,3),dtype=np.uint8),
    model_name="Facenet",
    enforce_detection=False
)

print("FaceNet Loaded Successfully")

# =====================================================
# LOAD ANTI-SPOOFING
# =====================================================

print("Loading Anti-Spoofing...")

anti_spoof_model = AntiSpoofPredict(0)
image_cropper = CropImage()

print("Anti-Spoof Loaded Successfully")


# =====================================================
# VERIFICATION SETTINGS
# =====================================================

DISTANCE_THRESHOLD = 0.40

VERIFICATION_TIME = 1.5

DISPLAY_RESULT_TIME = 5
hide_rectangle = False

NO_FACE_RESET_TIME = 1.0

# =====================================================
# COLOR STATE CONTROL
# =====================================================

STATE = "WAITING"   # WAITING / VERIFYING / SUCCESS / DONE
state_start_time = None

# =====================================================
# SYSTEM VARIABLES
# =====================================================

verification_name = None

verification_start = None
current_person_locked = None
recognized_name = None

recognized_time = None
attendance_just_marked = False

notification_title = ""
notification_message = ""
notification_color = (0, 255, 0)

last_face_time = None
status = "Waiting For Face"
label = ""
confidence = 0
box_color = (0,255,255)

progress = 0
STATE = "IDLE"

# Anti-spoof stabilization
spoof_history = []

last_spoken_name = None

# =====================================================
# HEAD MOVEMENT DETECTION
# =====================================================

head_left_done = False
head_right_done = False

head_movement_verified = False

head_instruction = "Turn Head Left"

# =====================================================
# START SYSTEM
# =====================================================

print("---------------------------------------")
print(" AI SMART ATTENDANCE SYSTEM STARTED ")
print("---------------------------------------")
print("Press Q to Exit")

# =====================================================
# MAIN LOOP
# =====================================================

while True:

    ret, frame = cap.read()

    # print("ret =", ret)

    if not ret:
        print("Camera frame lost. Reconnecting...")

        cap.release()

        cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

        time.sleep(1)

        continue

    now = datetime.now()

    current_date = now.strftime("%d-%m-%Y")
    current_time = now.strftime("%H:%M:%S")

    if recognized_name is not None:

        if time.time() - recognized_time > DISPLAY_RESULT_TIME:

            recognized_name = None
            recognized_time = None
            current_person_locked = None

            verification_name = None
            verification_start = None

            progress = 0
            STATE = "IDLE"

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = face_detector.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(50, 50)
    )

    # -------------------------------------
    # No face detected
    # -------------------------------------

    if len(faces) == 0:

        status = "Waiting For Face"

        verification_name = None
        verification_start = None
        progress = 0

    # -------------------------------------
    # Face detected
    # -------------------------------------

    for (x, y, w, h) in faces:

        # Prevent repeated attendance popup
        if recognized_name is not None:
            if time.time() - recognized_time < DISPLAY_RESULT_TIME:
                continue

        # Ignore tiny faces
        if w < 70 or h < 70:

            cv2.putText(
                frame,
                "MOVE CLOSER",
                (x, y - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 255),
                2
            )
            continue

        if w < 120 or h < 120:
            continue

        face = frame[y:y+h, x:x+w]

        # =====================================
        # ANTI SPOOF
        # =====================================

        try:

            bbox = [x, y, w, h]

            face1 = image_cropper.crop(
                org_img=frame,
                bbox=bbox,
                scale=2.7,
                out_w=80,
                out_h=80
            )

            result1 = anti_spoof_model.predict(
                face1,
                "anti_spoofing/anti_spoof_models/2.7_80x80_MiniFASNetV2.pth"
            )

            face2 = image_cropper.crop(
                org_img=frame,
                bbox=bbox,
                scale=4.0,
                out_w=80,
                out_h=80
            )

            result2 = anti_spoof_model.predict(
                face2,
                "anti_spoofing/anti_spoof_models/4_0_0_80x80_MiniFASNetV1SE.pth"
            )

            prediction = result1 + result2

            spoof_label = np.argmax(prediction)
            spoof_confidence = np.max(prediction)

            spoof_history.append(spoof_label)

            if len(spoof_history) > 10:
                spoof_history.pop(0)

            spoof_label = round(
                sum(spoof_history) / len(spoof_history)
            )

            confidence = spoof_confidence

            if spoof_label != 1 or confidence < 0.65:

                label = "FAKE FACE"
                status = "Show Real Face"

                cv2.rectangle(
                    frame,
                    (x, y),
                    (x + w, y + h),
                    (0, 0, 255),
                    2
                )

                cv2.putText(
                    frame,
                    "FAKE FACE DETECTED",
                    (x, y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 0, 255),
                    2
                )

                continue

        except Exception as e:

            print("Anti-Spoof Error:", e)
            continue

        # =====================================
        # FACE RECOGNITION
        # =====================================

        try:

            embedding = DeepFace.represent(
                img_path=face,
                model_name="Facenet",
                enforce_detection=False
            )[0]["embedding"]

            name = "Unknown"
            min_dist = float("inf")

            for person, db_vec in db.items():

                dist = cosine(embedding, db_vec)

                if dist < min_dist:

                    min_dist = dist
                    name = person

            confidence = max(
                0,
                (1 - min_dist) * 100
            )

            # =====================================
            # UNKNOWN PERSON
            # =====================================

            if min_dist > DISTANCE_THRESHOLD:

                verification_name = None
                verification_start = None
                progress = 0

                box_color = (0, 0, 255)

                label = "Unknown"
                status = "Unknown Person"

                continue

            # =====================================
            # RECOGNIZED PERSON
            # =====================================

            box_color = (0, 255, 0)

            label = f"{name} ({confidence:.1f}%)"

            if (
                name == current_person_locked
                and recognized_name == name
                and recognized_time is not None
            ):

                if time.time() - recognized_time < DISPLAY_RESULT_TIME:
                    continue

        except Exception as e:

            print("Recognition Error:", e)

            box_color = (0, 0, 255)

            label = "Unknown"
            status = "Recognition Error"
            progress = 0

            continue

        # =====================================
        # START VERIFICATION
        # =====================================

        if verification_name != name:

            verification_name = name
            verification_start = time.time()

            head_left_done = False
            head_right_done = False
            head_movement_verified = False

            STATE = "VERIFYING"

            box_color = (0, 255, 255)

            status = "Verifying..."
            progress = 0

        else:

            elapsed = time.time() - verification_start

            progress = min(
                int((elapsed / VERIFICATION_TIME) * 100),
                100
            )

            if (
                elapsed >= VERIFICATION_TIME
                and recognized_name is None
            ):

                # =====================================
                # NEW ATTENDANCE
                # =====================================

                if name not in marked_students:

                    with open(
                        CSV_FILE,
                        "a",
                        newline=""
                    ) as file:

                        writer = csv.writer(file)

                        writer.writerow([
                            name,
                            current_date,
                            current_time,
                            "Present"
                        ])

                    marked_students.add(name)
                    attendance_count += 1

                    STATE = "SUCCESS"

                    recognized_name = name
                    recognized_time = time.time()
                    current_person_locked = name

                    verification_name = None
                    verification_start = None
                    progress = 0

                    notification_title = "Attendance Recorded"
                    notification_message = f"Welcome {name}"
                    notification_color = (0, 255, 0)

                    status = "Attendance Recorded Successfully"

                    threading.Thread(
                        target=speak,
                        args=(f"Welcome {name}. Your attendance has been recorded.",),
                        daemon=True
                    ).start()

                # =====================================
                # ALREADY MARKED
                # =====================================

                else:

                    STATE = "ALREADY_MARKED"

                    recognized_name = name
                    recognized_time = time.time()
                    current_person_locked = name

                    verification_name = None
                    verification_start = None
                    progress = 0

                    notification_title = "Attendance Already Recorded"
                    notification_message = "Welcome Back"
                    notification_color = (0, 255, 0)

                    status = "Attendance Already Recorded"

                    threading.Thread(
                        target=speak,
                        args=(f"Welcome back {name}. Your attendance has already been recorded.",),
                        daemon=True
                    ).start()

                 

        # =====================================================
        # DRAW FACE RECTANGLE
        # =====================================================

        if STATE == "VERIFYING":
            box_color = (0, 255, 255)

        elif STATE == "SUCCESS":
            box_color = (255, 0, 0)

        elif STATE == "ALREADY_MARKED":
            box_color = (0, 255, 0)

        if not hide_rectangle:

            cv2.rectangle(
                frame,
                (x, y),
                (x + w, y + h),
                box_color,
                2
            )

        # -------------------------------
        # Draw Name
        # -------------------------------

        text_y = y - 10

        if text_y < 80:
            text_y = y + h + 25

        cv2.putText(
            frame,
            label,
            (x, text_y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            box_color,
            2
        )

        # -------------------------------
        # Draw Status
        # -------------------------------

        status_y = text_y + 30

        cv2.putText(
            frame,
            status,
            (x, status_y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            box_color,
            2
        )

        # -------------------------------
        # Progress Bar
        # -------------------------------

        if verification_name is not None and progress < 100:

            bar_width = w

            filled = int((progress / 100) * bar_width)

            cv2.rectangle(
                frame,
                (x, y + h + 45),
                (x + bar_width, y + h + 55),
                (180, 180, 180),
                2
            )

            cv2.rectangle(
                frame,
                (x, y + h + 45),
                (x + filled, y + h + 55),
                (0, 255, 0),
                -1
            )

    # =====================================================
    # TOP HEADER
    # =====================================================

    cv2.rectangle(
        frame,
        (0, 0),
        (frame.shape[1], 70),
        (40, 40, 40),
        -1
    )

    cv2.putText(
        frame,
        f"Attendance Today : {attendance_count}",
        (20, 60),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (0, 255, 0),
        2
    )

    # =====================================================
    # DATE & TIME
    # =====================================================

    cv2.putText(
        frame,
        f"Date : {current_date}",
        (frame.shape[1] - 260, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (255, 255, 255),
        2
    )

    cv2.putText(
        frame,
        f"Time : {current_time}",
        (frame.shape[1] - 260, 60),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (255, 255, 255),
        2
    )

    # =====================================================
    # TOP-RIGHT NOTIFICATION
    # =====================================================

    if recognized_name is not None:

        if time.time() - recognized_time < DISPLAY_RESULT_TIME:

            cv2.putText(
                frame,
                notification_title,
                (frame.shape[1] - 380, 95),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                notification_color,
                2
            )

            cv2.putText(
                frame,
                notification_message,
                (frame.shape[1] - 380, 125),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 255, 255),
                2
            )

    # =====================================================
    # FOOTER
    # =====================================================

    cv2.rectangle(
        frame,
        (0, frame.shape[0] - 35),
        (frame.shape[1], frame.shape[0]),
        (40, 40, 40),
        -1
    )

    cv2.putText(
        frame,
        "Press Q to Exit",
        (20, frame.shape[0] - 10),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (0, 255, 255),
        2
    )

    # =====================================================
    # SHOW WINDOW
    # =====================================================

    cv2.imshow(
        "Attendance System",
        frame
    )

    key = cv2.waitKey(1)

    if key & 0xFF == ord("q"):
        break

# =====================================================
# RELEASE
# =====================================================

cap.release()

cv2.destroyAllWindows()

print("---------------------------------------")
print("Attendance System Closed Successfully")
print("---------------------------------------")