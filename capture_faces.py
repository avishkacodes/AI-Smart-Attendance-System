import cv2
import os

# ==========================
# CHANGE STUDENT NAME HERE
# ==========================
student_name = "Ajinkya"
# student_name = "Aarush"

dataset_path = os.path.join("dataset", student_name)
os.makedirs(dataset_path, exist_ok=True)

face_detector = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)

camera = cv2.VideoCapture(0)

count = 0

print("Capturing images...")
print("Look in different directions and change expressions.")
print("Press 'q' to quit.")

while True:

    success, frame = camera.read()

    if not success:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = face_detector.detectMultiScale(
        gray,
        scaleFactor=1.2,
        minNeighbors=5,
        minSize=(120, 120)
    )

    for (x, y, w, h) in faces:

        # Add padding around face
        padding = 30

        x1 = max(0, x - padding)
        y1 = max(0, y - padding)

        x2 = min(frame.shape[1], x + w + padding)
        y2 = min(frame.shape[0], y + h + padding)

        # COLOR face (NOT grayscale)
        face = frame[y1:y2, x1:x2]

        # Resize
        face = cv2.resize(face, (224, 224))

        count += 1

        file_name = os.path.join(dataset_path, f"{count}.jpg")
        cv2.imwrite(file_name, face)

        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

        cv2.putText(
            frame,
            f"{count}/100",
            (10, 35),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0,255,0),
            2
        )

    cv2.imshow("Capture Dataset", frame)

    key = cv2.waitKey(80)

    if key == ord('q'):
        break

    if count >= 100:
        break

camera.release()
cv2.destroyAllWindows()

print(f"\nSuccessfully captured {count} images for {student_name}")