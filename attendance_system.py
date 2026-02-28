import os
from datetime import datetime

import cv2
import face_recognition
import numpy as np

# =========================
# Dataset Path
# =========================
path = r"D:\Dataset"  # Change this to your dataset path

images = []
classNames = []

print("Loading images...")

# =========================
# Load Images
# =========================
for root, _, files in os.walk(path):
    for file in files:
        if file.lower().endswith((".jpg", ".jpeg", ".png")):
            img_path = os.path.join(root, file)
            img = cv2.imread(img_path)

            if img is None:
                continue

            name = os.path.basename(root)
            images.append(img)
            classNames.append(name)

print("Students Loaded:", classNames)


# =========================
# Encode Function
# =========================
def findEncodings(imagesList):
    encodeList = []

    for img in imagesList:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        encodes = face_recognition.face_encodings(img)

        if len(encodes) > 0:
            encodeList.append(encodes[0])

    return encodeList


# =========================
# Attendance Function
# =========================
def markAttendance(name):
    fileName = "Attendance.csv"

    if not os.path.exists(fileName):
        with open(fileName, "w", encoding="utf-8") as f:
            f.write("Name,Time,Date")

    with open(fileName, "r+", encoding="utf-8") as f:
        dataList = f.readlines()
        nameList = []

        for line in dataList:
            entry = line.split(",")
            nameList.append(entry[0])

        if name not in nameList:
            now = datetime.now()
            timeString = now.strftime("%H:%M:%S")
            dateString = now.strftime("%d-%m-%Y")
            f.writelines(f"\n{name},{timeString},{dateString}")
            print(f"Attendance Marked for {name}")


# =========================
# Encode Known Faces
# =========================
encodeListKnown = findEncodings(images)
print("Encoding Complete")


# =========================
# Start Webcam
# =========================
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    raise RuntimeError("Unable to open webcam (index 0).")

while True:
    success, frame = cap.read()

    if not success:
        break

    imgSmall = cv2.resize(frame, (0, 0), None, 0.25, 0.25)
    imgSmall = cv2.cvtColor(imgSmall, cv2.COLOR_BGR2RGB)

    facesCurFrame = face_recognition.face_locations(imgSmall)
    encodesCurFrame = face_recognition.face_encodings(imgSmall, facesCurFrame)

    for encodeFace, faceLoc in zip(encodesCurFrame, facesCurFrame):
        if len(encodeListKnown) == 0:
            continue

        matches = face_recognition.compare_faces(encodeListKnown, encodeFace)
        faceDis = face_recognition.face_distance(encodeListKnown, encodeFace)

        matchIndex = np.argmin(faceDis)

        if matches[matchIndex]:
            name = classNames[matchIndex].upper()
            markAttendance(name)
        else:
            name = "UNKNOWN"

        y1, x2, y2, x1 = faceLoc
        y1, x2, y2, x1 = y1 * 4, x2 * 4, y2 * 4, x1 * 4

        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(
            frame,
            name,
            (x1, y1 - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (255, 255, 255),
            2,
        )

    cv2.imshow("Attendance System", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
