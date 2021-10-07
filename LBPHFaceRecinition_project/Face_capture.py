import cv2
import os
import numpy as np
face_cascade_file = 'Cascade Classifier/face_detect.xml'
face_cascade = cv2.CascadeClassifier(face_cascade_file)

folder_name = 'faces data'
total_images = 5
counter = 1
ids = 1

cam = cv2.VideoCapture(0)  # Akses Kamera
while True:
    ret, frame = cam.read()
    frame_copy = frame.copy()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Proses pencarian wajah
    faces = face_cascade.detectMultiScale(gray, 1.3, 3)
    for x, y, w, h in faces:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 3)

        if cv2.waitKey(1) & 0xff == ord('c'):
            roi_face = frame_copy[y:y + h, x:x + w]
            cv2.imwrite(f'{folder_name}/people.{ids}.{counter}.jpg', roi_face)
            counter += 1
            if counter > total_images:
                print(f'[INFO] {total_images} IMAGE CAPTURED!')
    cv2.imshow('Face Detect Video', frame)
    if cv2.waitKey(1) & 0xff == ord('x'):
        break

cam.release()
cv2.destroyAllWindows()