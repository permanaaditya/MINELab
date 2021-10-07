import cv2
import os
import numpy as np

face_cascade_file = 'Cascade Classifier/face_detect.xml'
face_cascade = cv2.CascadeClassifier(face_cascade_file)
known_names = ['Aditya Permana']  # List untuk nama yang ada di model

cam = cv2.VideoCapture(0)  # Akses Kamera
while True:
    ret, frame = cam.read()  # Membaca setiap frame dari stream kamera
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)  # Mengubah mode BGR ke GRAY (hitam putih)

    # Proses pencarian wajah
    faces = face_cascade.detectMultiScale(gray, 1.3, 3)  # <cascade_file>.detectMultiScale(<frame>, <scale_factor>, <min_neighbors>)
    for x, y, w, h in faces:  # Looping semua wajah yang terdeteksi
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 3)  # Gambar box untuk setiap wajah

        roi_gray = gray[y:y + h, x:x + w]
        ids, dist = recognizer.predict(roi_gray)  # Prediksi wajah siapoa
        cv2.putText(frame, f'{known_names[ids - 1]} {round(dist, 2)}',
                    (x - 20, y - 20), font, 1, (255, 255, 0), 3)  # Menaruh text pada frame

    cv2.imshow('Face Recognition Video', frame)  # Jendela untuk menampilkan hasil

    if cv2.waitKey(1) & 0xff == ord('x'):  # Exit dengan tombol x
        break

cam.release()  # Menyudahi akses kamera
cv2.destroyAllWindows()  # Menutup jendela
cam.release()