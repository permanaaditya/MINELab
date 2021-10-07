import cv2
import os
import numpy as np

# CAPTURE THE FACE
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
                break
    cv2.imshow('Face Detect Video', frame)
    # if cv2.waitKey(1) & 0xff == ord('x'):
    #    break

cam.release()
cv2.destroyAllWindows()

# TRAIN THE DATA
recognizer = cv2.face.LBPHFaceRecognizer_create()
images = os.listdir(folder_name)  # list semua path data wajah pada folder train data

image_arrays = []  # Containes semua array data wajah
image_ids = []  # Container semua ID data wajah
for image_path in images:  # Looping semua path data wajah
    splitted_path = image_path.split('.')
    print(splitted_path)
    image_id = int(splitted_path[1])  # Ambil ID data wajah

    image = cv2.imread(os.path.join(folder_name, image_path))
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    image_array = np.array(image, 'uint8')  # Ambil array data wajah

    image_arrays.append(image_array)  # Store array data wajah ke list/container
    image_ids.append(image_id)  # Store ID data wajah ke list/container

recognizer.train(image_arrays, np.array(image_ids))                 # Train recognizer
recognizer.save('recognizer/faces_data.yml')                        # Save model recognizer
print('[INFO] TRAIN RECOGNIZER SUCCESS!')

# TEST THE FACE RECOGNITION
recognizer.read('recognizer/faces_data.yml')                        # Load recognizer
font = cv2.FONT_HERSHEY_SIMPLEX                                     # Specify jenis font dari OpenCV
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
        cv2.putText(frame, f'{known_names[ids - 1]} {round(dist, 2)}', (x - 20, y - 20), font, 1, (255, 255, 0), 3)  # Menaruh text pada frame

    cv2.imshow('Face Recognition Video', frame)  # Jendela untuk menampilkan hasil

    if cv2.waitKey(1) & 0xff == ord('x'):  # Exit dengan tombol x
        break

cam.release()  # Menyudahi akses kamera
cv2.destroyAllWindows()  # Menutup jendela
cam.release()