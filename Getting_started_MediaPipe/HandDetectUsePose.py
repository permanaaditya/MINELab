import cv2
import mediapipe as mp
import numpy as np
import os
import tensorflow as tf

# print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))
# exit()

mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_pose = mp.solutions.pose

coordinateWrist = []
maxFileName = ''

def getImages(folder):
    for filename in os.listdir(folder):
        img = cv2.imread(os.path.join(folder, filename))
        if img is not None:
            stringPath = folder.split('/')
            parentFile = stringPath[len(stringPath)-1]
            if filename.find(parentFile+'.jpg') != -1:
                print(folder, filename)
                imagesPath.append(os.path.join(folder, filename))
                foldersPath.append(folder)
                images.append(img)
        else:
            getImages(os.path.join(folder, filename))
    return images, imagesPath, foldersPath

def extractSkinHand(images):
  min_YCrCb = np.array([0, 133, 77], np.uint8)
  max_YCrCb = np.array([235, 173, 127], np.uint8)
  image = images
  imageYCrCb = cv2.cvtColor(image, cv2.COLOR_BGR2YCR_CB)
  skinRegionYCrCb = cv2.inRange(imageYCrCb, min_YCrCb, max_YCrCb)
  skinYCrCb = cv2.bitwise_and(image, image, mask=skinRegionYCrCb)
  return skinYCrCb

images = []
imagesPath = []
foldersPath = []
image_folder = 'C:/Users/LEGION/PycharmProjects/pythonProject/InstrumentProject/data/erhu/12/'
images, imagesPath, foldersPath = getImages(image_folder)
print('Images Count:', len(imagesPath))
counter = 0
# For static images:
# IMAGE_FILES = imagesPath
# print(IMAGE_FILES)
# exit()
# IMAGE_FILES = ['C:/Users/LEGION/PycharmProjects/pythonProject/InstrumentProject/data/erhu/6/126/126.jpg']
BG_COLOR = (192, 192, 192) # gray
with mp_pose.Pose(
    static_image_mode=True,
    model_complexity=2,
    enable_segmentation=True,
    min_detection_confidence=0.5) as pose:
    for (imagePath, folderPath) in zip(imagesPath, foldersPath):
        image = cv2.imread(imagePath)
        image_height, image_width, _ = image.shape
        # Convert the BGR image to RGB before processing.
        results = pose.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

        if not results.pose_landmarks:
            continue

        annotated_image = image.copy()
        # Draw segmentation on the image.
        # To improve segmentation around boundaries, consider applying a joint
        # bilateral filter to "results.segmentation_mask" with "image".
        condition = np.stack((results.segmentation_mask,) * 3, axis=-1) > 0.1
        bg_image = np.zeros(image.shape, dtype=np.uint8)
        bg_image[:] = BG_COLOR
        annotated_image = np.where(condition, annotated_image, bg_image)

        # # Draw pose landmarks on the image.
        # mp_drawing.draw_landmarks(
        #     annotated_image,
        #     results.pose_landmarks,
        #     mp_pose.POSE_CONNECTIONS,
        #     landmark_drawing_spec=mp_drawing_styles.get_default_pose_landmarks_style())

        x_R1 = round(results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_WRIST].x * image_width + 60)
        x_R2 = round(results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_WRIST].y * image_height - 60)
        x_R3 = round(results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_WRIST].x * image_width - 60)
        x_R4 = round(results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_WRIST].y * image_height + 60)
        if x_R1 > annotated_image.shape[0] : x_R1 = annotated_image.shape[0]
        if x_R2 < 0 : x_R2 = 0
        if x_R3 < 0 : x_R3 = 0
        if x_R4 > annotated_image.shape[1] : x_R4 = annotated_image.shape[1]
        right_hand = annotated_image[x_R2:x_R4, x_R3:x_R1]
        # right_hand_skin = extractSkinHand(right_hand)
        cv2.imwrite(folderPath+'/right_hand.png', right_hand)

        x_L1 = round(results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_WRIST].x * image_width + 60)
        x_L2 = round(results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_WRIST].y * image_height - 60)
        x_L3 = round(results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_WRIST].x * image_width - 60)
        x_L4 = round(results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_WRIST].y * image_height + 90)
        if x_L1 > annotated_image.shape[0] : x_L1 = annotated_image.shape[0]
        if x_L2 < 0 : x_L2 = 0
        if x_L3 < 0 : x_L3 = 0
        if x_L4 > annotated_image.shape[1] : x_L4 = annotated_image.shape[1]
        left_hand = annotated_image[x_L2:x_L4, x_L3:x_L1]
        # left_hand_skin = extractSkinHand(left_hand)
        cv2.imwrite(folderPath+'/left_hand.png', left_hand)
        print(folderPath, ' extract successfully!')
        counter+=1
        # cv2.imwrite('output_pose' + str(idx) + '.png', annotated_image)
print('Total images:', counter)
