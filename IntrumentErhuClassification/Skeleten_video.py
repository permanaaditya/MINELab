import os
import sys
import shutil
import cv2
import math
import time
import numpy as np
import argparse
import json
import util.Util as Util
from json import JSONEncoder
from shapely.geometry import Point
from shapely.geometry.polygon import Polygon

import tensorflow as tf
from tensorflow.compat.v1.keras.backend import set_session
config = tf.compat.v1.ConfigProto()
config.gpu_options.allow_growth = True
sess = tf.compat.v1.Session(config=config)
set_session(sess)

import mediapipe as mp

from src import model
from src import util
from src.body import Body
from src.hand import Hand

class DataGeneratorClass:

    def __init__(self, args):
        self.DATA_BASE_DIRECTORY = Util.DATA_BASE_DIRECTORY
        self.SHOW_PROCESS = True
        self.VIDEO_DATA = []
        self.LABELS = ["BG", "Bow", "Body", "uBridge", "lBridge"]
        self.MAKE_VIDEO_OUTPUT = True
        self.FPS_VIDEO_RESULT = 15.0
        self.SAVE_DATA = False
        self.HAND_TRACKING_METHOD = "openpose" # mediapipe | openpose

        if hasattr(args, 'dir') and args.dir is not None and args.dir != "":
            self.DATA_BASE_DIRECTORY = args.dir

        if hasattr(args, 'show') and args.show is not None and args.show != "":
            self.SHOW_PROCESS = args.show

        self.mp_drawing = None
        self.mp_hands = None
        self.hands = None

        if self.HAND_TRACKING_METHOD == 'mediapipe' :
            self.mp_drawing = mp.solutions.drawing_utils
            self.mp_hands = mp.solutions.hands
            self.hands = self.mp_hands.Hands(max_num_hands=2, static_image_mode=False, min_detection_confidence=0.5, min_tracking_confidence=0.5)

        self.body_estimation = None
        self.hand_estimation = None

        if self.HAND_TRACKING_METHOD == 'openpose' :
            self.body_estimation = Body('model/body_pose_model.pth')
            self.hand_estimation = Hand('model/hand_pose_model.pth')

        self.build()
        self.run()

    def build(self):
        os.makedirs(self.DATA_BASE_DIRECTORY, exist_ok=True)
        os.makedirs(os.path.join(self.DATA_BASE_DIRECTORY, "output_" + self.HAND_TRACKING_METHOD), exist_ok=True)

    def run(self):
        self.VIDEO_DATA = [file for file in os.listdir(self.DATA_BASE_DIRECTORY) if file.endswith(".mp4")]

        for file in self.VIDEO_DATA :
            fileName = file[0:file.rindex('.')]
            jsonfile = fileName + ".json"
            numpyfile = fileName + ".npy"
            filePath = os.path.join(self.DATA_BASE_DIRECTORY, file)
            numpyFilePath = os.path.join(self.DATA_BASE_DIRECTORY, numpyfile)
            jsonFilePath = os.path.join(self.DATA_BASE_DIRECTORY, jsonfile)

            videoPath = os.path.join(self.DATA_BASE_DIRECTORY, "output_" + self.HAND_TRACKING_METHOD, fileName + ".mp4")
            videoOut = None

            if os.path.exists(jsonfile) and os.path.exists(numpyFilePath):
                print("\"{}\" is already  generated.".format(file))
            else :
                print("\"{}\" is generating".format(file))
                videoData = cv2.VideoCapture(filePath)
                properties = Util.getVideoProperties(videoData)

                counter = 0

                FPS = 0
                FPS_counter = 0
                start_time = time.time()

                while videoData.isOpened():
                    FPS_counter +=1
                    if (time.time() - start_time) > 1.0 :
                        FPS = math.ceil(FPS_counter / (time.time() - start_time))
                        FPS_counter = 0
                        start_time = time.time()

                    res, frame = videoData.read()

                    if not res:
                        if self.MAKE_VIDEO_OUTPUT and videoOut is not None :
                            videoOut.release()
                            videoOut = None
                        break

                    img = frame.copy()
                    videoIMG = np.zeros((img.shape[0], img.shape[1], 3), np.uint8)

                    if self.MAKE_VIDEO_OUTPUT and videoOut is None :
                        videoOut = cv2.VideoWriter(videoPath, cv2.VideoWriter_fourcc(*'MP4V'), self.FPS_VIDEO_RESULT, (img.shape[1], img.shape[0]))

                    keypoints = [[],[]]

                    if self.HAND_TRACKING_METHOD == 'mediapipe' :
                        results = self.hands.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                        if results.multi_hand_landmarks:
                            if len(results.multi_hand_landmarks) > 2 :
                                continue
                            for i, hand_landmarks in enumerate(results.multi_hand_landmarks):
                                for index, data_point in enumerate(hand_landmarks.landmark) :
                                    keypoints[i].append({'x': data_point.x, 'y': data_point.y, 'z': data_point.z})
                                self.mp_drawing.draw_landmarks(img, hand_landmarks, self.mp_hands.HAND_CONNECTIONS)

                                if self.MAKE_VIDEO_OUTPUT :
                                    self.mp_drawing.draw_landmarks(videoIMG, hand_landmarks, self.mp_hands.HAND_CONNECTIONS)

                    if self.HAND_TRACKING_METHOD == 'openpose' :
                        try:
                            candidate, subset = self.body_estimation(img)
                            img = util.draw_bodypose(img, candidate, subset)

                            hands_list = util.handDetect(candidate, subset, img)
                            keypoints = []

                            for x, y, w, is_left in hands_list:
                                peaks = self.hand_estimation(img[y:y+w, x:x+w, :])
                                peaks[:, 0] = np.where(peaks[:, 0]==0, peaks[:, 0], peaks[:, 0]+x)
                                peaks[:, 1] = np.where(peaks[:, 1]==0, peaks[:, 1], peaks[:, 1]+y)
                                keypoints.append(peaks)

                            img = util.draw_handpose(img, keypoints)
                            img = cv2.resize(img, (frame.shape[1], frame.shape[0]))

                            if self.MAKE_VIDEO_OUTPUT :
                                videoIMG = util.draw_bodypose(videoIMG, candidate, subset)
                                videoIMG = util.draw_handpose(videoIMG, keypoints)
                                videoIMG = cv2.resize(videoIMG, (frame.shape[1], frame.shape[0]))
                        except:
                            ''

                    counter += 1
                    Util.PrintPercent(counter, properties['Count'])

                    cv2.putText(videoIMG,"FPS - %d" % (FPS), (5, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0 ,255, 0), 1)

                    if self.SHOW_PROCESS :
                        cv2.imshow("Hand tracking", frame)
                        cv2.imshow("videoIMG", videoIMG)
                        key = cv2.waitKey(1)

                    if self.MAKE_VIDEO_OUTPUT :
                        videoOut.write(videoIMG)

                key = cv2.waitKey(1)
                if key == 27 :
                    break

            cv2.destroyAllWindows()
            print("\n")

if __name__ == '__main__':
    print("# Data generator is starting...\n")

    parser = argparse.ArgumentParser()
    parser.add_argument("--dir", "-d", default="data", type=str, help="Data folder")
    parser.add_argument("--show", "-s", default=True, type=bool, help="Show process")
    args = parser.parse_args()

    DataGeneratorClass(args)

    print("\n# Data generator is finished.")
