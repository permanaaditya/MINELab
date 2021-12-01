import os
import sys
import shutil
import cv2
import math
import time
import numpy as np
import argparse
import util.Util as Util
import pyaudio
import wave
import threading
import time
import subprocess

DATA_DIR = "data/new_video"
VIDEO_INPUT_1 = os.path.join(DATA_DIR, "1630047263.6414664_cam_1.mp4")
VIDEO_INPUT_2 = os.path.join(DATA_DIR, "1630047263.6414664_cam_2.mp4")
VIDEO_INPUT_3 = os.path.join(DATA_DIR, "1630047263.6414664_depth_2.mp4")
VIDEO_OUTPUT = os.path.join(DATA_DIR, "output_tone.mp4")

videoObj_1 = cv2.VideoCapture(VIDEO_INPUT_1)
videoObj_2 = cv2.VideoCapture(VIDEO_INPUT_2)
videoObj_3 = cv2.VideoCapture(VIDEO_INPUT_3)
videoOutObj = None

properties = Util.getVideoProperties(cv2.VideoCapture(VIDEO_INPUT_1))

while videoObj_1.isOpened() and videoObj_2.isOpened() :
    res_1, frames_1 = videoObj_1.read()
    res_2, frames_2 = videoObj_2.read()
    res_3, frames_3 = videoObj_3.read()

    if not res_1 or not res_2  or not res_3 :
        break

    rgb_depth = np.concatenate((frames_2, frames_3), axis=1)

    h1, w1 = frames_1.shape[:2]
    h2, w2 = rgb_depth.shape[:2]

    image = np.zeros((h1 + h2, w2, 3), np.uint8)

    image[: h1, int(w1/2) : int(w1/2) + w1] = frames_1
    image[h1 : h1 + h2, : w2] = rgb_depth

    if videoOutObj is None :
        videoOutObj = cv2.VideoWriter(VIDEO_OUTPUT, cv2.VideoWriter_fourcc(*'MP4V'), properties['FPS'], (image.shape[1], image.shape[0]))
    videoOutObj.write(image)

    cv2.imshow("Output", image)

    key = cv2.waitKey(1)
    if key == 27 :
        break

videoObj_1.release()
videoObj_2.release()
videoObj_3.release()
videoOutObj.release()
