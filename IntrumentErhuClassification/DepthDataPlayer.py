import os
import sys
import shutil
import cv2
import math
import time
import numpy as np
import util.Util as Util

fileName = "1628563139.242836_depth_2.mp4.npz"
DATA_PATH = os.path.join(Util.DATA_BASE_DIRECTORY, Util.DATA_NEW_VIDEO_DIRECTORY, fileName)

if not os.path.exists(DATA_PATH) :
    print("File is not found!")
    exit()

depthData = np.load(DATA_PATH)["arr_0"]

for data in depthData :
    cv2.imshow("Depth data", data)

    key = cv2.waitKey(30)
    if key == 27 :
        break
