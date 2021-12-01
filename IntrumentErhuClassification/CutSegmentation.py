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

import mediapipe as mp

import pixellib
from pixellib.instance import custom_segmentation

videoInput = cv2.VideoCapture("Erhu Video Segmentation 640 480 15.mp4")
properties = Util.getVideoProperties(videoInput)

LABELS = ["bow", "erhu", "body"]

segment_image = custom_segmentation()
segment_image.inferConfig(num_classes=3, class_names=LABELS)
segment_image.load_model("train/mask_rcnn_models/mask_rcnn_model.097-0.102301.h5")

videoOut_1 = None
videoSegmentation = {}

while videoInput.isOpened() :
    success, frame = videoInput.read()

    if not success :
        break

    image = frame.copy()

    try :

        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        blank = np.full((image.shape[1], image.shape[0], 3), 255, np.uint8)

        if videoOut_1 is None :
            videoOut_1 = cv2.VideoWriter("result/humanbody.mp4", cv2.VideoWriter_fourcc(*'MP4V'), properties['FPS'], (blank.shape[1], blank.shape[0]))
        videoOut_1.write(blank)


        seg_mask, seg_output = segment_image.segmentFrame(frame.copy())

        cv2.imshow("Segment", seg_output)

        segLeng = len(seg_mask['scores'])
        for i in range(segLeng):
            mask = frame.copy()
            id = seg_mask['class_ids'][i]
            label = LABELS[int(id)-1]

            if label not in videoSegmentation :
                videoSegmentation[label] = cv2.VideoWriter("result/seg_{}.mp4".format(label), cv2.VideoWriter_fourcc(*'MP4V'), properties['FPS'], (mask.shape[1], mask.shape[0]))

            if mask.shape[0] == seg_mask['masks'].shape[0] and mask.shape[1] == seg_mask['masks'].shape[1] :
                mask[seg_mask['masks'][:,:,i] == False] = (255, 255, 255)
                videoSegmentation[label].write(mask)
                #cv2.imshow(label, mask)


        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        cv2.imshow("Video", image)
        if cv2.waitKey(5) & 0xFF == 27:
            break

    except :
        print("Something is wrong...")

videoInput.release()
videoOut_1.release()
for label in videoSegmentation :
    videoSegmentation[label].release()