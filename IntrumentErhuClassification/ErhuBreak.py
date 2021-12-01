import numpy as np
import cv2
import shutil, os
import re
from pixellib.instance import custom_segmentation


def break_erhu(folder):
    imagesFileName = []
    imagesPath = []
    minImage = None
    counter = 0
    for filename in os.listdir(folder):
        # if filename.find("hand")==-1:
        img = cv2.imread(os.path.join(folder, filename))
        # sub_folders = [name for name in os.listdir(filename) if os.path.isdir(os.path.join(filename, name))]
        # print(filename)
        if img is not None:
            counter += 1
            # imagesFileName.append(filename)
            # imagesPath.append(os.path.join(folder, filename))
            if minImage is None:
                minImage = img
            else:
                if minImage.shape[1] > img.shape[1]:
                    # print('minImage:' + str(minImage.shape[1]) + ' - CurrImg:' + str(img.shape[1]))
                    if filename.find('hand') == -1 & filename.find('erhu') == -1 :
                        minImage = img
            if counter == 8 :
                h = minImage.shape[0]
                w = minImage.shape[1]
                topErhu = minImage[:h//2, :w]
                # crop depend on contours
                gray = cv2.cvtColor(topErhu, cv2.COLOR_BGR2GRAY)
                cnts = cv2.findContours(gray, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                cnts = cnts[0] if len(cnts) == 2 else cnts[1]
                cnts = sorted(cnts, key=cv2.contourArea, reverse=True)
                # Find bounding box and extract ROI
                for c in cnts:
                    x, y, w, h = cv2.boundingRect(c)
                    ROI = topErhu[y:y + h, x:x + w]
                    break
                cv2.imwrite(os.path.join(folder, 'erhuTop.jpg'), ROI)
                print('Write ' + filename + ' successfull at : ' + os.path.join(folder, 'erhuTop.jpg'))
        else:
            break_erhu(os.path.join(folder, filename))
    return imagesFileName, imagesPath

folder = 'C:/Users/LEGION/PycharmProjects/pythonProject/InstrumentProject/data/erhu/13/'
# crop_erhu(folder)
listImagesFileName, listPath = break_erhu(folder)
# print(listPath)
