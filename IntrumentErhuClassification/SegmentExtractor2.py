import numpy as np
import cv2
import shutil, os
import re
from pixellib.instance import custom_segmentation


def load_images_from_folder(folder):
    images = []
    imagesPath = []
    for filename in os.listdir(folder):
        img = cv2.imread(os.path.join(folder,filename))
        if img is not None:
            images.append(img)
            imagesPath.append(os.path.join(folder, filename))
    return images,imagesPath

path = 'C:/Users/LEGION/PycharmProjects/pythonProject/InstrumentProject/data/erhu/12/'
segment_image = custom_segmentation()
segment_image.inferConfig(num_classes= 3, class_names= ["body", "bow", "erhu"])
segment_image.load_model("mask_rcnn_model.020-0.141293.h5")
listImages, listPath = load_images_from_folder(path)
# imageCount = 0
for imagePath in listPath:
    arrStrPath = re.findall(r'\w+', imagePath)
    strCount = len(arrStrPath)
    filename = arrStrPath[strCount-2]
    newFolder = os.path.join(path, filename)
    os.mkdir(newFolder)
    print("Filename : "+filename+" is processed")
    seg, output = segment_image.segmentImage(imagePath, extract_segmented_objects=True, save_extracted_objects =False, output_image_name= newFolder + "/seg_" + filename+"_out.jpg")
    res = seg["extracted_objects"]
    imgidx = 0
    amax = None
    for a in res:
        imgidx+=1
        if amax is None :
            amax = a
        else:
            if (amax.shape[0]+amax.shape[1]) < (a.shape[0]+a.shape[1]) :
                amax = a
        print("Index:"+str(imgidx))
        print(a.shape)
        cv2.imwrite(os.path.join(newFolder, "seg_" + filename + "_" + str(imgidx) + "_.jpg"), a)

    min_YCrCb = np.array([0, 133, 77], np.uint8)
    max_YCrCb = np.array([235, 173, 127], np.uint8)
    imageYCrCb = cv2.cvtColor(amax, cv2.COLOR_BGR2YCR_CB)
    skinRegionYCrCb = cv2.inRange(imageYCrCb, min_YCrCb, max_YCrCb)
    skinYCrCb = cv2.bitwise_and(amax, amax, mask=skinRegionYCrCb)

    h = skinYCrCb.shape[0]
    w = skinYCrCb.shape[1]
    print("Width:"+str(w))
    print("Heigth:"+str(h))
    crop_img_left = skinYCrCb[h//2:, :w//2]
    crop_img_right = skinYCrCb[h//3:h-(h//4), (w//2):]

    cv2.imwrite(os.path.join(newFolder, "seg_" + filename + '_handL.jpg'), crop_img_left)
    cv2.imwrite(os.path.join(newFolder, "seg_" + filename + '_handR.jpg'), crop_img_right)
    # cv2.imshow("image", crop_img_left)
    # cv2.waitKey(0)
    w, h = 0, 0
    cv2.destroyAllWindows()
    shutil.move(imagePath, newFolder)

    # imageCount += 1
    # if imageCount == 4 :
    #     break
