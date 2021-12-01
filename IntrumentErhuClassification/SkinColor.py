# Required modules
import cv2
import numpy as np
import imutils
from PIL import Image
import matplotlib.pyplot as plt

min_YCrCb = np.array([0, 133, 77], np.uint8)
max_YCrCb = np.array([235, 173, 127], np.uint8)

# Get pointer to video frames from primary device
image = cv2.imread("image/21_02.jpg")
imageYCrCb = cv2.cvtColor(image, cv2.COLOR_BGR2YCR_CB)
skinRegionYCrCb = cv2.inRange(imageYCrCb, min_YCrCb, max_YCrCb)
skinYCrCb = cv2.bitwise_and(image, image, mask=skinRegionYCrCb)
print(skinYCrCb.shape)
cv2.imwrite("image/21_02_out.png", skinYCrCb)
img = cv2.imread('image/21_02_out.png')

gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
blurred = cv2.GaussianBlur(gray, (5, 5), 0)
thresh = cv2.threshold(blurred, 60, 255, cv2.THRESH_BINARY)[1]

# find contours in the thresholded image
cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
cnts = imutils.grab_contours(cnts)

# loop over the contours
for c in cnts:
    # compute the center of the contour
    M = cv2.moments(c)

    if int(M['m10']) != 0 :
        cX = int(M["m10"] / M["m00"])
        cY = int(M["m01"] / M["m00"])
    else:
        continue
    # draw the contour and center of the shape on the image
    cv2.drawContours(img, [c], -1, (0, 255, 0), 0)
    # cv2.circle(image, (cX, cY), 5, (255, 255, 255), -1)
    # cv2.putText(image, "center", (cX - 20, cY - 20),
    # cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

# Display the source image
cv2.imshow('images', img)

# waits for user to press any key
# (this is necessary to avoid Python kernel form crashing)
cv2.waitKey(0)

# closing all open windows
cv2.destroyAllWindows()
