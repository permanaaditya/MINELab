import cv2
import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt

img = cv.imread('smarties.jfif', cv2.IMREAD_GRAYSCALE)
_, mask = cv2.threshold(img, 220, 255, cv2.THRESH_BINARY_INV)

kernel = np.ones((5, 5), np.uint8)
dilation = cv2.dilate(mask, kernel, iterations=2)
erosion = cv2.erode(mask, kernel, iterations=1)
opening = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
closing = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
morgrad = cv2.morphologyEx(mask, cv2.MORPH_GRADIENT, kernel)
tophat = cv2.morphologyEx(mask, cv2.MORPH_TOPHAT, kernel)

titles = ['Original', 'Mask', 'Dilation', 'Erosion', 'Opening', 'Close', 'MorphGrad', 'TopHat']
images = [img, mask, dilation, erosion, opening, closing, morgrad, tophat]

for i in range(8):
    plt.subplot(3, 3, i+1)
    plt.imshow(images[i], 'gray')
    plt.title(titles[i])
    plt.xticks([])
    plt.yticks([])

plt.show()
# cv.imshow("Original Image", img_adp)
# cv.imshow("th6", th6)
# cv.imshow("th_adp", th_adp)
# cv.imshow("th3", th3)
# cv.imshow("th4", th4)
# cv.imshow("th5", th5)

# cv.waitKey(0)
# cv.destroyAllWindows()