import cv2 as cv
from matplotlib import pyplot as plt

img = cv.imread('gradient_image.jfif')
img_adp = cv.imread('sudoku.jpg', 0)
#=================== Threshold ===========================
_, th1 = cv.threshold(img, 100, 255, cv.THRESH_BINARY)
_, th2 = cv.threshold(img, 100, 255, cv.THRESH_BINARY_INV)
_, th3 = cv.threshold(img, 100, 255, cv.THRESH_TRUNC)
_, th4 = cv.threshold(img, 100, 255, cv.THRESH_TOZERO)
_, th5 = cv.threshold(img, 100, 255, cv.THRESH_TOZERO_INV)
#=================== Adaptive Threshold ==================
_, th6 = cv.threshold(img_adp, 100, 255, cv.THRESH_BINARY)
th_adp = cv.adaptiveThreshold(img_adp, 255, cv.ADAPTIVE_THRESH_MEAN_C, cv.THRESH_BINARY, 11, 2);
titles = ['Original', 'BINARY', 'BINARY_INV', 'TRUNC', 'TO_ZERO', 'TO_ZERO_INV']
images = [img, th1, th2, th3, th4, th5]
cv.imshow("Original Image", img_adp)
cv.imshow("th6", th6)
cv.imshow("th_adp", th_adp)
for i in range(6):
    plt.subplot(2, 3, i+1)
    plt.imshow(images[i], 'gray')
    plt.title(titles[i])
    plt.xticks([])
    plt.yticks([])
plt.show()
cv.waitKey(0)
cv.destroyAllWindows()