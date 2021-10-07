import cv2
import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt

img = cv.imread('lena.jpg')
img = cv.cvtColor(img, cv2.COLOR_BGR2RGB)

kernel = np.ones((5, 5), np.float32)/25
dst = cv.filter2D(img, -1, kernel)
blur = cv.blur(img, (5, 5))
gauss = cv.GaussianBlur(img, (5, 5), 0)
median = cv.medianBlur(img, 5)
bilateralFilter = cv.bilateralFilter(img, 9, 75, 75)

titles = ['Original', '2D Convolution', 'Blur', 'GaussBlur', 'Median', 'BilateralFilter']
images = [img, dst, blur, gauss, median, bilateralFilter]

for i in range(6):
    plt.subplot(2, 3, i+1)
    plt.imshow(images[i], 'gray')
    plt.title(titles[i])
    plt.xticks([])
    plt.yticks([])

plt.show()
