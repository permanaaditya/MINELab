import cv2 as cv
import numpy as np

img = cv.imread('lena.jpg')
layer = img.copy()
gp = [layer]

for i in range(6):
    layer = cv.pyrDown(layer)
    gp.append(layer)
    #cv.imshow(str(i), layer)

#lower = cv.pyrDown(img)
#lower2 = cv.pyrDown(lower)
#upper = cv.pyrUp(lower2)

layer = gp[5]
cv.imshow('Upper level Gaussian Pyramid', layer)
lp = [layer]

for i in range(5, 0, -1):
    gauss_extend = cv.pyrUp(gp[i])
    laplacian = cv.subtract(gp[i-1], gauss_extend)
    cv.imshow(str(i), laplacian)

cv.imshow('Original Image', img)
#cv.imshow('Lower 1 Image', lower)
#cv.imshow('Lower 2 Image', lower2)
#cv.imshow('Upper 2 Image', upper)

cv.waitKey(0)
cv.destroyAllWindows()