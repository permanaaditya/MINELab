import numpy as np
import cv2

img = cv2.imread('lena.jpg', 1)
img = cv2.line(img, (0, 0), (255, 255), (0, 0, 255), 10)
img = cv2.arrowedLine(img, (0, 0), (255, 255), (255, 0, 0), 10)
img = cv2.rectangle(img, (350, 0), (510, 120), (0, 0, 255), 10)
img = cv2.circle(img, (400,60), 80, (0, 255, 0), 10)
img = cv2.putText(img, 'Learning OpenCV', (10,100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 1, cv2.LINE_AA)

cv2.imshow('image', img)

cv2.waitKey(0)
cv2.destroyAllWindows()
