import cv2
import numpy as np

from LOL8 import contours

img = cv2.imread ('opencv-logo.png')
imgrey = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
ret, thresh = cv2.threshold(imgrey, 127, 255, 0)
contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
cv2.imshow('Image', img)
cv2.imshow('Image GRAY',imgrey)
cv2.waitKey()
cv2.destroyAllWindows()