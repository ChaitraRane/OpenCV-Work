import numpy as np
import cv2 as cv

img = cv.imread ('randum.png')

gray = cv.cvtColor (img, cv.COLOR_BGR2GRAY)
corners = cv.goodFeaturesToTrack(gray, 100, 0.01, 10)
corners = np.intp(corners)

for i in corners :
    x,y = i.ravel()
    cv.circle (img, (x,y), 3, 255, -1)

cv.imshow('dst', img)

if cv.waitKey(0) & 0xff == 27:
    cv.destroyAllWindows()