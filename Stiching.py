import cv2
import numpy as np
import imutils
import os

image_files = ["first.png", "second.png", "third.png"]
images = []

for file in image_files:
    if not os.path.exists(file):
        raise FileNotFoundError(f"{file} not found")

    img = cv2.imread(file)
    if img is None:
        raise Exception(f"Could not read {file}")

    images.append(img)
    cv2.imshow("Image", img)
    cv2.waitKey(0)

try:
    stitcher = cv2.Stitcher_create()
except:
    stitcher = cv2.createStitcher(False)

error, stitched_img = stitcher.stitch(images)

if error != cv2.Stitcher_OK:
    print("Images could not be stitched!")
    print("Error code:", error)
    exit()

cv2.imwrite("stitchedOutput.png", stitched_img)
cv2.imshow("Stitched Image", stitched_img)
cv2.waitKey(0)

stitched_img = cv2.copyMakeBorder(
    stitched_img, 10, 10, 10, 10,
    cv2.BORDER_CONSTANT, (0, 0, 0)
)

gray = cv2.cvtColor(stitched_img, cv2.COLOR_BGR2GRAY)
thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY)[1]

cv2.imshow("Threshold Image", thresh)
cv2.waitKey(0)

contours = cv2.findContours(
    thresh.copy(),
    cv2.RETR_EXTERNAL,
    cv2.CHAIN_APPROX_SIMPLE
)
contours = imutils.grab_contours(contours)

areaOI = max(contours, key=cv2.contourArea)

mask = np.zeros(thresh.shape, dtype="uint8")
x, y, w, h = cv2.boundingRect(areaOI)
cv2.rectangle(mask, (x, y), (x + w, y + h), 255, -1)

minRect = mask.copy()
sub = mask.copy()

while cv2.countNonZero(sub) > 0:
    minRect = cv2.erode(minRect, None)
    sub = cv2.subtract(minRect, thresh)

contours = cv2.findContours(
    minRect.copy(),
    cv2.RETR_EXTERNAL,
    cv2.CHAIN_APPROX_SIMPLE
)
contours = imutils.grab_contours(contours)

areaOI = max(contours, key=cv2.contourArea)
x, y, w, h = cv2.boundingRect(areaOI)

stitched_img = stitched_img[y:y + h, x:x + w]

cv2.imwrite("stitchedOutputProcessed.png", stitched_img)
cv2.imshow("Stitched Image Processed", stitched_img)
cv2.waitKey(0)
cv2.destroyAllWindows()