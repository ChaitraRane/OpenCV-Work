import cv2
import numpy as np

apple = cv2.imread('apple.png')
orange = cv2.imread('orange.png')
print(apple.shape)
print(orange.shape)
apple_orange = np.hstack((apple[:, :180], orange[:, 180:]))

# generate Gaussian pyramid
apple_copy = apple.copy()
gp_apple = [apple_copy]
for i in range(6):
    apple_copy = cv2.pyrDown(apple_copy)
    gp_apple.append(apple_copy)

orange_copy = orange.copy()
gp_orange = [orange_copy]
for i in range(6):
    orange_copy = cv2.pyrDown(orange_copy)
    gp_orange.append(orange_copy)

# generate Laplacian
apple_copy = gp_apple[5]
lp_apple = [apple_copy]
for i in range(5, 0, -1):
    gaussian_expanded = cv2.pyrUp(gp_apple[i])
    gaussian_expanded = cv2.resize(gaussian_expanded, (gp_apple[i-1].shape[1], gp_apple[i-1].shape[0]))
    laplacian = cv2.subtract(gp_apple[i-1], gaussian_expanded)
    lp_apple.append(laplacian)

orange_copy = gp_orange[5]
lp_orange = [orange_copy]
for i in range(5, 0, -1):
    gaussian_expanded = cv2.pyrUp(gp_orange[i])
    gaussian_expanded = cv2.resize(gaussian_expanded, (gp_orange[i-1].shape[1], gp_orange[i-1].shape[0]))
    laplacian = cv2.subtract(gp_orange[i-1], gaussian_expanded)
    lp_orange.append(laplacian)

# blend pyramids
apple_orange_pyramid = []
for apple_lap, orange_lap in zip(lp_apple, lp_orange):
    cols, rows, ch = apple_lap.shape
    laplacian = np.hstack((apple_lap[:, 0:int(cols / 2)], orange_lap[:, int(cols / 2):]))
    apple_orange_pyramid.append(laplacian)

# reconstruct image
apple_orange_reconstruct = apple_orange_pyramid[0]
for i in range(1, 6):
    apple_orange_reconstruct = cv2.pyrUp(apple_orange_reconstruct)
    apple_orange_reconstruct = cv2.resize(apple_orange_reconstruct,
                                          (apple_orange_pyramid[i].shape[1], apple_orange_pyramid[i].shape[0]))
    apple_orange_reconstruct = cv2.add(apple_orange_pyramid[i], apple_orange_reconstruct)

cv2.imshow("apple", apple)
cv2.imshow("orange", orange)
cv2.imshow("apple_orange", apple_orange)
cv2.imshow("apple_orange_reconstruct", apple_orange_reconstruct)
cv2.waitKey(0)
cv2.destroyAllWindows()
