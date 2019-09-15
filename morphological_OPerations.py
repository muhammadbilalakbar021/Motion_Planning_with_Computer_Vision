import cv2
import numpy as np
import os
import matplotlib.pyplot as plt

currentDirecory = os.getcwd()
print(currentDirecory)

img=cv2.imread("/media/bilal/Work_Space/Robotics_Path_Planning/openCV/test.jpeg",0)
kernel = np.ones((5,5),np.uint8)

# Scaling and resizing image
erosion = cv2.erode(img,kernel,iterations = 1)


dilation = cv2.dilate(img,kernel,iterations = 1)
opening = cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel)
closing = cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel)
gradient = cv2.morphologyEx(img, cv2.MORPH_GRADIENT, kernel)
tophat = cv2.morphologyEx(img, cv2.MORPH_TOPHAT, kernel)
blackhat = cv2.morphologyEx(img, cv2.MORPH_BLACKHAT, kernel)


cv2.imshow("Erosion",erosion)
cv2.waitKey(0)
cv2.destroyAllWindows()