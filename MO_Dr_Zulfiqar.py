import cv2
import numpy as np
import os
import matplotlib.pyplot as plt

currentDirecory = os.getcwd()
print(currentDirecory)

img = cv2.imread("/media/bilal/Work_Space/Robotics_Path_Planning/openCV/Morphological_Images/wirebond.tif", 0)
Opening_img = cv2.imread("/media/bilal/Work_Space/Robotics_Path_Planning/openCV/Morphological_Images/wirebond.tif", 0)
# kernel = np.ones((5,5),np.uint8)

kernel1 = np.array([[1, 1, 1],
                    [1, 1, 1],
                    [1, 1, 1], ], dtype=np.uint8)

ret, thresh1 = cv2.threshold(Opening_img, 50, 255, cv2.THRESH_BINARY)
erosion = cv2.erode(thresh1, kernel1, iterations=20)
dilation = cv2.dilate(thresh1,kernel1,iterations = 0)
opening = cv2.morphologyEx(thresh1, cv2.MORPH_OPEN, kernel1, iterations=4)

cv2.imshow("Erosion", opening)
cv2.waitKey(0)
cv2.destroyAllWindows()