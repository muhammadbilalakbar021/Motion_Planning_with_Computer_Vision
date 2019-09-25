import cv2
import numpy as np
import os
import matplotlib.pyplot as plt

currentDirecory = os.getcwd()
print(currentDirecory)

img = cv2.imread("/media/bilal/Work_Space/Robotics_Path_Planning/openCV/test.jpeg", 0)
# kernel = np.ones((5,5),np.uint8)

kernel1 = np.array([[1, 1, 1],
                    [1, 1, 1],
                    [1, 1, 1], ], dtype=np.uint8)

kernel2 = np.array([[1, 1, 1, 1, 1],
                    [1, 1, 1, 1, 1],
                    [1, 1, 1, 1, 1],
                    [1, 1, 1, 1, 1],
                    [1, 1, 1, 1, 1]], dtype=np.uint8)

ret, thresh1 = cv2.threshold(img, 50, 255, cv2.THRESH_BINARY)
erosion = cv2.erode(thresh1, kernel2, iterations=1)
dilation = cv2.dilate(thresh1,kernel2,iterations = 2)
opening = cv2.morphologyEx(dilation, cv2.MORPH_OPEN, kernel1, iterations=4)
closing = cv2.morphologyEx(opening, cv2.MORPH_CLOSE, kernel2, iterations=4)
dilation = cv2.dilate(closing,kernel1,iterations = 1)

cv2.imshow("Removing Noise", dilation)
cv2.waitKey(0)
cv2.destroyAllWindows()



# gradient = cv2.morphologyEx(erosion, cv2.MORPH_GRADIENT, kernel1)
# tophat = cv2.morphologyEx(img, cv2.MORPH_TOPHAT, kernel1, iterations=2)
# blackhat = cv2.morphologyEx(img, cv2.MORPH_BLACKHAT, kernel1)
