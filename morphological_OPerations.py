import cv2
import numpy
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

ret,thresh1 = cv2.threshold(img,50,255,cv2.THRESH_BINARY)
erosion = cv2.erode(thresh1, kernel1, iterations=1)
dilation = cv2.dilate(thresh1,kernel2,iterations = 2)
opening = cv2.morphologyEx(dilation, cv2.MORPH_OPEN, kernel1, iterations=2)
closing = cv2.morphologyEx(opening, cv2.MORPH_CLOSE, kernel2, iterations=2)
dilation1 = cv2.dilate(closing,kernel1,iterations = 1)
triangle = numpy.array([[0, 0], [200, 0], [0, 300]])
#triangle1= numpy.array([[102, 0], [0, 110], [300, 102]])
color = [255, 255, 255]
imgRemoving=cv2.fillConvexPoly(dilation1, triangle, color)
#imgRemoving=cv2.fillConvexPoly(dilation1, triangle, color)


cv2.imshow("Removing Noise", imgRemoving)
cv2.waitKey(0)
cv2.destroyAllWindows()



# gradient = cv2.morphologyEx(erosion, cv2.MORPH_GRADIENT, kernel1)
# tophat = cv2.morphologyEx(img, cv2.MORPH_TOPHAT, kernel1, iterations=2)
# blackhat = cv2.morphologyEx(img, cv2.MORPH_BLACKHAT, kernel1)
