import cv2
import numpy as np
from matplotlib import pyplot as plt
import matplotlib.image as mpimg

# Configuration Space 1
Configuration_Image_1 = cv2.imread("/media/bilal/Work_Space/Robotics_Path_Planning/openCV/Configuration_Space/Test1.jpg")
#img to NumPy
#numpyImg=mpimg.imread(img)
Configuration_Image_1 = cv2.cvtColor(Configuration_Image_1, cv2.COLOR_BGR2HSV)

# Global THresholding
ret, Configuration_Image_1_Thresh = cv2.threshold(Configuration_Image_1, 128, 200, cv2.THRESH_BINARY)
plt.subplot(2,1,1)
plt.title("Binary Threshing")
plt.imshow(Configuration_Image_1_Thresh)
plt.subplot(2,1,2)
plt.title("Binary Histogram")
values=Configuration_Image_1.mean(axis=2).flatten()
plt.hist(values,255)
plt.ylabel('No. of Pixels')
plt.xlabel('Pixel Values')
plt.show()

blue1 = np.array([10, 100, 20])
blue2 = np.array([25, 255, 255])

Configuration_Image_1_Mask = cv2.inRange(Configuration_Image_1, blue1, blue2)
plt.plot()
plt.title("Mask Threshing")
plt.imshow(Configuration_Image_1_Mask)
plt.show()

# Otsu Threshold
print("In Otsu Thresh")
print("IMAGE type",Configuration_Image_1.shape)
print("IMAGE DataType",Configuration_Image_1.dtype)
OtsuThresh = cv2.cvtColor(Configuration_Image_1, cv2.COLOR_BGR2GRAY)
retOtsu, otsuThres=cv2.threshold(OtsuThresh, 10, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
plt.plot()
plt.title("Otsu Thresh")
plt.imshow(OtsuThresh)
plt.show()

Global_Kernal_3by3 = np.array([[1, 1, 1],
                    [1, 1, 1],
                    [1, 1, 1], ], dtype=np.uint8)
erosion = cv2.erode(otsuThres, Global_Kernal_3by3, iterations=1)
dilation = cv2.dilate(erosion, Global_Kernal_3by3, iterations=3)
toBeShown=cv2.resize(dilation,(400,400))
cv2.imshow("Removing Noise", toBeShown)
cv2.waitKey(0)
cv2.destroyAllWindows()




















































































# Configuration Space 2
# Configuration_Image_2 = cv2.imread("/media/bilal/Work_Space/Robotics_Path_Planning/openCV/Configuration_Space/Test2.jpg")
# Configuration_Image_2 = cv2.cvtColor(Configuration_Image_2, cv2.COLOR_BGR2HSV)
#
#
# ret, Configuration_Image_2_Thresh = cv2.threshold(Configuration_Image_2, 128, 200, cv2.THRESH_BINARY)
# plt.subplot(2,1,1)
# plt.title("Binary Threshing")
# plt.imshow(Configuration_Image_2_Thresh)
# plt.subplot(2,1,2)
# plt.title("Binary Histogram")
# values=Configuration_Image_2.mean(axis=2).flatten()
# plt.hist(values,255)
# plt.ylabel('No. of Pixels')
# plt.xlabel('Pixel Values')
# plt.show()
#
# blue1 = np.array([10, 100, 20])
# blue2 = np.array([25, 255, 255])
#
# Configuration_Image_2_Mask = cv2.inRange(Configuration_Image_2, blue1, blue2)
# plt.plot()
# plt.title("Mask Threshing")
# plt.imshow(Configuration_Image_2_Mask)
# plt.show()
#
# Global_Kernal_3by3 = np.array([[1, 1, 1],
#                     [1, 1, 1],
#                     [1, 1, 1], ], dtype=np.uint8)
# erosion = cv2.erode(Configuration_Image_2_Mask, Global_Kernal_3by3, iterations=1)
# toBeShown=cv2.resize(erosion,(400,400))
# cv2.imshow("Removing Noise", toBeShown)
# cv2.waitKey(0)
# cv2.destroyAllWindows()
#
# # Configuration Space 3
# Configuration_Image_3 = cv2.imread("/media/bilal/Work_Space/Robotics_Path_Planning/openCV/Configuration_Space/Test3.jpg")
# Configuration_Image_3 = cv2.cvtColor(Configuration_Image_3, cv2.COLOR_BGR2HSV)
#
#
# ret, Configuration_Image_3_Thresh = cv2.threshold(Configuration_Image_3, 128, 200, cv2.THRESH_BINARY)
# plt.subplot(2,1,1)
# plt.title("Binary Threshing")
# plt.imshow(Configuration_Image_3_Thresh)
# plt.subplot(2,1,2)
# plt.title("Binary Histogram")
# values=Configuration_Image_3.mean(axis=2).flatten()
# plt.hist(values,255)
# plt.ylabel('No. of Pixels')
# plt.xlabel('Pixel Values')
# plt.show()
#
# blue1 = np.array([10, 100, 20])
# blue2 = np.array([25, 255, 255])
#
# Configuration_Image_3_Mask = cv2.inRange(Configuration_Image_3, blue1, blue2)
# plt.plot()
# plt.title("Mask Threshing")
# plt.imshow(Configuration_Image_3_Mask)
# plt.show()
#
# Global_Kernal_3by3 = np.array([[1, 1, 1],
#                     [1, 1, 1],
#                     [1, 1, 1], ], dtype=np.uint8)
# erosion = cv2.erode(Configuration_Image_3_Mask, Global_Kernal_3by3, iterations=1)
# toBeShown=cv2.resize(erosion,(400,400))
# cv2.imshow("Removing Noise", toBeShown)
# cv2.waitKey(0)
# cv2.destroyAllWindows()