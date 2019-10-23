import sys
import cv2
import numpy as np
np.set_printoptions(threshold=sys.maxsize)
from matplotlib import pyplot as plt
from array import *
import matplotlib.image as mpimg

# Configuration Space 1
Configuration_Image_1 = cv2.imread("/media/bilal/Work_Space/Robotics_Path_Planning/openCV/Configuration_Space/Test1.jpg")

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

plt.plot()
plt.title("Morphological Operation")
plt.imshow(toBeShown)
plt.show()
# cv2.imshow("Removing Noise", toBeShown)
#img to NumPy
numpyImg=np.asarray(toBeShown)
print(numpyImg.shape)

white_pixels = np.array(np.where(numpyImg != 0))
first_white_pixel = white_pixels[:,0]
last_white_pixel = white_pixels[:,-1]

# white_pixels=np.stack(white_pixels)
count=1
column=[]
row=[]
for i in range(0,int(white_pixels.size/2)):
    for j in range(0,int(white_pixels.size/2)):
        if(count==1):
            # print(white_pixels[i][j])
            row.insert(j,white_pixels[i][j])
        elif(count==2):
            # print(white_pixels[i][j])
            column.insert(j, white_pixels[i][j])
    count=count+1
print(len(column))
print(len(row))
print(column[0],row[0])


print("Plotting Graph")
for j in range(0,int(white_pixels.size/2)):
    print(column[j],row[j])
    plt.scatter(column[j],row[j])

plt.gca().invert_yaxis()
plt.show()
print("Done")

# mask = numpyImg ==255
# column=np.where(np.any(mask, axis=0))[0]
# row=np.where(np.any(mask, axis=1))[0]

# print("Length of Column",len(column))
# print("Length of row",len(row))


# cv2.waitKey(0)
# cv2.destroyAllWindows()

