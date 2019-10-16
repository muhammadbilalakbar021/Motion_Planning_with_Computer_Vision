import cv2
import numpy as np
from matplotlib import pyplot as plt

img = cv2.imread("/media/bilal/Work_Space/Robotics_Path_Planning/openCV/Configuration_Space/Test1.jpg")
img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

ret, thresh1 = cv2.threshold(img, 128, 200, cv2.THRESH_BINARY)
ret, thresh2 = cv2.threshold(img, 128, 200, cv2.THRESH_BINARY_INV)
ret, thresh3 = cv2.threshold(img, 128, 255, cv2.THRESH_TRUNC)
ret, thresh4 = cv2.threshold(img, 128, 255, cv2.THRESH_TOZERO)
ret, thresh5 = cv2.threshold(img, 128, 255, cv2.THRESH_TOZERO_INV)
titles = ['Original Image', 'BINARY', 'BINARY_INV', 'TRUNC', 'TOZERO', 'TOZERO_INV']
images = [img, thresh1, thresh2, thresh3, thresh4, thresh5]

for i in range(6):
    plt.subplot(2, 3, i + 1), plt.imshow(images[i], 'gray')
    plt.title(titles[i])
    plt.xticks([]), plt.yticks([])

plt.show()

img = cv2.medianBlur(img, 5)

ret, AdaptiveThresholding1 = cv2.threshold(img, 50, 255, cv2.THRESH_BINARY)
AdaptiveThresholding2 = cv2.adaptiveThreshold(img, 200, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 11, 2)
AdaptiveThresholding3 = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)

titles = ['Global Thresholding', 'Adaptive Mean Threshold', 'Adaptive Gaussian Thresholding']
images = [img, AdaptiveThresholding1, AdaptiveThresholding2, AdaptiveThresholding3]

for i in range(1, 2):
    plt.subplot(2, 2, i + 1), plt.imshow(images[i], 'gray')
    plt.title(titles[i])
    plt.xticks([]), plt.yticks([])

plt.show()

blue1 = np.array([10, 100, 20])
blue2 = np.array([25, 255, 255])

Configuration_Image_1_Mask = cv2.inRange(img, blue1, blue2)
# global thresholding
ret1, th1 = cv2.threshold(img, 50, 127, cv2.THRESH_BINARY)
# Otsu's thresholding
#ret2, th2 = cv2.threshold(img, 10, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
# Otsu's thresholding after Gaussian filtering
blur = cv2.GaussianBlur(img, (5, 5), 0)
#ret3, th3 = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

# plot all the images and their histograms
images = [img, 0, th1,
          img, 0, th1,
          blur, 0, th1]
titles = ['Original Noisy Image', 'Histogram', 'Global Thresholding (v=127)',
          'Original Noisy Image', 'Histogram', "Otsu's Thresholding",
          'Gaussian filtered Image', 'Histogram', "Otsu's Thresholding"]

for i in range(0,1):
    plt.subplot(3,3,i*3+1),plt.imshow(images[i*3],'gray')
    plt.title(titles[i*3]), plt.xticks([]), plt.yticks([])
    plt.subplot(3,3,i*3+2),plt.hist(images[i*3].ravel(),256)
    plt.title(titles[i*3+1]), plt.xticks([]), plt.yticks([])
    plt.subplot(3,3,i*3+3),plt.imshow(images[i*3+2],'gray')
    plt.title(titles[i*3+2]), plt.xticks([]), plt.yticks([])
plt.show()
