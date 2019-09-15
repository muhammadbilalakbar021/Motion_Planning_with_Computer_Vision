import numpy as np
import scipy
from skimage import io, color
from skimage import exposure
import matplotlib.pyplot as plt
import cv2
import numpy as np


img = io.imread('test.jpeg')    # Load the image
#img = color.rgb2gray(img)       # Convert the image to grayscale (1 channel)

edges = cv2.Canny(img,100,200)

# sharpenkernel = np.array([[0,-1,0],[-1,5,-1],[0,-1,0]])
# blurkernal=np.array([[0.625,0.125,0.0625],[0.125,0.25,0.125],[0.0625,0.0125,0.0625]])
# print(np.max(sharpenkernel))
#
# image_sharpen = scipy.signal.convolve2d(img, sharpenkernel, 'same')
# print ('\n First 5 columns and rows of the image_sharpen matrix: \n', image_sharpen[:5,:5]*255)

#$image_sharpen_equalized = exposure.equalize_adapthist(image_sharpen/np.max(np.abs(image_sharpen)), clip_limit=0.03)
plt.imshow(edges, cmap=plt.cm.gray)
plt.axis('on')
plt.show()