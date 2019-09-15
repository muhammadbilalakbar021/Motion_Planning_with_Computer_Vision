import cv2
import numpy as np
import os
import matplotlib.pyplot as plt

currentDirecory = os.getcwd()
print(currentDirecory)

# Scaling and resizing image
img=cv2.imread("Images/messi5.jpg")
resize=cv2.resize(img, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)

# Translation
img=cv2.imread("Images/messi5.jpg",0)
rows, cols = img.shape
M = np.float32([[1,0,100],[0,1,50]])
resize=cv2.warpAffine(img,M,(cols,rows))

# Rotating an image
M = cv2.getRotationMatrix2D((cols/4,rows/2),180,1)
resize = cv2.warpAffine(img,M,(cols,rows))
cv2.imshow("Rsize",resize)
cv2.waitKey(0)
cv2.destroyAllWindows()

# Affine Transformation
rows,cols,ch = img.shape

pts1 = np.float32([[50,50],[200,50],[50,200]])
pts2 = np.float32([[10,100],[200,50],[100,250]])

M = cv2.getAffineTransform(pts1,pts2)

dst = cv2.warpAffine(img,M,(cols,rows))

plt.subplot(121),plt.imshow(img),plt.title('Input')
plt.subplot(122),plt.imshow(dst),plt.title('Output')
plt.show()

cv2.imshow("Rsize",resize)
cv2.waitKey(0)
cv2.destroyAllWindows()