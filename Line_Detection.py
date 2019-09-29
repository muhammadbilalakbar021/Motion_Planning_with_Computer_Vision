import cv2
import numpy as np
import os

path=os.getcwd()
img = cv2.imread("/media/bilal/Work_Space/Robotics_Path_Planning/openCV/Morphological_Images/wirebond.tif")
gray=cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
edge=cv2.Canny(gray,50,150,apertureSize=3)
lines=cv2.HoughLines(edge,1,np.pi/180,200)
print(len(lines))

for r,theta in lines[0]:
    print(r,theta)
    # Stores the value of cos(theta) in a
    a = np.cos(theta)
    # Stores the value of sin(theta) in b
    b = np.sin(theta)
    # x0 stores the value rcos(theta)
    x0 = a * r
    # y0 stores the value rsin(theta)
    y0 = b * r
    # x1 stores the rounded off value of (rcos(theta)-1000sin(theta))
    x1 = int(x0 + 1000 * (-b))
    # y1 stores the rounded off value of (rsin(theta)+1000cos(theta))
    y1 = int(y0 + 1000 * (a))
    # x2 stores the rounded off value of (rcos(theta)+1000sin(theta))
    x2 = int(x0 - 1000 * (-b))
    # y2 stores the rounded off value of (rsin(theta)-1000cos(theta))
    y2 = int(y0 - 1000 * (a))
    # cv2.line draws a line in img from the point(x1,y1) to (x2,y2).
    # (0,0,255) denotes the colour of the line to be
    # drawn. In this case, it is red.
    cv2.line(img, (x1, y1), (x2, y2), (0, 0, 255), 2)

cv2.imwrite('linesDetected.jpg', img)