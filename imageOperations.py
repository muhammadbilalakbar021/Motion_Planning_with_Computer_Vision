import cv2

img=cv2.imread('/media/bilal/Work_Space/Robotics_Path_Planning/openCV/aero1.jpg', 1)
cv2.imshow('image',img)
wait=cv2.waitKey(0)
if wait is 27:
    cv2.destroyAllWindows()
elif wait == ord('s'):
    name=input("Enter the name for the image ?")
    cv2.imwrite(name+'.png',img)
    cv2.destroyAllWindows()