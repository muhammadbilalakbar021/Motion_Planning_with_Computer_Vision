import cv2
import matplotlib.pyplot as plt
import numpy as np

# Linear line representation
# plt.plot([1, 2, 3, 4], [1, 4, 9, 100])
# plt.ylabel("Yaxis")
# plt.xlabel("X axis")
#plt.show()

# points represeantaion
# plt.plot([1, 2, 3, 4], [1, 4, 9, 100], 'ro')
# plt.axis([0, 6, 0, 20])
# plt.ylabel("Yaxis")
# plt.xlabel("X axis")
#plt.show()

# Scatter / Cluster representation
# t = np.arange(0., 5., 0.2)
# plt.plot(t,t,'r--' , t , t**2 , 'bs' , t, t**3 , 'g^')
#plt.show()

# data = {'a': np.arange(50),
#         'c': np.random.randint(0, 50, 50),
#         'd': np.random.randn(50)}
# data['b'] = data['a'] + 100 * np.random.randn(50)
# data['d'] = np.abs(data['d']) * 100
#
# plt.scatter('a', 'b', c='c', s='d', data=data)
# plt.xlabel('entry a')
# plt.ylabel('entry b')
# plt.show()

# names = ['group_a', 'group_b', 'group_c']
# values = [1, 10, 100]
#
# plt.figure(figsize=(10, 5))
#
# plt.subplot(131)
# plt.bar(names, values)
# plt.subplot(132)
# plt.scatter(names, values)
# plt.subplot(133)
# plt.plot(names, values)
# plt.suptitle('Categorical Plotting')
# plt.show()

img = cv2.imread("/media/bilal/Work_Space/Robotics_Path_Planning/openCV/test.jpeg", 0)
plt.figure(figsize=(10,5))
plt.subplot(121)
plt.imshow(img)
plt.title("Origioanl IMage")
plt.subplot(122)
plt.hist(img.ravel(),255)
plt.title("HIstograM")
plt.show()