import os
import glob
import numpy as np
import matplotlib.pyplot as plt

path = 'D:/New folder/dataset/poses/00.txt'
txtfile = open(path, 'r')
trajectory = []
for line in txtfile:
    ground_truth = np.array([float(x) for x in line.split()])
    ground_truth = ground_truth.reshape((3,4))
    trajectory.append(ground_truth[:,-1])

trajectory = np.array(trajectory)

fig = plt.figure()
plt.autoscale(enable=True, axis='both', tight=True)
plt.axis('equal')

plt.xlabel('x')
plt.ylabel('z')

plt.plot(trajectory[:,0], trajectory[:,2])
plt.show()