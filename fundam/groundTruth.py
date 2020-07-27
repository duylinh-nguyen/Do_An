'''
# Module đọc ground truth từ file txt
'''

import os
import glob
import numpy as np
import matplotlib.pyplot as plt

def get(sequence):
    path = 'D:/New folder/dataset/poses/'+ str(sequence)+'.txt'
    txtfile = open(path, 'r')
    trajectory = []
    for line in txtfile:
        ground_truth = np.array([float(x) for x in line.split()])
        ground_truth = ground_truth.reshape((3,4))
        trajectory.append(ground_truth[:,-1])

    return np.array(trajectory)

# Test function
if __name__ == '__main__':
    trajectory = get()
    fig = plt.figure()
    plt.autoscale(enable=True, axis='both', tight=True)
    plt.axis('equal')

    plt.xlabel('x')
    plt.ylabel('z')

    plt.plot(trajectory[:,0], trajectory[:,2])
    plt.show()