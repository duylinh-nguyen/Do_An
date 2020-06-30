import os
import glob

path = 'C:/Users/NGUYEN DUY LINH/Desktop/SLAM/SLAM-LAB/data/'
data_path = os.path.join(img_dir,'*g')
files = glob.glob(data_path)
line = txtfile.readline()
ground_truth = np.array([float(x) for x in line.split()])
ground_truth = ground_truth.reshape((3,4))