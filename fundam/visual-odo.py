import numpy as np
import cv2
import matplotlib.pyplot as plt
import os
import glob
import recoverPose as rcvpose
import groundTruth

''' 
# Các giá trị hằng số
'''

sequence = '10'

# Max iteration
# max_iter = len(files)-1
max_iter = 5

# Số đặc trưng tối đa 
number_of_points = 500

# Ma trận hiệu chỉnh/ thông số nội 
K = np.array([[718.8560, 0.0, 607.1928],
              [0.0, 719.2800, 185.2157],
              [0.0, 0.0, 1.0]])

font = cv2.FONT_HERSHEY_SIMPLEX
img_dir = "D:/New folder/dataset/sequences/"+sequence+"/image_0" 
data_path = os.path.join(img_dir,'*g')
files = glob.glob(data_path)
iter = 0
trajectory = []
Pose = []

# Gốc tọa độ theo tọa độ thuần nhất = [0,0,0,1]
homo_origin = np.zeros((1,4))
homo_origin[:,-1] = 1

# Mảng chứ các vị trí thực tế: quỹ đạo thực tế
g_trajectory, g_pose = groundTruth.get(sequence)

'''
# Khởi tạo biểu đồ
'''
fig = plt.figure(0)
plt.autoscale(enable=True, axis='both', tight=True)
plt.axis('equal')

plt.xlabel('x')
plt.ylabel('z')


for f in files:

    '''
    # Đọc và hiển thị ảnh
    '''
    img = cv2.imread(f)

    cv2.imshow("output", img)

    # im = img
    # frame_name = 'image ' + os.path.basename(f)[0:6]
    # cv2.putText(im, frame_name ,(40,40), font, 1,(0,0,255),2)
    # resized = cv2.resize(im, (np.int(im.shape[1]/2), np.int(im.shape[0]/2)), interpolation = cv2.INTER_AREA)

    ''' Test for the first "max_iter" images in dataset''' 
    if (iter == max_iter):
        break

    elif (iter == 0): #ietr = 0
        """
        Tạo mảng [R|T] cho ảnh tại gốc tham chiếu có R = I, T = 0
        homo_recent_pose =
            [[1. 0. 0. 0.]
            [0. 1. 0. 0.]
            [0. 0. 1. 0.]
            [0. 0. 0. 1.]]
        """
        img2 = img
        X = np.zeros((1,4))
        homo_recent_pose = np.identity(4)
        homo_recent_pose[:3,:] = 0
        R = homo_recent_pose[:3,:3]
        T = homo_recent_pose[:3,-1]
        global_pose = homo_recent_pose
        mask = [1]
        # print(global_pose)

    elif (iter == 1):
        img1 = img2
        img2 = img
        R, T, X, mask = rcvpose.recoverPose(img1, img2, number_of_points)
        scale = np.linalg.norm(g_trajectory[iter]-g_trajectory[iter-1])
        T = scale * T
        recent_pose = np.concatenate((R.T, (-R.T @ T.T)), axis = 1)
        # global_pose = hom_recent_pose
        global_pose = np.concatenate((recent_pose, homo_origin), axis = 0)
        # print(np.linalg.inv(global_pose))
        # print("recent of "+ str(iter) + "\n", recent_pose)
        # print("pose of ietr " + str(iter) + "\n", Pose[iter])

    else: 
        img0 = img1
        img1 = img2
        img2 = img
        R, T, X, mask = rcvpose.recoverPose(img1, img2, number_of_points)
        scale = np.linalg.norm(g_trajectory[iter]-g_trajectory[iter-1])
        T = scale * T
        recent_pose = np.concatenate((R, T.T), axis = 1)
        homo_recent_pose = np.concatenate((recent_pose, homo_origin), axis = 0)
        global_pose = Pose[iter-1] @ np.linalg.inv(homo_recent_pose)
        # print("recent", recent_pose)
        # print("pose of ietr " + str(iter) + "\n", Pose[iter])

    Pose.append(global_pose)
    previous_X = X
    position = global_pose[:3,-1]
    trajectory.append(position) 


    # Visualize 
    plt.title("image "+str(iter))

    '''
    # Vẽ điểm đã được Triangulate
    '''
    # plotx =[]
    # for i in range(len(X)):
    #     if mask[i]==1:
    #         plotx.append(X[i])
    # plotx = np.array(plotx)
    
    # plotx = np.array([(global_pose @ k).flatten() for k in plotx])
    # plt.scatter(plotx[:5,0], plotx[:5,2], color= 'black', s =1)

    '''
    # Vẽ quỹ đạo di chuyển bằng các vị trí T' = -R.transpose x T
    '''
    recover_poses = plt.scatter(position[0], position[2], color= 'red', s=5)
    ground_truth = plt.scatter(g_trajectory[iter][0], g_trajectory[iter][2], color= 'green', s=5)

    plt.legend((recover_poses, ground_truth),
           ('recover poses', 'ground truth',),
           scatterpoints=1,
           loc='upper left',
           ncol=1,
           fontsize=8)


    plt.pause(.00001)

    iter+=1
    cv2.waitKey(1)

'''
# Vẽ biểu đồ cột thể hiện sai số cho vị trí của từng ảnh
'''
# Tính khoảng các euclide giữa thực tế và tính toán
rms_e = np.linalg.norm(trajectory - g_trajectory[:max_iter], axis = 1)
fig1 = plt.figure(1)
ind = np.arange(max_iter)
plt.bar(ind, rms_e)
plt.title("Sai số t")

new_Pose = np.array(Pose[1:])
norm  = [0]
for i in range(max_iter-1):
    norm.append(np.linalg.norm(new_Pose[i,:3,:3]- g_pose[i,:3,:3]))
norm = np.array(norm)
fig2 = plt.figure(2)
plt.bar(ind, norm)
plt.title("Sai số R")


plt.show()
cv2.waitKey()
cv2.destroyAllWindows()