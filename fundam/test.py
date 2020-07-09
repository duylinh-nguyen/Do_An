import numpy as np
import numpy.matlib
import cv2
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection, Line3DCollection
import matplotlib.pyplot as plt
from scipy.optimize import leastsq
import extractFeature

def Euclide2Homo(local_pairs):

    homo = []

    for pair in local_pairs:
        pair[0] = (pair[0][0], pair[0][1], 1)
        pair[1] = (pair[1][0], pair[1][1], 1)
        homo.append([pair[0], pair[1]])  

    return homo  


def computeEssentialMat(_fundamental_matrix, _K):
    _essential_matrix = (_K).T @ _fundamental_matrix @ _K
    u, s, v = np.linalg.svd(np.array(_essential_matrix))
    _essential_matrix = u @ np.diag([1,1,0]) @ v

    return _essential_matrix


def estimateCamPose(essential_matrix):
    w = np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]])
    u, s, vh = np.linalg.svd(essential_matrix)
    T = [u[:, -1],
         -u[:, -1],
         u[:, -1],
         -u[:, -1]]
    R = [(u @ w) @ vh,
         (u @ w) @ vh,
         (u @ w.T) @ vh,
         (u @ w.T) @ vh]
    for idx in range(len(R)):
        if (round(np.linalg.det(R[idx])) == -1):
            R[idx] = -R[idx]
            T[idx] = -T[idx]
    for t in T:
        t.reshape((3,1))
    return R, T


def triangulation(K, R1, R2, T1, T2, x1, x2):

    if (len(x1) != len(x2)):
        print("x1, x2 dont have the same dimension!")
        return -1

    T1 = np.reshape(T1, (3,1))
    T2 = np.reshape(T2, (3,1))

    P1 = K @ np.concatenate((R1, T1), axis = 1)

    P2 = K @ np.concatenate((R2, T2), axis = 1)

    A = []
    X = []
   

    for i in range(len(x1)):
        A = [ x1[i][1]*P1[2] - P1[1],
              P1[0] - x1[i][0]*P1[2],
              x2[i][1]*P2[2] - P2[1],
              P2[0] - x2[i][0]*P2[2]]
        u, s, v = np.linalg.svd(A)
        X.append(v[-1]/v[-1][-1])

    return np.array(X)


def checkCheirality(K, Rset, Tset, x1, x2):
    ''' Check for cheirality constrain
        to pick most possible pose from Rset & Tset
        *note: depth = R3*(X+T) > 0 '''
    
    R1 = np.identity(3)
    T1 = np.zeros((1, 3))

    X = np.array([])
    max_vote = 0
    ret_R = []
    ret_T = []
    ret_X = []

    for i in range(len(Rset)):
        vote = 0
        temp_X = []
        mask = np.zeros((len(x1)))
        X = triangulation(K, R1, Rset[i], T1, Tset[i], x1, x2)
        for j in range(len(X)):
            z = Rset[i][-1] @ X[j][0:3] + Tset[i][-1]
            if (z>0 and X[j,2] > 0):
                temp_X.append(X[j])
                mask[j] = 1
                vote += 1    
        if vote > max_vote:
            max_vote = vote 
            ret_R = Rset[i]
            ret_T = Tset[i]
            ret_X = temp_X
            final_mask = mask
        print("Vote for pose", i, "=", vote)
    print("Max vote: ", max_vote)
    print("Total inliers: ", np.shape(X))
    return ret_R, ret_T, ret_X, final_mask


def reprojectionError(X, inlier, R, T, K):
    ''' Reprojection Error function 
        for Non-linear Triangulation.
        inlier: (2x1) pair of inliers 
        X: linear triangulated 3D point
        *note: Hàm này áp dụng trên 1 điểm duy nhất'''

    R0 = np.identity(3)
    T0 = np.zeros((3, 1))
    
    T0 = np.reshape(T0, (3,1))
    T = np.reshape(T, (3,1))

    # 1st camera's projection matrix
    P0 = K @ np.concatenate((R0, T0), axis = 1)
    # 2nd camera's projection matrix
    P = K @ np.concatenate((R, T), axis = 1)

    # e = d(P0.X - x1)^2 + d(P.X - x2)^2
    return np.linalg.norm((P0 @ X) - inlier[0])**2 + np.linalg.norm((P @ X) - inlier[1])**2

#BACKUP.PY
'''BACKUP CODE NGAY ĐÂY'''
# -------------------------------------------------------------------------
''' MAIN '''
# -------------------------------------------------------------------------
# Constants

# number of feature extracted 
number_of_points = 5000


# Calibration matrix 
K = np.array([[568.996140852    , 0             , 643.21055941],
              [0                ,568.988362396  , 477.982801038], 
              [0                , 0             , 1            ]])

# Camera translation and rotation set 
Tset = []
Rset = []

path = 'C:\\Users\\NGUYEN DUY LINH\\Desktop\\Tai-lieu-do-an\\kitti\\00\\image_0\\'
path = 'C:\\Users\\NGUYEN DUY LINH\\Desktop\\SLAM\\SLAM-LAB\\data\\'

# Read images and initial SIFT feature detector
img1 = cv2.imread(path + 'image5.bmp')
img2 = cv2.imread(path + 'image6.bmp')

# gray1 = cv2.cvtColor(img1,cv2.COLOR_BGR2GRAY)
# gray2 = cv2.cvtColor(img2,cv2.COLOR_BGR2GRAY)

gray1 = cv2.cvtColor(img1,cv2.COLOR_RGB2GRAY)
gray2 = cv2.cvtColor(img2,cv2.COLOR_RGB2GRAY)

euclide_pt_pairs, _= extractFeature.getPointPair(gray1, gray2, number_of_points)

x1 = np.array([x[0] for x in euclide_pt_pairs])
x2 = np.array([x[1] for x in euclide_pt_pairs])

'''Test Cv function'''
homo_pt_pairs = np.array(Euclide2Homo(euclide_pt_pairs))
# fund_matrix, inliers_mask = cv2.findFundamentalMat(homo_pt_pairs[:,0], homo_pt_pairs[:,1], cv2.FM_RANSAC, 1, 0.999)
essential_matrix, inliers_mask = cv2.findEssentialMat(np.round(x1, 2), np.round(x2, 2), K, cv2.RANSAC, 0.999, 1.0, None)

'''RECOVER THE POSE'''
Rset, Tset = estimateCamPose(essential_matrix)
R, T, X, tMask = checkCheirality(K, Rset, Tset, x1, x2)
print("Rotation matrix R: \n", R)
print("Translation vector T: \n", T)

X2, R2, T2, tMask2 = cv2.recoverPose(essential_matrix, x1, x2, K)
print("OpenCV2 R1: \n", R2)
print("Opencv2 T1: \n", T2.flatten())

# tInliers = []
# ''' Apply mask to inliers to get triangulated inliers 
#     relative to X'''
# for i in range(len(tMask)):
#     if (tMask[i]): 
#         tInliers.append(inliers[i])

# print("X[0] =", X[0])
# out = leastsq(reprojectionError, X[0], args=(tInliers[0], R, T, K))

'''VISUALIZE THE CAMERA POSE'''

fig = plt.figure(1)
plt.axis('equal')

T = -R.T @T

X = np.array(X)

plt.scatter(X[:,0], X[:,2], s=1)
plt.scatter(0, 0,color= 'red', s=2 )
plt.scatter(T[0], T[2],color= 'blue' , s=2)
plt.scatter(T2[0], T2[2],color= 'yellow', s=2 )

plt.tight_layout()


fig2 = plt.figure(2)
# cv.drawMatchesKnn expects list of lists as matches.
# img3 = cv2.drawMatchesKnn(img1, kp1, img2, kp2, good_matches, None, [0,255,0])
# img4 = cv2.drawMatchesKnn(img1, kp1, img2, kp2, inl_match, None, [0,0,255])
vis = np.concatenate((img1, img2), axis=0)
# cv2.namedWindow("output", cv2.WINDOW_NORMAL) 
# cv2.imshow("output", img4)
plt.imshow(cv2.cvtColor(vis, cv2.COLOR_BGR2RGB))
plt.show()

cv2.waitKey()
cv2.destroyAllWindows()