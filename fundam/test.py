import numpy as np
import numpy.matlib
import cv2
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from estimateFundamentalMat import *


# Extract features and matching features 
def getPointPair(gray_image1, gray_image2, number_of_points):

    sift = cv2.xfeatures2d.SIFT_create(number_of_points, contrastThreshold = 0.04, sigma = 1.6)
    kp1, des1 = sift.detectAndCompute(gray_image1,None)
    kp2, des2 = sift.detectAndCompute(gray_image2,None)

    # Matching using BFmatcher or FLANN based matcher
    bf = cv2.BFMatcher()
    # flann = cv2.FlannBasedMatcher()
    matches = bf.knnMatch(des1, des2, k=2)

    # Apply ratio test
    good_matches = []
    for m,n in matches:
        if m.distance < 0.8*n.distance:
            good_matches.append([m])

    # Get points pairs from good matches
    local_pairs = []
    for m in good_matches:
        p1 = kp1[m[0].queryIdx].pt
        p2 = kp2[m[0].trainIdx].pt
        local_pairs.append([p1, p2])

    return local_pairs, good_matches, kp1, kp2


def Euclide2Homo(local_pairs):

    homo = []

    for pair in local_pairs:
        pair[0] = (pair[0][0], pair[0][1], 1)
        pair[1] = (pair[1][0], pair[1][1], 1)
        homo.append([pair[0], pair[1]])  

    return homo  


def findEssentialMat(_fundamental_matrix, _K):
    _essential_matrix = (_K).T @ _fundamental_matrix @ _K
    u, s, v = np.linalg.svd(np.array(_essential_matrix))
    _essential_matrix = u @ np.diag([1,1,0]) @ v

    return _essential_matrix


def estimateCamPose(essential_matrix):
    w = np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]])
    u, s, vh = np.linalg.svd(essential_matrix)
    T = [u[:, 2],
         -u[:, 2],
         u[:, 2],
         -u[:, 2]]
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

    # A = []
    # X = []
   

    # for i in range(len(x1)):
    #     A = [ x1[i][1]*P1[2] - P1[1],
    #           P1[0] - x1[i][0]*P1[2],
    #           x2[i][1]*P2[2] - P2[1],
    #           P2[0] - x2[i][0]*P2[2]]
    #     u, s, v = np.linalg.svd(A)
    #     X.append(v[-1]/v[-1][-1])]


    X = cv2.triangulatePoints(P1, P2, x1.T, x2.T).T

    print(np.shape(X))

    for i in range(len(X)):
        X[i] /= X[i,3]

    return X



def checkCheirality(K, Rset, Tset, inliers):

    x1 = np.array([x[0][:2] for x in inliers])
    x2 = np.array([x[1][:2] for x in inliers])
    
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
        X = triangulation(K, R1, Rset[i], T1, Tset[i], x1, x2)
        for j in range(len(X)):
            z = Rset[i][2] @ (X[j][0:3] - Tset[i])
            if (z > 0 and X[j,3] > 0):
                temp_X.append(X[j])
                vote += 1    
        if vote > max_vote:
            max_vote = vote 
            ret_R = Rset[i]
            ret_T = Tset[i]
            ret_X = temp_X
        print("vote for pose ", i, "th = ", vote)
    print("Max vote: ", max_vote)
    print("Total inliers: ", np.shape(X))
    return ret_R, ret_T, ret_X

#BACKUP.PY
'''BACKUP CODE NGAY ĐÂY'''
# -------------------------------------------------------------------------
''' MAIN '''
# -------------------------------------------------------------------------
# Constants

# number of feature extracted 
number_of_points = 5000

# iteration & threshold in RANSAC PnP
iteration = 1000
threshold = 0.05

# Calibration matrix 
K = np.array([[568.996140852    , 0             , 643.21055941],
              [0                ,568.988362396  , 477.982801038], 
              [0                , 0             , 1            ]])

# Camera translation and rotation set 
Tset = []
Rset = []

path = 'C:/Users/NGUYEN DUY LINH/Desktop/SLAM/SLAM-LAB/data/'

# Read images and initial SIFT feature detector
img1 = cv2.imread(path + 'image1.bmp')
img2 = cv2.imread(path + 'image2.bmp')

gray1 = cv2.cvtColor(img1,cv2.COLOR_BGR2GRAY)
gray2 = cv2.cvtColor(img2,cv2.COLOR_BGR2GRAY)


euclide_pt_pairs, good_matches, kp1, kp2 = getPointPair(gray1, gray2, number_of_points)

'''Test Cv function'''
homo_pt_pairs = np.array(Euclide2Homo(euclide_pt_pairs))
fund_matrix, inliers_mask = cv2.findFundamentalMat(homo_pt_pairs[:,0], homo_pt_pairs[:,1], cv2.FM_RANSAC, 1, 0.999)

'''get inliers from homo_pt_pairs and inliers_mask'''
inl_match = [] # inliers' match
inliers = [] # inlier points in pair
for i in range(len(inliers_mask)):
    if (inliers_mask[i]): 
        inl_match.append(good_matches[i])
        inliers.append(homo_pt_pairs[i])

'''My function backup code is here'''

essential_matrix = findEssentialMat(fund_matrix, K)


R1, R2, T1 = cv2.decomposeEssentialMat(essential_matrix)


print("OpenCV R1: \n", R1)
print("Opencv T1: \n", T1)

Rset, Tset = estimateCamPose(essential_matrix)
# print("Shape of T", np.shape(T))
R, T, X = checkCheirality(K, Rset, Tset, inliers)
print("My R: \n", R)
print("My T: \n", T)



X.sort(key = lambda x: x[1])
X = np.array(X[4:])

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

ax.scatter3D(X[:,0], X[:,1], X[:,2], 'red')
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_zlabel('z');
plt.show()


# cv.drawMatchesKnn expects list of lists as matches.
# img3 = cv2.drawMatchesKnn(img1, kp1, img2, kp2, good_matches, None, [0,255,0])
# img4 = cv2.drawMatchesKnn(img1, kp1, img2, kp2, inl_match, None, [0,0,255])
# vis = np.concatenate((img3, img4), axis=0)
# cv2.namedWindow("output", cv2.WINDOW_NORMAL) 
# cv2.imshow("output", img4)
# plt.imshow(cv2.cvtColor(img4, cv2.COLOR_BGR2RGB))
# plt.show()
# # cv2.namedWindow("output1", cv2.WINDOW_NORMAL) 
# # cv2.imshow("output1", img4)

cv2.waitKey()
cv2.destroyAllWindows()
#cv2.drawMatchesKnn(img1, kp1, img2, kp2, good, None, [0,255,0])