'''
# Module phục hồi R,t và thực hiện Linear Triangulation
'''
import cv2
import numpy as np
import extractFeature
import estimateFundamentalMat
import random
random.seed(1)

'''
# Hằng số
'''
init = 0
prev_ds2 = 0
prev_X = 0
scale = 0
prev_norm_T = 0

def computeEssentialMat(_fundamental_matrix, _K):
    _essential_matrix = (_K).T @ _fundamental_matrix @ _K
    u, _, v = np.linalg.svd(np.array(_essential_matrix))
    _essential_matrix = u @ np.diag([1,1,0]) @ v
    return _essential_matrix

def Euclide2Homo(local_pairs):
    '''
    # Chuyển từ tọa độ Euclide sang tọa độ thuần nhất
    '''
    homo = []
    for pair in local_pairs:
        pair[0] = (pair[0][0], pair[0][1], 1)
        pair[1] = (pair[1][0], pair[1][1], 1)
        homo.append([pair[0], pair[1]])  
    return homo  

def estimateCamPose(essential_matrix):
    '''
    # Phân giải ma trận E, tìm 4 khả năng của R và T
    '''
    w = np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]])
    u, s, vh = np.linalg.svd(essential_matrix)
    T = [u[:,-1],
         -u[:,-1],
         u[:,-1],
         -u[:,-1]]
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
    '''
    # Linear Triangulation 
    '''
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
        _, _, v = np.linalg.svd(A)
        X.append(v[-1]/v[-1][-1])

    # X = cv2.triangulatePoints(P1, P2, x1.T, x2.T).T
    # print(np.shape(X))
    # for i in range(len(X)):
    #     X[i] /= X[i,3]

    return np.array(X)


def checkCheirality(K, Rset, Tset, x1, x2):
    ''' 
    Kiểm tra điều kiện ràng buộc
    Tìm ra R và T từ Rset & Tset
    *note: depth = R3*(X+T) > 0 
    '''
    
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
            z = Rset[i][-1] @ (X[j][0:3] + Tset[i])
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
    #     print("Vote for pose", i, "=", vote)
    # print("Max vote: ", max_vote)
    # print("Total inliers: ", np.shape(X))
    # print(ret_X)
    return ret_R, ret_T, ret_X, final_mask

def recoverPose(img1, img2, number_of_points):

    global init
    global prev_ds2
    global prev_X
    global scale
    global prev_norm_T

    if img1 is None or img2 is None:
        return -1

    K = np.array([[718.8560, 0.0, 607.1928],
             [0.0, 718.8560, 185.2157],
             [0.0, 0.0, 1.0]])

    # Camera translation and rotation set 
    Tset = []
    Rset = []

    gray1 = cv2.cvtColor(img1,cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(img2,cv2.COLOR_BGR2GRAY)

    euclide_pt_pairs, ds2 = extractFeature.getPointPair(gray1, gray2, number_of_points)

    x1 = np.array([x[0] for x in euclide_pt_pairs])
    x2 = np.array([x[1] for x in euclide_pt_pairs])
    
    essential_matrix, _ = cv2.findEssentialMat(np.round(x1, 2), np.round(x2, 2), K, cv2.RANSAC, 0.999, 1.0, None)

    homo_pt_pairs = np.array(Euclide2Homo(euclide_pt_pairs))

    # fund_matrix, inliers_mask = cv2.findFundamentalMat(homo_pt_pairs[:,0], homo_pt_pairs[:,1], cv2.FM_RANSAC, 1, 0.999)
    # fund_matrix, inliers_mask = estimateFundamentalMat.RANSACMethod(homo_pt_pairs, 0.01, 200)
    # essential_matrix = computeEssentialMat(fund_matrix, K)

    Rset, Tset = estimateCamPose(essential_matrix)
    R, T, X, tMask = checkCheirality(K, Rset, Tset, x1, x2)
    t_ds2 = []
    for i in range(len(tMask)):
        if (tMask[i]): 
            t_ds2.append(ds2[i])

    return R, T.reshape((1,3)), X, tMask

