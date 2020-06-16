import numpy as np
import cv2
import random
random.seed(1)

Fo = np.array([])

def compute_fundamental(x1, x2):
    """    Computes the fundamental matrix from corresponding points
        (x1,x2 3*n arrays) using the 8 point algorithm.
        Each row in the A matrix below is constructed as
        [x'*x, x'*y, x', y'*x, y'*y, y', x, y, 1] """

    x1 = x1.T
    x2 = x2.T

    n = x1.shape[1]
    if x2.shape[1] != n:
        raise ValueError("Number of points don't match.")

    # build matrix for equations
    A = np.zeros((n, 9))
    for i in range(n):
        A[i] = [x1[0, i] * x2[0, i], x1[0, i] * x2[1, i], x1[0, i] * x2[2, i],
                x1[1, i] * x2[0, i], x1[1, i] * x2[1, i], x1[1, i] * x2[2, i],
                x1[2, i] * x2[0, i], x1[2, i] * x2[1, i], x1[2, i] * x2[2, i]]

    # compute linear least square solution
    U, S, V = np.linalg.svd(A)
    F = V[-1].reshape(3, 3)

    # constrain F
    # make rank 2 by zeroing out last singular value
    U, S, V = np.linalg.svd(F)
    S[2] = 0
    F = np.dot(U, np.dot(np.diag(S), V))
    
    return F



def compute_normalized_fundamental(_samples):
    ''' Normalize image coordinate with 
        T = [S1 0   0]      [0  0   -M1[0]]
            [0  S2  0]   x  [0  0   -M2[1]]
            [0  0   0]      [0   0   0    ]
        S1, S2: VARIANCE OF X1[0], X1[1]
        M1, M2: MEAN OF X1, X2 '''  

    x1 = _samples[:,0]
    x2 = _samples[:,1]

    F, _ = cv2.findFundamentalMat(x1[:,:2], x2[:,:2], cv2.FM_8POINT)
    # print("Opencv: \n", F)

    # normalize x1
    M1 = np.mean(x1, 0)
    S1 = np.sqrt(2.0)/(np.std(x1)*8)
    T1 = np.array([[S1, 0 ,-S1*M1[0]], [0, S1, -S1*M1[1]], [0, 0, 1]])
    x1 = x1 @ T1

    # normalize x2
    M2 = np.mean(x2, 0)
    S2 = np.sqrt(2.0)/(np.std(x2)*8)
    T2 = np.array([[S2, 0 ,-S2*M2[0]], [0, S2, -S2*M2[1]], [0, 0, 1]])
    x2 = x2 @ T2


    Ft = compute_fundamental(x1, x2)
    Ft = T1.T @ Ft @ T2
    # print('Me: \n', Ft/Ft[2,2])
    return Ft/Ft[2,2]




def RANSACMethod(homo_pt_pairs, _threshold, _iteration):
    ''' RANSAC method for getting good Fundamental matrix '''    
    ret_f = []
    ret_mask = []
    n = 0
    
    for i in range(_iteration):
        samples = np.array(random.sample(homo_pt_pairs, 8))
        # print(samples)
        f = compute_normalized_fundamental(samples)
        mask = np.zeros((len(homo_pt_pairs),1))
        i = 0
       
        for pair in homo_pt_pairs:
            temp2 = np.array(pair[0]).T @ np.array(f) @ np.array(pair[1])
            # print('temp2: , ', temp2)
            if (abs(temp2) < _threshold):
                mask[i] = 1 
                i+=1
                
        if (n < i):
            n = i
            ret_f = f
            ret_mask = mask
       
    # return the good pairs, correspondent matches and fundamental matrix   
    return ret_f, ret_mask

