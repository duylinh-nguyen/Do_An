# Convert points from euclidian to homogenious coordinate
def Euclide2Homo(local_pairs):
    homo = []

    for pair in local_pairs:
        pair[0] = (pair[0][0], pair[0][1], 1)
        pair[1] = (pair[1][0], pair[1][1], 1)
        homo.append([pair[0], pair[1]])  

    return homo


# Compute essential matrix frome Fundamental matrix and Calibration matrix


#Estimate possible camera poses

'''My function'''
# homo_pt_pairs = Euclide2Homo(euclide_pt_pairs)
# fund_matrix, inliers_mask= RANSACMethod(homo_pt_pairs, threshold, iteration)


# print(np.shape(mask))
# print(np.shape(euclide_pt_pairs))

# x1 = [[2,3], [4,8], [9,4]]
# print(np.mean(x1, 0))

# print(np.std(x1, 1)**2)
# x2 = [[9,5], [2,5], [8,3]]

# for pair in homo_pt_pairs:
#     s = pair[0].T @ fund_matrix @ pair[1]
#     s2  = pair[0].T @ Fo @ pair[1]
#     print('Me: ', s, '| Opencv: ', s2)

# for i in range(len(homo_pt_pairs)):
#     result = homo_pt_pairs[i][0].T @ F @ homo_pt_pairs[i][1]
#     print(result) 

'''----------------------------------'''
Matx43d A(u.x*P(2,0)-P(0,0),    u.x*P(2,1)-P(0,1),      u.x*P(2,2)-P(0,2),
          u.y*P(2,0)-P(1,0),    u.y*P(2,1)-P(1,1),      u.y*P(2,2)-P(1,2),
          u1.x*P1(2,0)-P1(0,0), u1.x*P1(2,1)-P1(0,1),   u1.x*P1(2,2)-P1(0,2),
          u1.y*P1(2,0)-P1(1,0), u1.y*P1(2,1)-P1(1,1),   u1.y*P1(2,2)-P1(1,2))