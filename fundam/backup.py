
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

    # '''get inliers from homo_pt_pairs and inliers_mask'''
    # inl_match = [] # inliers' match
    # inliers = [] # inlier points in pair
    # for i in range(len(inliers_mask)):
    #     if (inliers_mask[i]): 
    #         inl_match.append(good_matches[i])
    #         inliers.append(homo_pt_pairs[i])



    # ax = fig.add_subplot(111, projection='3d')
# define first camera vectors
# C =np.array([[0], [0], [0]])
# z = np.array([0,0,1])
# y = np.array([0,1,0])
# x = np.array([1,0,0])

# V = np.array([x,y,z])

# plt.quiver(*C, V[:,0], V[:,1], V[:,2], color = ['r', 'g', 'b'])
# v = np.array([[x[-1]-0.5, y[-1]-0.5, z[-1]], [x[-1]+0.5, y[-1]-0.5, z[-1]], [x[-1]+0.5, y[-1]+0.5, z[-1]],  [x[-1]-0.5, y[-1]+0.5, z[-1]], C[:,0]])
# v1 = [(R @ i + T) for i in v]
# v1 = np.array(v1)
# ax.scatter3D(v[:, 0], v[:, 1], v[:, 2])
# ax.scatter3D(v1[:, 0], v1[:, 1], v1[:, 2])

# # generate list of sides' polygons of camera symbols
# verts = [ [v[0],v[1],v[4]], [v[0],v[3],v[4]],
#  [v[2],v[1],v[4]], [v[2],v[3],v[4]], [v[0],v[1],v[2],v[3]]]
# verts1 = [ [v1[0],v1[1],v1[4]], [v1[0],v1[3],v1[4]],
#  [v1[2],v1[1],v1[4]], [v1[2],v1[3],v1[4]], [v1[0],v1[1],v1[2],v1[3]]]

# # plot sides
# ax.add_collection3d(Poly3DCollection(verts, facecolors='cyan', linewidths=1, edgecolors='g', alpha=.25))
# ax.add_collection3d(Poly3DCollection(verts1, facecolors='green', linewidths=1, edgecolors='r', alpha=.25))