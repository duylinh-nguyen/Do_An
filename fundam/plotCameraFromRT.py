from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection, Line3DCollection
import numpy as np
import matplotlib.cm as cm

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# vertices of a pyramid
# define camera vectors
z = np.array([0,0,1])
y = np.array([0,1,0])
x = np.array([1,0,0])
centroid =np.array([[0], [0], [0]])
V = np.array([x,y,z])

cmap = cm.get_cmap(name='rainbow')
plt.quiver(*centroid, V[:,0], V[:,1], V[:,2], color = ['r', 'g', 'b'])
v = np.array([[x[-1]-0.5, y[-1]-0.5, z[-1]], [x[-1]+0.5, y[-1]-0.5, z[-1]], [x[-1]+0.5, y[-1]+0.5, z[-1]],  [x[-1]-0.5, y[-1]+0.5, z[-1]], centroid[:,0]])
ax.scatter3D(v[:, 0], v[:, 1], v[:, 2])

# generate list of sides' polygons of our pyramid
verts = [ [v[0],v[1],v[4]], [v[0],v[3],v[4]],
 [v[2],v[1],v[4]], [v[2],v[3],v[4]], [v[0],v[1],v[2],v[3]]]

# plot sides
ax.add_collection3d(Poly3DCollection(verts, facecolors='cyan', linewidths=1, edgecolors='g', alpha=.25))

plt.show()