import matplotlib.pyplot as plt
import numpy as np
import os
import sys
import pytransform3d.rotations as pr
from stereo_utils import *

sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))
from camera_basic.utils import *


# define params
f = 2
img_size = (5, 5)

# define cam A
angles = [np.pi/2, -np.pi/6]
order = 'yz'
offset1 = np.array([0, -10, 0])

R1 = create_rotation_transformation_matrix(angles, order)
R1_ = np.identity(4)
R1_[:3, :3] = R1
T1_ = create_translation_matrix(offset1)

# define cam B
angles = [np.pi/2, np.pi/6]
order = 'yz'
offset2 = np.array([0, 10, 0])

R2 = create_rotation_transformation_matrix(angles, order)
R2_ = np.identity(4)
R2_[:3, :3] = R2
T2_ = create_translation_matrix(offset2)


# visualize the environment
point = np.array([[-6, 5, 2]])  # wrt the world 

# plane wrt cam A
xx1, yy1, Z1 = create_image_grid(f, img_size)
pt1_h =convert_grid_to_homog(xx1, yy1, Z1, img_size)
pt1_h_transformed = T1_ @ R1_ @ pt1_h
xxt1, yyt1, Zt1 = convert_homog_to_grid(pt1_h_transformed, img_size)

# plane wrt cam B
xx2, yy2, Z2 = create_image_grid(f, img_size)
pt2_h =convert_grid_to_homog(xx2, yy2, Z2, img_size)
pt2_h_transformed = T2_ @ R2_ @ pt2_h
xxt2, yyt2, Zt2 = convert_homog_to_grid(pt2_h_transformed, img_size)

fig = plt.figure(figsize=(8, 6))
ax = fig.add_subplot(111, projection='3d')
ax.set(xlim=(-10, 5), ylim=(-15, 15), zlim=(-3, 10))

ax = pr.plot_basis(ax, R1, offset1, label='cam A')
ax = pr.plot_basis(ax, R2, offset2, label='cam B')

ax.plot_surface(xxt1, yyt1, Zt1, alpha=0.75)
ax.plot_surface(xxt2, yyt2, Zt2, alpha=0.75)

ax.plot(*make_line(offset1, offset2), color='red', alpha=0.5, label='baseline')

ax.scatter(*point[0], color='black')
ax.plot(*make_line(point, offset1), color='purple', alpha=0.25)
ax.plot(*make_line(point, offset2), color='purple', alpha=0.25)

c1_intn_world = offset1 + (point[0] - offset1) * 0.16
ax.scatter(*c1_intn_world, color='green')
c2_intn_world = offset2 + (point[0] - offset2) * 0.26
ax.scatter(*c2_intn_world, color='green')

ax.set_title('stereo geometry')
ax.set_xlabel('X-axis')
ax.set_ylabel('Y-axis')
ax.set_zlabel('Z-axis')

plt.legend()


# Compute projections
K = compute_intrinsic_parameter_matrix(f, 0, 1, 0, 0)  # adapted the simple intrinsic for both cams

# proj for cam A
E1 = np.linalg.inv(T1_ @ R1_)
E1_ = E1[:-1, :]
M1 = K @ E1_
proj_camA = compute_w2i_projection(point.reshape(3, -1), M1)

# proj for cam B
E2 = np.linalg.inv(T2_ @ R2_)
E2_ = E2[:-1, :]
M2 = K @ E2_
proj_camB = compute_w2i_projection(point.reshape(3, -1), M2)

# visualize projections
h, w = img_size
nrows = 1
ncols = 2

fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(6, 4))

ax1 = axes[0]
ax1.set(xlim=(-(h//2), w//2), ylim=(-(h//2), w//2))
ax1.set_title('cam A')
ax1.scatter(*proj_camA.reshape(-1))

ax2 = axes[1]
ax2.set(xlim=(-(h//2), w//2), ylim=(-(h//2), w//2))
ax2.set_title('cam B')
ax2.scatter(*proj_camB.reshape(-1))

plt.tight_layout()


# Essential Matrix ##
point_hg = to_hg_coords(point.T)

# the points wrt each cam
point_c1 = E1_ @ point_hg
print('coordinates of the point wrt cam A: \n', point_c1, '\n')
point_c2 = E2_ @ point_hg
print('coordinates of the point wrt cam B: \n', point_c2, '\n')

# define the relative pose
Ec = (E2 @ np.linalg.inv(E1))[:-1, :]
Rc = Ec[:, :-1]
Tc = Ec[:, -1]

# validate the essential matrix
is_vector_close(point_c2, Rc @ point_c1.reshape(-1) + Tc)

Tm = get_cross_product_matrix(Tc)
essential_matrix = Tm @ Rc

is_vector_close(point_c2.T @ essential_matrix @ point_c1, np.array([[0]]))

#convert coordinates of the intersection points from world to cam
c1_intn_world_hg = to_hg_coords(np.expand_dims(c1_intn_world, axis=1))
c2_intn_world_hg = to_hg_coords(np.expand_dims(c2_intn_world, axis=1))

c1_intn_hg = E1 @ c1_intn_world_hg
c2_intn_hg = E2 @ c2_intn_world_hg

c1_intn = c1_intn_hg[:-1, :]
c2_intn = c2_intn_hg[:-1, :]

is_vector_close(c2_intn.T @ essential_matrix @ c1_intn, np.array([[0]]))


# visualize epipolar lines
nrows = 1
ncols = 2
h, w = img_size

fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(6, 4))

# epipolar line in cam A given a point wrt cam B
ax1 = axes[0]
ax1.set_title("cam A")
ax1.set(xlim = (-(h // 2), w // 2), ylim = (-(h // 2), w // 2))

coeffs = (point_c2.T @ essential_matrix).reshape(-1)
x, y = plot_line(coeffs, (-1, 1))

u,v = to_eucld_coords(c1_intn).reshape(-1)

ax1.plot(x, y, label='epipolar line')
ax1.scatter(u, v, color='orange', label='point')


# epipolar line in cam B given a point wrt cam A
ax1 = axes[1]
ax1.set_title("cam B")
ax1.set(xlim = (-(h // 2), w // 2), ylim = (-(h // 2), w // 2))

coeffs = (essential_matrix @ point_c1).reshape(-1)
x, y = plot_line(coeffs, (-1, 1))

u,v = to_eucld_coords(c2_intn).reshape(-1)

ax1.plot(x, y, label='epipolar line')
ax1.scatter(u, v, color='orange', label='point')

plt.tight_layout()

plt.show()