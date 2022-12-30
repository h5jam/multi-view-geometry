import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import minimize
import pytransform3d.rotations as pr
from utils import *


make_line = lambda u, v: np.vstack((u, v)).T

np.random.seed(2)

# define params
angles = [np.pi/4]
order = 'y'

offset = np.array([0, -8, 0])

f = 2
s = 0
a = 1
cx = 0
cy = 0
img_size = (10, 10)


# create GT matrix
R = create_rotation_transformation_matrix(angles, order)
R_ = np.identity(4)
R_[:3, :3] = R

T_ = create_translation_matrix(offset)
E = np.linalg.inv(R_ @ T_)
E = E[:-1, :]

K = compute_intrinsic_parameter_matrix(f, s, a, cx, cy)

# generate points
n_points = 12
rand_points = generate_random_points(n_points, (-10, 0), (-10, 10), (f, 10))

# setup the space
xx, yy, Z = create_image_grid(f, img_size)
pt_h = convert_grid_to_homog(xx, yy, Z, img_size)
pt_h_transformed = R_ @ T_ @ pt_h

xxt, yyt, Zt = convert_homog_to_grid(pt_h_transformed, img_size)

fig = plt.figure(figsize=(8, 6))
ax = fig.add_subplot(111, projection='3d')

ax.set(xlim=(-10, 5), ylim=(-15, 5), zlim=(0, 10))

ax = pr.plot_basis(ax, R, offset)
ax.plot_surface(xxt, yyt, Zt, alpha=0.75)

c = 0
for i in range(n_points):
    point = rand_points[:, c]
    ax.scatter(*point, color='orange')
    ax.plot(*make_line(offset, point), color='purple', alpha=0.25)
    c += 1

ax.set_title('Setup')
ax.set_xlabel('X-axis')
ax.set_ylabel('Y-axis')
ax.set_zlabel('Z-axis')


# projection
rand_points_cam = compute_coordinates_wrt_camera(rand_points, E, is_homogeneous=False)
projection = compute_image_projection(rand_points_cam, K)

# visualize GT
# fig = plt.figure(figsize=(8, 6))
# ax = fig.add_subplot(111)

# for i in range(n_points):
#     ax.scatter(*projection.reshape(-1, 2)[i], color='orange')

# ax.set_title('GT projection of points in the image')


# direct linear calibration using eigenvalue
A = create_algebraic_mat(rand_points, projection)
A_ = np.matmul(A.T, A)
eigenvalues, eigenvectors = np.linalg.eig(A_)

m = eigenvectors[:, 11]
M = m.reshape(3, 4)

preds = compute_w2i_projection(rand_points, M, is_homogeneous=False)

# visualize direct linear calibration
# fig = plt.figure(figsize=(8,6))
# ax = fig.add_subplot(111)

# for i in range(n_points):
#     if i == 0:
#         o_label = 'groundtruth'
#         g_label = 'predictions'
#     else:
#         o_label = ''
#         g_label = ''
    
#     ax.scatter(*projection.reshape(-1, 2)[i], color='orange', alpha=0.75, label=o_label)
#     ax.scatter(*preds.reshape(-1, 2)[i], color='green', alpha=0.75, label=g_label)

# ax.set_title('GT vs Preds via direct linear calibration')
# ax.legend()

# minimize geometric error
result = minimize(geometric_error, m, args=(rand_points, projection))
M_ = result.x.reshape(3, 4)

optimized_pred = compute_w2i_projection(rand_points, M_, is_homogeneous=False)

# visualize comparison
fig = plt.figure(figsize=(8,6))
fig, axes = plt.subplots(nrows=2, ncols=1, figsize=(12, 8))

for i in range(n_points):
    axes[0].scatter(*projection.reshape(-1, 2)[i], color='orange', label='groundtruth')
    axes[1].scatter(*optimized_pred.reshape(-1, 2)[i], color='green', label='predictions')

axes[0].set_title('groundtruth')
axes[1].set_title('predictions')

plt.tight_layout()

plt.show()
