import matplotlib.pyplot as plt
from utils import *
import pytransform3d.rotations as pr

# define params
n_points = 6
xlim = (-5, 5)
ylim = (-5, 5)
elevation = 5 # depth
origin = np.array([0, 0, 0])


img_size = (7, 7)
f = 2 # focal length

points = create_same_plane_points(n_points, xlim, ylim, elevation)
xx, yy, Z = create_image_grid(f, img_size)


# visualize 3d space
fig = plt.figure(figsize=(8, 6))
ax = fig.add_subplot(111, projection='3d')

ax.plot_surface(xx, yy, Z, alpha=0.75)

ax.set(xlim=(-10, 10), ylim=(-10, 10), zlim=(0, 10))
ax = pr.plot_basis(ax)

c = 0
for i in range(n_points):
    for j in range(n_points):
        point = points[:, c]
        ax.scatter(*point, color="orange")
        ax.plot(*make_line(origin, point), color='purple', alpha=0.25)
        c += 1

ax.set_xlabel('X-axis')
ax.set_ylabel('Y-axis')
ax.set_zlabel('Z-axis')

# visualize projection
K = compute_intrinsic_parameter_matrix(2, 0, 1, 0, 0)  # ideal intrinsic parameter
projection_points = compute_image_projection(points, K)

h, w =img_size

flg = plt.figure(figsize=(6,4))
ax = plt.subplot()
ax.set(xlim = (-(h // 2), w // 2), ylim = (-(h // 2), w // 2))

for k in range(n_points * n_points):
    ax.scatter(*projection_points[:,k])

plt.show()
