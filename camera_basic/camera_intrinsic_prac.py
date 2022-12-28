import matplotlib.pyplot as plt
from utils import *


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

fig = plt.figure(figsize=(8, 6))
ax = fig.add_subplot(111, projection='3d')

ax.plot_surface(xx, yy, Z, alpha=0.75)
# ax = pr.plot_basis(ax)

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

plt.show()