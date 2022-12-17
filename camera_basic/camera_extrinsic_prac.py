import matplotlib.pyplot as plt
import numpy as np
from utils import *


angles = [np.pi/4]
order = 'y'

offset = np.array([0, -8, 0])

f = 2
img_size = (7, 7)


R = create_rotation_transformation_matrix(angles, order)
R_ = np.identity(4)
R_[:3, :3] = R

T_ = create_translation_matrix(offset)

xx, yy, Z = create_image_grid(f, img_size)
pt_h = convert_grid_to_homog(xx, yy, Z, img_size)
pt_h_trans = R_ @ T_ @ pt_h

xxt, yyt, Zt = convert_homog_to_grid(pt_h_trans, img_size)


E = np.linalg.inv(R_ @ T_)
print(E)

# visualization
fig = plt.figure(figsize=(6, 4))
ax = fig.add_subplot(111, projection='3d')

ax.set(xlim=(-10, 5), ylim=(-15, 5), zlim=(0, 10))

ax.plot_surface(xx, yy, Z, alpha=0.75)
ax.plot_surface(xxt, yyt, Zt, alpha=0.75)

ax.set_title("Camera Transformation")
ax.set_xlabel("X-axis")
ax.set_ylabel("Y-axis")
ax.set_zlabel("Z-axis")

plt.show()