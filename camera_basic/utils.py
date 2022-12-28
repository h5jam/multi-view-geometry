import numpy as np
import pytransform3d.rotations as pr
from pytransform3d.plot_utils import plot_vector


def get_rot_x(angle):
    Rx = np.zeros(shape=(3, 3))
    Rx[0, 0] = 1
    Rx[1, 1] = np.cos(angle)
    Rx[1, 2] = -np.sin(angle)
    Rx[2, 1] = np.sin(angle)
    Rx[2, 2] = np.cos(angle)

    return Rx


def get_rot_y(angle):
    Ry = np.zeros(shape=(3, 3))
    Ry[1, 1] = 1
    Ry[0, 0] = np.cos(angle)
    Ry[0, 2] = -np.sin(angle)
    Ry[2, 0] = np.sin(angle)
    Ry[2, 2] = np.cos(angle)

    return Ry


def get_rot_z(angle):
    Rz = np.zeros(shape=(3, 3))
    Rz[2, 2] = 1
    Rz[0, 0] = np.cos(angle)
    Rz[0, 1] = -np.sin(angle)
    Rz[1, 0] = np.sin(angle)
    Rz[1, 1] = np.cos(angle)

    return Rz
    

def create_rotation_transformation_matrix(angles, order):
    '''
    Rotation transformation matrix

    angles: list of radians
    order: which to rotate in the standard global axes
    '''

    fn_mapping = {'x':get_rot_x, 'y':get_rot_y, 'z':get_rot_z}
    net = np.identity(3)

    for angle, axis in list(zip(angles, order))[::-1]:
        R = fn_mapping.get(axis)
        net = np.matmul(net, R(angle))

    return net


def create_translation_matrix(offset):
    T = np.identity(4)
    T[:3, 3] = offset
    return T


def create_image_grid(f, img_size):
    h,w = img_size
    xx,yy = np.meshgrid(range(-(h//2), w//2 + 1), range(-(h//2), w//2 + 1))
    Z = np.ones(shape=img_size) * f
    return xx, yy, Z


def convert_grid_to_homog(xx, yy, Z, img_size):
    h,w = img_size
    pi = np.ones(shape=(4, h*w))
    c = 0

    for i in range(h):
        for j in range(w):

            x = xx[i, j]
            y = yy[i, j]
            z = Z[i, j]

            point = np.array([x, y, z])
            pi[:3, c] = point
            c += 1

    return pi


def convert_homog_to_grid(pts, img_size):
    xxt = pts[0, :].reshape(img_size)
    yyt = pts[1, :].reshape(img_size)
    Zt = pts[2, :].reshape(img_size)

    return xxt, yyt, Zt


def create_same_plane_points(n_points, xlim, ylim, elevation):
    '''
    return points lied on the same plane
    '''

    x = np.linspace(xlim[0], xlim[1], n_points)
    y = np.linspace(ylim[0], ylim[1], n_points)

    xxs, yys = np.meshgrid(x, y)
    zzs = elevation * np.ones(shape=(n_points, n_points))

    same_plane_points = np.ones(shape=(3, n_points*n_points))
    c = 0

    for i in range(n_points):
        for j in range(n_points):
            xs = xxs[i, j]
            ys = yys[i, j]
            zs = zzs[i, j]
            same_plane_points[:, c] = np.array([xs, ys, zs])
            c += 1
    
    return same_plane_points