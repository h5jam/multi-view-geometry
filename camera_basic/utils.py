import numpy as np
# from pytransform3d.plot_utils import plot_vector


make_line = lambda u, v: np.vstack((u, v)).T

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


def compute_intrinsic_parameter_matrix(f, s, a, cx, cy):
    K = np.identity(3)
    K[0, 0] = f
    K[0, 1] = s
    K[0, 2] = cx
    K[1, 1] = a * f
    K[1, 2] = cy

    return K    


def compute_image_projection(points, K):
    h_points_i = K @ points

    h_points_i[0, :] = h_points_i[0, :] / h_points_i[2, :]
    h_points_i[1, :] = h_points_i[1, :] / h_points_i[2, :]

    points_i = h_points_i[:2, :]

    return points_i


def generate_random_points(n_points, xlim, ylim, zlim):
    X = np.random.randint(xlim[0], xlim[1], size=n_points)
    Y = np.random.randint(ylim[0], ylim[1], size=n_points)
    Z = np.random.randint(zlim[0], zlim[1], size=n_points)
    
    return np.vstack((X, Y, Z))


def compute_coordinates_wrt_camera(world_points, E, is_homogeneous=False):
    if not is_homogeneous:
        points_h = np.vstack((world_points, np.ones(world_points.shape[1])))
    
    points_c = E @ points_h
    return points_c


def create_algebraic_mat(world_points, projections):
    assert world_points.shape[1] == projections.shape[1]
    n_points = world_points.shape[1]
    A = np.ones(shape=(2*n_points, 12))

    c = 0

    for i in range(n_points):
        w = world_points[:, i]
        p = projections[:, i]

        X, Y, Z = w
        u, v = p

        row = np.zeros(shape=(2, 12))
        row[0, 0], row[0, 1], row[0, 2], row[0, 3] = X, Y, Z, 1
        row[0, 8], row[0, 9], row[0, 10], row[0, 11] = -u*X, -u*Y, -u*Z, -u

        row[1, 4], row[1, 5], row[1, 6], row[1, 7] = X, Y, Z, 1
        row[1, 8], row[1, 9], row[1, 10], row[1, 11] = -v*X, -v*Y, -v*Z, -v

        A[c:c+2, :]= row
        c += 2
    
    return A


def compute_w2i_projection(world_points, M, is_homogeneous=False):
    if not is_homogeneous:
        points_h = np.vstack((world_points, np.ones(world_points.shape[1])))
    
    h_points_i = M @ points_h

    h_points_i[0, :] = h_points_i[0, :] / h_points_i[2, :]
    h_points_i[1, :] = h_points_i[1, :] / h_points_i[2, :]

    points_i = h_points_i[:2, :]

    return points_i


def geometric_error(m, world_points, projections):
    assert world_points.shape[1] == projections.shape[1]
    error = 0
    n_points = world_points.shape[1]

    for i in range(n_points):
        X, Y, Z = world_points[:, i]
        u, v = projections[:, i]

        u_ = m[0]*X + m[1]*Y + m[2]*Z + m[3]
        v_ = m[4]*X + m[5]*Y + m[6]*Z + m[7]
        d = m[8]*X + m[9]*Y + m[10]*Z + m[11]
        u_ = u_/d
        v_ = v_/d

        error += np.sqrt(np.square(u - u_) + np.square(v - v_))
    
    return error