import numpy as np


def to_hg_coords(points):
    return np.concatenate((points, np.ones((1, points.shape[1]))), axis=0)

def to_eucld_coords(points_hg):
    z = points_hg[-1, :]
    return points_hg[:2, :] / z

def is_vector_close(v1, v2):
    v1 = v1.reshape(-1)
    v2 = v2.reshape(-1)
    assert len(v1) == len(v2)
    assert np.isclose(v1, v2).sum() == len(v1)

def get_cross_product_matrix(vector):
    assert len(vector.reshape(-1)) == 3

    A = np.zeros((3, 3))
    a1, a2, a3 = vector
    A[0][1] = -a3
    A[0][2] = a2
    A[1][0] = a3
    A[1][2] = -a1
    A[2][0] = -a2
    A[2][1] = a1

    return A

def plot_line(coeffs, xlim):
    a, b, c = coeffs
    x = np.linspace(xlim[0], xlim[1], 100)
    y = (a*x+c) / -b
    return x, y

def compute_fundamental_matrix(points1, points2):
    assert points1.shape[0] == points2.shape[0]  # check the number of points for correspondences

    u1 = points1[:, 0]
    v1 = points1[:, 1]
    u2 = points2[:, 0]
    v2 = points2[:, 1]
    one = np.ones_like(u1)

    A = np.c_[u1*u2, v1*u2, u2, u1*v2, v1*v2, v2, u1, v1, one]

    U, S, V = np.linalg.svd(A, full_matrices=True)
    f = V[-1, :]
    F = f.reshape(3, 3)

    U, S, V = np.linalg.svd(F, full_matrices=True)
    S[-1] = 0
    F = U @ np.diag(S) @ V

    return F

def compute_fundamental_matrix(points1, points2):
    assert points1.shape[0] == points2.shape[0]  # check the number of points for correspondences

    c1 = np.mean(points1, axis=0)
    c2 = np.mean(points2, axis=0)

    # scaling factor
    s1 = np.sqrt(2 / np.mean(np.sum((points1 - c1) ** 2, axis=1)))
    s2 = np.sqrt(2 / np.mean(np.sum((points2 - c2) ** 2, axis=1)))

    # for cross product
    T1 = np.array([
        [s1, 0, -s1 * c1[0]],
        [0, s1, -s1 * c1[1]],
        [0, 0 ,1]
    ])
    T2 = np.array([
        [s2, 0, -s2 * c2[0]],
        [0, s2, -s2 * c2[1]],
        [0, 0, 1]
    ])  

    normd_points1 = T1 @ points1.T
    normd_points2 = T2 @ points2.T

    # to resolve variance in the correspondences
    normd_F = compute_fundamental_matrix(normd_points1.T, normd_points2.T)

    return T2.T @ normd_F @ T1

def compute_epipole(F):
    U, S, V = np.linalg.svd(F)
    e = V[-1, :]
    e = e / e[2]
    return e
