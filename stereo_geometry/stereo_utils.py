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