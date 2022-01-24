import matplotlib.pyplot as plt
import numpy as np


def condition_points(points):
    """ Conditioning: Normalization of coordinates for numeric stability 
    by substracting the mean and dividing by half of the component-wise
    maximum absolute value.
    Args:
        points: (l, 3) numpy array containing unnormalized homogeneous coordinates.

    Returns:
        ps: (l, 3) numpy array containing normalized points in homogeneous coordinates.
        T: (3, 3) numpy array, transformation matrix for conditioning
    """
    t = np.mean(points, axis=0)[:-1]  # mean value of every column  store the first two values
    s = 0.5 * np.max(np.abs(points), axis=0)[:-1]
    T = np.eye(3)
    T[0:2,2] = -t
    T[0:2, 0:3] = T[0:2, 0:3] / np.expand_dims(s, axis=1) # change row vector into column vector
    # ps = points @ T.T
    ps = np.dot(T, points.T).T
    return ps, T


def enforce_rank2(A):
    """ Enforces rank 2 to a given 3 x 3 matrix by setting the smallest
    eigenvalue to zero.
    Args:
        A: (3, 3) numpy array, input matrix

    Returns:
        A_hat: (3, 3) numpy array, matrix with rank at most 2
    """

    U, D, V_T = np.linalg.svd(A)
    D_F = D
    D_F[-1] = 0
    A_hat = np.dot(U, np.dot(np.diag(D_F), V_T))
    return A_hat
    #
    # You code here
    #



def compute_fundamental(p1, p2):
    """ Computes the fundamental matrix from conditioned coordinates.
    Args:
        p1: (n, 3) numpy array containing the conditioned coordinates in the left image
        p2: (n, 3) numpy array containing the conditioned coordinates in the right image

    Returns:
        F: (3, 3) numpy array, fundamental matrix
    """
    ps_1, _ = condition_points(p1)
    ps_2, _ = condition_points(p2)
    dim = np.shape(p1)[0]
    A = np.zeros((dim, 9))
    for i in range(dim):
        A[i, :] = [ps_1[i, 0]*ps_2[i, 0], ps_1[i, 1]*ps_2[i, 0], ps_2[i, 0], ps_1[i, 0]*ps_2[i, 1],
                   ps_1[i, 1]*ps_2[i, 1], ps_2[i, 1], ps_1[i, 0], ps_1[i, 1], 1]

    U, D, V_T = np.linalg.svd(A)
    F = np.array(V_T.T[:, 8]).reshape(3, 3)
    return F
    #
    # You code here
    #



def eight_point(p1, p2):
    """ Computes the fundamental matrix from unconditioned coordinates.
    Conditions coordinates first.
    Args:
        p1: (n, 3) numpy array containing the unconditioned homogeneous coordinates in the left image
        p2: (n, 3) numpy array containing the unconditioned homogeneous coordinates in the right image

    Returns:
        F: (3, 3) numpy array, fundamental matrix with respect to the unconditioned coordinates
    """
    ps_1, T_1 = condition_points(p1)
    ps_2, T_2 = condition_points(p2)
    F_con = compute_fundamental(ps_1, ps_2)
    F_rank2 = enforce_rank2(F_con)
    F = np.dot(T_2.T, np.dot(F_rank2, T_1))
    return F
    #
    # You code here
    #




def draw_epipolars(F, p1, img):
    """ Computes the coordinates of the n epipolar lines (X1, Y1) on the left image border and (X2, Y2)
    on the right image border.
    Args:
        F: (3, 3) numpy array, fundamental matrix 
        p1: (n, 2) numpy array, cartesian coordinates of the point correspondences in the image
        img: (H, W, 3) numpy array, image data

    Returns:
        X1, X2, Y1, Y2: (n, ) numpy arrays containing the coordinates of the n epipolar lines
            at the image borders
    """
    # n = np.shape(p1)[0]
    # a = np.ones((n, 1))
    # p1_hom = np.insert(p1, 2, values=a, axis=1)
    # l = np.dot(F, p1_hom.T).T

    W = np.shape(img)[1]
    nb_line = np.shape(p1)[0]
    X1 = np.zeros(nb_line)
    Y1 = np.empty(nb_line)
    X2 = np.full((nb_line), W)
    Y2 = np.empty(nb_line)

    for i in range(nb_line):
        one_point = np.concatenate([p1[i], [1]], axis=0)  # create homogeneous coordinates
        l_1 = np.dot(F,
                     one_point)  # we get a vector that define the line, so we have to find c in the line equation : ax+by+c = 0 because the vector value are (-b, a)
        a = l_1[0]
        b = l_1[1]
        c = l_1[2]

        Y1[i] = (-c) / b
        Y2[i] = -(c + (a * W)) / b

    return X1, X2, Y1, Y2

    #
    # You code here
    #



def compute_residuals(p1, p2, F):
    """
    Computes the maximum and average absolute residual value of the epipolar constraint equation.
    Args:
        p1: (n, 3) numpy array containing the homogeneous correspondence coordinates from image 1
        p2: (n, 3) numpy array containing the homogeneous correspondence coordinates from image 2
        F:  (3, 3) numpy array, fundamental matrix

    Returns:
        max_residual: maximum absolute residual value
        avg_residual: average absolute residual value
    """
    n = np.shape(p1)[0]
    residual = np.zeros((n, 1))
    for i in range(n):
        residual[i] = np.abs(np.dot(p1[i, :], np.dot(F, p2[i, :].T)))

    max_residual = np.max(residual)
    avg_residual = np.average(residual)

    return max_residual, avg_residual
    #
    # You code here
    #


def compute_epipoles(F):
    """ Computes the cartesian coordinates of the epipoles e1 and e2 in image 1 and 2 respectively.
    Args:
        F: (3, 3) numpy array, fundamental matrix

    Returns:
        e1: (2, ) numpy array, cartesian coordinates of the epipole in image 1
        e2: (2, ) numpy array, cartesian coordinates of the epipole in image 2
    """

    # reference: http: // pillowlab.princeton.edu / teaching / statneuro2018 / slides / notes03a_SVDandLinSys.pdf
    #            https: // math.stackexchange.com / questions / 1771013 / how - is -the - null - space - related - to - singular - value - decomposition

    U,S,V_T = np.linalg.svd(F)
    e2_h = V_T.T[:, -1]
    e2_c = (e2_h / e2_h[-1])[0:2]
    e1_h = U[:, -1]
    e1_c = (e1_h / e1_h[-1])[0:2]
    return e1_c, e2_c

    # e2 = V_T.T[:, -1]
    # e1 = U[:, -1]
    # return e1, e2


