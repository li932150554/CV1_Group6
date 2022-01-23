import matplotlib.pyplot as plt
import numpy as np
import numpy.linalg as linalg


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
    t = np.mean(points, axis=0)[:-1]
    s = 0.5 * np.max(np.abs(points), axis=0)[:-1]
    T = np.eye(3)
    T[0:2,2] = -t
    T[0:2, 0:3] = T[0:2, 0:3] / np.expand_dims(s, axis=1)
    ps = points @ T.T
    return ps, T


def enforce_rank2(A):
    """ Enforces rank 2 to a given 3 x 3 matrix by setting the smallest
    eigenvalue to zero.
    Args:
        A: (3, 3) numpy array, input matrix

    Returns:
        A_hat: (3, 3) numpy array, matrix with rank at most 2
    """

    U, D, V_T = linalg.svd(A)
    D[2] = 0
    A_hat = U*D*V_T
    return A_hat



def compute_fundamental(p1, p2):
    """ Computes the fundamental matrix from conditioned coordinates.
    Args:
        p1: (n, 3) numpy array containing the conditioned coordinates in the left image
        p2: (n, 3) numpy array containing the conditioned coordinates in the right image

    Returns:
        F: (3, 3) numpy array, fundamental matrix
    """
    A = np.empty([8, 9])
    for i in range(8):
        A[i][0] = p1[i][0]*p2[i][0]
        A[i][1] = p1[i][1]*p2[i][0]
        A[i][2] = p2[i][0]
        A[i][3] = p1[i][0]*p2[i][1]
        A[i][4] = p1[i][1]*p2[i][1]
        A[i][5] = p2[i][1]
        A[i][6] = p1[i][0]
        A[i][7] = p1[i][1]
        A[i][8] = 1

    U, D, V_T = linalg.svd(A)
    f = V_T[:, 8]
    return f.reshape((3, 3))




def eight_point(p1, p2):
    """ Computes the fundamental matrix from unconditioned coordinates.
    Conditions coordinates first.
    Args:
        p1: (n, 3) numpy array containing the unconditioned homogeneous coordinates in the left image
        p2: (n, 3) numpy array containing the unconditioned homogeneous coordinates in the right image

    Returns:
        F: (3, 3) numpy array, fundamental matrix with respect to the unconditioned coordinates
    """

    # condition the coordinates
    ps_1, T_1 = condition_points(p1)
    ps_2, T_2 = condition_points(p2)


    # find F
    F = compute_fundamental(ps_1,ps_2)
    # enforce rank 2
    F_hat = enforce_rank2(F)
    # return the orignal coordinates
    return F_hat*T_1



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
    W = np.shape(img)[1]
    nb_line = np.shape(p1)[0]
    X1 = np.zeros(nb_line)
    Y1 = np.empty(nb_line)
    X2 = np.full((nb_line), W)
    Y2 = np.empty(nb_line)

    for i in range(nb_line-1):
        one_point = np.concatenate([p1[i], [1]], axis = 0)
        l_1 = np.dot(F,one_point) # we get a vector that define the line, so we have to find c in the line equation : ax+by+c = 0 because the vector value are (-b, a)
        a = l_1[1]
        b = -l_1[0]
        c = -(a*p1[i][0])-(b*p1[i][1])

        Y1[i] = (-c)/b
        Y2[i] = -(c + (a*W))/b
    
    return X1, X2, Y1, Y2
        



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
    max_value = -1
    sum_value = 0
    constraint = np.empty(n)

    for i in range(n):
        constraint[i] = abs(np.dot(np.dot(p1[i].T,F),p2[i]))
        if max_value<constraint[i]:
            max_value = constraint[i]
        sum_value = sum_value + constraint[i]

    return max_value, (sum_value/n)

def compute_epipoles(F):
    """ Computes the cartesian coordinates of the epipoles e1 and e2 in image 1 and 2 respectively.
    Args:
        F: (3, 3) numpy array, fundamental matrix

    Returns:
        e1: (2, ) numpy array, cartesian coordinates of the epipole in image 1
        e2: (2, ) numpy array, cartesian coordinates of the epipole in image 2
    """

    U, D, V_T = linalg.svd(F)
    e_2 = V_T[:, 2]
    e_1 = V_T[:, 2].T

    return e_1, e_2
