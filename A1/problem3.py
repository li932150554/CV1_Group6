import numpy as np
from scipy import linalg


def load_points(path):
    '''
    Load points from path pointing to a numpy binary file (.npy). 
    Image points are saved in 'image'
    Object points are saved in 'world'
    Returns:
        image: A Nx3 array of 2D points form image coordinate 
        world: A N*4 array of 3D points form world coordinate
    '''
    # for homogeneous coordinates should image be (N, 3) and world be (N, 4)

    data = np.load(path)
    image_pts = data["image"]
    world_pts = data["world"]
    # H_i, W_i = image_pts.shape  # image_pts is (75, 3)
    # H_w, W_w = world_pts.shape  # world_pts is (75, 4)
    # print(H_i, W_i)
    # print(H_w, W_w)

    # sanity checks
    assert image_pts.shape[0] == world_pts.shape[0] # check if the numbers of row are euqal 

    # homogeneous coordinates
    assert image_pts.shape[1] == 3 and world_pts.shape[1] == 4
    return image_pts, world_pts


def create_A(x, X):
    """Creates (2*N, 12) matrix A from 2D/3D correspondences
    that comes from cross-product
    
    Args:
        x and X: N 2D and 3D point correspondences (homogeneous)
        
    Returns:
        A: (2*N, 12) matrix A
    """

    N, _ = x.shape
    assert N == X.shape[0]

    A = np.empty((2*N, 12)) # the size of A: (2*N, 12)
    for i in range(N):
        A[2*i, :] = np.array([[0, 0, 0, 0], -X[i, :], x[i][1]*X[i, :]]).reshape(1, 12)
        A[2*i+1, :] = np.array([X[i, :], [0, 0, 0, 0], -x[i][0]*X[i, :]]).reshape(1, 12)

    assert A.shape[0] == 2*N and A.shape[1] == 12
    print("A = ", A)
    return A


def homogeneous_Ax(A):
    """Solve homogeneous least squares problem (Ax = 0, s.t. norm(x) == 0),
    using SVD decomposition as in the lecture.
    Args:
        A: (2*N, 12) matrix A
    
    Returns:
        P: (3, 4) projection matrix P
    """
    # reference: https://docs.scipy.org/doc/numpy-1.10.1/reference/generated/numpy.linalg.svd.html#numpy.linalg.svd
    U, S, V_T = np.linalg.svd(A)
    p = V_T.T[:,11] # from the lecture should p be the last right singular vector V_12
    P = p.reshape((3, 4))
    # print("P = ", P)
    return P

def solve_KR(P):
    """Using th RQ-decomposition find K and R 
    from the projection matrix P.
    Hint 1: you might find scipy.linalg useful here.
    Hint 2: recall that K has 1 in the the bottom right corner.
    Hint 3: RQ decomposition is not unique (up to a column sign).
    Ensure positive element in K by inverting the sign in K columns 
    and doing so correspondingly in R.
    Args:
        P: 3x4 projection matrix.
    
    Returns:
        K: 3x3 matrix with intrinsics
        R: 3x3 rotation matrix 
    """
    # reference: https://docs.scipy.org/doc/scipy-0.14.0/reference/generated/scipy.linalg.qr.html
    M = np.empty((3, 3))
    for i in range(3):
        for j in range(3):
            M[i][j] = P[i][j]

    K, R = linalg.qr(M)
    I = np.diag([np.sign(K[0][0]), np.sign(K[1][1]), np.sign(K[2][2])]) # get the sign of the diagonal value of K
    K = K*I # make sure the diagonal value of K be positive
    R = I.T*R # K*R = (K*I)*(I.T*R) as long as I is orthonormal

    return K, R

def solve_c(P):
    """Find the camera center coordinate from P
    by finding the nullspace of P with SVD.
    Args:
        P: 3x4 projection matrix
    
    Returns:
        c: 3x1 camera center coordinate in the world frame
    """
    # reference: https://docs.scipy.org/doc/scipy/reference/generated/scipy.linalg.null_space.html
    c_ = linalg.null_space(P)
    c = c_[:3, ]

    return c