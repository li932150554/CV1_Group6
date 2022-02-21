import numpy as np
from scipy.ndimage import convolve
from scipy.interpolate import griddata

######################
# Basic Lucas-Kanade #
######################

def compute_derivatives(im1, im2):
    """Compute dx, dy and dt derivatives.
    
    Args:
        im1: first image
        im2: second image
    
    Returns:
        Ix, Iy, It: derivatives of im1 w.r.t. x, y and t
    """
    assert im1.shape == im2.shape

    fx = np.array([[0.5, 0, -0.5]])
    fy = fx.transpose()
    # Ix = np.empty_like(im1)
    # Iy = np.empty_like(im1)
    # It = np.empty_like(im1)
    Ix = convolve(im1, fx, mode='mirror')  # mirror padding the same as assignment 3
    Iy = convolve(im1, fy, mode='mirror')
    It = im2 - im1


    #
    # Your code here
    #
    
    assert Ix.shape == im1.shape and \
           Iy.shape == im1.shape and \
           It.shape == im1.shape

    return Ix, Iy, It

def compute_motion(Ix, Iy, It, patch_size=15, aggregate="gauss", sigma=2):
    """Computes one iteration of optical flow estimation.
    
    Args:
        Ix, Iy, It: image derivatives w.r.t. x, y and t
        patch_size: specifies the side of the square region R in Eq. (1)
        aggregate: 0 or 1 specifying the region aggregation region
        sigma: if aggregate=='gaussian', use this sigma for the Gaussian kernel
    Returns:
        u: optical flow in x direction
        v: optical flow in y direction
    
    All outputs have the same dimensionality as the input
    """
    assert Ix.shape == Iy.shape and \
            Iy.shape == It.shape

    H, W = np.shape(Ix)
    u = np.zeros((H, W))
    v = np.zeros((H, W))
    padding_size = int(patch_size / 2)
    padded_Ix = Ix.copy()
    padded_Ix = np.pad(padded_Ix, (padding_size, padding_size), 'symmetric')  # which padding model to use?
    padded_Iy = Iy.copy()
    padded_Iy = np.pad(padded_Iy, (padding_size, padding_size), 'symmetric')
    padded_It = It.copy()
    padded_It = np.pad(padded_It, (padding_size, padding_size), 'symmetric')

    padded_Ixx = np.power(padded_Ix, 2)
    padded_Iyy = np.power(padded_Iy, 2)
    padded_Ixy = np.multiply(padded_Ix, padded_Iy)
    padded_Ixt = np.multiply(padded_Ix, padded_It)
    padded_Iyt = np.multiply(padded_Iy, padded_It)


    if(aggregate == "gauss"):
        gauss_kernel = np.zeros((patch_size,patch_size))
        gauss_kernel = gaussian_kernel(patch_size, sigma)

        patch_Ixx = np.zeros((patch_size,patch_size))
        patch_Ixy = np.zeros((patch_size,patch_size))
        patch_Iyy = np.zeros((patch_size,patch_size))
        patch_Ixt = np.zeros((patch_size,patch_size))
        patch_Iyt = np.zeros((patch_size,patch_size))

        for i in range(H):
            for j in range(W):
                patch_Ixx = (padded_Ixx[i: i + patch_size, j: j + patch_size])*gauss_kernel
                patch_Ixy = (padded_Ixy[i: i + patch_size, j: j + patch_size])*gauss_kernel
                patch_Iyy = (padded_Iyy[i: i + patch_size, j: j + patch_size])*gauss_kernel
                patch_Ixt = (padded_Ixt[i: i + patch_size, j: j + patch_size])*gauss_kernel
                patch_Iyt = (padded_Iyt[i: i + patch_size, j: j + patch_size])*gauss_kernel

                a = np.array([[patch_Ixx.sum(), patch_Ixy.sum(), patch_Ixy.sum(), patch_Iyy.sum()]]).reshape(2, 2)
                b = np.array([[patch_Ixt.sum(), patch_Iyt.sum()]]).reshape(2, 1)
                U = - np.dot(np.linalg.inv(a), b)
                u[i, j] = U[0]
                v[i, j] = U[1]
    else:
        for i in range(H):
            for j in range(W):
                a = np.array([[padded_Ixx[i: i + patch_size, j: j + patch_size].sum(), padded_Ixy[i: i + patch_size, j: j + patch_size].sum(),
                            padded_Ixy[i: i + patch_size, j: j + patch_size].sum(), padded_Iyy[i: i + patch_size, j: j + patch_size].sum()]]).reshape(2, 2)

                b = np.array([[padded_Ixt[i: i + patch_size, j: j + patch_size].sum(), padded_Iyt[i: i + patch_size, j: j + patch_size].sum()]]).reshape(2, 1)
                U = - np.dot(np.linalg.inv(a), b)
                u[i, j] = U[0]
                v[i, j] = U[1]

    
    
    assert u.shape == Ix.shape and \
            v.shape == Ix.shape
    return u, v

def warp(im, u, v):
    """Warping of a given image using provided optical flow.
    
    Args:
        im: input image
        u, v: optical flow in x and y direction
    
    Returns:
        im_warp: warped image (of the same size as input image)
    """
    assert im.shape == u.shape and \
            u.shape == v.shape

    # reference: https://docs.scipy.org/doc/scipy/reference/generated/scipy.interpolate.griddata.html
    # im_warp = np.empty_like(im)
    H, W = np.shape(im)
    points = np.zeros((H*W, 2))
    values = np.array(im).reshape(-1, 1)  # make the values of im1 into a column vektor
    x_0, y_0 = np.mgrid[0:H, 0:W]
    grid_x = (x_0 + u).reshape(-1, 1)
    grid_y = (y_0 + v).reshape(-1, 1)

    # generate the points in griddata, based on original coordinates from im1 and correspond offset u, v
    for i in range(H*W):
        points[i, 0] = grid_x[i]
        points[i, 1] = grid_y[i]

    im_warp = griddata(points, values, (x_0, y_0), method='nearest')[:, :, 0]  # parameters: points, values, xi, method

    assert im_warp.shape == im.shape
    return im_warp

def compute_cost(im1, im2):
    """Implementation of the cost minimised by Lucas-Kanade."""
    assert im1.shape == im2.shape

    d = 0.0
    d = np.power(im1-im2, 2).sum()

    assert isinstance(d, float)
    return d

####################
# Gaussian Pyramid #
####################

#
# this function implementation is intentionally provided
#
def gaussian_kernel(fsize, sigma):
    """
    Define a Gaussian kernel
    Args:
        fsize: kernel size
        sigma: deviation of the Guassian
    Returns:
        kernel: (fsize, fsize) Gaussian (normalised) kernel
    """

    _x = _y = (fsize - 1) / 2
    x, y = np.mgrid[-_x:_x + 1, -_y:_y + 1]
    G = np.exp(-0.5 * (x**2 + y**2) / sigma**2)

    return G / G.sum()

def downsample_x2(x, fsize=5, sigma=1.4):
    """
    Downsampling an image by a factor of 2
    Hint: Don't forget to smooth the image beforhand (in this function).
    Args:
        x: image as numpy array (H x W)
        fsize and sigma: parameters for Guassian smoothing
                         to apply before the subsampling
    Returns:
        downsampled image as numpy array (H/2 x W/2)
    """

    x = convolve(x, gaussian_kernel(fsize, sigma), mode='mirror')
    x_ds = x[::2, ::2]

    return x_ds

def gaussian_pyramid(img, nlevels=3, fsize=5, sigma=1.4):
    '''
    A Gaussian pyramid is a sequence of downscaled images
    (here, by a factor of 2 w.r.t. the previous image in the pyramid)
    Args:
        img: face image as numpy array (H * W)
        nlevels: num of level Gaussian pyramid, in this assignment we will use 3 levels
        fsize: gaussian kernel size, in this assignment we will define 5
        sigma: sigma of guassian kernel, in this assignment we will define 1.4
    Returns:
        GP: list of gaussian downsampled images in ascending order of resolution
    '''

    GP = []
    for i in range(nlevels):
        if i == 0:
            GP.append(img)
        else:
            GP.append(downsample_x2(GP[i-1], fsize, sigma))

    # We reverse the array to have the smaller size first
    return GP[::-1]

###############################
# Coarse-to-fine Lucas-Kanade #
###############################

def coarse_to_fine(im1, im2, pyramid1, pyramid2, n_iter=3):
    """Implementation of coarse-to-fine strategy
    for optical flow estimation.
    
    Args:
        im1, im2: first and second image
        pyramid1, pyramid2: Gaussian pyramids corresponding to im1 and im2
        n_iter: number of refinement iterations
    
    Returns:
        u: OF in x direction
        v: OF in y direction
    """
    assert im1.shape == im2.shape

    nb_reso = len(pyramid1)
    U = []
    V = []
    Cost = []
    img_1 = pyramid1[0]
    for i in range(n_iter):   
        img_2 = pyramid2[0]
        for y in range(nb_reso-1):
            Ix, Iy, It = compute_derivatives(img_1, img_2)
            a, b = compute_motion(Ix, Iy, It)

            img_1 = pyramid1[y+1]
            img_2 = pyramid2[y+1]
            #Upscale u and v 
            H,W = np.shape(a)
            u_up = np.zeros((H*2, W*2))
            v_up = np.zeros((H*2, W*2))
            for h in range(H):
                for w in range(W):
                    u_up[h][w] = a[h][w]*2
                    u_up[h+1][w] = a[h][w]*2
                    u_up[h][w+1] = a[h][w]*2
                    u_up[h+1][w+1] = a[h][w]*2
                    
                    v_up[h][w] = b[h][w]*2
                    v_up[h+1][w] = b[h][w]*2
                    v_up[h][w+1] = b[h][w]*2
                    v_up[h+1][w+1] = b[h][w]*2

            img_1 = warp(img_1, u_up,v_up)

        Ix, Iy, It = compute_derivatives(img_1, img_2)
        u, v = compute_motion(Ix, Iy, It)

        #downscale u and v for wraping the first level of the gaussian pyramid
        u_down = u
        v_down = v
        for i in range(nb_reso-1):
            u_down = u_down[::2, ::2]
            v_down = v_down[::2, ::2]

        img_1 = warp(pyramid1[0], u_down, v_down)
    

    assert u.shape == im1.shape and \
            v.shape == im1.shape
    return u, v