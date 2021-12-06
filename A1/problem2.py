import numpy as np
from scipy.ndimage import convolve


def loaddata(path):
    """ Load bayerdata from file

    Args:
        Path of the .npy file
    Returns:
        Bayer data as numpy array (H,W)
    """
    return np.load(path)


def separatechannels(bayerdata):
    """ Separate bayer data into RGB channels so that
    each color channel retains only the respective
    values given by the bayer pattern and missing values
    are filled with zero

    Args:
        Numpy array containing bayer data (H,W)
    Returns:
        red, green, and blue channel as numpy array (H,W)
    """
    nrow, ncol = np.shape(bayerdata)
    r = np.zeros((nrow,ncol))
    g = np.zeros((nrow,ncol))
    b = np.zeros((nrow,ncol))
    for i in range(nrow):
        for y in range(ncol):
            if i%2 == 0 :
                if y%2 == 0 :
                    g[i,y] = bayerdata[i,y]
                else:
                    r[i,y] = bayerdata[i,y]
            else:
                if y%2 == 0 :
                    b[i,y] = bayerdata[i,y]
                else:
                    g[i,y] = bayerdata[i,y]
    
    return r,g,b


def assembleimage(r, g, b):
    """ Assemble separate channels into image

    Args:
        red, green, blue color channels as numpy array (H,W)
    Returns:
        Image as numpy array (H,W,3)
    """
    return np.dstack((r,g,b))


def interpolate(r, g, b):
    """ Interpolate missing values in the bayer pattern
    by using bilinear interpolation

    Args:
        red, green, blue color channels as numpy array (H,W)
    Returns:
        Interpolated image as numpy array (H,W,3)
    """

    k = np.array([[1,1,1],[1,1,0],[1,0,0]])

    r = convolve(r, k, mode='constant', cval=0.0)
    g = convolve(g, k, mode='constant', cval=0.0)
    b = convolve(b, k, mode='constant', cval=0.0)

    return assembleimage(r,g,b)
