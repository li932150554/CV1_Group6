import math
import numpy as np
from scipy import ndimage
from scipy.ndimage.filters import convolve


def gauss2d(sigma, fsize):
  """
  Args:
    sigma: width of the Gaussian filter
    fsize: dimensions of the filter
  Returns:
    g: *normalized* Gaussian filter
  """
  # reference: https://stackoverflow.com/questions/47369579/how-to-get-the-gaussian-filter 
  kernel = np.fromfunction(lambda x, y: (1/(2*np.pi*sigma**2))*np.exp((-1*((x-(fsize-1)/2)**2+(y-(fsize-1)/2)**2))/(2*sigma**2)), (fsize, fsize))
  # kernel = np.zeros((fsize, fsize))
  # for i in range(fsize):
  #   for j in range(fsize):
  #     kernel[i][j] = (1/(2*np.pi*sigma**2))*np.exp((-1*((i-(fsize-1)/2)**2+(j-(fsize-1)/2)**2))/(2*sigma**2))
  kernel /= np.sum(kernel)
  return kernel
  #
  # You code here
  #


def createfilters():
  """
  Returns:
    fx, fy: filters as described in the problem assignment
  """
  x = np.array([0.5, 0, -0.5]).reshape(1,3) # central difference
  # 1D-Gauss filter mit sigma=0.9
  Y = gauss2d(0.9, 3)
  y = Y[:,1]
  y = y/sum(y)
  y = np.array(y).reshape(3,1)

  fx = np.empty((3, 3))
  fy = np.empty((3, 3))

  for i in range(3):
    for j in range(3):
      fx[i,j] = x[0][i] * y[j][0]
      fy[i,j] = x[0][j] * y[i][0]
  return fx, fy


def filterimage(I, fx, fy):
  """ Filter the image with the filters fx, fy.
  You may use the ndimage.convolve scipy-function.
  Args:
    I: a (H,W) numpy array storing image data
    fx, fy: filters
  Returns:
    Ix, Iy: images filtered by fx and fy respectively
  """

  
  imgx = convolve(I, fx)
  imgy = convolve(I, fy)
  return imgx, imgy
  #
  # You code here
  #


def detectedges(Ix, Iy, thr):
  """ Detects edges by applying a threshold on the image gradient magnitude.
  Args:
    Ix, Iy: filtered images
    thr: the threshold value
  Returns:
    edges: (H,W) array that contains the magnitude of the image gradient at edges and 0 otherwise
  """
  print
  edges = np.sqrt(Ix**2 + Iy**2)
  edges
  H, W = np.shape(edges)
  for i in range(H):
    for j in range(W):
      if edges[i][j] < thr:
        edges[i][j] = 0
  return edges

  #
  # You code here
  #


def nonmaxsupp(edges, Ix, Iy):
  """ Performs non-maximum suppression on an edge map.
  search the local maximums
  Args:
    edges: edge map containing the magnitude of the image gradient at edges and 0 otherwise
    Ix, Iy: filtered images
  Returns:
    edges2: edge map where non-maximum edges are suppressed
  """
  
  edges2 = edges
  theta = np.arctan(Iy / Ix)
  theta = theta*180 / np.pi # convert into angle
  H, W = np.shape(edges2)
  for i in range(1, H-1): # the pixel values on the boundary will not be processed
    for j in range(1, W-1):
      # handle top-to-bottom edges: theta in [-90, -67.5] or (67.5, 90]
      if -90 <= theta[i][j] <= -67.5 or 67.5 < theta[i][j] <= 90:
        if edges2[i][j] < edges2[i][j + 1] or edges2[i][j] < edges2[i][j - 1]:
           edges[i][j] = 0

      # handle left-to-right edges: theta in (-22.5, 22.5]
      if -22.5 < theta[i][j] <= 22.5:
        if edges2[i][j] < edges2[i + 1][j] or edges2[i][j] < edges2[i - 1][j]:
          edges[i][j] = 0
      
      # handle bottomleft-to-topright edges: theta in (22.5, 67.5]
      if 22.5 < theta[i][j] <= 67.5:
        if edges2[i][j] < edges2[i + 1][j + 1] or edges2[i][j] < edges2[i - 1][j - 1]:
          edges[i][j] = 0

      # handle topleft-to-bottomright edges: theta in [-67.5, -22.5]
      if -67.5 < theta[i][j] <= -22.5:
        if edges2[i][j] < edges2[i + 1][j - 1] or edges2[i][j] < edges2[i - 1][j + 1]:
          edges[i][j] = 0

  return edges