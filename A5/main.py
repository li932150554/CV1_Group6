import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from utils import *


# Helper functions
def show_two(im1, im2):
    fig = plt.figure(figsize=(14, 6), dpi=80, facecolor='w', edgecolor='k')

    (ax1, ax2) = fig.subplots(1, 2)

    ax1.imshow(im1)
    ax2.imshow(im2)

    plt.show()


#
# Problem 1
#

import problem1 as p1


def problem1():

    # Loading the image and scaling to [0, 1]
    im1 = np.array(Image.open("data/frame09.png").convert('L')) / 255.0
    im2 = np.array(Image.open("data/frame10.png").convert('L')) / 255.0

    #
    # Basic implementation
    #
    Ix, Iy, It = p1.compute_derivatives(im1, im2)  # gradients
    u, v = p1.compute_motion(Ix, Iy, It)  # flow

    # stacking for visualisation
    of = np.stack([u, v], axis=-1)
    # convert to RGB using wheel colour coding
    rgb_image = flow_to_color(of, clip_flow=5)
    # display
    show_two(im1, rgb_image)

    # warping 1st image to the second
    im1_warped = p1.warp(im1, u, v)
    cost = p1.compute_cost(im1_warped, im2)
    print("Cost (Basic): {:4.3e}".format(cost))

    #
    # Iterative coarse-to-fine implementation
    #
    n_iter = 5  # number of iterations
    n_levels = 3  # levels in Gaussian pyramid

    pyr1 = p1.gaussian_pyramid(im1, nlevels=n_levels)
    pyr2 = p1.gaussian_pyramid(im2, nlevels=n_levels)

    u, v = p1.coarse_to_fine(im1, im2, pyr1, pyr2, n_iter)

    # warping 1st image to the second
    im1_warped = p1.warp(im1, u, v)
    cost = p1.compute_cost(im1_warped, im2)
    print("Cost (Coarse-to-fine): {:4.3e}".format(cost))

    # stacking for visualisation
    of = np.stack([u, v], axis=-1)
    # convert to RGB using wheel colour coding
    rgb_image = flow_to_color(of, clip_flow=5)
    # display
    show_two(im1, rgb_image)

if __name__ == "__main__":
    problem1()