import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

#
# Problem 1
#

import problem1 as P1
def problem1():

    def cart2hom(p_c):
        p_h = np.concatenate([p_c, np.ones((p_c.shape[0], 1))], axis=1)
        return p_h

    def plot_epipolar(F, points1, points2, im1, im2):
        n = points1.shape[0]
        fig, (ax1, ax2) = plt.subplots(1, 2)
        ax1.imshow(im1, interpolation=None)
        ax1.axis("off")
        ax1.scatter(points1[:,0], points1[:,1])
        ax1.set_title("Epipolar lines in the left image")
        # determine coordinates of the epipolar lines
        # at the left and right borders of image 1
        X1, X2, Y1, Y2 = P1.draw_epipolars(F, points2, im1)
        for i in range(n):
            ax1.plot([X1[i], X2[i]], [Y1[i], Y2[i]], "r")
        ax1.set_ylim(top=0)

        ax2.imshow(im2, interpolation=None)
        ax2.axis("off")
        ax2.scatter(points2[:,0], points2[:,1])
        ax2.set_title("Epipolar lines in the right image")
        # determine coordinates of the epipolar lines
        # at the left and right borders of image 2
        X1, X2, Y1, Y2 = P1.draw_epipolars(F.T, points1, im2)
        for i in range(n):
            ax2.plot([X1[i], X2[i]], [Y1[i], Y2[i]], "r")
        ax2.set_ylim(top=0)

    im1 = plt.imread("data/a4p1a.png")
    im2 = plt.imread("data/a4p1b.png")

    data = np.load("data/points.npz")
    points1 = data['points1']
    points2 = data['points2']

    fig, (ax1, ax2) = plt.subplots(1, 2)
    ax1.imshow(im1, interpolation=None)
    ax1.axis("off")
    ax1.scatter(points1[:,0], points1[:,1])
    ax1.set_title("Keypoints in the left image")
    ax2.imshow(im2, interpolation=None)
    ax2.axis("off")
    ax2.scatter(points2[:,0], points2[:,1])
    ax2.set_title("Keypoints in the right image")

    # convert to homogeneous coordinates
    p1 = cart2hom(points1)
    p2 = cart2hom(points2)
    # compute fundamental matrix
    F = P1.eight_point(p1, p2)
    # plot the epipolar lines
    plot_epipolar(F, points1, points2, im1, im2)

    # compute the residuals
    max_residual, avg_residual = P1.compute_residuals(p1, p2, F)
    print('Max residual: ', max_residual)
    print('Average residual: ', avg_residual)

    # compute the epipoles
    e1, e2 = P1.compute_epipoles(F)
    print('Epipole in image 1: ', e1)
    print('Epipole in image 2: ', e2)

    plt.show()


#
# Problem 2
#
import problem2 as p2

def rgb2gray(im):
    return np.mean(im, -1)
def load_image(path):
    return plt.imread(path)
    
def disparity_read(filename):
    """ Return disparity read from filename. """
    f_in = np.array(Image.open(filename))
    d_r = f_in[:,:,0].astype('float64')
    d_g = f_in[:,:,1].astype('float64')
    d_b = f_in[:,:,2].astype('float64')

    disparity = d_r * 4 + d_g / (2**6) + d_b / (2**14)
    return disparity




def problem2():
    """Example code implementing the steps in Problem 2"""

    # Given parameter. No need to change
    max_disp = 15

    alpha = p2.optimal_alpha()
    print("Alpha: {:4.3f}".format(alpha))

    # Window size. You can freely change, but it should be an odd number
    window_size = 11

    # from utils.py
    im_left = rgb2gray(load_image("data/p2_left.png"))
    im_right = rgb2gray(load_image("data/p2_right.png"))
    disparity_gt = disparity_read("data/p2_gt.png")
   
    padded_img_l = p2.pad_image(im_left, window_size, padding_mode='symmetric')
    padded_img_r = p2.pad_image(im_right, window_size, padding_mode='symmetric') 

    disparity_res = p2.compute_disparity(padded_img_l, padded_img_r, max_disp, window_size, alpha)
    aepe = p2.compute_aepe(disparity_gt, disparity_res)
    print("AEPE: {:4.3f}".format(aepe))
        
    interp = 'bilinear'
    fig, axs = plt.subplots(nrows=2, sharex=True)
    axs[0].set_title('disparity_gt')
    axs[0].imshow(disparity_gt, vmin=0, vmax=20)
    axs[1].set_title('disparity_res')
    axs[1].imshow(disparity_res, vmin=0, vmax=20)
    plt.show()

if __name__ == "__main__":
    # problem1()
    problem2()
