import matplotlib.pyplot as plt
import numpy as np

#
# Problem 1
#
import problem1 as p1

def problem1(fsize=5, nlevel=3, sigma=1.4):
    """Example code implementing the steps in Problem 1"""
    """
    # show gaussian kernel
    gaussian_kernel = p1.gaussian_kernel(5, 1.4)
    plt.imshow(gaussian_kernel, cmap=plt.get_cmap('jet'), interpolation='nearest')
    plt.colorbar()
    plt.show()

    # show downsample x2
    img_path = 'data/facial_images/01.pgm'
    img = plt.imread(img_path)
    new_img = p1.downsample_x2(img)
    plt.figure()
    plt.subplot(1, 2, 1)
    plt.imshow(img)
    plt.subplot(1, 2, 2)
    plt.imshow(new_img)
    plt.show()

    # show gaussian pyramid
    GP = p1.gaussian_pyramid(img, 3, 5, 1.4)
    plt.figure()
    for i, g in enumerate(GP):
        plt.subplot(1, 3, i+1)
        plt.imshow(g)
    plt.show()
    """
    # finding matches
    root_dir = 'data'
    imgs, feats = p1.load_data(root_dir)

    # answers
    dist = p1.Distance()
    print(dist.answer())

    m = p1.find_matching_with_scale(imgs, feats)
    plt.figure()
    for i, g in enumerate(m):
        print(g[0])
        plt.subplot(1, 2, 1)
        plt.imshow(g[1])
        plt.subplot(1, 2, 2)
        plt.imshow(g[2])
        plt.show()

#
# Problem 2
#
import problem2 as p2
from utils import *

def problem2():
    """Example code implementing the steps in Problem 2"""

    # Task 1
    y, hw = p2.load_faces("./data/yale_faces")
    print("Loaded array: ", y.shape)

    # Using 2 random images for testing
    test_face2 = y[0, :]
    test_face = y[-1, :]
    show_images(np.stack([test_face, test_face2], 0), hw, (3, 1))

    # Task 3. compute PCA
    u, lmb = p2.compute_pca(y)

    # Tasks 4 and 5
    # percentiles of components
    ps = [0.5, 0.7, 0.8, 0.95]
    ims = []
    for i, p in enumerate(ps):
        b = p2.basis(u, lmb, p)
        ims.append(p2.project(test_face, b))

    show_images(np.stack(ims, 0), hw)

    # fix some basis
    b = p2.basis(u, lmb, 0.95)

    # Task 7. Image search
    top5 = p2.search(y, test_face, b, 5)
    show_images(top5, hw)

    # Taksk 8. Interpolation
    ints = p2.interpolate(test_face2, test_face, b, 5)
    show_images(ints, hw)

    plt.show()

if __name__ == "__main__":
    problem1()
    #problem2()
