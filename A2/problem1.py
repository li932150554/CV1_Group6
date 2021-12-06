import os
import numpy as np
import matplotlib.pyplot as plt
from numpy.core.numeric import argwhere
from numpy.lib.function_base import append
from scipy.signal import convolve2d
from scipy.ndimage.filters import convolve
import numpy.linalg as linalg

from PIL import Image
from scipy.signal.filter_design import bilinear_zpk

def load_data(path):
    '''
    Load data from folder data, face images are in the folder facial_images, face features are in the folder facial_features.
    

    Args:
        path: path of folder data

    Returns:
        imgs: list of face images as numpy arrays 
        feats: list of facial features as numpy arrays 
    '''

    imgs = []
    feats = []

    imgs_file_path = os.path.join(path, "facial_images")
    # print("imgs_file_path: " ,imgs_file_path)
    imgs_list = os.listdir(imgs_file_path)
    # print("imgs_list: ", imgs_list)

    feats_file_path = os.path.join(path, "facial_features")
    feats_list = os.listdir(feats_file_path)

    for i in imgs_list:
        img_path = os.path.join(imgs_file_path, i)
        # print("current img_path: ", img_path)
        im = plt.imread(img_path)
        imgs.append(im)
    
    for i in feats_list:
        feat_path = os.path.join(feats_file_path, i)
        im = plt.imread(feat_path)
        feats.append(im)

    return imgs, feats

def gaussian_kernel(fsize, sigma):
    '''
    Define a Gaussian kernel

    Args:
        fsize: kernel size
        sigma: sigma of Gaussian kernel

    Returns:
        The Gaussian kernel
    '''
    # reference: https://stackoverflow.com/questions/47369579/how-to-get-the-gaussian-filter 
    kernel = np.fromfunction(lambda x, y: (1/(2*np.pi*sigma**2))*np.exp((-1*((x-(fsize-1)/2)**2+(y-(fsize-1)/2)**2))/(2*sigma**2)), (fsize, fsize))
    # kernel = np.zeros((fsize, fsize))
    # for i in range(fsize):
    #   for j in range(fsize):
    #     kernel[i][j] = (1/(2*np.pi*sigma**2))*np.exp((-1*((i-(fsize-1)/2)**2+(j-(fsize-1)/2)**2))/(2*sigma**2))
    kernel /= np.sum(kernel)

    return kernel

def downsample_x2(x, factor=2):
    '''
    Downsampling an image by a factor of 2

    Args:
        x: image as numpy array (H * W)

    Returns:
        downsampled image as numpy array (H/2 * W/2)
    '''

    height, width  = x.shape
    
    downsample = np.empty([int(height/2), int(width/2)])

    for i in range(0,height-1,2):
        for j in range(0,width-1,2):
            downsample[int(i/2)][int(j/2)] = (int(x[i][j]) + int(x[i+1][j]) + int(x[i][j+1]) + int(x[i+1][j+1]))/4
            #print(x[i][j], '+', x[i+1][j], '+', x[i][j+1], '+', x[i+1][j+1])
            #total = (int(x[i][j]) + int(x[i+1][j]) + int(x[i][j+1]))
            #print(total)



    return downsample


def gaussian_pyramid(img, nlevels, fsize, sigma):
    '''
    A Gaussian pyramid is constructed by combining a Gaussian kernel and downsampling.
    Tips: use scipy.signal.convolve2d for filtering image.

    Args:
        img: face image as numpy array (H * W)
        nlevels: number of levels of Gaussian pyramid, in this assignment we will use 3 levels
        fsize: Gaussian kernel size, in this assignment we will define 5
        sigma: sigma of Gaussian kernel, in this assignment we will define 1.4

    Returns:
        GP: list of Gaussian downsampled images, it should be 3 * H * W
    '''
    GP = []
    currentImg = img
    GP.append(currentImg)
    for level in range(nlevels-1):
        downImg = downsample_x2(currentImg)
        GP.append(downImg)
        GaussImg = convolve(downImg, gaussian_kernel(fsize, sigma))
        currentImg = GaussImg

    return GP

def template_distance(v1, v2):
    '''
    Calculates the distance between the two vectors to find a match.
    Browse the course slides for distance measurement methods to implement this function.
    Tips: 
        - Before doing this, let's take a look at the multiple choice questions that follow. 
        - You may need to implement these distance measurement methods to compare which is better.

    Args:
        v1: vector 1
        v2: vector 2

    Returns:
        Distance
    '''
    
    """ Dot-Product """
    #distance = v2.T*v1 / (linalg.norm(v1)*linalg.norm(v2))
    distance = np.dot(v1, v2) / (linalg.norm(v1)*linalg.norm(v2))
     
    """ the sum of squared differences(SSD) """

    #distance = linalg.norm(v1 - v2)**2

    return distance


def sliding_window(img, feat, step=1):
    ''' 
    A sliding window for matching features to windows with SSDs. When a match is found it returns to its location.
    
    Args:
        img: face image as numpy array (H * W)
        feat: facial feature as numpy array (H * W)
        step: stride size to move the window, default is 1
    Returns:
        min_score: distance between feat and window
    '''
    H_img, W_img = np.shape(img)
    H_feat, W_feat = np.shape(feat) 

    # Case of the feat is bigger than the image
    if(H_img < H_feat and W_img < W_feat): 
        img = np.pad(img, [(0, H_feat-H_img), (0, W_feat-W_img)], mode = 'constant')
        H_img, W_img = np.shape(img)

    # cut off the boundary
    score=[]
    for x in range(0, (H_img - H_feat + 1), step):
        for y in range(0, (W_img - W_feat + 1), step):
            
            sample = img[x:x+H_feat, y:y+W_feat]
            temp = template_distance(feat.flatten(), sample.flatten())
            score.append(temp) # calculate the distance
    
    return min(score)



class Distance(object):

    # choice of the method
    METHODS = {1: 'Dot Product', 2: 'SSD Matching'}

    # choice of reasoning
    REASONING = {
        1: 'it is more computationally efficient',
        2: 'it is less sensitive to changes in brightness.',
        3: 'it is more robust to additive Gaussian noise',
        4: 'it can be implemented with convolution',
        5: 'All of the above are correct.'
    }

    def answer(self):
        '''Provide your answer in the return value.
        This function returns one tuple:
            - the first integer is the choice of the method you will use in your implementation of distance.
            - the following integers provide the reasoning for your choice.
        Note that you have to implement your choice in function template_distance

        For example (made up):
            (1, 1) means
            'I will use Dot Product because it is more computationally efficient.'
        '''

        return (None, None)  # TODO


def find_matching_with_scale(imgs, feats):
    ''' 
    Find face images and facial features that match the scales 
    
    Args:
        imgs: list of face images as numpy arrays
        feats: list of facial features as numpy arrays 
    Returns:
        match: all the found face images and facial features that match the scales: N * (score, g_im, feat)
        score: minimum score between face image and facial feature
        g_im: face image with corresponding scale
        feat: facial feature
    '''


    GP_sum = []
    imgs_length = np.shape(imgs)[0]
    feats_length = np.shape(feats)[0]
    for i in range(imgs_length): # create in sum 5x3 gaussian pyramids / -1 because in range start at 0
        GP = gaussian_pyramid(imgs[i], 3, 5, 1.4)
        GP_sum.append(GP)  # at the end GP_sum be (5, 3)

    GP_sum = np.array(GP_sum, dtype=object).flatten() #in order to reshape must change into np.array
    # print("size of feats:", feats_length)
    # print("shape of GP_sum:", np.shape(GP_sum))
    # plt.figure()
    # plt.imshow(GP_sum[0])
    # plt.show()

    score_all = np.zeros((5, 15)) 
    match = []
    score = []
    g_im = []
    feat_all =[]
    for i in range(feats_length):
        # score_sum = []
        for j in range(np.size(GP_sum)):
            score_all[i,j] = sliding_window(GP_sum[j], feats[i], step=1)
            # score_sum.append(sliding_window(GP_sum[j], feats[i], step=1)) # Every feats go through all gaussian pyramids
        # score_sum = np.array(score_sum)
        index_min = np.argmin(score_all[i,:])
        """
        score.append(score_all[i,index_min])
        g_im.append(GP_sum[j])
        
        #g_im.append(GP_sum[list.index(score_sum[i])])
        #g_im.append(GP_sum[np.argwhere(score_sum = np.min(score_sum))])
        """
        feat_all.append(feats[i])
        match.append((score_all[i,index_min], GP_sum[index_min], feats[i]))

    print("score:", score)


    #match = [(score, g_im, feat_all)]

    return match