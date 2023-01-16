import matplotlib.pyplot as plt
from skimage import io
from skimage.transform import resize
from skimage.transform import warp, ProjectiveTransform
from stereo_utils import *
from skimage.color import rgb2gray, rgba2rgb
import cv2


'''
images are taken from https://github.com/ethan-li-coding/SemiGlobalMatching
'''

# load image
im1 = io.imread('stereo_geometry/imgs/first.png')
im1 = rgb2gray(rgba2rgb(im1))
im2 = io.imread('stereo_geometry/imgs/second.png')
im2 = rgb2gray(rgba2rgb(im2))

# match points
