import matplotlib.pyplot as plt
from skimage import io
from skimage.transform import resize
from skimage.transform import warp, ProjectiveTransform
from stereo_utils import *
from skimage.color import rgb2gray, rgba2rgb
import argparse
import cv2


def main(args):
    '''
    A given pair of stereo images is taken from https://github.com/ethan-li-coding/SemiGlobalMatching
    '''

    # load image
    base_path = 'stereo_geometry/imgs/'
    im1 = io.imread(f'{base_path}{args.img1}')
    im2 = io.imread(f'{base_path}{args.img2}')

    if im1.shape[-1] == 4:
        im1 = rgb2gray(rgba2rgb(im1))
    else:
        im1 = rgb2gray(im1)

    if im2.shape[-1] == 4:
        im2 = rgb2gray(rgba2rgb(im2))
    else:
        im2 = rgb2gray(im2)

    # match points
    matched_pt1, matched_pt2 = compute_matching_points((im1*255).astype(np.uint8), (im2*255).astype(np.uint8))

    assert len(matched_pt1) == len(matched_pt2)

    matched_pt1 = coord2homo(matched_pt1)
    matched_pt2 = coord2homo(matched_pt2)

    # epipolar geometry
    F = compute_nomrd_fundamental_matrix(matched_pt1, matched_pt2)
    e1 = compute_epipole(F)
    e2 = compute_epipole(F.T)

    # print(np.round(e2.T @ F @ e1))

    # stereo rectification
    H1, H2 = compute_matching_homographies(e2, F, im2, matched_pt1, matched_pt2)

    new_points1 = H1 @ matched_pt1.T
    new_points2 = H2 @ matched_pt2.T
    new_points1 /= new_points1[2,:]
    new_points2 /= new_points2[2,:]
    new_points1 = new_points1.T
    new_points2 = new_points2.T

    # warp images based on the homography matrix
    im1_warped = warp(im1, ProjectiveTransform(matrix=np.linalg.inv(H1)))
    im2_warped = warp(im2, ProjectiveTransform(matrix=np.linalg.inv(H2)))


    h, w = im1.shape

    nrows = 2
    ncols = 1
    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(6, 8))


    # plot image 1
    ax1 = axes[0]
    ax1.set_title("Image 1 warped")
    ax1.imshow(im1_warped, cmap="gray")

    # plot image 2
    ax2 = axes[1]
    ax2.set_title("Image 2 warped")
    ax2.imshow(im2_warped, cmap="gray")

    # plot the epipolar lines and points
    # n = new_points1.shape[0]
    # for i in range(n):
    #     p1 = new_points1[i]
    #     p2 = new_points2[i]

    #     ax1.hlines(p2[1], 0, w, color="orange")
    #     ax1.scatter(*p1[:2], color="blue")

    #     ax2.hlines(p1[1], 0, w, color="orange")
    #     ax2.scatter(*p2[:2], color="blue")

    plt.tight_layout()
    plt.show()


if __name__=='__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--img1', default='first.png', help=' : the name of the first image')
    parser.add_argument('--img2', default='second.png', help=' : the name of the second image')

    args = parser.parse_args()
    

    main(args)
