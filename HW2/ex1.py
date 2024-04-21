import cv2
import matplotlib.pyplot as plt
import numpy as np
import math
import os


def read_img(img_path):
    """
        Read grayscale image
        Inputs:
        img_path: str: image path
        Returns:
        img: cv2 image
    """
    return cv2.imread(img_path, 0)


def padding_img(img, filter_size=3):
    """
    The surrogate function for the filter functions.
    The goal of the function: replicate padding the image such that when applying the kernel with the size of filter_size, the padded image will be the same size as the original image.
    WARNING: Do not use the exterior functions from available libraries such as OpenCV, scikit-image, etc. Just do from scratch using function from the numpy library or functions in pure Python.
    Inputs:
        img: cv2 image: original image
        filter_size: int: size of square filter
    Return:
        padded_img: cv2 image: the padding image
    """
    # Need to implement here
    pad = (filter_size - 1) / 2

    top_and_right = round(pad)
    bot_and_left = int(pad)

    # 1st solution
    h, w = img.shape[:2]
    padded_img = np.zeros((h + filter_size - 1, w + filter_size - 1))

    padded_img[top_and_right: h + top_and_right, bot_and_left: w + bot_and_left] = img

    # pad left
    for i in range(bot_and_left):
        padded_img[top_and_right: h + top_and_right, i] = padded_img[top_and_right: h + top_and_right, bot_and_left]
    # pad right
    for i in range(w + bot_and_left, padded_img.shape[1]):
        padded_img[top_and_right: h + top_and_right, i] = padded_img[top_and_right: h + top_and_right, w + bot_and_left - 1]
    # pad top
    for i in range(top_and_right):
        padded_img[i, :] = padded_img[top_and_right, :] 
    # pad bot
    for i in range(h + top_and_right, padded_img.shape[0]):
        padded_img[i, :] = padded_img[h + top_and_right - 1, :]
    padded_img = padded_img.astype(img.dtype)

    # 2nd solution
    # padded_img = cv2.copyMakeBorder(img, top_and_right, bot_and_left, bot_and_left, top_and_right, cv2.BORDER_REPLICATE)    # top, bot, left, right

    return padded_img

def mean_filter(img, filter_size=3):
    """
    Smoothing image with mean square filter with the size of filter_size. Use replicate padding for the image.
    WARNING: Do not use the exterior functions from available libraries such as OpenCV, scikit-image, etc. Just do from scratch using function from the numpy library or functions in pure Python.
    Inputs:
        img: cv2 image: original image
        filter_size: int: size of square filter,
    Return:
        smoothed_img: cv2 image: the smoothed image with mean filter.
    """
    # Need to implement here
    smoothed_img = np.zeros_like(img)

    padded_img = padding_img(img, filter_size)
    h, w = padded_img.shape[:2]

    for i in range(0, h - filter_size + 1):
        for j in range(0, w - filter_size + 1):
            mean = np.mean(padded_img[i : i + filter_size, j : j + filter_size])
            smoothed_img[i, j] = mean

    return smoothed_img

def median_filter(img, filter_size=3):
    """
        Smoothing image with median square filter with the size of filter_size. Use replicate padding for the image.
        WARNING: Do not use the exterior functions from available libraries such as OpenCV, scikit-image, etc. Just do from scratch using function from the numpy library or functions in pure Python.
        Inputs:
            img: cv2 image: original image
            filter_size: int: size of square filter
        Return:
            smoothed_img: cv2 image: the smoothed image with median filter.
    """
    # Need to implement here
    smoothed_img = np.zeros_like(img)

    padded_img = padding_img(img, filter_size)
    h, w = padded_img.shape[:2]

    for i in range(0, h - filter_size + 1):
        for j in range(0, w - filter_size + 1):
            median = np.median(padded_img[i : i + filter_size, j : j + filter_size])
            smoothed_img[i, j] = median

    return smoothed_img


def psnr(gt_img, smooth_img):
    """
        Calculate the PSNR metric
        Inputs:
            gt_img: cv2 image: groundtruth image
            smooth_img: cv2 image: smoothed image
        Outputs:
            psnr_score: PSNR score
    """
    # Need to implement here

    mse = np.mean((gt_img - smooth_img) ** 2)
    max_pixel = 255
    psnr_score = 10 * math.log10(max_pixel ** 2 / mse)
    return psnr_score 

def show_res(before_img, after_img, title="nothing.png"):
    """
        Show the original image and the corresponding smooth image
        Inputs:
            before_img: cv2: image before smoothing
            after_img: cv2: corresponding smoothed image
        Return:
            None
    """
    # plt.title("Comparison between original and {} filter".format(title.split(".")[0]))
    plt.figure(figsize=(12, 9))
    plt.subplot(1, 2, 1)
    plt.imshow(before_img, cmap='gray')
    plt.title('Before')

    plt.subplot(1, 2, 2)
    plt.imshow(after_img, cmap='gray')
    plt.title('After')
    plt.savefig("HW2/outputs/" + title)
    plt.show()


if __name__ == '__main__':
    img_noise = "HW2/ex1_images/noise.png" # <- need to specify the path to the noise image
    img_gt = "HW2/ex1_images/ori_img.png" # <- need to specify the path to the gt image
    img = read_img(img_noise)
    filter_size = 3

    # Mean filter
    mean_smoothed_img = mean_filter(img, filter_size)
    show_res(img, mean_smoothed_img, "mean.png")
    print('PSNR score of mean filter: ', psnr(img, mean_smoothed_img))
    
    # Median filter
    median_smoothed_img = median_filter(img, filter_size)
    show_res(img, median_smoothed_img, "median.png")
    print('PSNR score of median filter: ', psnr(img, median_smoothed_img))

