
import numpy as np
from imageio import imread, imwrite
from skimage.color import rgb2gray
import matplotlib.pyplot as plt


GRAY_REP = 1
RGB_REP = 2
RGB_DIM = 3
GRAY_DIM = 1
HIGHEST_COLOR = 255
EPSILON = 0.0001

TO_YIQ =[[0.299, 0.596, 0.212],
        [0.587, -0.275, -0.523],
        [0.114, -0.321, 0.311]]
TO_RGB = np.linalg.inv(TO_YIQ)


def read_image(filename, representation):
    """
    :param filename: the filename of an image on disk (could be grayscale or RGB)
    :param representation: representation code, either 1 or 2 defining whether the output should be a grayscale
    :return: Normalized image with grayscal = [0, 1] in the wanted representation mode.
    """
    try:
        im = imread(filename)
    except (ValueError, FileNotFoundError) as e:
        print(e)
        exit(-1)
    else:
        if representation == GRAY_REP:
            if len(im.shape) == RGB_DIM:
                im = rgb2gray(im)
        if im.dtype == np.uint8:
            im = (im / HIGHEST_COLOR).astype(np.float64)
        return im


def imdisplay(filename, representation):
    converted = read_image(filename, representation)
    plt.figure()
    plt.imshow(converted)
    plt.axis('off')
    plt.show()


def rgb2yiq(imRGB):
    gray_img = np.dot(imRGB, TO_YIQ)
    return gray_img


def yiq2rgb(imYIQ):
    rgb_im = np.dot(imYIQ, TO_RGB)
    return rgb_im


def histogram_equalize(im_orig):
    dim = len(im_orig.shape)
    if dim == RGB_DIM:
        yiq_img = rgb2yiq(im_orig)
        im_cpy = yiq_img[:, :, 0]
    else:
        im_cpy = im_orig.copy()

    hist_orig, bounds = np.histogram(im_cpy, 256, (0, 1))
    com_orig = np.cumsum(hist_orig)
    c_m = np.min(com_orig[np.nonzero(com_orig)])
    T = (HIGHEST_COLOR * (com_orig - c_m) / (com_orig[-1] - c_m))

    im_eq = (T[(im_cpy * HIGHEST_COLOR).astype(np.uint8)])

    im_eq = im_eq.astype(np.float64)
    im_eq /= HIGHEST_COLOR
    hist_eq, bins = np.histogram(im_eq, 256, (0, 1))

    if dim == RGB_DIM:
        yiq_img[:, :, 0] = im_eq
        im_eq = yiq2rgb(yiq_img)

    im_eq = np.clip(im_eq, a_min=0, a_max=1)
    return im_eq, hist_orig, hist_eq


def find_nearest(array, value):
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return idx


def quantize (im_orig, n_quant, n_iter):
    dim = len(im_orig.shape)
    if dim == RGB_DIM:
        yiq_img = rgb2yiq(im_orig)
        im_cpy = yiq_img[:, :, 0]
    else:
        im_cpy = im_orig.copy()
    his, bins = np.histogram(im_cpy, 256, (0, 1))
    comulative_his = np.cumsum(his)
    error = []
    iter_count = 0
    q = np.zeros(n_quant, dtype=int)
    z = np.zeros(n_quant + 1, dtype=int)
    prev_z = np.zeros(n_quant + 1, dtype=int)
    frac = comulative_his[-1] / n_quant
    for i in range(1, n_quant):
        z[i] = find_nearest(comulative_his, frac * i)
        if i > 0 and z[i] <= z[i-1]:
            z[i] += 1
    z[-1] = HIGHEST_COLOR

    intensities = np.arange(0, len(his))
    numerator = np.multiply(his, intensities)

    while iter_count < n_iter and not np.array_equal(z, prev_z):
        prev_z = z.copy()
        cur_err = 0
        for i in range(len(q)):
            cur_numer = np.sum(numerator[z[i]+1: z[i+1]+1])
            cur_denom = np.sum(his[z[i]+1: z[i+1]+1])
            if i == 0:
                cur_numer += numerator[0]
                cur_denom += his[0]
            try:
                q[i] = round(cur_numer / cur_denom)
            except ValueError as e:
                print("Division by 0. n_quant too high for this image.")
                print(e)
                exit(-1)

        for i in range(1, len(q)):
            z[i] = int((q[i-1] + q[i]) / 2)

        for i in range(len(z) - 1):
            cur_err += np.sum((np.square(intensities[(z[i]+1): (z[i+1]+1)] - q[i])) * his[(z[i]+1): (z[
                                                                                                         i+1]+1)])
            if i == 0:
                cur_err += np.square(intensities[0] - q[i]) * his[0]
        error.append(cur_err)
        iter_count += 1

    im_quant = (im_cpy * HIGHEST_COLOR)
    for i in range(n_quant):
        im_quant[np.logical_and(z[i] < im_quant, im_quant <= z[i + 1])] = q[i]
    im_quant /= HIGHEST_COLOR

    if dim == RGB_DIM:
        yiq_img[:, :, 0] = im_quant
        im_quant = yiq2rgb(yiq_img)

    return [im_quant, error]
        

