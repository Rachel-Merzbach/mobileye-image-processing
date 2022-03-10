"""build a kernel and using it to convolve the image and find the traffic lights in the image"""

import numpy as np
from scipy import signal as sg
from skimage.feature import peak_local_max
from skimage import img_as_float


def create_kernel():
    """create a kernel"""
    kernel = np.array([
        [1/2,  1/2,   1/4,   1/4,   1/8,   1/8],
        [1/2,  1/4,   1/4,   1/8,   1/8,  -1/2],
        [1/4,  1/4,   1/8,   1/8,   1/8,  -1/2],
        [1/4,  1/8,   1/8,   0,     0,    -1/2],
        [1/8,  1/8,   1/8,  -1/2,  -1/2,  -1/2],
        [1/8,  1/8,  -1/2,  -1/2,  -1/2,  -1/2]
    ])

    return kernel


def find_tfl_lights(c_image: np.ndarray):
    """find the green and red lights for a given image"""

    kernel = create_kernel()

    red_x = []
    red_y = []
    green_x = []
    green_y = []

    if c_image is not None:
        red = img_as_float(c_image[:, :, 0])
        green = img_as_float(c_image[:, :, 1])

        conv_red = sg.convolve(red, kernel, mode='same')
        conv_green = sg.convolve(green, kernel, mode='same')

        # get the lightest coordinates of the convolved images
        red_coordinates = peak_local_max(conv_red, min_distance=70, num_peaks=10)
        green_coordinates = peak_local_max(conv_green, min_distance=70, num_peaks=10)

        # for every red coordinate, add to red_x and red_y if the red color is dominant
        for coordinate in red_coordinates:
            rgb_pixel = c_image[coordinate[0]][coordinate[1]] # get the pixel (rgb) from origin image by the coordinate of red_coordinates
            if rgb_pixel[0] > rgb_pixel[1] and rgb_pixel[0] > rgb_pixel[2]:
                red_x.append(coordinate[1])
                red_y.append(coordinate[0])
        # for every green coordinate, add to green_x and green_y if the green color is dominant
        for coordinate in green_coordinates:
            rgb_pixel = c_image[coordinate[0]][coordinate[1]] # get the pixel (rgb) from origin image by the coordinate of green_coordinates
            if rgb_pixel[1] >= rgb_pixel[0]:
                green_x.append(coordinate[1])
                green_y.append(coordinate[0])  

    return red_x, red_y, green_x, green_y



