"""build the dataset"""

import numpy as np
import os
import matplotlib.pyplot as plt
from model.find_tfl_lights import find_tfl_lights
from model.images_data import ImagesData

ROOT_PATH = "data_dir/train"


def insert_to_data_set(image, label):
    """  add data and labels to the files (at first time, create 2 binary files)"""

    with open(f"{ROOT_PATH}/data.bin", "ab") as data_file:
        np.array(image, dtype=np.uint8).tofile(data_file)
    with open(f"{ROOT_PATH}/labels.bin", "ab") as labels_file:
        labels_file.write(label.to_bytes(1, byteorder='big', signed=False))


def read_img_and_label(index, crop_size):
    """read the image and label from files using index and crop_size"""

    image = np.memmap(f"{ROOT_PATH}/data.bin",  dtype=np.uint8, mode='r', shape=crop_size,
                        offset=crop_size[0]*crop_size[1]*crop_size[2]*index)
    label = np.memmap(f"{ROOT_PATH}/labels.bin", dtype=np.uint8, mode='r', shape=(1,), offset=index)
    return image, label


def is_tfl_divide(x_axis: np.ndarray, y_axis: np.ndarray, label: np.ndarray):
    """return the points that are tfl and the points that are not"""

    x_true, y_true, x_false, y_false = [], [], [], []
    for x, y in zip(x_axis, y_axis):
        if label[y, x] == 19: # if it's a tfl
            x_true.append(x)
            y_true.append(y)
        else:
            x_false.append(x)
            y_false.append(y)
    return x_true, y_true, x_false, y_false


def crop_image(img: np.ndarray, x: int, y: int):
    """crop the image for given x and y"""
    crop_size = 40
    h, w, d = img.shape # define dimensions of image
    
    # define a matrix of zeros, padding the size of origin image with 40 in each side
    padded_img = np.zeros((crop_size*2+h, crop_size*2+w, 3), dtype=np.uint8) 

    # replace the original image in the center of the matrix
    padded_img[crop_size: -crop_size, crop_size: -crop_size] = img

    #crop the matrix to be 81*81. the center point is given (x,y)
    cropped = padded_img[y: y+crop_size*2+1, x: x+crop_size*2+1]
    return cropped


def crop_and_set(x_axis, y_axis, label, img):
    """crop the image for *every* suspect (x,y) point """
    for x, y in zip(x_axis, y_axis):
        cropped_img = crop_image(img, x, y)
        insert_to_data_set(cropped_img, label)


def build_dataset():
    for city in os.listdir("../leftImg8bit/train"):
        for img, label in ImagesData.images_and_labeled(city):
            x_red, y_red, x_green, y_green = find_tfl_lights(np.array(img))
            x = np.append(x_red, x_green)
            y = np.append(y_red, y_green)
            x_true, y_true, x_false, y_false = is_tfl_divide(x, y, np.array(label))
            min_size = min(len(x_true), len(x_false)) # send the same quantity of true and false tfls for correct training set
            crop_and_set(x_true[:min_size], y_true[:min_size], 1, np.array(img))
            crop_and_set(x_false[:min_size], y_false[:min_size], 0, np.array(img))
