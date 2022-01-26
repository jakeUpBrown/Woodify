import cv2
import numpy as np
import matplotlib.pyplot as plt
import skimage
from imageLoader import read_file, SamplePicture


def get_image_color_map(img):
    height = img.shape[0]
    width = img.shape[1]
    color_dict = dict()
    color_map = np.zeros((height, width), np.uint8)
    latest_color_id = 0
    for i in range(0, height):
        for j in range(0, width):
            rgb = img[i][j]
            rgb_tup = tuple(rgb)
            # find color id
            if rgb_tup in color_dict:
                color_map[i][j] = color_dict[rgb_tup]
            else:
                color_dict[rgb_tup] = latest_color_id
                color_map[i][j] = latest_color_id
                latest_color_id += 1
    print(color_map)
    return np.array(color_map), color_dict

def color_map_to_img(color_map, color_dict):
    height = len(color_map)
    width = len(color_map[0])
    inv_color_dict = {v: k for k, v in color_dict.items()}

    img = np.zeros((height, width, 3), np.uint8)
    for i in range(0, height):
        for j in range(0, width):
            img[i][j] = inv_color_dict[color_map[i][j]]

    return img


def path_finding(x, y, img, group_nums):
    print('path_finding')

# do a searching algorithm that then adds that group # to the groupNums array
def find_connected_points(img):
    width = 1000
    height = 1000
    groupNums = [[None] * width] * height
    currentGroup = 0
    for x in range(0, width):
        for y in range(0, height):
            # skip if pixel has already been checked
            if groupNums[x][y] is not None:
                continue

            currentColor = img[x][y]
            # run path finding algorithm that assigns current groupNum to all pixels found


            currentGroup += 1
            print('here')

