import cv2
import numpy as np
import matplotlib.pyplot as plt
import skimage
import math
from imageLoader import read_file, SamplePicture
from imageProcessor import edge_mask, color_quantization
from findConnectedPoints import color_map_to_img, get_image_color_map
from woodMatcher import get_wood_matches

def convert_labels_to_group_nums(labels):
    height = len(labels)
    width = len(labels[0])

    # take the first value in the group 1 -> 0, 4 -> 1, 7 -> 2
    group_nums = np.zeros((height, width), np.uint8)

    for i in range(0, height):
        for j in range(0, width):
            group_nums[i][j] = math.floor(labels[i][j][0] / 3)

    return group_nums


k = 4
blur_d = 5
blur_iter = 3

img = read_file(SamplePicture.GIRL_FACE)
ogImg = read_file(SamplePicture.GIRL_FACE)
e = edge_mask(img)

for i in range(0,blur_iter):

    if(i > 0):
        img = cv2.bilateralFilter(img, d=blur_d, sigmaColor=500, sigmaSpace=200)
        #plt.title("blurred - " + str(i))
        #plt.imshow(img)
        #plt.show()

    img = color_quantization(img, k=k)
    #plt.title("color quantize - " + str(i))
    #plt.imshow(img)
    #plt.show()

color_map = get_image_color_map(img)
labels, num_labels = skimage.measure.label(img, return_num=True, connectivity=2)
print('before num_labels=', num_labels)
img_holes_filled = skimage.morphology.remove_small_objects(labels, 100)
img_holes_filled, num_labels = skimage.measure.label(img_holes_filled, return_num=True, connectivity=1)
print('after num_labels=', num_labels)

group_nums = convert_labels_to_group_nums(img_holes_filled)

get_wood_matches(ogImg, group_nums)

plt.imshow(img_holes_filled)
plt.show()
# cartoon(img, e)

