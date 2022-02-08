import cv2
import numpy as np
import matplotlib.pyplot as plt
import skimage
import math

from cartoonizer import cartoonize
from imageLoader import read_file, SamplePicture, WoodType, wood_type_from_name, write_output_picture
from imageProcessor import edge_mask, color_quantization, add_edges
from findConnectedPoints import color_map_to_img, get_image_color_map
from shaper.polygonGenerator import generatePolygons
from woodMatcher import get_wood_matches
from datetime import datetime

def convert_labels_to_group_nums(labels):
    height = len(labels)
    width = len(labels[0])

    # take the first value in the group 1 -> 0, 4 -> 1, 7 -> 2
    group_nums = np.zeros((height, width), np.uint8)

    for i in range(0, height):
        for j in range(0, width):
            group_nums[i][j] = math.floor(labels[i][j][0] / 3)

    return group_nums


def get_mask_from_group_nums(group_nums, current_group):
    height = len(group_nums)
    width = len(group_nums[0])

    mask = np.zeros((height, width), dtype="uint8")
    for h in range(0, height):
        for w in range(0, width):
            if group_nums[h][w] == current_group:
                mask[h][w] = 255

    return mask


# should return wood image that covers the bounding box of the group, but is also proportional to the real size of the
# wood.
def get_masked_wood_image(group_nums, current_group, wood_type, regionprops, real_img_shape):
    height = len(group_nums)
    width = len(group_nums[0])
    dim = (width, height)

    wood_type_enum = WoodType[wood_type]
    wood_img = read_file(wood_type_enum)

    #TODO: enhance this to create a realistically sized wood image
    wood_img = cv2.resize(wood_img, dim)

    mask = get_mask_from_group_nums(group_nums, current_group)

    return wood_img, mask


def create_wood_preview(group_nums, group_wood_pairs, max_group_num, regionprops, real_img_shape):
    print('creating wood preview')
    height = len(group_nums)
    width = len(group_nums[0])
    cumulative_wood_img = np.zeros((height, width, 3), np.uint8)
    for j in range(0, max_group_num + 1):
        print('generating wood layer for group ' + str(j))
        wood_img, wood_img_mask = get_masked_wood_image(group_nums, j, group_wood_pairs[j], regionprops, None)
        # add to cumulative_wood_img

        for h in range(0, height):
            for w in range(0, width):
                if wood_img_mask[h][w] == 255:
                    cumulative_wood_img[h][w] = wood_img[h][w]

        #plt.imshow(cumulative_wood_img)
        #plt.show()

    print('returning wood preview')
    return cumulative_wood_img


image_filename = SamplePicture.GIRL_FACE
img = read_file(image_filename)
ogImg = read_file(image_filename)
e = edge_mask(img)

cartoon_img, max_group_num = cartoonize(img)

group_nums = convert_labels_to_group_nums(cartoon_img)

# get polygons
generatePolygons(group_nums)
group_wood_pairs = get_wood_matches(ogImg, group_nums)

regionprops = skimage.measure.regionprops(cartoon_img)
wood_preview = create_wood_preview(group_nums, group_wood_pairs, max_group_num, regionprops, None)

plt.imshow(wood_preview)
plt.show()

# img_with_edges = add_edges(wood_preview, e)
# plt.imshow(img_with_edges)
# plt.show()

now = datetime.now()
write_output_picture(wood_preview, image_filename + "_v1-0__" + now.strftime("%Y-%m%d_%H-%M"))
# cartoon(img, e)

