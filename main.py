import cv2
import numpy as np
import matplotlib.pyplot as plt
import skimage
import math

from cartoonizer import cartoonize
from imageLoader import SamplePicture, WoodType, wood_type_from_name, write_output_picture, \
    read_sample_wood_pic, read_sample_photo
from imageProcessor import edge_mask, color_quantization, add_edges, pixel_size_to_router_bit_conversion
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
    wood_img = read_sample_wood_pic(wood_type_enum)

    #TODO: enhance this to create a realistically sized wood image
    wood_img = cv2.resize(wood_img, dim)

    mask = get_mask_from_group_nums(group_nums, current_group)

    if current_group == 0:
        wood_img_preview = np.zeros((height, width, 3), np.uint8)
        for h in range(0, height):
            for w in range(0, width):
                if mask[h][w] == 255:
                    wood_img_preview[h][w] = wood_img[h][w]

        plt.title('Group Num: ' + str(current_group))
        plt.imshow(wood_img_preview)
        plt.show()

    return wood_img, mask


def create_wood_preview(group_nums, group_wood_pairs, max_group_num, regionprops, real_img_shape):
    print('creating wood preview')
    height = len(group_nums)
    width = len(group_nums[0])
    cumulative_wood_img = np.zeros((height, width, 3), np.uint8)
    for group_key in group_wood_pairs:
        print('generating wood layer for group ' + str(group_key))
        wood_img, wood_img_mask = \
            get_masked_wood_image(group_nums, group_key, group_wood_pairs[group_key], regionprops, None)
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
img = read_sample_photo(image_filename)
ogImg = read_sample_photo(image_filename)

# img = pixel_size_to_router_bit_conversion(img, (1/16), 12, 12)
# plt.imshow(img)
# plt.show()

e = edge_mask(img)

cartoon_img, max_group_num = cartoonize(img)

group_nums = convert_labels_to_group_nums(cartoon_img)

group_wood_pairs = get_wood_matches(ogImg, group_nums)

regionprops = skimage.measure.regionprops(cartoon_img)
wood_preview = create_wood_preview(group_nums, group_wood_pairs, max_group_num, regionprops, None)

plt.imshow(wood_preview)
plt.show()

now = datetime.now()
image_filename_no_ext = image_filename.value.partition('.')[0]
write_output_picture(wood_preview, image_filename_no_ext + "_v1-0__" + now.strftime("%Y-%m-%d %H-%M") + ".png")

# get polygons
generatePolygons(group_nums, group_wood_pairs, 1)


