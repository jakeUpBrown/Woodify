import cv2
import numpy as np
import math


def convert_labels_to_group_nums(labels, return_group_ct=False):
    height = len(labels)
    width = len(labels[0])

    # take the first value in the group 1 -> 0, 4 -> 1, 7 -> 2
    group_nums = np.zeros((height, width), np.int64)
    if return_group_ct:
        group_cts = {}
        for i in range(0, height):
            for j in range(0, width):
                group_num = math.floor(labels[i][j][0] / 3)
                if group_num not in group_cts:
                    group_cts[group_num] = 1
                else:
                    group_cts[group_num] += 1
                group_nums[i][j] = group_num

        return group_nums, group_cts

    else:
        for i in range(0, height):
            for j in range(0, width):
                group_nums[i][j] = math.floor(labels[i][j][0] / 3)

        return group_nums


def pixel_size_to_router_bit_conversion2(img, bit_size, max_real_height, max_real_width):
    height = len(img)
    width = len(img[0])

    aspect_ratio = width / height
    max_theor_width = max_real_height * aspect_ratio
    max_theor_height = max_real_width / aspect_ratio

    real_width = min(max_theor_width, max_real_width)
    real_height = min(max_theor_height, max_real_height)

    if real_width < max_real_width:
        scale_ratio = real_width/max_real_width
    elif real_height < max_real_height:
        scale_ratio = real_height/max_real_height
    else:
        raise Exception('SOMETHING WENT WRONG in pixel_size_to_router_bit_conversion')

    return cv2.resize(img, (0,0), fx=scale_ratio, fy=scale_ratio)


def pixel_size_to_router_bit_conversion(img, bit_size, max_real_height, max_real_width):
    height = len(img)
    width = len(img[0])

    aspect_ratio = width / height
    max_theor_width = max_real_height * aspect_ratio
    max_theor_height = max_real_width / aspect_ratio

    real_width = min(max_theor_width, max_real_width)
    real_height = min(max_theor_height, max_real_height)

    new_img_size = (int(real_width / bit_size), int(real_height / bit_size))

    return cv2.resize(img, new_img_size)


# Create Edge Mask
def edge_mask(img, line_size=7, blur_value=7):
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    gray_blur = cv2.medianBlur(gray, blur_value)

    edges = cv2.adaptiveThreshold(gray_blur, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, line_size, blur_value)

    return edges


# Reduce the Color Palette
def color_quantization(img, k):
    # transform image
    data = np.float32(img).reshape((-1, 3))

    # determine criteria
    criteria = (cv2.TermCriteria_EPS + cv2.TERM_CRITERIA_MAX_ITER, 20, 0.001)

    # implementing K-means
    ret, label, center = cv2.kmeans(data, k, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
    center = np.uint8(center)

    result = center[label.flatten()]
    result = result.reshape(img.shape)

    return result


# Combine Edge Mask with Quantiz Img
def add_edges(img, edges, edge_shade_multiplier=.95):
    height = len(img)
    width = len(img[0])

    for i in range(0, height):
        for j in range(0, width):
            if edges[i][j] == True:
                og_rgb = img[i][j]
                new_rgb = [int(element * edge_shade_multiplier) for element in og_rgb]
                img[i][j] = new_rgb

    return img


# How to detect islands?
def detect_islands(group_nums):
    print('detect_islands')
    # traverse through each group number and


