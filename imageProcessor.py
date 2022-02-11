import cv2
import numpy as np


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
def add_edges(img, edges):
    return cv2.bitwise_and(img, img, mask=edges)
    # plt.imshow(c)
    # plt.show()



# How to detect islands?
def detect_islands(group_nums):
    print('detect_islands')
    # traverse through each group number and


