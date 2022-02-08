import cv2
import numpy as np
import matplotlib.pyplot as plt
from imageLoader import read_file, SamplePicture

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


