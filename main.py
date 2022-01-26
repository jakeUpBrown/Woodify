import cv2
import numpy as np
import matplotlib.pyplot as plt
import skimage
from imageLoader import read_file, SamplePicture
from imageProcessor import edge_mask, color_quantization
from findConnectedPoints import color_map_to_img, get_image_color_map


k = 4
blur_d = 5
blur_iter = 3

img = read_file(SamplePicture.JUNGLE_BOOK)
e = edge_mask(img)

for i in range(0,blur_iter):

    if(i > 0):
        img = cv2.bilateralFilter(img, d=blur_d, sigmaColor=500, sigmaSpace=200)
        plt.title("blurred - " + str(i))
        plt.imshow(img)
        plt.show()

    img = color_quantization(img, k=k)
    plt.title("color quantize - " + str(i))
    plt.imshow(img)
    plt.show()

color_map = get_image_color_map(img)
labels, num_labels = skimage.measure.label(img, return_num=True, connectivity=1)

print(num_labels)
plt.imshow(labels)
plt.show()
# cartoon(img, e)

