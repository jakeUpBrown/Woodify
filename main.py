import cv2
import numpy as np
import matplotlib.pyplot as plt

def read_file(filename):
    img = cv2.imread(filename)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    plt.imshow(img)
    plt.show()
    return img


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
def cartoon(blurred, edges):
    c = cv2.bitwise_and(blurred, blurred, mask=edges)
    plt.imshow(c)
    plt.show()


k = 4
blur_d = 3
blur_iter = 3

plt.interactive(False)
# Reduce the noise
zion = "Zion 1.jpg"
girlFace = "Girl Face 1.jpg"
parrots = "parrots.jpg"

img = read_file(zion)
e = edge_mask(img)
plt.imshow(e)
plt.show()


img = color_quantization(img, k=k)

plt.imshow(img)
plt.show()

blurred = cv2.bilateralFilter(img, d=blur_d, sigmaColor=500, sigmaSpace=200)
plt.imshow(blurred)
plt.show()

img = color_quantization(blurred, k=k)
plt.title("color quantize again")
plt.imshow(img)
plt.show()

# cartoon(img, e)

