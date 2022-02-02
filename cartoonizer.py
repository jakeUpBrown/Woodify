import cv2
import skimage

from imageProcessor import color_quantization

def remove_small_objects(img, min_size=200):
    labels, num_labels = skimage.measure.label(img, return_num=True, connectivity=2)
    max_group_num = int(num_labels / 3) - 1
    print('before max_group_num=', max_group_num)
    img_holes_filled = skimage.morphology.remove_small_objects(labels, min_size=min_size, connectivity=1)
    img_holes_filled, num_labels = skimage.measure.label(img_holes_filled, return_num=True, connectivity=1)
    max_group_num = int(num_labels / 3) - 1
    print('after max_group_num=', max_group_num)
    return img_holes_filled, max_group_num


def cartoonize(img):
    wood_pieces_limit = 50
    blur_d = 5
    k = 5

    use_blur = False
    while True:
        if use_blur:
            img = cv2.bilateralFilter(img, d=blur_d, sigmaColor=500, sigmaSpace=200)
            # plt.title("blurred - " + str(i))
            # plt.imshow(img)
            # plt.show()
        use_blur = True

        img = color_quantization(img, k=k)

        # get the num_labels
        img, max_group_num = remove_small_objects(img)
        # if it's within the desired limit return image
        if max_group_num < wood_pieces_limit:
            return img, max_group_num

