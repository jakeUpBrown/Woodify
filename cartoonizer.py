import cv2
import skimage

from imageProcessor import color_quantization

def remove_small_objects(img, min_size=200):
    labels, num_labels = skimage.measure.label(img, return_num=True, connectivity=2)
    max_group_num = int(num_labels / 3) - 1
    print('before max_group_num=', max_group_num)
    img_holes_filled = skimage.morphology.remove_small_objects(labels, min_size=min_size, connectivity=2)
    img_holes_filled, num_labels = skimage.measure.label(img_holes_filled, return_num=True, connectivity=2)
    max_group_num = int(num_labels / 3) - 1
    print('after max_group_num=', max_group_num)
    return img_holes_filled, max_group_num


def cartoonize(img):
    total_pixels = len(img) * len(img[0])

    wood_pieces_limit = 50
    blur_d = 5
    blur_d_inc = .5
    k = 5
    min_size = int(total_pixels / 1000)
    min_size_inc = int(min_size / 8)

    iters = 1
    min_iters = 1

    use_blur = False
    while True:
        if use_blur:
            img = cv2.bilateralFilter(img, d=int(blur_d), sigmaColor=500, sigmaSpace=200)
            # plt.title("blurred - " + str(i))
            # plt.imshow(img)
            # plt.show()
        use_blur = True

        img = color_quantization(img, k=k)

        # get the num_labels
        img_small_objects_removed, max_group_num = remove_small_objects(img, min_size=min_size)
        # if it's within the desired limit return image
        if (max_group_num < wood_pieces_limit) and iters >= min_iters:
            return img_small_objects_removed, max_group_num

        min_size += min_size_inc
        blur_d += blur_d_inc
        iters += 1

