import cv2
import skimage

from imageProcessor import color_quantization, convert_labels_to_group_nums
from woodMatcher import get_group_neighbors


def remove_small_holes_with_one_neighbor(group_nums, group_keys):
    # get_group_neighbors(group_nums, )
    return group_nums


def remove_small_objects(img, wood_pieces_limit, min_size=200):
    labels, num_labels = skimage.measure.label(img, background=-1, return_num=True, connectivity=1)

    max_group_num = int(num_labels / 3) - 1
    print('before max_group_num=', max_group_num)
    img_holes_removed = skimage.morphology.remove_small_objects(labels, min_size=min_size, connectivity=1)
    labels, num_labels = skimage.measure.label(img_holes_removed, background=-1, return_num=True, connectivity=1)
    max_group_num = int(num_labels / 3) - 1
    print('after max_group_num=', max_group_num)

    group_nums, group_cts = convert_labels_to_group_nums(labels, return_group_ct=True)
    group_neighbors = get_group_neighbors(group_nums, group_cts)

    small_label_set = set()
    for k, v in group_cts.items():
        if v < min_size:
            small_label_set.add(k)

    invalid_group_ct = 0
    invalid_sn_group_ct = 0
    sn_group_pairs = dict()
    for k, v in group_cts.items():
        if v < min_size:
            # print('group num: ' + str(k) + ' has size of ' + str(v))
            # check if their neighbor dict only has 1 key (meaning they only have one neighbor)
            neighbors = group_neighbors[k]
            # filter keys included in small_label_set
            filtered_neighbors = set(filter(lambda key: key not in small_label_set, neighbors))

            if len(filtered_neighbors) < len(neighbors):
                print('yea baby')
            length = len(filtered_neighbors)
            if length <= 1:
                invalid_sn_group_ct += 1
                if length == 0:
                    sn_group_pairs[k] = next(iter(neighbors))
                else:
                    sn_group_pairs[k] = next(iter(filtered_neighbors))

            invalid_group_ct += 1

    real_count = max_group_num - invalid_sn_group_ct
    print('found ' + str(invalid_group_ct) + '/' + str(max_group_num) + ' invalid groups')
    print('found ' + str(max_group_num - invalid_group_ct) + ' real shapes')
    print('single neighbors: ' + str((invalid_sn_group_ct / invalid_group_ct) * 100) + '%')
    print('real label count: ' + str(real_count))

    if real_count <= wood_pieces_limit:
        height = len(group_nums)
        width = len(group_nums[0])
        for i in range(0, height):
            for j in range(0, width):
                current_group = group_nums[i][j]
                if current_group in sn_group_pairs:
                    group_nums[i][j] = sn_group_pairs[current_group]

    return group_nums, real_count


def cartoonize(img):
    wood_pieces_limit = 75
    iters = 1
    min_iters = 2

    # color_quantization values
    k = 5

    # blur values
    blur_d = 3
    blur_d_inc = .5
    sigma_color = 500
    sigma_space = 200

    # remove_small_objects values
    min_size = 150
    min_size_inc = int(min_size / 8)

    use_blur = False
    while True:
        if use_blur:
            img = cv2.bilateralFilter(img, d=int(blur_d), sigmaColor=sigma_color, sigmaSpace=sigma_space)
            # plt.title("blurred - " + str(i))
            # plt.imshow(img)
            # plt.show()
        use_blur = True

        img = color_quantization(img, k=k)

        # get the num_labels
        img_small_objects_removed, max_group_num = remove_small_objects(img, wood_pieces_limit, min_size=min_size)
        # if it's within the desired limit return image
        if (max_group_num < wood_pieces_limit) and iters >= min_iters:
            return img_small_objects_removed, max_group_num

        min_size += min_size_inc
        blur_d += blur_d_inc
        iters += 1
