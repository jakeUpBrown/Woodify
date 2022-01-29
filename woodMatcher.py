import json
import os

import cv2
import numpy as np
from colormath.color_diff import delta_e_cie1976
from colormath.color_objects import LabColor

from commonColorFinder import get_color_freqs
from imageLoader import read_file, WoodType
from imageProcessor import color_quantization
import matplotlib.pyplot as plt


class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)


wood_hists_path = 'wood_hists.json'


def generate_wood_hists():
    wood_hists = dict()
    for wood in WoodType:
        woodImg = read_file(wood)
        # woodImg = cv2.cvtColor(woodImg, cv2.COLOR_RGB2LAB)
        hist = get_color_freqs(woodImg, 5, use_lab_values=True)
        wood_hists[wood.name] = hist

    with open(wood_hists_path, 'w') as outfile:
        json.dump(wood_hists, outfile, indent=2, cls=NumpyEncoder)

    return wood_hists


def load_wood_hists():
    if not os.path.exists(wood_hists_path):
        return generate_wood_hists()

    with open(wood_hists_path, 'r') as infile:
        wh = json.load(infile)
        conv_wh = {}
        for e in wh:
            e_list = wh[e]
            conv_wh[e] = list()
            for h in e_list:
                conv_h = tuple([h[0], np.asarray(h[1])])
                conv_wh[e].append(conv_h)
        return conv_wh


def compare_hists(hist1, hist2):
    c_val = 0
    for h1 in hist1:
        color1 = LabColor(lab_l=h1[1][0], lab_a=h1[1][1], lab_b=h1[1][2])
        for h2 in hist2:
            color2 = LabColor(lab_l=h2[1][0], lab_a=h2[1][1], lab_b=h2[1][2])
            delta_e = delta_e_cie1976(color1, color2)
            c_val += h1[0] * h2[0] * delta_e
    return c_val

# send in list of colors and list of available wood
def get_wood_list(colors, woods):
    # check that there are more wood options than colors
    if len(colors) < len(woods):
        print('Too many colors. Not enough wood options')
        return None

    match_values = ([None] * len(colors)) * len(woods)
    i = 0
    for color in colors:
        j = 0
        for wood in woods:
            # match_values[i][j] = compare_color_and_wood(color, wood)
            j += 1

        i += 1

    print(match_values)
    # return pairs of colors to wood


def get_group_neighbors(group_nums, max_group):
    height = len(group_nums)
    width = len(group_nums[0])

    group_neighbors = np.zeros(shape=(max_group, max_group), dtype=bool)
    for i in range(0, height):
        for j in range(0, width):
            current_group = group_nums[i][j]
            if i > 0:
                top_neighbor = group_nums[i-1][j]
                if top_neighbor != current_group:
                    group_neighbors[top_neighbor][current_group] = True
                    group_neighbors[current_group][top_neighbor] = True
            if i < height - 1:
                bottom_neighbor = group_nums[i + 1][j]
                group_neighbors[bottom_neighbor][current_group] = True
                group_neighbors[current_group][bottom_neighbor] = True

            if j > 0:
                left_neighbor = group_nums[i][j-1]
                group_neighbors[left_neighbor][current_group] = True
                group_neighbors[current_group][left_neighbor] = True

            if j < width - 1:
                right_neighbor = group_nums[i][j+1]
                group_neighbors[right_neighbor][current_group] = True
                group_neighbors[current_group][right_neighbor] = True

    return group_neighbors


def get_wood_matches(img, group_nums):

    wood_hists = load_wood_hists()

    # convert to k=100(?) color quantization
    lab_img = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)
    k100img = color_quantization(lab_img, 100)

    cv2.imwrite("outputPhotos/k100_girl_face.jpg", k100img)
    # for each group, find all original pixels and try to find the best wood to match that group of pixels
    height = len(group_nums)
    width = len(group_nums[0])

    group_wood_map = np.zeros((height, width), np.uint8)

    group_ct = {}
    for i in range(0, height):
        for j in range(0, width):
            current_group = group_nums[i][j]
            if current_group not in group_ct:
                group_ct[current_group] = 1
            else:
                group_ct[current_group] += 1
    # for each group id (starting at 0), traverse entire image and collect list of all colors
    # initial idea: get the average color and compare it to average colors of the wood images

    group_hists = {}
    current_group = 0
    while (True):
        group_color_ct = {}

        for i in range(0, height):
            for j in range(0, width):
                if group_nums[i][j] == current_group:
                    # add to map of color freq
                    current_color = tuple(k100img[i][j])
                    if current_color not in group_color_ct:
                        group_color_ct[current_color] = 1
                    else:
                        group_color_ct[current_color] += 1

        if len(group_color_ct) == 0:
            break

        # get average of colors
        srt = {}
        top5sum = 0
        n = 0
        for k, v in sorted(group_color_ct.items(), reverse=True, key=lambda item: item[1]):
            if n < 5:
                srt[k] = v
                top5sum += v
            n += 1

        # convert each of them into percent of total top 5 colors ct
        hist = []
        for k, v in srt.items():
            t = ((v / top5sum), np.asarray(k))
            hist.append(t)

        group_hists[current_group] = hist
        current_group += 1

    max_group_num = current_group - 1

    # compare hist to all wood types on files and add to color_to_wood table
    color_wood_comp = dict()
    print('done with group_hists')
    group_idx = 0
    for g_num in group_hists:
        g_hist = group_hists[g_num]
        for w_val in wood_hists:
            w_hist = wood_hists[w_val]
            comp_val = compare_hists(g_hist, w_hist)
            color_wood_comp[tuple([g_num, w_val])] = comp_val
            print('Group ' + str(g_num) + ' vs ' + w_val + " = " + str(comp_val))
        group_idx += 1

    group_neighbors = get_group_neighbors(group_nums, max_group_num + 1)
    color_wood_comp = sorted(color_wood_comp.items(), key=lambda item:item[1])

    ineligible_pairs = set()
    color_wood_pairs = dict()
    while len(color_wood_comp) > 0:
        # get the first item in the color_wood_comp, since it is sorted, and that represents the best group-wood pair
        pair = color_wood_comp.pop(0)
        matched_group = pair[0][0]
        if matched_group in color_wood_pairs or pair[0] in ineligible_pairs:
            continue
        matched_wood = pair[0][1]
        color_wood_pairs[matched_group] = matched_wood

        # find all neighbors
        for j in range(0, max_group_num + 1):
            if group_neighbors[matched_group][j] and j not in color_wood_pairs:
                # add j,WOOD tuple key from color_wood_comp
                ineligible_pairs.add(tuple([j, matched_wood]))

    cumulative_wood_img = None

    # create mask for group number
    for i in range(0, max_group_num + 1):
        print(i)
        mask = np.zeros(img.shape[:2], dtype="uint8")
        for h in range(0, height):
            for w in range(0, width):
                if group_nums[h][w] == i:
                    mask[h][w] = 255

        current_wood = WoodType(color_wood_pairs[i])
        read_file(current_wood)
        masked = cv2.bitwise_and(img, img, mask=mask)
        plt.imshow(masked)
        plt.show()
    # combine mask with wood image
    # combine all masked images into 1

    print('tester')
# function that takes in image, converts to k=100(?) color quantization
# takes a certain group and gets the top 5 represented colors in that section and creates a histogram
# compares that to all know wood histograms that we have
# creates group_to_wood table that has a value for every group_id to every possible wood
# traverses the chart and grabs the highest value and assigns that wood value to that group
# for that group, reduce all of it's neighboring groups to 0 compare_score for the selected wood
# grab the next highest value from the group_to_wood table

# final output is group_id -> wood_id pairs for every group

# once we've done that, we need to create masked images of the wood textures for every group
# combine them into 1 image?


# take group_neighbors and the hist comparisons and pick out the best matches for each wood, excluding neighbors
# create masked images of each wood pattern and stitch them together