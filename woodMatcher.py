import json
import os

import cv2
import numpy as np
from colormath.color_diff import delta_e_cie1976
from colormath.color_objects import LabColor

from commonColorFinder import get_color_freqs
from imageLoader import WoodType, read_sample_wood_pic
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
        woodImg = read_sample_wood_pic(wood)
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


# returns dict of dict (k = group_num, v = (k = group_num, v = isNeighbor))
def get_group_neighbors(group_nums, group_keys):
    height = len(group_nums)
    width = len(group_nums[0])

    group_neighbors = {}
    for group_key in group_keys:
        group_neighbors[group_key] = {}
    for i in range(0, height):
        for j in range(0, width):
            current_group = group_nums[i][j]
            if i > 0:
                top_neighbor = group_nums[i - 1][j]
                if top_neighbor != current_group:
                    group_neighbors[top_neighbor][current_group] = True
                    group_neighbors[current_group][top_neighbor] = True
            if i < height - 1:
                bottom_neighbor = group_nums[i + 1][j]
                group_neighbors[bottom_neighbor][current_group] = True
                group_neighbors[current_group][bottom_neighbor] = True

            if j > 0:
                left_neighbor = group_nums[i][j - 1]
                group_neighbors[left_neighbor][current_group] = True
                group_neighbors[current_group][left_neighbor] = True

            if j < width - 1:
                right_neighbor = group_nums[i][j + 1]
                group_neighbors[right_neighbor][current_group] = True
                group_neighbors[current_group][right_neighbor] = True

    return group_neighbors


def get_wood_matches(img, group_nums):
    """
       get_wood_matches takes in image and group numbers and returns a map between group number and wood type

       :param img: RGB original image
       :param group_nums: 2d array indicating which piece each pixel belongs to
       :return: dict where key = group number, and value = WoodType
    """
    print('getting wood matches')
    wood_hists = load_wood_hists()

    # convert to k=100(?) color quantization
    print('convert RGB to LAB')
    lab_img = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)

    print('color quantize w/ 100 colors')
    k100img = color_quantization(lab_img, 100)

    # for each group, find all original pixels and try to find the best wood to match that group of pixels
    height = len(group_nums)
    width = len(group_nums[0])

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

    print('group_ct', group_ct)
    group_color_cts = {}

    for i in range(0, height):
        for j in range(0, width):
            current_group = group_nums[i][j]
            if current_group not in group_color_cts:
                group_color_ct = group_color_cts[current_group] = {}
            else:
                group_color_ct = group_color_cts[current_group]

            current_color = tuple(k100img[i][j])

            if current_color not in group_color_ct:
                group_color_ct[current_color] = 1
            else:
                group_color_ct[current_color] += 1

    group_hists = {}
    for (group_num, group_color_ct) in group_color_cts.items():
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

        group_hists[group_num] = hist

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

    group_neighbors = get_group_neighbors(group_nums, group_color_cts.keys())
    color_wood_comp = sorted(color_wood_comp.items(), key=lambda item: item[1])

    ineligible_pairs = set()
    color_wood_pairs = dict()
    print('matching wood')
    while len(color_wood_comp) > 0:
        # get the first item in the color_wood_comp, since it is sorted, and that represents the best group-wood pair
        pair = color_wood_comp.pop(0)
        matched_group = pair[0][0]
        if matched_group in color_wood_pairs or pair[0] in ineligible_pairs:
            continue
        matched_wood = pair[0][1]
        color_wood_pairs[matched_group] = matched_wood

        # find all neighbors
        for neighbor in group_neighbors[matched_group].keys():
            if neighbor not in color_wood_pairs:
                ineligible_pairs.add(tuple([neighbor, matched_wood]))

    print('done matching woods')
    return color_wood_pairs
