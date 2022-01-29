import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import json
from commonColorFinder import get_color_freqs
from imageProcessor import color_quantization
from imageLoader import read_file, WoodType


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
    # will need some fancier way to compare these 2 hists since cv2.compareHist seems to need the same rgb values
    # in each hist?
    c_val = 0
    for h1 in hist1:
        for h2 in hist2:
            h1[0]
    return 1

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


# wood_imgs = load_all_woods()


def get_wood_matches(img, group_nums):

    wood_hists = load_wood_hists()

    # convert to k=100(?) color quantization
    k100img = color_quantization(img, 100)

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

    # compare hist to all wood types on files and add to color_to_wood table
    print('done with group_hists')
    group_idx = 0
    for g_num in group_hists:
        g_hist = group_hists[g_num]
        for w_val in wood_hists:
            w_hist = wood_hists[w_val]
            comp_val = compare_hists(g_hist, w_hist)
            print('Group ' + str(g_num) + ' vs ' + w_val + " = " + str(comp_val))
        group_idx += 1
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
