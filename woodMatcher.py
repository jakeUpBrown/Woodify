import os
import cv2
import numpy as np
import matplotlib.pyplot as plt

def load_all_woods():
    woodSampleDir = "woodSamples"
    woodImgs = []
    for filename in os.listdir(woodSampleDir):
        img = cv2.imread(os.path.join(woodSampleDir, filename))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        mean_color = cv2.mean
        woodImgs.extend((img, ))
    return woodImgs

def compare_color_and_wood(color, wood):
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
            match_values[i][j] = compare_color_and_wood(color, wood)
            j += 1

        i += 1

    print(match_values)
    # return pairs of colors to wood

wood_imgs = load_all_woods()



def get_wood_matches(img, group_nums):
    # for each group, find all original pixels and try to find the best wood to match that group of pixels
    height = len(group_nums)
    width = len(group_nums[0])

    group_wood_map = np.zeros((height, width), np.uint8)

    # for each group id (starting at 0), traverse entire image and collect list of all colors
    # initial idea: get the average color and compare it to average colors of the wood images

    current_group = 0
    while(True):
        group_color_list = []

        for i in range(0, len(group_nums)):
            for j in range(0, len(group_nums[0])):
                if group_nums[i][j] == current_group:
                    group_color_list.extend(img[i][j])

        if len(group_color_list) == 0:
            break

        # get average of colors
        current_group += 1


    for i in range(0, len(group_nums)):
        for j in range(0, len(group_nums[0])):
