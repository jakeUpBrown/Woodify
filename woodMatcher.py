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
        woodImgs.extend(img)
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
