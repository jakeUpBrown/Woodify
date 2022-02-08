import numpy as np
from PIL import Image

# need 2 circles around so that you can get full list of 8 neighbor coordinates up to dir = 8
coord_circle = [
    (1, 0),
    (1, 1),
    (0, 1),
    (-1, 1),
    (-1, 0),
    (-1, -1),
    (0, -1),
    (1, -1),  # full circle here
    (1, 0),
    (1, 1),
    (0, 1),
    (-1, 1),
    (-1, 0),
    (-1, -1),
    (0, -1),
    (1, -1)
]


def get_neighbor_coord_list(dir = 0):
    last_index = dir + 8
    return coord_circle[dir:last_index]


def get_deltas_from_dir(dir):
    return coord_circle[dir]


def generatePolygons(group_nums):
    height = len(group_nums)
    width = len(group_nums[0])

    isEdge = np.zeros((height, width), bool)

    for i in range(0, height):
        for j in range(0, width):
            # check if [i][j] is on edge of screen
            if i == 0 or i == (height-1) or j == 0 or j == (width-1):
                isEdge[i][j] = True
                continue

            current_group = group_nums[i][j]
            # or, if it is bordering a different group
            # check all 8 neighbors (top, top-right, right, etc)
            for (dI, dJ) in get_neighbor_coord_list():
                if group_nums[i+dI][j+dJ] != current_group:
                    isEdge[i][j] = True
                    break

    edgeArray = np.array(isEdge, dtype=np.uint8) * 255

    edgeImg = Image.fromarray(edgeArray, mode='L')
    edgeImg.save("./edgeImg.jpg")

    isLine = np.zeros((height, width), bool)
