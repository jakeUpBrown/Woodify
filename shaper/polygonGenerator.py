import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

# need 2 circles around so that you can get full list of 8 neighbor coordinates up to dir = 8
diag_coord_circle = [
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

orthogonal_coord_deltas = [
    (1, 0),
    (-1, 0),
    (0, 1),
    (0, -1)
]


def get_diag_neighbor_coord_list(dir=0):
    last_index = dir + 8
    return diag_coord_circle[dir:last_index]


def get_deltas_from_dir(dir):
    return diag_coord_circle[dir]


def generate_edges_array(group_nums, print_debug=False, connectivity=1):
    height = len(group_nums)
    width = len(group_nums[0])

    # key = group_num, value = (key = neighbor_group, value = (x,y))
    edge_starting_points = dict()
    isEdge = np.zeros((height, width), bool)

    for i in range(0, height):
        for j in range(0, width):
            # check if [i][j] is on edge of screen
            if i == 0 or i == (height - 1) or j == 0 or j == (width - 1):
                isEdge[i][j] = True
                continue

            current_group = group_nums[i][j]
            # or, if it is bordering a different group
            # check all 8 neighbors (top, top-right, right, etc)
            if connectivity == 1:
                coord_list = orthogonal_coord_deltas
            elif connectivity == 2:
                coord_list = get_diag_neighbor_coord_list()
            else:
                print('Connectivity has to be 1 or 2')
                return

            for (dI, dJ) in coord_list:
                neighbor_group = group_nums[i + dI][j + dJ]
                if neighbor_group != current_group:
                    isEdge[i][j] = True

                    current_group_dict = None
                    if current_group not in edge_starting_points:
                        current_group_dict = edge_starting_points[current_group] = dict()
                    else:
                        current_group_dict = edge_starting_points[current_group]

                    if neighbor_group not in current_group_dict:
                        current_group_dict[neighbor_group] = (i + dI, j + dJ)
                    break

    if print_debug:
        edgeArray = np.array(isEdge, dtype=np.uint8) * 255

        edgeImg = Image.fromarray(edgeArray, mode='L')
        edgeImg.save("./edgeImg.jpg")

    return isEdge, edge_starting_points


# return a list of x,y coordinates that represent the outer edge of the polygon for a given group
def get_polygon_points_for_group(group_nums, starting_points, isEdge, isLine, group_num):
    (start_x, start_y) = list(starting_points.values())[0]
    cur_x = start_x
    cur_y = start_y

    # create list and add starting point to list
    points = list()
    points.append((cur_x, cur_y))
    # start in any direction?
    dir = 0

    while True:
        # get list of neighbor deltas
        neighbor_deltas = get_diag_neighbor_coord_list(dir)
        found_neighbor = False
        for d_x, d_y in neighbor_deltas:
            n_x = cur_x + d_x
            n_y = cur_y + d_y
            if group_nums[n_x][n_y] == group_num and isEdge[n_x][n_y] == True and isLine[n_x][n_y] == False:
                # found a neighbor in the same group that is an edge but has not been added to the line yet

                # add neighbor to the list of points
                points.append((n_x, n_y))

                isLine[n_x][n_y] = True
                cur_x = n_x
                cur_y = n_y

                # set dir to the direction from neighbor to previous point, then incremented by 1 TODO: maybe by 2?
                # because we want to start 1 position clockwise from original point
                # adding 4 does 180 degree turn,
                dir = (dir + 4 + 1) % 8
                found_neighbor = True
                break

            dir = (dir + 1) % 8

        # for now, assume any deadend found should be the end
        # TODO: expand this to account for deadends
        if not found_neighbor:
            points.append((start_x, start_y))
            break

        # set cur_x and cur_y to neighbor and repeat with dir being the direction from neighbor to previous point
    return points


def create_polygon_from_points(group_nums, points, current_group, isEdge, size_multiplier):
    height = len(group_nums)
    width = len(group_nums[0])

    polygon_vals = np.zeros((height, width, 3), np.uint8)

    for i in range(0, height):
        for j in range(0, width):
            if group_nums[i][j] == current_group:
                if isEdge[i][j]:
                    polygon_vals[i][j] = (122, 122, 122)
                else:
                    polygon_vals[i][j] = (25, 25, 25)

    for point in points:
        polygon_vals[point[0]][point[1]] = (255, 255, 255)

    last_idx = len(points) - 2
    polygon_vals[points[last_idx][0]][points[last_idx][1]] = (255, 0, 0)

    polygon_vals[points[0][0]][points[0][1]] = (0, 255, 0)

    plt.imshow(polygon_vals)
    plt.show()

    print('create polygon')


def generatePolygons(group_nums, group_wood_pairs, size_multiplier):
    height = len(group_nums)
    width = len(group_nums[0])

    isEdge, edge_starting_points = generate_edges_array(group_nums)
    isLine = np.zeros((height, width), bool)

    for group_num in group_wood_pairs:
        starting_points = edge_starting_points[group_num]
        points = get_polygon_points_for_group(group_nums, starting_points, isEdge, isLine, group_num)
        create_polygon_from_points(group_nums, points, group_num, isEdge, size_multiplier)
