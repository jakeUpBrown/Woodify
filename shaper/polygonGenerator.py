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


def get_dir_from_delta(delta):
    i = 0
    for d in diag_coord_circle:
        if d == delta:
            return i
        i += 1



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
    point_dirs = list()
    point_dirs.append(-1)
    # start in any direction?
    direct = 0

    passed_neighbors = list()
    while True:
        # get list of neighbor deltas
        neighbor_deltas = get_diag_neighbor_coord_list(direct)
        found_neighbor = False
        adjacent_neighbor_found = False
        neighbor_dir = -1
        found_neighbor_coords = None
        for d_x, d_y in neighbor_deltas:
            n_x = cur_x + d_x
            n_y = cur_y + d_y
            if group_nums[n_x][n_y] == group_num \
                    and isEdge[n_x][n_y] == True \
                    and isLine[n_x][n_y] == False \
                    and adjacent_neighbor_found == False:
                # found a neighbor in the same group that is an edge but has not been added to the line yet
                adjacent_neighbor_found = True

                if found_neighbor:
                    # this is an additional neighbor found, add to passed_neighbors
                    passed_neighbors.append(((cur_x, cur_y), (n_x, n_y)))
                else:
                    # add neighbor to the list of points
                    points.append((n_x, n_y))
                    point_dirs.append(direct)
                    isLine[n_x][n_y] = True
                    found_neighbor_coords = (n_x, n_y)
                    found_neighbor = True
            else:
                # every time a neighbor is not found, reset adjacent_neighbor_found to false
                adjacent_neighbor_found = False

            direct = (direct + 1) % 8

        if not found_neighbor:
            # either a deadend has been a found, or we should have passed_neighbors populated

            passed_neighbors_to_remove = list()
            # filter out passed_neighbors where the neighbor has been added already
            for passed_neighbor in passed_neighbors:
                _, neighbor_coords = passed_neighbor
                for pt in points:
                    if pt == neighbor_coords:
                        passed_neighbors_to_remove.append(passed_neighbor)
                        break

            for pn in passed_neighbors_to_remove:
                passed_neighbors.remove(pn)

            if len(passed_neighbors) > 0:
                last_passed_neighbor = passed_neighbors.pop()
                src_coords, passed_neighbor_coords = last_passed_neighbor
                points_since_passed_neighbor = set()
                # go through and add all previous points until you hit src_coords
                for pt in reversed(points):
                    points_since_passed_neighbor.add(pt)
                    if pt == src_coords:
                        break
                create_polygon_from_points(group_nums, points, group_num, isEdge, 1.0)
                points_since_passed_neighbor.remove((cur_x, cur_y))

                # do same path-finding as usual, except instead of looking at group_nums, isEdge, etc, you look
                # at whether or not the point is in points_since_passed_neighbor, until you find src_coords

                # increment dir 1, since dir should be pointing at the previous point of the deadend, and we want
                # to start 1 over clockwise from that point
                # temp_dir = (direct + 1) % 8
                temp_dir = (point_dirs[len(point_dirs) - 1] + 5) % 8
                while True:
                    neighbor_deltas = get_diag_neighbor_coord_list(temp_dir)
                    n = None
                    neighbor_found = False
                    for (d_x, d_y) in neighbor_deltas:
                        n = (cur_x + d_x, cur_y + d_y)
                        if n in points_since_passed_neighbor:
                            # points_since_passed_neighbor.remove(n)
                            points.append(n)
                            point_dirs.append(temp_dir)
                            neighbor_found = True
                            break
                        temp_dir = (temp_dir + 1) % 8

                    if not neighbor_found:
                        print('Couldnt find way home')
                    if n == src_coords:
                        found_neighbor_coords = passed_neighbor_coords
                        points.append(passed_neighbor_coords)
                        isLine[passed_neighbor_coords[0]][passed_neighbor_coords[1]] = True
                        neighbor_dir = get_dir_from_delta(
                            (passed_neighbor_coords[0] - src_coords[0], passed_neighbor_coords[1] - src_coords[1])
                        )
                        point_dirs.append(neighbor_dir)
                        break
                    cur_x, cur_y = n
                    temp_dir = (temp_dir + 4 + 1) % 8

            if len(passed_neighbors) == 0:
                # check if we are back next to the starting point
                if abs(cur_x - start_x) <= 1 and abs(cur_y - start_y) <= 1:
                    #TODO: add append neighbor_dirs?
                    points.append((start_x, start_y))
                    break
                else:
                    print('Group ' + str(group_num) + 'Couldnt find the way back to the starting point')

        # set dir to the direction from neighbor to previous point, then incremented by 1
        # because we want to start 1 position clockwise from original point
        # adding 4 does 180 degree turn,
        direct = (neighbor_dir + 4 + 1) % 8
        cur_x, cur_y = found_neighbor_coords

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
