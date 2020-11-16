import numpy as np
import math
import sys


def distance(x1, y1, x2, y2):
    return math.sqrt(math.pow(x1 - x2, 2) + math.pow(y1 - y2, 2))


def othercar_rad_grid(x, y):
    return (math.atan2(y, x) + math.pi) / (2 * math.pi)


def point_line_dist(x, y, a, b, c):
    return abs(a * x + b * y + c) / math.sqrt(math.pow(a, 2) + math.pow(b, 2))


def lat_lng_orientation(x, y, orient_car, reference_path):
    ref = np.array(reference_path)
    min_value = 1000000000
    min_index = 0
    lng_dist = 0

    for i in range(ref.shape[0]):

        diff = distance(x, y, ref[i][0], ref[i][1])

        if min_value > diff:
            min_value = diff
            min_index = i

    # This case will not happen in training so ignore this.
    if min_index == ref.shape[0] - 1:
        return 0, 0, 0

    # Nearest reference point from mycar
    x1 = ref[min_index][0]
    y1 = ref[min_index][1]

    # Next point of the nearest reference point
    x2 = ref[min_index + 1][0]
    y2 = ref[min_index + 1][1]

    # line connecting (x1, y1), (x2, y2)
    y_in_line = ((y2 - y1) / (x2 - x1)) * (x - x1) + y1

    # lat_dist : -25 ~ 25 => -1 ~ +1
    lat_dist = point_line_dist(x, y, (y2 - y1) / (x2 - x1), -1, y1 - (y2 - y1) / (x2 - x1))

    if y_in_line < y1:
        lat_dist = -1 * lat_dist

    lat_dist = lat_dist / 25

    # lng dist : 0 ~ 70 => 0 ~ 1
    # print("min_index", min_index)
    for j in range(min_index, ref.shape[0] - 1):
        lng_dist = lng_dist + distance(ref[j][0], ref[j][1], ref[j + 1][0], ref[j + 1][1])
        # print("lng_dist", lng_dist)

    lng_dist = lng_dist / 70

    orient_car = orient_car + 180
    ref_rad = othercar_rad_grid(ref[min_index + 1][0] - ref[min_index][0], ref[min_index + 1][1] - ref[min_index][1])

    # orient_car : -pi ~ +pi => 0 ~ 2pi => 0 ~ 1
    # ref_rad : -pi ~ +pi => 0 ~ 2pi => 0 ~ 1
    # orientation : -1 ~ +1
    orientation = math.radians(orient_car) / (2 * math.pi) - ref_rad

    return lat_dist, lng_dist, orientation

# from calculate_distance import lat_lng_orientation


# import numpy as np
# import math
# import sys
#
#
# def distance(x1, y1, x2, y2):
#     return math.sqrt(math.pow(x1-x2, 2) + math.pow(y1-y2, 2))
#
#
# def othercar_rad_grid(x, y):
#     return (math.atan2(y, x) + math.pi) / (2 * math.pi)
#
#
# def lat_lng_orientation(x, y, orient_car, reference_path):
#     ref = np.array(reference_path)
#     min_value = 1000000000
#     min_index = 0
#     lng_dist = 0
#
#     for i in range(ref.shape[0]):
#
#         diff = distance(x, y, ref[i][0], ref[i][1])
#
#         if min_value > diff:
#             min_value = diff
#             min_index = i
#
#     print("min_index", min_index)
#     for j in range(min_index, ref.shape[0] - 1):
#         lng_dist = lng_dist + distance(ref[j][0], ref[j][1], ref[j+1][0], ref[j+1][1])
#         print("lng_dist", lng_dist)
#
#     ref_rad = othercar_rad_grid(ref[min(min_index + 1, ref.shape[0] - 1)][0] - ref[min_index][0], ref[min(min_index + 1, ref.shape[0] - 1)][1] - ref[min_index][1])
#
#     orientation = math.radians(orient_car) - ref_rad
#
#     sys.exit()
#     return min_value, lng_dist, orientation
#
#
# # from calculate_distance import lat_lng_orientation
#
# # lat, lng, ori = lat_lng_orientation(actor_snapshot.get_transform().location.x, actor_snapshot.get_transform().location.y, actor_snapshot.get_transform.rotation.yaw, reference_path)