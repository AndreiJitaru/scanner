import numpy as np
import cv2


def convert_to_list_of_tuples(corners):
    result = []
    for a in range(0, len(corners)):
        corner = tuple(corners[a])
        result.append(corner)
    return result


def convert_to_list_of_arrays(corners):
    result = np.zeros((4, 2), np.int32)
    for a in range(0, len(corners)):
        y, x = corners[a]
        result[a][0] = y
        result[a][1] = x
    return result


def draw_corners(img, corners, color):
    for a in range(0, len(corners)):
        point_a = corners[a % 4]
        point_b = corners[(a + 1) % 4]
        # cv2.circle(img, point_a, 20, color, 2)
        cv2.line(img, point_a, point_b, color, 2)