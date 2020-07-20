import math

import cv2
import numpy as np
from ipm_builder import convert_from_image_corners_to_ipm_corners, convert_from_ipm_corners_to_image_corners, \
    compute_projection_matrix, compute_ipm_and_corners, compute_ipm_fast_numba
from shared import draw_corners, convert_to_list_of_tuples
from shared import convert_to_list_of_arrays

last_good_ipm_corners = []
last_good_img_corners = []
q = []
projection_matrix = np.zeros((3, 4), np.float32)


def refresh_corners():
    global last_good_img_corners, last_good_ipm_corners, q
    last_good_img_corners = []
    last_good_ipm_corners = []
    q = []


def are_corners_valid(corners):
    angles = []
    for a in range(0, len(corners)):
        point_a = corners[a % 4]
        point_b = corners[(a + 1) % 4]
        point_c = corners[(a + 2) % 4]
        dot_product = (point_b[1] - point_a[1]) * (point_b[1] - point_c[1]) + (point_b[0] - point_a[0]) * (
                point_b[0] - point_c[0])
        angle = np.rad2deg(
            np.arccos(
                dot_product / (
                        math.sqrt(math.pow(point_b[1] - point_a[1], 2) + math.pow(point_b[0] - point_a[0], 2)) *
                        math.sqrt(math.pow(point_b[1] - point_c[1], 2) + math.pow(point_b[0] - point_c[0], 2))
                )
            )
        )
        angles.append(angle)
    is_rectangle = True
    for angle in angles:
        if not (88 <= abs(angle) <= 92):
            is_rectangle = False
    return is_rectangle


def check_if_set_of_corners_are_similar(corners1, corners2):
    for i in range(0, len(corners1)):
        corner1 = corners1[i]
        corner2 = corners2[i]
        if not (abs(corner1[0] - corner2[0]) <= 15 and abs(corner1[1] - corner2[1]) <= 15):
            return False
    return True


def compute_mean_corners(corners):
    corner_sums = np.zeros((4, 2), np.int32)
    for i in range(0, len(corners)):
        set_of_corners = corners[i]
        for j in range(0, len(set_of_corners)):
            corner = set_of_corners[j]
            temp_y = corner_sums[j][0] + corner[0]
            temp_x = corner_sums[j][1] + corner[1]
            corner_sums[j][0] = temp_y
            corner_sums[j][1] = temp_x
    result = []
    for i in range(0, 4):
        temp = np.int(corner_sums[i][0] / len(corners)), np.int(corner_sums[i][1] / len(corners))
        result.append(temp)
    return result


def validate_ipm_corners(first_frame, current_ipm_corners):
    global last_good_ipm_corners, q
    if first_frame:
        last_good_ipm_corners = current_ipm_corners
        good_ipm_corners = current_ipm_corners
    else:
        if are_corners_valid(current_ipm_corners):
            good_ipm_corners = current_ipm_corners
        else:
            good_ipm_corners = current_ipm_corners
            point_a = current_ipm_corners[0]  # BL
            point_b = current_ipm_corners[1]  # BR
            if (point_b[1] - point_a[1]) != 0:
                m_ab = (point_b[0] - point_a[0]) / (point_b[1] - point_a[1])
                temp_a = last_good_ipm_corners[0]
                temp_b = last_good_ipm_corners[1]
                temp_c = last_good_ipm_corners[2]
                temp_d = last_good_ipm_corners[3]
                length_bc = np.sqrt(
                    math.pow(temp_b[0] - temp_c[0], 2) + math.pow(temp_b[1] - temp_c[1], 2))
                m_bc = -1 / m_ab
                dx = (length_bc / np.sqrt(1 + (m_bc * m_bc)))
                dy = m_bc * dx
                good_ipm_corners[2] = (int(point_b[0] + dy), int(point_b[1] + dx))
                length_ad = np.sqrt(
                    math.pow(temp_a[0] - temp_d[0], 2) + math.pow(temp_a[1] - temp_d[1], 2))
                m_ad = -1 / m_ab
                dx = (length_ad / np.sqrt(1 + (m_ad * m_ad)))
                dy = m_ad * dx
                good_ipm_corners[3] = (int(point_a[0] + dy), int(point_a[1] + dx))
        if check_if_set_of_corners_are_similar(last_good_ipm_corners, good_ipm_corners):
            if len(q) < 5:
                q.append(good_ipm_corners)
            good_ipm_corners = compute_mean_corners(q)
        else:
            q = [good_ipm_corners]
        last_good_ipm_corners = good_ipm_corners
    return good_ipm_corners


def compute_rotation_angle(point_a, point_b):
    m = (point_a[1] - point_b[1]) / (point_a[0] - point_b[0])
    angle = np.rad2deg(np.arctan(m))
    return "%.2f" % angle


def validate_image_corners(first_frame, img, current_image_corners):
    global projection_matrix, last_good_img_corners
    if first_frame:
        projection_matrix = compute_projection_matrix(img, current_image_corners)
    current_ipm_corners = convert_from_image_corners_to_ipm_corners(projection_matrix, current_image_corners)
    final_ipm_corners = validate_ipm_corners(first_frame, current_ipm_corners)
    final_image_corners = convert_from_ipm_corners_to_image_corners(img, projection_matrix, final_ipm_corners)

    if len(final_image_corners) < 4:
        final_image_corners = last_good_img_corners
        raise Exception
    else:
        last_good_img_corners = final_image_corners

    return final_image_corners


def validate_image_corners_using_completely_built_ipm(first_frame, img, current_image_corners):
    global projection_matrix
    if first_frame:
        projection_matrix = compute_projection_matrix(img, current_image_corners)
    ipm_image, current_ipm_corners = compute_ipm_and_corners(projection_matrix, img, current_image_corners)
    final_ipm_corners = validate_ipm_corners(first_frame, current_ipm_corners)
    final_image_corners = convert_from_ipm_corners_to_image_corners(img, projection_matrix, final_ipm_corners)

    # draw_corners(img, current_image_corners, (0, 0, 255))
    # draw_corners(img, final_image_corners, (255, 0, 0))
    # draw_corners(ipm_image, current_ipm_corners, (0, 0, 255))
    # draw_corners(ipm_image, final_ipm_corners, (255, 0, 0))
    cv2.imshow("ipm!!!", ipm_image)
    return final_image_corners