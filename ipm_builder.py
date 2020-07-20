import math as math
import numpy as np
from numba import njit
from config import MAX_HEIGHT_3D, MAX_WIDTH_3D, MIN_WIDTH_3D, MIN_HEIGHT_3D, CELL_SIZE
from shared import convert_to_list_of_tuples, convert_to_list_of_arrays


def get_line_equations(corners):
    point_a = corners[0]
    point_b = corners[3]
    line_ab = point_a[0] - point_b[0], point_b[1] - point_a[1], point_a[1] * point_b[0] - point_b[1] * point_a[0]
    point_c = corners[1]
    point_d = corners[2]
    line_cd = point_c[0] - point_d[0], point_d[1] - point_c[1], point_c[1] * point_d[0] - point_d[1] * point_c[0]
    return line_ab, line_cd


def get_vanishing_point(corners):
    line_ab, line_cd = get_line_equations(corners)
    # cv2.line(img, tuple([int(-line_ab[2] / line_ab[1]), 0]), tuple([0, int(-line_ab[2] / line_ab[0])]), (0, 255,
    # 0), 2)
    # cv2.line(img, tuple([int(-line_cd[2] / line_cd[1]), 0]), tuple([0, int(-line_cd[2] / line_cd[0])]), (0,
    # 255, 0), 2)
    print("line_ab: ", line_ab[0], " * x +", line_ab[1], " * y + (", line_ab[2], ")")
    print("line_cd: ", line_cd[0], " * x +", line_cd[1], " * y + (", line_cd[2], ")")
    determinant = line_ab[0] * line_cd[1] - line_cd[0] * line_ab[1]
    if determinant == 0:
        return False
    else:
        x = (line_ab[1] * line_cd[2] - line_cd[1] * line_ab[2]) / determinant
        y = (line_cd[0] * line_ab[2] - line_ab[0] * line_cd[2]) / determinant
        print("vanishing point:", (x, y))
        return y, x


def get_focal_length(img):
    height, width = img.shape[0], img.shape[1]
    d = math.sqrt(math.pow(width, 2) + math.pow(height, 2))
    cot = 1 / math.tan(np.deg2rad(39))
    return d / 2 * cot


def get_projected_paper_width(img, corners):
    height, width = img.shape[0], img.shape[1]
    line_ab, line_cd = get_line_equations(corners)
    first_x = -(line_ab[0] * height + line_ab[2]) / line_ab[1]
    second_x = -(line_cd[0] * height + line_cd[2]) / line_cd[1]
    first_x = -(line_ab[1] * height + line_ab[2]) / line_ab[0]
    second_x = -(line_cd[1] * height + line_cd[2]) / line_cd[0]
    return second_x - first_x


def get_pitch_angle(height, row, focal_length):
    aux = (height / 2 - row) / focal_length
    return np.rad2deg(np.arctan(aux))


def get_yaw_angle(width, col, focal_length):
    aux = (width / 2 - col) / focal_length
    return np.rad2deg(np.arctan(aux))


def get_vertical_half_field_of_view(height, focal_length):
    aux = (height / 2) / focal_length
    return np.rad2deg(np.arctan(aux))


def compute_projection_matrix(img, corners):
    height, width = img.shape[0], img.shape[1]
    vanishing_point = get_vanishing_point(corners)
    focal_length = get_focal_length(img)
    pitch = get_pitch_angle(height, vanishing_point[1], focal_length)
    vertical_half_field_of_view = get_vertical_half_field_of_view(height, focal_length)
    w = get_projected_paper_width(img, corners)
    d = math.sqrt(math.pow(focal_length, 2) + pow(height / 2, 2))
    D = (d * 300) / w
    sigma = 90 - vertical_half_field_of_view - pitch
    h = D * math.cos(np.deg2rad(sigma))
    yaw = get_yaw_angle(width, vanishing_point[0], focal_length)

    A = np.float32([[focal_length, 0, width / 2],
                    [0, focal_length, height / 2],
                    [0, 0, 1]])
    Rx = np.float32([[1, 0, 0],
                     [0, np.cos(np.deg2rad(pitch)), -np.sin(np.deg2rad(pitch))],
                     [0, np.sin(np.deg2rad(pitch)), np.cos(np.deg2rad(pitch))]])
    Ry = np.float32([[np.cos(np.deg2rad(yaw)), 0, np.sin(np.deg2rad(yaw))],
                     [0, 1, 0],
                     [-np.sin(np.deg2rad(yaw)), 0, np.cos(np.deg2rad(yaw))]])
    R = np.matmul(Rx, Ry)
    Tcw = np.float32([[0], [-h], [0]])
    Twc = -1 * np.matmul(R, Tcw)
    X = np.concatenate((R, Twc), axis=1)
    P = np.matmul(A, X)
    print("image width =", width, "| image height =", height)
    print("focal length = ", focal_length)
    print("pitch = ", pitch)
    print("yaw = ", yaw)
    print("vertical half field of view = ", vertical_half_field_of_view)
    print("w = ", w)
    print("d = ", d)
    print("D = ", D)
    print("h = ", h)
    print(P)
    return P


def compute_ipm_slow(img, P, corners):
    height, width = img.shape[0], img.shape[1]
    width_3D = MAX_WIDTH_3D - MIN_WIDTH_3D
    height_3D = MAX_HEIGHT_3D - MIN_HEIGHT_3D
    IPM_width = int(width_3D / CELL_SIZE)
    IPM_height = int(height_3D / CELL_SIZE)
    result = np.zeros((IPM_height, IPM_width, 3), np.uint8)
    d = {}
    for corner in corners:
        d[corner] = []
    print(IPM_height * IPM_width)
    for y in range(0, IPM_height):
        for x in range(0, IPM_width):
            x_3D = (x - IPM_width / 2) * CELL_SIZE
            z_3D = MIN_HEIGHT_3D + y * CELL_SIZE
            point_3D = np.float32([
                [x_3D],
                [0],
                [z_3D],
                [1]
            ])
            point_IPM = np.matmul(P, point_3D)
            if point_IPM[2] != 0:
                x_new = int(np.round(point_IPM[0] / point_IPM[2]))
                y_new = int(np.round(point_IPM[1] / point_IPM[2]))
                if 0 <= x_new < width and 0 <= y_new < height:
                    result[y][x] = img[y_new][x_new]
                    for corner in corners:
                        if corner[1] - 3 <= y_new <= corner[1] + 3 and corner[0] - 3 <= x_new <= corner[0] + 3:
                            d[corner].append((x_new, y_new, x, y))
    ipm_corners = []
    for corner in d:
        min_dist = math.inf
        best_candidate = (-1, -1, -1, -1)
        candidates = d[corner]
        for candidate in candidates:
            dist = np.sqrt(np.power(corner[0] - candidate[0], 2) + np.power(corner[1] - candidate[1], 2))
            if dist < min_dist:
                min_dist = dist
                best_candidate = candidate
        ipm_corners.append((best_candidate[2], best_candidate[3]))
    print(ipm_corners)
    return result, ipm_corners


@njit(['Tuple((uint8[:,:,:], int32[:,:]))(uint8[:,:,:], float32[:,:], int32[:,:])'])
def compute_ipm_fast_numba(img, P, corners):
    height, width = img.shape[0], img.shape[1]
    width_3D = MAX_WIDTH_3D - MIN_WIDTH_3D
    height_3D = MAX_HEIGHT_3D - MIN_HEIGHT_3D
    IPM_width = int(width_3D / CELL_SIZE)
    IPM_height = int(height_3D / CELL_SIZE)
    result = np.zeros((IPM_height, IPM_width, 3), np.uint8)
    candidates = np.zeros((4, 2000), np.int32)
    indices = np.zeros(4, np.int32)
    for y in range(0, IPM_height):
        for x in range(0, IPM_width):
            x_3D = (x - IPM_width / 2) * CELL_SIZE
            z_3D = MIN_HEIGHT_3D + y * CELL_SIZE
            point_3D = np.zeros((4, 1), dtype=np.float32)
            point_3D[0][0] = x_3D
            point_3D[1][0] = 0
            point_3D[2][0] = z_3D
            point_3D[3][0] = 1
            point_IPM = P @ point_3D
            if point_IPM[2][0] != 0:
                x_temp = point_IPM[0][0] / point_IPM[2][0]
                y_temp = point_IPM[1][0] / point_IPM[2][0]
                x_new = np.int(np.round(x_temp))
                y_new = np.int(np.round(y_temp))
                if 0 <= x_new < width and 0 <= y_new < height:

                    # Bilinear Interpolation
                    x_0 = np.int(np.floor(x_temp))
                    x_1 = x_0 + 1
                    y_0 = np.int(np.floor(y_temp))
                    y_1 = y_0 + 1
                    if x_1 < width and y_1 < height:
                        f_Q00 = img[y_0][x_0]
                        f_Q01 = img[y_0][x_1]
                        f_Q10 = img[y_1][x_0]
                        f_Q11 = img[y_1][x_1]
                        f_x_y0 = (x_1 - x_temp) * f_Q00 + (x_temp - x_0) * f_Q01
                        f_x_y1 = (x_1 - x_temp) * f_Q10 + (x_temp - x_0) * f_Q11
                        result[y][x] = (y_1 - y_temp) * f_x_y0 + (y_temp - y_0) * f_x_y1

                    # Neirest Neighbour Interpolation
                    # result[y][x] = img[y_new][x_new]

                    for a in range(0, len(corners)):
                        corner = corners[a]
                        if corner[1] - 3 <= y_new <= corner[1] + 3 and corner[0] - 3 <= x_new <= corner[0] + 3:
                            i = indices[a]
                            candidates[a][i] = x_new
                            i += 1
                            candidates[a][i] = y_new
                            i += 1
                            candidates[a][i] = x
                            i += 1
                            candidates[a][i] = y
                            i += 1
                            indices[a] = i
    ipm_corners = np.zeros((4, 2), np.int32)
    best_candidate = np.zeros(4, np.int32)
    for a in range(0, len(corners)):
        corner = corners[a]
        min_dist = math.inf
        best_candidate[0] = -1
        best_candidate[1] = -1
        best_candidate[2] = -1
        best_candidate[3] = -1
        for i in range(0, candidates.shape[1], 4):
            if candidates[a][i] != 0:
                dist = np.sqrt(
                    np.power(corner[0] - candidates[a][i], 2) + np.power(corner[1] - candidates[a][i + 1], 2))
                if dist < min_dist:
                    min_dist = dist
                    best_candidate[0] = candidates[a][i]
                    best_candidate[1] = candidates[a][i + 1]
                    best_candidate[2] = candidates[a][i + 2]
                    best_candidate[3] = candidates[a][i + 3]
        ipm_corners[a][0] = best_candidate[2]
        ipm_corners[a][1] = best_candidate[3]
    return result, ipm_corners


def compute_ipm_and_corners(projection_matrix, img, current_image_corners):
    current_image_corners_as_arrays = convert_to_list_of_arrays(current_image_corners)
    ipm_image, current_ipm_corners_as_arrays = \
        compute_ipm_fast_numba(img, projection_matrix, np.asarray(current_image_corners_as_arrays))
    current_ipm_corners = convert_to_list_of_tuples(current_ipm_corners_as_arrays)
    current_ipm_corners.reverse()
    return ipm_image, current_ipm_corners


def convert_from_image_corners_to_ipm_corners(P, image_corners):
    width_3D = MAX_WIDTH_3D - MIN_WIDTH_3D
    IPM_width = int(width_3D / CELL_SIZE)
    ipm_corners = []
    for corner in image_corners:
        x, y = corner
        A = np.array([[P[0][0] - x * P[2][0], P[0][2] - x * P[2][2]],
                      [P[1][0] - y * P[2][0], P[1][2] - y * P[2][2]]])
        B = np.array([[x * P[2][3] - P[0][3]],
                      [y * P[2][3] - P[1][3]]])
        inv_A = np.linalg.inv(A)
        X = inv_A.dot(B)
        # y_ipm = int(np.round(X[0] / CELL_SIZE + IPM_width / 2))
        # x_ipm = int(np.round((X[1] - MIN_HEIGHT_3D) / CELL_SIZE))
        x_ipm = int(np.round(X[0] / CELL_SIZE + IPM_width / 2))
        y_ipm = int(np.round((X[1] - MIN_HEIGHT_3D) / CELL_SIZE))
        ipm_corners.append((x_ipm, y_ipm))
    ipm_corners.reverse()
    return ipm_corners


def convert_from_ipm_corners_to_image_corners(img, P, ipm_corners):
    height, width = img.shape[0], img.shape[1]
    width_3D = MAX_WIDTH_3D - MIN_WIDTH_3D
    IPM_width = int(width_3D / CELL_SIZE)
    img_corners = []
    for corner in ipm_corners:
        y = corner[1]
        x = corner[0]
        x_3D = (x - IPM_width / 2) * CELL_SIZE
        z_3D = MIN_HEIGHT_3D + y * CELL_SIZE
        point_3D = np.float32([
            [x_3D],
            [0],
            [z_3D],
            [1]
        ])
        point_IPM = np.matmul(P, point_3D)
        if point_IPM[2] != 0:
            x_new = int(np.round(point_IPM[0] / point_IPM[2]))
            y_new = int(np.round(point_IPM[1] / point_IPM[2]))
            if 0 <= x_new < width and 0 <= y_new < height:
                img_corners.append((x_new, y_new))
    img_corners.reverse()
    return img_corners