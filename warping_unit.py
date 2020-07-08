import cv2
import numpy as np
import math
from config import INTERPOLATION_METHOD
from shared import convert_to_list_of_arrays

width_warped = 0
height_warped = 0
projection_matrix = np.zeros((3, 4), np.float32)


def crop_rect(img, rect):
    center, size, angle = rect[0], rect[1], rect[2]
    center, size = tuple(map(int, center)), tuple(map(int, size))
    height, width = img.shape[0], img.shape[1]
    M = cv2.getRotationMatrix2D(center, angle, 1)
    img_rot = cv2.warpAffine(img, M, (width, height))
    img_crop = cv2.getRectSubPix(img_rot, size, center)
    return img_crop


def order_points(pts):
    rect = np.zeros((4, 2), dtype="float32")
    rect[0] = pts[0]
    rect[1] = pts[1]
    rect[2] = pts[2]
    rect[3] = pts[3]
    return rect


def warp_document(first_frame, img, image_corners):
    global width_warped, height_warped
    rect = order_points(image_corners)
    (tl, tr, br, bl) = rect
    length_cd = np.sqrt(math.pow(br[0] - bl[0], 2) + math.pow(br[1] - bl[1], 2))
    length_ab = np.sqrt(math.pow(tr[0] - tl[0], 2) + math.pow(tr[1] - tl[1], 2))
    length_bc = np.sqrt(math.pow(tr[0] - br[0], 2) + math.pow(tr[1] - br[1], 2))
    length_da = np.sqrt(math.pow(tl[0] - bl[0], 2) + math.pow(tl[1] - bl[1], 2))
    if first_frame:
        width_warped = max(int(length_cd), int(length_ab))
        height_warped = max(int(length_bc), int(length_da))
    dst = np.array([
        [0, 0],
        [width_warped - 1, 0],
        [width_warped - 1, height_warped - 1],
        [0, height_warped - 1]], dtype="float32")
    M = cv2.getPerspectiveTransform(rect, dst)
    warped = cv2.warpPerspective(img, M, (width_warped, height_warped), flags=INTERPOLATION_METHOD)
    return warped


def warp_document_using_ipm(ipm_image, final_ipm_corners):
    temp = convert_to_list_of_arrays(final_ipm_corners)
    rect = cv2.minAreaRect(temp)
    warped = crop_rect(ipm_image, rect)
    # temp1 = cv2.flip(warped, -1)
    return warped
