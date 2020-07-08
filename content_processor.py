import cv2
from config import INTERPOLATION_METHOD


def process_document_content(img, operation_type=None, contrast=None, brightness=None):
    flip_vertical = cv2.flip(img, -1)
    f_height, f_width = img.shape[0], img.shape[1]
    resized = cv2.resize(flip_vertical, None, fx=1050 / f_width, fy=735 / f_height, interpolation=INTERPOLATION_METHOD)
    if operation_type is None or operation_type == 0:
        gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
        denoised = cv2.fastNlMeansDenoising(gray, 3, 7, 21)
        binarized = cv2.adaptiveThreshold(denoised, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 15, 2)
        temp1 = cv2.pyrUp(binarized)
        for i in range(0, 1):
            temp1 = cv2.medianBlur(temp1, 9)
        result = cv2.pyrDown(temp1)
    elif operation_type == 1:
        contrast = 2 * (contrast - 1) / 99 + 1
        result = cv2.convertScaleAbs(resized, alpha=contrast, beta=brightness)
    return result
