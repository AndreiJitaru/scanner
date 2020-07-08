from corner_detector import find_corners_in_image
from corner_validator import validate_image_corners, validate_image_corners_using_completely_built_ipm
from warping_unit import warp_document, warp_document_using_ipm
from content_processor import process_document_content


first_frame = True


def scan(img, operation_type=None, contrast=None, brightness=None):
    global first_frame
    current_image_corners = find_corners_in_image(img)
    final_image_corners = validate_image_corners(first_frame, img, current_image_corners)
    warped = warp_document(first_frame, img, final_image_corners)
    print("current img corners", current_image_corners)
    print("final img corners", final_image_corners)
    # final_image_corners = validate_image_corners_using_completely_built_ipm(first_frame, img, current_image_corners)
    # warped = warp_document_using_ipm(img, final_image_corners)
    final = process_document_content(warped, operation_type, contrast, brightness)
    first_frame = False
    return final

