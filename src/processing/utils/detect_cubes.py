import cv2
import numpy as np
import os
from .colorMasking import get_color_masks

file_path = os.path.join(os.path.dirname(__file__),"fov_masks")

FRONT_CUBES_MASK_PATH = os.path.join(file_path, "cubes_only_front_mask.png")
SIDE_CUBES_MASK_PATH = os.path.join(file_path, "cubes_only_side_mask.png")

# def detect_cubes(image, image_name, output_dir = "output", detect_objects = True
def detect_cubes(image_bgr, camera, output_dir = None):

    # image = image_bgr.copy()
    image = image_bgr
    blue_mask, yellow_mask, red_mask = get_color_masks(image)

    if camera == "front":
        mask = cv2.imread(FRONT_CUBES_MASK_PATH, cv2.IMREAD_GRAYSCALE)
    elif camera == "side":
        mask = cv2.imread(SIDE_CUBES_MASK_PATH, cv2.IMREAD_GRAYSCALE)

    blue_mask = cv2.bitwise_and(blue_mask, mask)
    yellow_mask = cv2.bitwise_and(yellow_mask, mask)
    red_mask = cv2.bitwise_and(red_mask, mask)

    # if camera == "side":
    #     cv2.imwrite("/home/aristotelis/ros2_ws/src/rwr_system/src/processing/utils/blue_mask.jpg", blue_mask)
    #     cv2.imwrite("/home/aristotelis/ros2_ws/src/rwr_system/src/processing/utils/red_mask.jpg", red_mask)
    #     cv2.imwrite("/home/aristotelis/ros2_ws/src/rwr_system/src/processing/utils/yellow_mask.jpg", yellow_mask)

    
    color_detected = {"blue": False, "yellow": False, "red": False}

    for mask, color in zip([blue_mask, yellow_mask, red_mask], ["blue", "yellow", "red"]):
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        for contour in contours:
            area = cv2.contourArea(contour)
            if area > 700:
                color_detected[color] = True

        # Save the annotated image
        if output_dir:
            cv2.imwrite(os.path.join(output_dir, f"{color}_annotated_image.jpg"), mask)
        
    true_colors = [color for color, detected in color_detected.items() if detected]

    if len(true_colors) == 1:
        return true_colors[0]

    # # Define the priority order
    # priority_order = ["blue", "yellow", "red"]
    
    # # Iterate through the priority list and return the first detected color
    # for color in priority_order:
    #     if color_detected[color]:
    #         return color

    return None