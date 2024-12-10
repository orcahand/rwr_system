import cv2
import numpy as np
import os


WRIST_MASK_PATH = "src/logger/fov_masks/mask_wrist_camera.png"
FRONT_MASK_PATH = "src/logger/fov_masks/mask_front_camera.png"
SIDE_MASK_PATH = "src/logger/fov_masks/mask_side_camera.png"

def generate_color_masks(images, camera_id, replace = False, output_dir=None):
    """
    Generate color masks for the provided image based on defined color ranges.

    Args:
        image (numpy.ndarray): Input image.
        output_dir (str, optional): Directory to save masks. Defaults to None.

    Returns:
        dict: A dictionary containing masks for each color.
    """

    if camera_id == "wrist":
        mask = cv2.imread(WRIST_MASK_PATH)
        x, y, w, h = calculate_crop_coordinates(WRIST_MASK_PATH, tolerance=1)
    elif camera_id == "front":
        mask = cv2.imread(FRONT_MASK_PATH)
        x, y, w, h = calculate_crop_coordinates(FRONT_MASK_PATH, tolerance=1)
    elif camera_id == "side":
        mask = cv2.imread(SIDE_MASK_PATH)
        x, y, w, h = calculate_crop_coordinates(SIDE_MASK_PATH, tolerance=1)
    else:
        mask = np.ones_like(images[0], dtype=np.uint8) * 255
        print("Warning: No mask provided. Using default mask the keeps everything.")
    
    if images is None or len(images) == 0:
        print("Error: No images provided.")
        return images


    modified_images = []  # List to store the modified images
    
    for idx, image in enumerate(images):

        image_masked = cv2.bitwise_and(image, mask)
        
        # Crop the image based on black surrounding so we lose less information when resizing.
        image_masked = image_masked[y:y+h, x:x+w]

        image_masked = cv2.cvtColor(image_masked, cv2.COLOR_BGR2RGB)

        if camera_id == "wrist":
            image_masked = cv2.rotate(image_masked, cv2.ROTATE_90_CLOCKWISE)
            # if idx == 1:
            #     cv2.imwrite(os.path.join(output_dir, f"{camera_id}_{idx}.jpg"), image_masked)

        if camera_id == "side":
            # if idx == 1:
            #     cv2.imwrite(os.path.join(output_dir, f"{camera_id}_{idx}.jpg"), image_masked)
            image_masked = cv2.rotate(image_masked, cv2.ROTATE_180)
            # if idx == 1:
            #     cv2.imwrite(os.path.join(output_dir, f"{camera_id}_{idx}_rotated.jpg"), image_masked)
        # if camera_id == "front":
        #     if idx == 1:
        #         cv2.imwrite(os.path.join(output_dir, f"{camera_id}_{idx}.jpg"), image_masked)


        hsv = cv2.cvtColor(image_masked, cv2.COLOR_BGR2HSV)
        lab = cv2.cvtColor(image_masked, cv2.COLOR_BGR2LAB)
        
        color_ranges = {
            "blue": [(69, 62, 45), (131, 255, 233)],
            "yellow": [(9, 109, 89), (71, 255, 255)],
            "red": [(10, 150, 125), (255, 200, 200)] # Red is LAB colorspace values
        }

        color_masks = {}
        
        for color, (lower, upper) in color_ranges.items():
            lower = np.array(lower)
            upper = np.array(upper)
            
            # Use LAB for red, HSV for other colors
            if color == "red":
                color_mask = cv2.inRange(lab, lower, upper)
            else:
                color_mask = cv2.inRange(hsv, lower, upper)
            
            color_masks[color] = color_mask
            
            # Save the mask if output directory is provided
            if output_dir and idx%50 == 0:
                os.makedirs(output_dir, exist_ok=True)
                cv2.imwrite(os.path.join(output_dir, f"{camera_id}_{color}_{idx}_mask.jpg"), color_mask)
    
        if replace:
            # Create a 3-channel image with blue, yellow, and red masks
            # Ensure masks are single channel and have the same dimensions as the original image
            blue_mask = color_masks.get("blue", np.zeros_like(mask))
            yellow_mask = color_masks.get("yellow", np.zeros_like(mask))
            red_mask = color_masks.get("red", np.zeros_like(mask))
            
            # Stack the masks to form a 3-channel image
            masks_combined = cv2.merge([blue_mask, yellow_mask, red_mask])
            modified_images.append(masks_combined)
        else:
            # Append masks as additional channels to the original image
            # Convert masks to 3-channel by duplicating if necessary
            blue_mask = color_masks.get("blue", np.zeros_like(mask))
            yellow_mask = color_masks.get("yellow", np.zeros_like(mask))
            red_mask = color_masks.get("red", np.zeros_like(mask))
            
            # Stack the original image and masks along the channel axis
            # First, ensure masks are single channel
            # Then, stack them as separate channels
            image_masked = cv2.cvtColor(image_masked, cv2.COLOR_RGB2BGR)
            appended_image = np.dstack((image_masked, blue_mask, yellow_mask, red_mask))
            modified_images.append(appended_image)

    return modified_images


import cv2
import json
import os

def calculate_crop_coordinates(mask_path, tolerance=1):
    """
    Calculate the crop boundaries based on the fixed mask.

    Parameters:
    - mask_path (str): Path to the mask image.
    - tolerance (int): Threshold to consider a pixel as non-black.

    Returns:
    - crop_coords (dict): Dictionary with 'x', 'y', 'w', 'h' keys.
    """
    # Load the mask image
    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

    if mask is None:
        raise FileNotFoundError(f"Mask image at '{mask_path}' not found.")

    # Apply threshold to ensure binary mask
    _, thresh = cv2.threshold(mask, tolerance, 255, cv2.THRESH_BINARY)

    # Find non-zero (non-black) coordinates
    coords = cv2.findNonZero(thresh)

    if coords is None:
        raise ValueError("No non-black pixels found in the mask.")

    # Get bounding rectangle
    x, y, w, h = cv2.boundingRect(coords)

    return x, y, w, h 


# def detect_objects_from_masks(image, color_masks, detect_objects=True, output_dir=None):
#     """
#     Detect objects based on color masks and annotate the image.

#     Args:
#         image (numpy.ndarray): Input image.
#         color_masks (dict): Dictionary of masks for each color.
#         detect_objects (bool, optional): Whether to detect objects. Defaults to True.
#         output_dir (str, optional): Directory to save annotated images. Defaults to None.

#     Returns:
#         dict: A dictionary of detected objects categorized as cubes or trays.
#     """
#     detected_objects = {"cubes": [], "trays": []}
#     annotated_image = image.copy()
    
#     for color, mask in color_masks.items():
#         if detect_objects:
#             contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
#             for contour in contours:
#                 area = cv2.contourArea(contour)
#                 if area > 400:  # Filter based on area
#                     # Calculate centroid
#                     M = cv2.moments(contour)
#                     if M["m00"] != 0:
#                         Cx = int(M["m10"] / M["m00"])
#                         Cy = int(M["m01"] / M["m00"])
#                     else:
#                         Cx, Cy = (0, 0)  # Fallback
                
#                     # Annotation color
#                     annotation_color = {
#                         "red": (0, 0, 255),
#                         "blue": (255, 120, 10),
#                         "yellow": (0, 200, 200)
#                     }[color]
                    
#                     # Get bounding box details
#                     rotated_rect = cv2.minAreaRect(contour)
#                     box_points = cv2.boxPoints(rotated_rect)
#                     box_points = np.int0(box_points)
#                     width, height = rotated_rect[1]
#                     aspect_ratio = width / height if height != 0 else 1
                    
#                     # Classify object
#                     if width + height < 100 and 0.6 < aspect_ratio < 1.6:
#                         detected_objects["cubes"].append((color, rotated_rect[0][0], rotated_rect[0][1], width, height, Cx, Cy))
#                         cv2.putText(annotated_image, "Small Cube", (Cx, Cy - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
#                     elif width + height < 200 and 0.6 < aspect_ratio < 1.6:
#                         detected_objects["cubes"].append((color, rotated_rect[0][0], rotated_rect[0][1], width, height, Cx, Cy))
#                         cv2.putText(annotated_image, "Big Cube", (Cx, Cy - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
#                     elif width + height > 200:
#                         detected_objects["trays"].append((rotated_rect[0][0], rotated_rect[0][1], width, height, Cx, Cy))
#                         cv2.putText(annotated_image, "Tray", (Cx, Cy - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                    
#                     # Annotate
#                     cv2.drawContours(annotated_image, [box_points], 0, annotation_color, 2)
#                     cv2.circle(annotated_image, (Cx, Cy), 5, annotation_color, -1)
        
#         # Save annotated image
#         if output_dir:
#             os.makedirs(output_dir, exist_ok=True)
#             cv2.imwrite(os.path.join(output_dir, f"annotated_image.jpg"), annotated_image)
    
#     return detected_objects
