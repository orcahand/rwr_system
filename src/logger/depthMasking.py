import torch
import cv2
import numpy as np
import matplotlib.pyplot as plt
from torchvision.transforms import Compose, Normalize, Resize, ToTensor
from PIL import Image
import numpy as np
import os 

# Load MiDaS model
model_type = "MiDaS_small"   # Lightweight model "DPT_Lite"
model = torch.hub.load("intel-isl/MiDaS", model_type, pretrained=True)
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
model.to(device)
model.eval()

print("Device: ", device)

# Define transformation for the input image
def transform_image(image):
    transform = Compose([
        Resize(224),  # Resize to 256 for efficiency
        ToTensor(),
        Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    return transform(image)

def get_image_depth_masked(images, image_name = None ,output_dir = None):
    count = 0
    images_masked = []
    for i in range(len(images)):
        image = images[i] # torch.from_numpy(images[i]).to(device)
        count += 1
        if output_dir is not None:
            output_dir_depth_map = output_dir + "_depth_map"
            output_dir_depth_masked = output_dir +"_depth_masked"
        
            os.makedirs(output_dir_depth_map, exist_ok=True)
            os.makedirs(output_dir_depth_masked, exist_ok=True)

            if image_name is None:
                raise ValueError("Image name not provided")

        if image is None:
            raise ValueError(f"Image not found")


        ### HARDCODED PARAMETERS ### 
        cropped_image = image[:, 190:] # before 45 it is the spools of the hand etc. Discard them
        resized_image = cv2.resize(cropped_image, (224,224) , interpolation=cv2.INTER_LINEAR)

        image = resized_image

        original_size = (image.shape[1], image.shape[0])  # Save original dimensions (width, height)
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # Convert to RGB

        # Assuming 'frame' is your numpy array
        frame_pil = Image.fromarray(image_rgb)
        input_image = transform_image(frame_pil).unsqueeze(0)  # Add batch dimension

        input_image = input_image.to(device)
        # Perform inference
        with torch.no_grad():
            prediction = model(input_image)
            depth_map = prediction.squeeze().cpu().numpy()  # Remove batch dimension

        # Resize depth map to the original image size
        depth_map_resized = cv2.resize(depth_map, original_size)

        # Normalize depth map to range [0, 1]
        depth_map_normalized = (depth_map_resized - depth_map_resized.min()) / (
            depth_map_resized.max() - depth_map_resized.min()
        )

        # Apply a threshold to create a binary mask
        threshold = 0.3  # Threshold value (adjust as needed)
        mask = depth_map_normalized > threshold

        # Mask the original image
        masked_image = image_rgb.copy()
        masked_image[~mask] = 0  # Set pixels below the threshold to black

        if output_dir is not None:
            # Save results

            cv2.imwrite(os.path.join(output_dir_depth_map,f"{image_name}_depth_map.jpg"), (depth_map_normalized * 255).astype(np.uint8))
            cv2.imwrite(os.path.join(output_dir_depth_masked,f"{image_name}_masked_image.jpg") , masked_image)

        # cv2.imwrite( f"masked_images_validation/{count}_masked_image.jpg" , masked_image)

        # flipped = cv2.flip(masked_image, 0)
        images_masked.append(masked_image)

    print("DEBUG3")
    return np.array(images_masked)