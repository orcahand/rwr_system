import torch
import cv2
import numpy as np
import os

# Load MiDaS model
model_type = "MiDaS_small"   # Lightweight model "DPT_Lite"
model = torch.hub.load("intel-isl/MiDaS", model_type, pretrained=True)
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
model.to(device)
model.eval()

def get_image_depth_masked(images, image_names=None, output_dir=None, batch_size=16):
    if output_dir is not None:
        output_dir_depth_map = output_dir + "_depth_map"
        output_dir_depth_masked = output_dir + "_depth_masked"
        os.makedirs(output_dir_depth_map, exist_ok=True)
        os.makedirs(output_dir_depth_masked, exist_ok=True)
    
    images_masked = []
    original_sizes = []
    transformed_images = []

    # Preprocess images
    midas_transforms = torch.hub.load("intel-isl/MiDaS", "transforms")
    transform = midas_transforms.small_transform

    for image in images:
        if image is None:
            raise ValueError("One or more images are None.")
        
        cropped_image = image[:, 190:]  # Discard the spool region
        resized_image = cv2.resize(cropped_image, (224, 224), interpolation=cv2.INTER_LINEAR)
        original_sizes.append((image.shape[1], image.shape[0]))  # Store original size (width, height)
        image_rgb = cv2.cvtColor(resized_image, cv2.COLOR_BGR2RGB)  # Convert to RGB
        transformed_images.append(transform(image_rgb).to(device))

    # Process images in batches of size 16
    for batch_start in range(0, len(transformed_images), batch_size):
        batch_end = batch_start + batch_size
        batch_images = transformed_images[batch_start:batch_end]  # Get current batch
        batch_tensor = torch.stack(batch_images).squeeze(1)  # Create batched tensor [batch_size, channels, height, width]

        with torch.no_grad():
            predictions = model(batch_tensor)  # Run inference for the batch

        # Process predictions
        for i, prediction in enumerate(predictions):
            depth_map = prediction.squeeze().cpu().numpy()  # Remove batch dimension
            original_size = original_sizes[batch_start + i]  # Get original size
            depth_map_resized = cv2.resize(depth_map, original_size)  # Resize to original size

            # Normalize depth map to range [0, 1]
            depth_map_normalized = (depth_map_resized - depth_map_resized.min()) / (
                depth_map_resized.max() - depth_map_resized.min()
            )

            # Apply a threshold to create a binary mask
            threshold = 0.3  # Threshold value (adjust as needed)
            mask = depth_map_normalized > threshold

            # Mask the original image
            original_image = images[batch_start + i][:, 45:]  # Apply the same crop as preprocessing
            masked_image = original_image.copy()
            masked_image[~mask] = 0  # Set pixels below the threshold to black

            # Save results if output_dir is provided
            if output_dir is not None:
                image_name = image_names[batch_start + i]
                cv2.imwrite(
                    os.path.join(output_dir_depth_map, f"{image_name}_depth_map.jpg"),
                    (depth_map_normalized * 255).astype(np.uint8),
                )
                cv2.imwrite(
                    os.path.join(output_dir_depth_masked, f"{image_name}_masked_image.jpg"),
                    masked_image,
                )
            
            images_masked.append(masked_image)

    return np.array(images_masked)



# if __name__ == "__main__":
#     # Define input and output paths
#     input_dir = "input_images"  # Directory containing input images
#     output_dir = "output"       # Directory to save output images
#     batch_size = 16             # Number of images to process per batch

#     # Load images
#     image_paths = glob.glob(os.path.join(input_dir, "*.jpg"))  # Adjust extension if needed
#     images = [cv2.imread(img_path) for img_path in image_paths]
#     image_names = [os.path.splitext(os.path.basename(img_path))[0] for img_path in image_paths]

#     if not images:
#         print("No images found in the input directory.")
#     else:
#         # Execute the depth masking function
#         print("Processing images...")
#         masked_images = get_image_depth_masked(images, image_names, output_dir, batch_size=batch_size)

#         print(f"Processing completed. Results saved in {output_dir}.")
#         print(f"Total images processed: {len(masked_images)}")