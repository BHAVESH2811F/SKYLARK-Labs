import os
from PIL import Image
import torch
import random
import numpy as np
import torchvision.transforms as T
import torchvision

class ImageShuffler(object):
    def __init__(self):
        pass

    def __call__(self, img):
        # Convert PIL Image to NumPy array
        img_np = np.array(img)

        # Get image dimensions
        channels, height, width = img_np.shape

        # Calculate dimensions of each part (assuming 3x3 grid)
        part_height = height // 3
        part_width = width // 3

        # Divide the image into 9 parts
        parts = [
            img_np[:, i * part_height:(i + 1) * part_height, j * part_width:(j + 1) * part_width]
            for i in range(3) for j in range(3)
        ]

        # Shuffle the parts
        random.shuffle(parts)

        # Concatenate the parts back into a single image
        shuffled_img = np.concatenate([
            np.concatenate([parts[0], parts[1], parts[2]], axis=2),
            np.concatenate([parts[3], parts[4], parts[5]], axis=2),
            np.concatenate([parts[6], parts[7], parts[8]], axis=2)
        ], axis=1)

        # Convert numpy array back to tensor
        shuffled_img = torch.from_numpy(shuffled_img)

        return shuffled_img

# Define input and output directories
input_dir = '/media/ml-vm/3fea60f1-fd93-47c9-b3a5-c2fde1bd92a9/weapon_orig_aug_testing/weapon_test_original_images'
output_dir = '/media/ml-vm/3fea60f1-fd93-47c9-b3a5-c2fde1bd92a9/weapon_orig_aug_testing/weapon_test_augmented_image'

# Create output directory if it does not exist


# Transformation pipeline
transform = T.Compose([
    T.ToTensor(),  # Convert PIL Image to tensor first
    ImageShuffler(),  # Apply custom shuffling
])

# List all images in the input directory
for filename in os.listdir(input_dir):
    #print(filename)
    
    # Open the image using PIL
    img_path = os.path.join(input_dir, filename)
    img = Image.open(img_path).convert('RGB')  # Ensure RGB mode for PIL compatibility

    # Apply transformation
    augmented_img = transform(img)
    base_name, ext = os.path.splitext(filename)
    img_name = f"aug_{base_name}{ext}"
    # Save the augmented image to output directory
    output_path = os.path.join(output_dir, img_name)
    format = ext[1:] if ext else 'jpeg'  # Remove the leading dot and default to PNG
    torchvision.utils.save_image(augmented_img, output_path,format = format)

    print(f"Augmented and saved: {filename}")
    

print("Augmentation complete. Shuffled images saved to", output_dir)








# input_dir = 'saved_images2'
# output_dir = 'saved_output3'