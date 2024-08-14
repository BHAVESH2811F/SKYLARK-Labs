
import os
import shutil
import random

# Paths
source_dir = '/media/ml-vm/3fea60f1-fd93-47c9-b3a5-c2fde1bd92a9/cls/other'# Path to your dataset directory
target_dir = '/media/ml-vm/3fea60f1-fd93-47c9-b3a5-c2fde1bd92a9/knives_dataset/others_mixed_knives'  # Path to the directory where test images will be saved

# Ensure the target directory exists
if not os.path.exists(target_dir):
    os.makedirs(target_dir)

# List all images in the source directory
all_images = [img for img in os.listdir(source_dir) if os.path.isfile(os.path.join(source_dir, img))]
print(f"Number of images in source directory: {len(all_images)}")

# Randomly select 50 images
test_images = random.sample(all_images, 25)

# Move the selected images to the target directory with new names
for i, img in enumerate(test_images):
    new_name = f"G_{i}"  # Preserve the original file extension
    source_path = os.path.join(source_dir, img)
    target_path = os.path.join(target_dir, new_name)
    
    print(f"Moving {source_path} to {target_path}")
    
    try:
        shutil.move(source_path, target_path)
    except Exception as e:
        print(f"Error moving {source_path} to {target_path}: {e}")
    
    if not os.path.exists(target_path):
        print(f"Failed to move {img} to {target_path}")

print(f"Moved and renamed {len(test_images)} images to {target_dir}")