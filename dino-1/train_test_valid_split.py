import os
import random
import shutil
from pathlib import Path

def create_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)

def split_data(source_dir, train_dir, val_dir, test_dir, train_pct=0.7, val_pct=0.15, test_pct=0.15):
    # Ensure the percentages add up to 1.0
    assert train_pct + val_pct + test_pct == 1.0, "The sum of the percentages must be 1.0"

    # Create directories if they do not exist
    create_dir(train_dir)
    create_dir(val_dir)
    create_dir(test_dir)

    # Get all files from source directory
    all_files = [f for f in os.listdir(source_dir) if os.path.isfile(os.path.join(source_dir, f))]
    random.shuffle(all_files)

    # Calculate the number of files for each split
    total_files = len(all_files)
    train_count = int(total_files * train_pct)
    val_count = int(total_files * val_pct)
    test_count = total_files - train_count - val_count

    # Split the data
    train_files = all_files[:train_count]
    val_files = all_files[train_count:train_count + val_count]
    test_files = all_files[train_count + val_count:]

    # Move files to respective directories
    for file in train_files:
        shutil.move(os.path.join(source_dir, file), os.path.join(train_dir, file))
    for file in val_files:
        shutil.move(os.path.join(source_dir, file), os.path.join(val_dir, file))
    for file in test_files:
        shutil.move(os.path.join(source_dir, file), os.path.join(test_dir, file))

    print(f"Moved {train_count} files to {train_dir}")
    print(f"Moved {val_count} files to {val_dir}")
    print(f"Moved {test_count} files to {test_dir}")

# Define directories
source_directory = '/media/ml-vm/3fea60f1-fd93-47c9-b3a5-c2fde1bd92a9/other_cls/other'
train_directory = '/media/ml-vm/3fea60f1-fd93-47c9-b3a5-c2fde1bd92a9/other_cls/train_dataset'
val_directory = '/media/ml-vm/3fea60f1-fd93-47c9-b3a5-c2fde1bd92a9/other_cls/validation_dataset'
test_directory = '/media/ml-vm/3fea60f1-fd93-47c9-b3a5-c2fde1bd92a9/other_cls/test_dataset'

# Split data
split_data(source_directory, train_directory, val_directory, test_directory, train_pct=0.7, val_pct=0.15, test_pct=0.15)
