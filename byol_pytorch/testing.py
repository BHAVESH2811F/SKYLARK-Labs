from loading_dataset import UnsupervisedImageDataset
import os
import torch
import torch.nn.functional as F
import torchvision
from torch.utils.data.dataloader import DataLoader
from torch.utils.tensorboard import SummaryWriter
from data.shuffle_patches_transform import CustomTransform
from torch.utils.data import ConcatDataset

import torch
from torchvision import datasets, transforms

# Data transformations
train_transforms = transforms.Compose([
        transforms.Resize((448, 448)),  # Resize images to 448x448
        
    ])

# Load the dataset from the directories
folder_1 = '/home/ml-vm/Desktop/Bhavesh/archive(1)/PetImages/Cat_train'
folder_2 = '/home/ml-vm/Desktop/Bhavesh/archive(1)/PetImages/Dog_train'
print(folder_1)
dataset1 = UnsupervisedImageDataset([folder_1], transform=train_transforms)
dataset2 = UnsupervisedImageDataset([folder_2], transform=train_transforms)
combined_train_dataset = ConcatDataset([dataset1, dataset2])

class OriginalAndAugmentedDataset(torch.utils.data.Dataset):
            def __init__(self, dataset, transform):
                self.dataset = dataset
                self.transform = transform
                self.to_tensor = transforms.ToTensor()

            def __len__(self):
                return len(self.dataset)

            def __getitem__(self, idx):
                img,_= self.dataset[idx]
                original = self.to_tensor(img)
                augmented = self.transform(img)
                augmented =   self.to_tensor(augmented) if not isinstance(augmented, torch.Tensor) else augmented
                return (original, augmented)

# Define the augmentation transformation
augmentation_transform = transforms.Compose([
    CustomTransform(),
    
    #transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Create the custom dataset
dataset = OriginalAndAugmentedDataset(combined_train_dataset, augmentation_transform)

# Create the dataloader
train_loader = torch.utils.data.DataLoader(dataset, batch_size=64, drop_last=False, shuffle=True)

#train_loader = DataLoader(train_dataset, batch_size=self.batch_size,
    #                          drop_last=False, shuffle=True)

niter = 0
model_checkpoints_folder = os.path.join('/home/ml-vm/Desktop', 'checkpoints')





#for (batch_view_1, batch_view_2), _ in train_loader:
for batch in train_loader:
    batch_original, batch_augmented = batch

    #batch_original = batch_original.to(device)
    #batch_augmented = batch_augmented.to(self.device)
    print("Batch Original:")
    print(f"  Type: {type(batch_original)}")
    print(f"  Shape: {batch_original.shape}")
    print(f"  Dtype: {batch_original.dtype}")

    print("Batch Augmented:")
    print(f"  Type: {type(batch_augmented)}")
    print(f"  Shape: {batch_augmented.shape}")
    print(f"  Dtype: {batch_augmented.dtype}")
    break