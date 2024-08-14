import os
from PIL import Image
from torch.utils.data import Dataset
from torch.utils.data import ConcatDataset
from torchvision import transforms
from PIL import UnidentifiedImageError

class UnsupervisedImageDataset(Dataset):
    def __init__(self, folder_paths, transform=None):
        self.image_paths = []
        for folder_path in folder_paths:
            for root, _, files in os.walk(folder_path):
                for file in files:
                    if file.endswith(('jpg', 'jpeg', 'png', 'bmp')):
                        self.image_paths.append(os.path.join(root, file))
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        try:
            image = Image.open(img_path).convert('RGB')
        except UnidentifiedImageError:
            #print(f"Skipping corrupted image file: {img_path}")
            return self.__getitem__((idx + 1) % len(self))
        if self.transform:
            image = self.transform(image)
        #image = image.permute(1, 2, 0) 
        return image,0


'''
# Data transformations
#train_transforms = transforms.Compose([
        transforms.Resize((448, 448)),  # Resize images to 448x448
        transforms.ToTensor()  # Convert images to PyTorch tensors
    ])


# Load the dataset from the directories
folder_1 = '/home/ml-vm/Desktop/Bhavesh/archive(1)/PetImages/Cat_train'
folder_2 = '/home/ml-vm/Desktop/Bhavesh/archive(1)/PetImages/Dog_train'
print(folder_1)
dataset1 = UnsupervisedImageDataset([folder_1], transform=train_transforms)
dataset2 = UnsupervisedImageDataset([folder_2], transform=train_transforms)

combined_train_dataset = ConcatDataset([dataset1, dataset2])
for i in range(5):  # Print the first 5 examples
    print(i)
    img,_ = combined_train_dataset[i]
    print(f"Image {i}:")
    print(f"  Type: {type(img)}")
    print(f"  Shape: {img.shape}")
    print(f"  Dtype: {img.dtype}")

'''
