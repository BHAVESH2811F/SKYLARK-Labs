import os

import torch
import yaml
from torchvision import datasets
from data.multi_view_data_injector import MultiViewDataInjector
from data.transforms import get_simclr_data_transforms
from data.shuffle_patches_transform import CustomTransform
from models.mlp_head import MLPHead
from models.resnet_base_network import ResNet18
from trainer import BYOLTrainer
from torchvision import  transforms
from torch.utils.data import DataLoader
from torch.utils.data import ConcatDataset
from loading_dataset import UnsupervisedImageDataset

print(torch.__version__)
torch.manual_seed(0)
# os.environ["CUDA_VISIBLE_DEVICES"] = "1,2,3"


'''
#loading the train dataset from local drive

train_transforms = transforms.Compose([
transforms.Resize((448, 448)),  # Resize images to 224x224
transforms.ToTensor() # Convert images to PyTorch tensors
])

# Load the dataset from the directories
folder_1 = '/home/ml-vm/Desktop/Bhavesh/archive(1)/PetImages/Cat_train'
folder_2 = '/home/ml-vm/Desktop/Bhavesh/archive(1)/PetImages/Dog_train'
print(folder_1)
dataset1 = UnsupervisedImageDataset([folder_1], transform=train_transforms)
dataset2 = UnsupervisedImageDataset([folder_2], transform=train_transforms)

# Combine the datasets
combined_train_dataset = ConcatDataset([dataset1, dataset2])
'''

def main():
    config = yaml.load(open("./config/config.yaml", "r"), Loader=yaml.FullLoader)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(device)
    print(f"Training with: {device}")

    #data_transform = get_simclr_data_transforms(**config['data_transforms'])
    
    #loading the train dataset from local drive
    
    train_transforms = transforms.Compose([
    transforms.Resize((642,642)) # Resize images to 224x224
    # Convert images to PyTorch tensors
    ])
    
    # Load the dataset from the directories
    folder_1 = '/home/ml-vm/Desktop/Bhavesh/archive(1)/PetImages/Cat_train'
    folder_2 = '/home/ml-vm/Desktop/Bhavesh/archive(1)/PetImages/Dog_train'
    print(folder_1)
    dataset1 = UnsupervisedImageDataset([folder_1], transform=train_transforms)
    dataset2 = UnsupervisedImageDataset([folder_2], transform=train_transforms)

# Combine the datasets
    combined_train_dataset = ConcatDataset([dataset1, dataset2])
    


    # online network
    online_network = ResNet18(**config['network']).to(device)
    pretrained_folder = config['network']['fine_tune_from']

    # load pre-trained model if defined
    if pretrained_folder:
        try:
            checkpoints_folder = os.path.join('./runs', pretrained_folder, 'checkpoints')

            # load pre-trained parameters
            load_params = torch.load(os.path.join(os.path.join(checkpoints_folder, 'model.pth')),
                                     map_location=torch.device(torch.device(device)))

            online_network.load_state_dict(load_params['online_network_state_dict'])

        except FileNotFoundError:
            print("Pre-trained weights not found. Training from scratch.")

    # predictor network
    predictor = MLPHead(in_channels=online_network.projetion.net[-1].out_features,
                        **config['network']['projection_head']).to(device)

    # target encoder
    target_network = ResNet18(**config['network']).to(device)

    optimizer = torch.optim.SGD(list(online_network.parameters()) + list(predictor.parameters()),
                                **config['optimizer']['params'])

    trainer = BYOLTrainer(online_network=online_network,
                          target_network=target_network,
                          optimizer=optimizer,
                          predictor=predictor,
                          device=device,
                          **config['trainer'])

    trainer.train(combined_train_dataset)


if __name__ == '__main__':
    main()






