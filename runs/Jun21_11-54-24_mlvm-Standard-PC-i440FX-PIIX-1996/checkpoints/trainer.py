import os
import torch
from tqdm import tqdm
import time
import torch.nn.functional as F
import torchvision
from torch.utils.data.dataloader import DataLoader
from torch.utils.tensorboard import SummaryWriter
from data.shuffle_patches_transform import CustomTransform
from utils import _create_model_training_folder
import torch
from torchvision import datasets, transforms
from loading_dataset import UnsupervisedImageDataset

class BYOLTrainer:
    def __init__(self, online_network, target_network, predictor, optimizer, device, **params):
        self.online_network = online_network
        self.target_network = target_network
        self.optimizer = optimizer
        self.device = device
        self.predictor = predictor
        self.max_epochs = params['max_epochs']
        self.writer = SummaryWriter()
        self.m = params['m']
        self.batch_size = params['batch_size']
        self.num_workers = params['num_workers']
        self.checkpoint_interval = params['checkpoint_interval']
        _create_model_training_folder(self.writer, files_to_same=["./config/config.yaml", "main.py", 'trainer.py'])

    @torch.no_grad()
    def _update_target_network_parameters(self):
        """
        Momentum update of the key encoder
        """
        for param_q, param_k in zip(self.online_network.parameters(), self.target_network.parameters()):
            param_k.data = param_k.data * self.m + param_q.data * (1. - self.m)

    @staticmethod
    def regression_loss(x, y):
        x = F.normalize(x, dim=1)
        y = F.normalize(y, dim=1)
        return 2 - 2 * (x * y).sum(dim=-1)

    def initializes_target_network(self):
        # init momentum network as encoder net
        for param_q, param_k in zip(self.online_network.parameters(), self.target_network.parameters()):
            param_k.data.copy_(param_q.data)  # initialize
            param_k.requires_grad = False  # not update by gradient

    def train(self, train_dataset):

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
    ])

# Create the custom dataset
        dataset = OriginalAndAugmentedDataset(train_dataset, augmentation_transform)

# Create the dataloader
        train_loader = torch.utils.data.DataLoader(dataset, batch_size=64, drop_last=False, shuffle=True)
        num_batches = (len(dataset) + self.batch_size - 1) // self.batch_size  # Ceiling division to cover all samples
       
        niter = 0
        model_checkpoints_folder = os.path.join('/home/ml-vm/Desktop/Bhavesh/', 'checkpoints')

        self.initializes_target_network()

        for epoch_counter in range(self.max_epochs):

            start_time = time.time()  # Start timing the epoch

            # Create a tqdm progress bar for the batches within the current epoch
            batch_progress = tqdm(train_loader, desc=f"Epoch {epoch_counter+1}/{self.max_epochs}", leave=True)
            for batch in batch_progress:
                #batch_progress = tqdm(range(64), desc=f"Epoch {epoch_counter+1}/{self.max_epochs}", leave=True)
                batch_original, batch_augmented = batch
                
                batch_original = batch_original.to(self.device)
                batch_augmented = batch_augmented.to(self.device)
                '''
                if niter == 0:
                    grid = torchvision.utils.make_grid(batch_view_1[:32])
                    self.writer.add_image('views_1', grid, global_step=niter)

                    grid = torchvision.utils.make_grid(batch_view_2[:32])
                    self.writer.add_image('views_2', grid, global_step=niter)
                '''

                loss = self.update(batch_original, batch_augmented)
                #self.writer.add_scalar('loss', loss, global_step=niter)

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                self._update_target_network_parameters()  # update the key encoder
                #niter += 1
            # time.sleep(0.1)  # Replace this with actual training code
                # The progress bar automatically shows the completion percentage

            epoch_time = time.time() - start_time  # End timing the epoch
            
            print(f"Epoch {epoch_counter+1} took {epoch_time:.2f} seconds")
            print("End of epoch {}".format(epoch_counter+1))

        # save checkpoints
        self.save_model(os.path.join('/home/ml-vm/Desktop/Bhavesh', 'model_9_tiles_6_shuffle.pth'))

    def update(self, batch_original, batch_augmented):
        # compute query feature
        predictions_from_original = self.predictor(self.online_network(batch_original))
        #predictions_from_augmented = self.predictor(self.online_network(batch_augmented))

        # compute key features
        with torch.no_grad():
            targets_to_augmented = self.target_network(batch_augmented)
            #targets_to_view_1 = self.target_network(batch_view_2)

        loss = self.regression_loss(predictions_from_original, targets_to_augmented)
        #loss += self.regression_loss(predictions_from_view_2, targets_to_view_2)
        return loss.mean()

    def save_model(self, PATH):

        torch.save({
            'online_network_state_dict': self.online_network.state_dict(),
            'target_network_state_dict': self.target_network.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
        }, PATH)

