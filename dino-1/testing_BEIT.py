from transformers import BeitImageProcessor, BeitModel, BeitConfig# Load and save the model, processor, and config
import torch
model = BeitModel.from_pretrained('microsoft/beit-base-patch16-224-pt22k-ft22k')
processor = BeitImageProcessor.from_pretrained('microsoft/beit-base-patch16-224-pt22k-ft22k')
config = model.config# Save the model
torch.save(model.state_dict(), 'beit_model.pt')# Save the processor
processor.save_pretrained('beit_processor')# Save the config
config.save_pretrained('beit_config')
import utils
import os
from PIL import Image
import torch
from torch.utils.data import DataLoader, Dataset
from transformers import BeitImageProcessor, BeitModel
from tqdm import tqdm
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import os
import sys
import argparse

import torch
from torch import nn
import torch.distributed as dist
import torch.backends.cudnn as cudnn
from torchvision import datasets
from torchvision import transforms as pth_transforms
from torchvision import models as torchvision_models

import utils
import vision_transformer as vits
# Define paths to your image folders

class1_folder = r'/media/ml-vm/3fea60f1-fd93-47c9-b3a5-c2fde1bd92a9/cls_falses/test/others'
class2_folder = r"/media/ml-vm/3fea60f1-fd93-47c9-b3a5-c2fde1bd92a9/cls_falses/train/weapon"
# Custom dataset class
class ImageFolderDataset(Dataset):
    def __init__(self, folder_path, label):
        self.folder_path = folder_path
        self.image_files = [f for f in os.listdir(folder_path) if f.endswith(('.png', '.jpg', '.jpeg'))]
        self.label = label
    
    def __len__(self):
        return len(self.image_files)
    
    def __getitem__(self, idx):
        img_path = os.path.join(self.folder_path, self.image_files[idx])
        image = Image.open(img_path).convert('RGB')
        return image, self.label

# Initialize BEiT model and processor
processor = BeitImageProcessor.from_pretrained('microsoft/beit-base-patch16-224-pt22k-ft22k')
model = BeitModel.from_pretrained('microsoft/beit-base-patch16-224-pt22k-ft22k')
# torch.save(model, "BEiT_feature_extractor.pt")
model.eval()  # Set model to evaluation mode
# Use GPU if available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = model.to(device)
# Prepare datasets and dataloaders
dataset1 = ImageFolderDataset(class1_folder,label = 0)
dataset2 = ImageFolderDataset(class2_folder,label = 1)
dataloader1 = DataLoader(dataset1, batch_size=32, shuffle=False, collate_fn=lambda x: list(zip(*x)))
dataloader2 = DataLoader(dataset2, batch_size=32, shuffle=False, collate_fn=lambda x: list(zip(*x)))
# Function to compute embeddings
def compute_embeddings(dataloader):
    embeddings = []
    filenames = []
    labels = []
    with torch.no_grad():
        for batch, batch_labels  in tqdm(dataloader, desc="Computing embeddings"):
            inputs = processor(images=batch, return_tensors="pt", padding=True).to(device)
            outputs = model(**inputs, output_hidden_states=True)
            batch_embeddings = outputs.last_hidden_state.mean(dim=1)
            embeddings.append(batch_embeddings.cpu().numpy())
            #filenames.extend(batch_filenames)
            labels.extend(batch_labels)
    return embeddings, torch.tensor(labels)
# Compute embeddings for both classes
print("Computing embeddings for Class 1...")
test_features, test_labels  = compute_embeddings(dataloader1)
print("Computing embeddings for Class 2...")
train_features, train_labels  = compute_embeddings(dataloader2)
# Normalize embeddings
test_features = test_features / np.linalg.norm(test_features, axis=1, keepdims=True)
train_features = train_features / np.linalg.norm(train_features, axis=1, keepdims=True)
# Compute similarity matrix
##similarity_matrix = cosine_similarity(test_features, train_features)
# Find the most similar pairs
'''
num_pairs_to_show = 1
for i in range(len(test_features)):
    most_similar_indices = similarity_matrix[i].argsort()[-num_pairs_to_show:][::-1]
    print(f"\nMost similar images to Class 1 image {filenames1[i]}:")
    for j, idx in enumerate(most_similar_indices, 1):
        similarity_score = similarity_matrix[i][idx]
        print(f"  {j}. Class 2 image {filenames2[idx]}: Similarity = {similarity_score:.4f}")
        dicti2[filenames1[i]][filenames2[idx]] = similarity_score
# Optional: Compute average similarity between classes
average_similarity = np.mean(similarity_matrix)
print(f"\nAverage similarity between Class 1 and Class 2: {average_similarity:.4f}")
'''

@torch.no_grad()
def knn_classifier(train_features, train_labels, test_features, test_labels, k, T, num_classes=2):
    top1, top5, total = 0.0, 0.0, 0
    train_features = train_features.t()
    num_test_images, num_chunks = test_labels.shape[0], 100
    imgs_per_chunk =  int(num_test_images // num_chunks)
    
    # Initialize class-wise counters and confusion matrix
    class_correct = [0] * num_classes
    class_total = [0] * num_classes
    confusion_matrix = torch.zeros(num_classes, num_classes)

    retrieval_one_hot = torch.zeros(k, num_classes).to(train_features.device)
    for idx in range(0, num_test_images, imgs_per_chunk):
        # get the features for test images
        features = test_features[idx: min((idx + imgs_per_chunk), num_test_images), :]
        targets = test_labels[idx: min((idx + imgs_per_chunk), num_test_images)]
        batch_size = targets.shape[0]

        # calculate the dot product and compute top-k neighbors
        similarity = torch.mm(features, train_features)
        distances, indices = similarity.topk(k, largest=True, sorted=True)

        candidates = train_labels.view(1, -1).expand(batch_size, -1)
        retrieved_neighbors = torch.gather(candidates, 1, indices)

        retrieval_one_hot.resize_(batch_size * k, num_classes).zero_()
        retrieval_one_hot.scatter_(1, retrieved_neighbors.view(-1, 1), 1)
        distances_transform = distances.clone().div_(T).exp_()
        probs = torch.sum(
            torch.mul(
                retrieval_one_hot.view(batch_size, -1, num_classes),
                distances_transform.view(batch_size, -1, 1),
            ),
            1,
        )
        _, predictions = probs.sort(1, True)

        # find the predictions that match the target
        correct = predictions.eq(targets.data.view(-1, 1))
        top1 += correct.narrow(1, 0, 1).sum().item()
        total += targets.size(0)

        # Update class-wise counters and confusion matrix
        for i in range(batch_size):
            label = targets[i].item()
            pred = predictions[i, 0].item()
            class_total[label] += 1
            confusion_matrix[label, pred] += 1
            if correct[i, 0].item():
                class_correct[label] += 1

    top1 = top1 * 100.0 / total

    # Compute class-wise accuracy
    class_accuracies = [0] * num_classes
    for i in range(num_classes):
        if class_total[i] > 0:
            class_accuracies[i] = 100.0 * class_correct[i] / class_total[i]
        else:
            class_accuracies[i] = 0.0

    return top1, class_accuracies, confusion_matrix
tpo1, class_accuracies, confusion_matrix = knn_classifier(trainNN_features,train_labels,test_features,test_labels,k=10,T= 0.07,num_classes=2)
print("confusion_matrix:-",confusion_matrix)
print("top1%", tpo1)
print("class_accuricies:-",class_accuracies)
'''print(co)
#  =======================================================================================

import os
import sys
import argparse

import torch
from torch import nn
import torch.distributed as dist
import torch.backends.cudnn as cudnn
from torchvision import datasets
from torchvision import transforms as pth_transforms
from torchvision import models as torchvision_models

import utils
import vision_transformer as vits


def extract_feature_pipeline(args):
    # ============ preparing data ... ============
    transform = pth_transforms.Compose([
        pth_transforms.Resize(256, interpolation=3),
        pth_transforms.CenterCrop(224),
        pth_transforms.ToTensor(),
        pth_transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ])
    dataset_train = ReturnIndexDataset(os.path.join(args.data_path, "train"), transform=transform)
    dataset_val = ReturnIndexDataset(os.path.join(args.data_path, "test"), transform=transform)
    sampler = torch.utils.data.DistributedSampler(dataset_train, shuffle=False)
    data_loader_train = torch.utils.data.DataLoader(
        dataset_train,
        sampler=sampler,
        batch_size=args.batch_size_per_gpu,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=False,
    )
    data_loader_val = torch.utils.data.DataLoader(
        dataset_val,
        batch_size=args.batch_size_per_gpu,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=False,
    )
    print(f"Data loaded with {len(dataset_train)} train and {len(dataset_val)} val imgs.")

    # ============ building network ... ============
    
    if "vit" in args.arch:
        model = vits.__dict__[args.arch](patch_size=args.patch_size, num_classes=0)
        print(f"Model {args.arch} {args.patch_size}x{args.patch_size} built.")
    elif "xcit" in args.arch:
        model = torch.hub.load('facebookresearch/xcit:main', args.arch, num_classes=0)
    elif args.arch in torchvision_models.__dict__.keys():
        model = torchvision_models.__dict__[args.arch](num_classes=0)
        model.fc = nn.Identity()
    else:
        print(f"Architecture {args.arch} non supported")
        sys.exit(1)

    
    model.cuda()
    utils.load_pretrained_weights(model, args.pretrained_weights, args.checkpoint_key, args.arch, args.patch_size)
    model.eval()
    '''
'''
      # ============ Initialize BEiT model and processor ===========================
    processor = BeitImageProcessor.from_pretrained('microsoft/beit-base-patch16-224-pt22k-ft22k')
    model = BeitModel.from_pretrained('microsoft/beit-base-patch16-224-pt22k-ft22k')
    model.eval()  # Set model to evaluation mode

    # Use GPU if available
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    # ============ extract features ... ============
    print("Extracting features for train set...")
    train_features = extract_features(model, data_loader_train, args.use_cuda)
    print("Extracting features for val set...")
    test_features = extract_features(model, data_loader_val, args.use_cuda)

    if utils.get_rank() == 0:
        train_features = nn.functional.normalize(train_features, dim=1, p=2)
        test_features = nn.functional.normalize(test_features, dim=1, p=2)

    train_labels = torch.tensor([s[-1] for s in dataset_train.samples]).long()
    test_labels = torch.tensor([s[-1] for s in dataset_val.samples]).long()
    # save features and labels
    if args.dump_features and dist.get_rank() == 0:
        torch.save(train_features.cpu(), os.path.join(args.dump_features, "trainfeat.pth"))
        torch.save(test_features.cpu(), os.path.join(args.dump_features, "testfeat.pth"))
        torch.save(train_labels.cpu(), os.path.join(args.dump_features, "trainlabels.pth"))
        torch.save(test_labels.cpu(), os.path.join(args.dump_features, "testlabels.pth"))
    return train_features, test_features, train_labels, test_labels


@torch.no_grad()
def extract_features(model, data_loader, use_cuda=True, multiscale=False):
    metric_logger = utils.MetricLogger(delimiter="  ")
    features = None
    for samples, index in metric_logger.log_every(data_loader, 10):
        samples = samples.cuda(non_blocking=True)
        index = index.cuda(non_blocking=True)
        if multiscale:
            feats = utils.multi_scale(samples, model)
        else:
            feats = model(samples)

        # init storage feature matrix
        if dist.get_rank() == 0 and features is None:
            features = torch.zeros(len(data_loader.dataset), feats.shape[-1])
            if use_cuda:
                features = features.cuda(non_blocking=True)
            print(f"Storing features into tensor of shape {features.shape}")

        # get indexes from all processes
        y_all = torch.empty(dist.get_world_size(), index.size(0), dtype=index.dtype, device=index.device)
        y_l = list(y_all.unbind(0))
        y_all_reduce = torch.distributed.all_gather(y_l, index, async_op=True)
        y_all_reduce.wait()
        index_all = torch.cat(y_l)

        # share features between processes
        feats_all = torch.empty(
            dist.get_world_size(),
            feats.size(0),
            feats.size(1),
            dtype=feats.dtype,
            device=feats.device,
        )
        output_l = list(feats_all.unbind(0))
        output_all_reduce = torch.distributed.all_gather(output_l, feats, async_op=True)
        output_all_reduce.wait()

        # update storage feature matrix
        if dist.get_rank() == 0:
            if use_cuda:
                features.index_copy_(0, index_all, torch.cat(output_l))
            else:
                features.index_copy_(0, index_all.cpu(), torch.cat(output_l).cpu())
    return features


def extract_features(model, dataloader, use_cuda):
    model.eval()
    embeddings = []
    with torch.no_grad():
        for batch, _ in tqdm(dataloader, desc="Extracting features"):
            inputs = processor(images=batch, return_tensors="pt", padding=True).to(device)
            outputs = model(**inputs, output_hidden_states=True)
            batch_embeddings = outputs.last_hidden_state.mean(dim=1)
            embeddings.append(batch_embeddings.cpu())
    return torch.vstack(embeddings)
'''

