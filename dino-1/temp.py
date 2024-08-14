from transformers import BeitImageProcessor, BeitModel, BeitConfig # Load and save the model, processor, and config
import torch
import os
from PIL import Image
import torch
from torch.utils.data import DataLoader, Dataset
from transformers import BeitImageProcessor, BeitModel
from tqdm import tqdm
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import os
import torch
import torchvision.transforms as T

from PIL import Image
import numpy as np
import torch
from tqdm import tqdm

class1_folder = "/media/ml-vm/3fea60f1-fd93-47c9-b3a5-c2fde1bd92a9/cls_falses/train/other"
#class2_folder = "/home/ml-vm-pve2/Desktop/Prince/beit/test1"

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
    return np.vstack(embeddings), torch.tensor(labels)

def compute_embeddings_one(image_path, label, processor, model, device):
    embeddings = []
    filenames = []
    labels_list = []
    
    model.eval()
    
    with torch.no_grad():
        # for image_path, label in zip(image_paths, labels):
            # Load and preprocess the image
        image = Image.open(image_path).convert("RGB")
        inputs = processor(images=image, return_tensors="pt", padding=True).to(device)
        
        # Compute the embeddings
        outputs = model(**inputs, output_hidden_states=True)
        image_embedding = outputs.last_hidden_state.mean(dim=1)
        
        embeddings.append(image_embedding.cpu().numpy())
        filenames.append(image_path)
        labels_list.append(label)
    
    return np.vstack(embeddings), torch.tensor(labels_list)



processor = BeitImageProcessor.from_pretrained('microsoft/beit-base-patch16-224-pt22k-ft22k')
model = BeitModel.from_pretrained('microsoft/beit-base-patch16-224-pt22k-ft22k')
model.eval()

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = model.to(device)
# Prepare datasets and dataloaders
dataset1 = ImageFolderDataset(class1_folder,label = 0)
#dataset2 = ImageFolderDataset(class2_folder,label = 1)
dataloader1 = DataLoader(dataset1, batch_size=32, shuffle=False, collate_fn=lambda x: list(zip(*x)))
#dataloader2 = DataLoader(dataset2, batch_size=32, shuffle=False, collate_fn=lambda x: list(zip(*x)))
# Function to compute embeddings
is_bucket = False

if is_bucket == False :
    train_features, train_labels  = compute_embeddings(dataloader1)
    train_features = train_features / np.linalg.norm(train_features, axis=1, keepdims=True)
    torch.save(train_features, 'True_bucket_1.pth')


image_path = "/media/ml-vm/3fea60f1-fd93-47c9-b3a5-c2fde1bd92a9/cls/other/AM_01_178f1565_4899.jpg"
query_features,_ = compute_embeddings_one(image_path,0,processor, model,device )
query_features = query_features / np.linalg.norm(query_features, axis=1, keepdims=True)
True_bucket_path='True_bucket_1.pth'
ref_features = torch.load(True_bucket_path)


# # Compute similarity matrix
similarity_matrix = cosine_similarity(query_features, ref_features)
print(similarity_matrix)