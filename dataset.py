import os
import json
import torch
from torch.utils.data import Dataset
from PIL import Image


class ImageDataset(Dataset):
    def __init__(self, root_dir, labels_file, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        with open(labels_file, 'r') as f:
            self.labels = json.load(f)
        self.image_names = list(self.labels.keys())

    def __len__(self):
        return len(self.image_names)

    def __getitem__(self, idx):
        img_name = self.image_names[idx]
        img_path = os.path.join(self.root_dir, f"{img_name}.png")
        image = Image.open(img_path).convert("RGB")
        
        bbox = self.labels[img_name]
        x_min, y_min, x_max, y_max = bbox
        bbox = torch.tensor([x_min, y_min, x_max, y_max], dtype=torch.float32)
        
        if self.transform:
            image = self.transform(image)
        
        return image, bbox
    