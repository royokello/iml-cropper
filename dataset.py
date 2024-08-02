import os
import json
import torch
from torch.utils.data import Dataset
from PIL import Image, ImageOps


class ImageDataset(Dataset):
    def __init__(self, img_dir, json_file, transform=None):
        self.img_dir = img_dir
        self.transform = transform
        with open(json_file, 'r') as f:
            self.labels = json.load(f)
        self.img_list = list(self.labels.keys())

    def __len__(self):
        return len(self.img_list)

    def __getitem__(self, idx):
        img_id = self.img_list[idx]
        img_path = os.path.join(self.img_dir, f"{img_id}.png")
        img = Image.open(img_path).convert("RGB")
        
        coordinates = torch.tensor(self.labels[img_id], dtype=torch.float32)
        
        if self.transform:
            img = self.transform(img)
        
        return img, coordinates
    