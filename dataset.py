import os
import json
import torch
from torch.utils.data import Dataset
from PIL import Image


class ImageDataset(Dataset):
    def __init__(self, low_res_dir, labels_file, transform=None):
        self.low_res_dir = low_res_dir
        self.transform = transform
        with open(labels_file, 'r') as f:
            self.labels = json.load(f)
        self.image_names = list(self.labels.keys())

    def __len__(self):
        return len(self.image_names)

    def __getitem__(self, idx):
        img_name = self.image_names[idx]
        img_path = os.path.join(self.low_res_dir, f"{img_name}.png")
        image = Image.open(img_path).convert("RGB")

        if self.transform:
            image = self.transform(image)
        
        bbox = self.labels[img_name]
        x1, y1, x2, y2 = bbox
        bbox = torch.tensor([x1, y1, x2, y2], dtype=torch.float32)

        return image, bbox

    