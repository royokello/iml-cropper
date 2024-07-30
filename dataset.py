import os
import json
import torch
from torch.utils.data import Dataset
from PIL import Image

class ImageDataset(Dataset):
    def __init__(self, image_dir, labels_file, transform=None):
        self.image_dir = image_dir
        self.transform = transform
        self.labels = self.load_labels(labels_file)
        self.image_ids = list(self.labels.keys())

    def load_labels(self, labels_file):
        with open(labels_file, 'r') as f:
            return {int(k): v for k, v in json.load(f).items()}

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        img_id = self.image_ids[idx]
        img_name = f"{img_id}.png"
        image_path = os.path.join(self.image_dir, img_name)
        image = Image.open(image_path).convert('RGB')

        if self.transform:
            image = self.transform(image)

        crop_coords = torch.tensor(self.labels[img_id], dtype=torch.float32)
        return image, crop_coords