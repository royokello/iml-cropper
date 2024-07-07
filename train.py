import argparse
import os
import json
import time
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision.models.detection import FasterRCNN
import torchvision.transforms as transforms

from utils import load_model

class CropDataset(Dataset):
    def __init__(self, image_dir, labels_file, transform=None):
        self.image_dir = image_dir
        self.transform = transform
        with open(labels_file, 'r') as file:
            self.labels = json.load(file)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        img_name = os.path.join(self.image_dir, self.labels[idx]['filename'])
        image = Image.open(img_name).convert("RGB")
        bbox = self.labels[idx]['bbox']
        
        if self.transform:
            image = self.transform(image)
        
        return image, torch.tensor(bbox)

def train(working_dir: str, epochs: int, checkpoints: int):
    """
    Start or continue training the masking model.
    Models are located in the 'model' subdirectory of the working directory.
    Model filename format is seconds of unix time so the latest has the latest unix time.
    Images are located in the 'images' subdirectory.
    epochs - maximum training limit
    checkpoints - save model after this many epochs
    """
    model_dir = os.path.join(working_dir, 'model')
    os.makedirs(model_dir, exist_ok=True)
    image_dir = os.path.join(working_dir, 'images')
    labels_file = os.path.join(working_dir, 'labels.json')

    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor()
    ])

    dataset = CropDataset(image_dir=image_dir, labels_file=labels_file, transform=transform)

    dataloader = DataLoader(dataset, batch_size=4, shuffle=True)

    model: FasterRCNN = load_model(model_dir)
    
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    model.to(device)

    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(params, lr=0.005, momentum=0.9, weight_decay=0.0005)
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.1)

    model.train()
    for epoch in range(epochs):
        for images, targets in dataloader:
            images = list(image.to(device) for image in images)
            formatted_targets = []
            for target in targets:
                boxes = target.unsqueeze(0).float().to(device)  # Convert to float and add batch dimension
                labels = torch.ones(1, dtype=torch.int64).to(device)  # Single label '1' for foreground
                formatted_targets.append({'boxes': boxes, 'labels': labels})

            loss_dict = model(images, formatted_targets)
            losses = torch.stack([loss for loss in loss_dict.values() if loss is not None]).sum()

            optimizer.zero_grad()
            losses.backward()
            optimizer.step()

        if (epoch + 1) % checkpoints == 0:
            checkpoint_path = os.path.join(model_dir, f"{int(time.time())}.pth")
            torch.save(model.state_dict(), checkpoint_path)
            print(f"Saved checkpoint to {checkpoint_path}")

        lr_scheduler.step()

    final_model_path = os.path.join(working_dir, 'model', f"{int(time.time())}.pth")
    torch.save(model.state_dict(), final_model_path)
    print(f"Saved final model to {final_model_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train the model.")
    parser.add_argument("-w", "--working_dir", type=str, required=True, help="Directory where the training data is located.")
    parser.add_argument("-e", "--epochs", type=int, required=True, help="Number of epochs to train.")
    parser.add_argument("-c", "--checkpoints", type=int, required=True, help="Number of checkpoints to save.")
    
    args = parser.parse_args()
    train(args.working_dir, args.epochs, args.checkpoints)