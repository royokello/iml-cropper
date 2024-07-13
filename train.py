import argparse
import csv
import logging
import os
import json
import time
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision.models.detection import FasterRCNN
import torchvision.transforms as transforms

from utils import load_model, setup_logging

class CropDataset(Dataset):
    def __init__(self, image_dir, labels_file, transform=None):
        self.image_dir = image_dir
        self.transform = transform
        self.labels = []

        with open(labels_file, 'r') as file:
            reader = csv.reader(file)
            next(reader, None)  # Skip the header row if your CSV has one
            for row in reader:
                # Assuming the CSV columns are in the order: image_id, x, y, delta
                # Convert x, y, delta to float and store them with the image filename
                self.labels.append({
                    'i': row[0],
                    'x': float(row[1]),
                    'y': float(row[2]),
                    'd': float(row[3])
                })

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        label_info = self.labels[idx]
        img_name = os.path.join(self.image_dir, f"{label_info['i']}.png")
        image = Image.open(img_name).convert('RGB')
        
        target = {
            'boxes': torch.tensor([[label_info['x'], label_info['y'], label_info['x'] + label_info['d'], label_info['y'] + label_info['d']]], dtype=torch.float32),
            'labels': torch.tensor([1], dtype=torch.int64)
        } 

        if self.transform:
            image = self.transform(image)

        return image, target

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
    image_dir = os.path.join(working_dir, '256p')
    labels_file = os.path.join(working_dir, 'labels.csv')
    setup_logging(working_dir)

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
        for batch_idx, (images, targets) in enumerate(dataloader):
            images = list(image.to(device) for image in images)
            targets_list = [{k: v[i].to(device) for k, v in targets.items()} for i in range(len(images))]

            loss_dict = model(images, targets_list)
            losses = sum(loss for loss in loss_dict.values())
            

            optimizer.zero_grad()
            losses.backward()
            optimizer.step()

            # Log training progress
            logging.info(f'Epoch: {epoch}, Batch: {batch_idx}, Loss: {losses.item()}')

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