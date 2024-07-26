import argparse
import os
import json
import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from PIL import Image
from tqdm import tqdm

from transformers import ViTModel, ViTConfig
import torch.nn as nn
import torch.nn.functional as F

from utils import generate_model_name

class CropperViT(nn.Module):
    def __init__(self, num_coords=4):
        super().__init__()
        self.vit = ViTModel.from_pretrained('google/vit-base-patch16-256')
        self.regressor = nn.Linear(self.vit.config.hidden_size, num_coords)

    def forward(self, pixel_values):
        outputs = self.vit(pixel_values=pixel_values)
        crop_coords = self.regressor(outputs.last_hidden_state[:, 0])
        return crop_coords

def train(working_dir: str, epochs: int, checkpoint: int, base_model: str | None):
    """
    Start or continue training the masking model.
    Models are located in the 'model' subdirectory of the working directory.
    Images are located in the 'images' subdirectory.
    epochs - maximum training limit
    checkpoint - save model after this many epochs
    base_model - specify model to continue training from
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Setup directories
    model_dir = os.path.join(working_dir, 'model')
    image_dir = os.path.join(working_dir, 'images')
    labels_file = os.path.join(working_dir, 'labels.json')
    os.makedirs(model_dir, exist_ok=True)

    # Initialize model
    model = CropperViT().to(device)
    if base_model:
        model.load_state_dict(torch.load(os.path.join(model_dir, base_model)))
        print(f"Loaded base model: {base_model}")

    # Define dataset and dataloader
    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    class ImageDataset(Dataset):
        def __init__(self, image_dir, labels_file, transform=None):
            self.image_dir = image_dir
            self.transform = transform
            self.labels = self.load_labels(labels_file)
            self.labeled_images = self.get_labeled_images()

        def load_labels(self, labels_file):
            with open(labels_file, 'r') as f:
                return {int(k): v for k, v in json.load(f).items()}

        def get_labeled_images(self):
            return [f"{id}.png" for id in self.labels.keys() 
                    if os.path.exists(os.path.join(self.image_dir, f"{id}.png"))]

        def __len__(self):
            return len(self.labeled_images)

        def __getitem__(self, idx):
            img_name = self.labeled_images[idx]
            image_path = os.path.join(self.image_dir, img_name)
            image = Image.open(image_path).convert('RGB')
            if self.transform:
                image = self.transform(image)
            
            # Extract the ID from the filename (remove '.png' and convert to int)
            img_id = int(img_name[:-4])
            crop_coords = torch.tensor(self.labels[img_id])
            return image, crop_coords

    dataset = ImageDataset(image_dir, labels_file, transform=transform)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True, num_workers=4)

    # Define loss function and optimizer
    criterion = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    # Training loop
    for epoch in range(epochs):
        model.train()
        epoch_loss = 0
        for images, labels in tqdm(dataloader, desc=f"Epoch {epoch+1}/{epochs}"):
            images, labels = images.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()

        print(f"Epoch {epoch+1} loss: {epoch_loss/len(dataloader):.4f}")

        # Save checkpoint
        if (epoch + 1) % checkpoint == 0:
            checkpoint_name = generate_model_name(base_model, len(dataset), epoch + 1)
            checkpoint_path = os.path.join(model_dir, f"{checkpoint_name}.pth")
            torch.save(model.state_dict(), checkpoint_path)
            print(f"Checkpoint saved: {checkpoint_path}")

    # Save final model
    final_model_name = generate_model_name(base_model, len(dataset), epochs)
    final_model_path = os.path.join(model_dir, f"{final_model_name}.pth")
    torch.save(model.state_dict(), final_model_path)
    print(f"Final model saved: {final_model_path}")

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Train the model.")
    parser.add_argument("-w", "--working_dir", type=str, required=True, help="Directory where the training data is located.")
    parser.add_argument("-e", "--epochs", type=int, required=True, help="Number of epochs to train.")
    parser.add_argument("-c", "--checkpoint", type=int, required=True, help="Number of checkpoints to save.")
    parser.add_argument("-b", "--base_model", type=str, help="Name of the base model if continuing training, or None if starting from scratch.")
    
    args = parser.parse_args()
    
    train(
        working_dir=args.working_dir,
        epochs=args.epochs,
        checkpoint=args.checkpoint,
        base_model=args.base_model
    )