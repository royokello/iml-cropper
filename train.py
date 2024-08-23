import argparse
import torch
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import os
from torch.utils.data import Dataset, DataLoader
from dataset import ImageDataset
from utils import generate_model_name, get_model_by_name, log_print, setup_logging
import torch.nn.functional as F
from model import CropperNet
import torch.nn as nn

def bbox_loss(pred, target):
    return F.smooth_l1_loss(pred, target)

def main(working_dir: str, num_epochs: int, checkpoint: int, base_model: str|None):
    cropper_dir = os.path.join(working_dir, 'cropper')
    setup_logging(cropper_dir)
    log_print("training started ...")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log_print(f"using {device}")

    # Setup directories
    models_dir = os.path.join(cropper_dir, 'models')
    os.makedirs(models_dir, exist_ok=True)

    low_res_dir = os.path.join(cropper_dir, 'input', '256p')
    labels_file = os.path.join(cropper_dir, 'labels.json')

    # Define transforms
    transform = transforms.Compose([
        transforms.ToTensor(),
    ])

    # Create dataset and dataloader
    dataset = ImageDataset(low_res_dir, labels_file, transform=transform)
    print(f" * loaded {len(dataset)} images")
    dataloader = DataLoader(dataset, batch_size=16, shuffle=True, num_workers=4)

    # Initialize the model, optimizer, and loss function
    if base_model:
        model = get_model_by_name(device=device, directory=models_dir, name=base_model)
    else:
        model = CropperNet().to(device)

    model.train()
        
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001) # type: ignore

    for epoch in range(num_epochs):
        running_loss = 0.0
        for images, bbox in dataloader:
            images = images.to(device)
            bbox = bbox.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, bbox)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        avg_loss = running_loss / len(dataloader)
        log_print(f"epoch [{epoch + 1}/{num_epochs}], loss: {avg_loss:.4f}")

        if (epoch + 1) % checkpoint == 0:
            checkpoint_name = generate_model_name(base_model, len(dataset), epoch + 1)
            checkpoint_path = os.path.join(models_dir, f"{checkpoint_name}.pth")
            torch.save(model.state_dict(), checkpoint_path)
            log_print(f"checkpoint saved: {checkpoint_path}")

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Train the model.")
    parser.add_argument("-w", "--working_dir", type=str, required=True, help="Directory where the training data is located.")
    parser.add_argument("-e", "--epochs", type=int, required=True, help="Number of epochs to train.")
    parser.add_argument("-c", "--checkpoint", type=int, required=True, help="Number of checkpoints to save.")
    parser.add_argument("-b", "--base_model", type=str, help="Name of the base model if continuing training, or None if starting from scratch.")
    
    args = parser.parse_args()
    
    main(
        working_dir=args.working_dir,
        num_epochs=args.epochs,
        checkpoint=args.checkpoint,
        base_model=args.base_model
    )