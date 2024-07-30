import argparse
from datetime import datetime
import os
import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm
from dataset import ImageDataset
from utils import generate_model_name, get_model_by_name, log_print

def train(working_dir: str, epochs: int, checkpoint: int, base_model: str|None):
    """
    Start or continue training the masking model.
    Models are located in the 'model' subdirectory of the working directory.
    Images are located in the 'images' subdirectory.
    epochs - maximum training limit
    checkpoint - save model after this many epochs
    base_model - specify model to continue training from
    """
    log_print("training started ...")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log_print(f"using {device}")
    
    # Setup directories
    model_dir = os.path.join(working_dir, 'model')
    image_dir = os.path.join(working_dir, '256p')
    labels_file = os.path.join(working_dir, 'labels.json')
    os.makedirs(model_dir, exist_ok=True)

    # Initialize model
    model = get_model_by_name(device=device, directory=model_dir, name=base_model)

    # Define dataset and dataloader
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    ])

    dataset = ImageDataset(image_dir, labels_file, transform=transform)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True, num_workers=4)

    # Define loss function and optimizer
    criterion = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    # Training loop
    for epoch in range(epochs):
        model.train()
        epoch_loss = 0

        for images, labels in tqdm(dataloader):

            images, labels = images.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()

        log_print(f" * epoch {epoch+1}, loss: {epoch_loss/len(dataloader):.4f}")

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