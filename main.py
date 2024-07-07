import os
import json
import time
from PIL import Image
import numpy as np
import torch
import torchvision
from torch.utils.data import Dataset, DataLoader
from torchvision.models.detection import fasterrcnn_resnet50_fpn, FasterRCNN
import torchvision.transforms as transforms

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

def get_latest_checkpoint(model_dir):
    model_files = [f for f in os.listdir(model_dir) if f.endswith('.pth')]
    if not model_files:
        return None
    latest_model_file = max(model_files, key=lambda x: int(x.split('.')[0]))
    return os.path.join(model_dir, latest_model_file)

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

    model = fasterrcnn_resnet50_fpn(pretrained=True)
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = torch.nn.Linear(in_features, 2) 
    
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    model.to(device)

    # Load the latest checkpoint if available
    latest_checkpoint = get_latest_checkpoint(model_dir)
    if latest_checkpoint:
        model.load_state_dict(torch.load(latest_checkpoint))
        print(f"Loaded model from {latest_checkpoint}")

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

def load_model(model_dir: str) -> FasterRCNN:
    model = fasterrcnn_resnet50_fpn(pretrained=False)
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = torch.nn.Linear(in_features, 2)  # 2 classes: background and crop region

    latest_checkpoint = get_latest_checkpoint(model_dir)
    if latest_checkpoint:
        model.load_state_dict(torch.load(latest_checkpoint))
        print(f"Loaded model from {latest_checkpoint}")
    model.eval()
    return model

def image_to_tensor(image: Image.Image, device: torch.device) -> torch.Tensor:
    """
    Transforms a PIL Image to a torch Tensor, adds a batch dimension, and moves it to the specified device.
    """
    image = image.resize((256, 256))
    image_tensor = torch.tensor(np.array(image)).permute(2, 0, 1).float() / 255.0  # HWC to CHW and normalize
    image_tensor = image_tensor.unsqueeze(0)
    image_tensor = image_tensor.to(device)
    return image_tensor

def crop(working_dir: str):
    """
    Crops images in the images subdirectory into the output subdirectory.
    """
    image_dir = os.path.join(working_dir, 'images')
    output_dir = os.path.join(working_dir, 'output')
    model_dir = os.path.join(working_dir, 'model')
    os.makedirs(output_dir, exist_ok=True)
    
    # Load the trained model
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    model = load_model(model_dir).to(device)

    for image_name in os.listdir(image_dir):
        if image_name.endswith(('.jpg', '.png')):
            image_path = os.path.join(image_dir, image_name)
            image_object = Image.open(image_path).convert("RGB")
            image_tensor: torch.Tensor = image_to_tensor(image=image_object, device=device)

            with torch.no_grad():
                predictions = model(image_tensor)
            
            boxes = predictions[0]['boxes'].cpu().numpy()
            scores = predictions[0]['scores'].cpu().numpy()

            # Filter boxes with a confidence score above a threshold (e.g., 0.5)
            threshold = 0.5
            boxes = boxes[scores >= threshold]

            if len(boxes) > 0:
                box = boxes[0]  # Taking the first box for simplicity
                cropped_image = image_object.crop((box[0], box[1], box[2], box[3]))
                output_path = os.path.join(output_dir, image_name)
                cropped_image.save(output_path)
                print(f"Cropped and saved {image_name}")

def label(working_dir: str):
    """
    Opens a server with a page to assist in adding manual labels.
    Opens on the first image.
    Left and Right Buttons.
    Skip to next or previous labeled.
    Square cropping tool.
    """

