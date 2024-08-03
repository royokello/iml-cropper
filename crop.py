import argparse
import os
from PIL import Image
import numpy as np
import torch

from lib import load_model

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

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Crop images in a specified directory.")
    parser.add_argument("-w", "--working_dir", type=str, required=True, help="Directory where the images to be cropped are located.")
    
    args = parser.parse_args()
    crop(args.working_dir)