import argparse
import os
from PIL import Image
import numpy as np
import torch

from utils import get_model_by_latest, get_model_by_name
from predict import predict


def crop(working_dir: str, model_name: str|None=None):
    """
    """
    models_dir = os.path.join(working_dir, 'models')
    
    # Load the trained model
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    if model_name:
        model = get_model_by_name(device=device, directory=models_dir, name=model_name)
    else:
        model = get_model_by_latest(device=device, directory=models_dir)

    low_input_dir = os.path.join(working_dir, 'input', '256p')

    low_output_dir = os.path.join(working_dir, 'output', '256p')
    std_output_dir = os.path.join(working_dir, 'output', '512p')
    os.makedirs(low_output_dir, exist_ok=True)
    os.makedirs(std_output_dir, exist_ok=True)

    for img_name in os.listdir(low_input_dir):

        low_input_path = os.path.join(low_input_dir, img_name)
        
        prediction = predict(device=device, model=model, image_path=low_input_path)

        high_input_path = os.path.join(working_dir, 'input', '1024p', img_name)

        # Open the image
        high_input_image = Image.open(high_input_path)

        # Apply the predicted crop
        x1, y1, x2, y2 = [int(p * 1024) for p in prediction]
        cropped_image = high_input_image.crop((x1, y1, x2, y2))

        resolutions = [
            (256, low_output_dir),
            (512, std_output_dir)
        ]

        for (output_res, img_output_dir) in resolutions:
            # Resize the cropped image to the specified resolution
            resized_image = cropped_image.resize((output_res, output_res), Image.LANCZOS) # type: ignore
            
            # Save the resized cropped image to the output directory
            img_output_path = os.path.join(img_output_dir, img_name)
            resized_image.save(img_output_path)

            print(f"cropped and resized image saved to {img_output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Crop images in a specified directory.")
    parser.add_argument("-w", "--working_dir", type=str, required=True, help="Directory where the images to be cropped are located.")
    
    args = parser.parse_args()
    crop(args.working_dir)