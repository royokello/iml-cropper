import json
import logging
import os
import time
import torch

from model import CropperNet

def setup_logging(working_dir):
    log_file_path = os.path.join(working_dir, 'training.log')

    logging.basicConfig(
        filename=log_file_path,
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
def generate_model_name(base_model: str | None, samples: int, epochs: int) -> str:
    """
    Generate a unique model name based on current timestamp, base model (if any), number of samples, and epochs.
    """
    result = f"{int(time.time())}"
    if base_model:
        result += f"_b={base_model}"
    
    result += f"_s={samples}_e={epochs}"
    
    return result

def get_model_by_name(device: torch.device, directory: str, name: str) -> torch.nn.Module:
    """
    
    """
    
    model = CropperNet()  # Initialize your model architecture

    for file in os.listdir(directory):
        if file.startswith(name):
            model_path = os.path.join(directory, file)
            break
    else:
        raise ValueError(f"No model starting with {name} found in {directory}")

    model.load_state_dict(torch.load(model_path, map_location=device))

    model = model.to(device)
    
    return model

def get_model_by_latest(device: torch.device, directory: str|None=None) -> torch.nn.Module:
    """
    Load a model whose model name is the latest time from the specified directory and move it to the specified device.
    """
    model = CropperNet()

    if directory and os.path.exists(directory):
        model_files = [f for f in os.listdir(directory) if f.endswith('.pth')]
        if not model_files:
            raise ValueError(f"No model files found in {directory}")

        latest_model = max(model_files, key=lambda x: int(x.split('_')[0]))
        print(f"latest model: {latest_model}")
        
        model_path = os.path.join(directory, latest_model)

        model.load_state_dict(torch.load(model_path, map_location=device))

    model = model.to(device)
    
    return model

def get_labels(directory: str) -> dict[int, list[int]]:
    """
    Load labels from a JSON file in the specified directory.
    """
    labels_file = os.path.join(directory, 'cropper', 'labels.json')
    if os.path.exists(labels_file):
        with open(labels_file, 'r') as f:
            labels = json.load(f)

        labels = {int(k): v for k, v in labels.items()}
    else:
        labels = {}
    return labels

def save_labels(directory: str, labels: dict[int, list[int]]):
    """
    Save labels to a JSON file in the specified directory.
    """
    labels_file = os.path.join(directory, 'cropper', 'labels.json')
    with open(labels_file, 'w') as f:
        json.dump(labels, f, indent=4)

def log_print(message):
    print(message)
    logging.info(message)