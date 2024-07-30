from datetime import datetime
import json
import logging
import os
import time
import torch
from torchvision.models.detection import fasterrcnn_resnet50_fpn, FasterRCNN
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
import torchvision.transforms as transforms
from PIL import Image

from model import CropperViT


def get_latest_checkpoint(model_dir) -> str | None:
    model_files = [f for f in os.listdir(model_dir) if f.endswith('.pth')]
    if not model_files:
        return None
    latest_model_file = max(model_files, key=lambda x: int(x.split('.')[0]))
    return os.path.join(model_dir, latest_model_file)

def load_model(model_dir: str, ) -> FasterRCNN:
    model = fasterrcnn_resnet50_fpn(pretrained=False)
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, 3)

    latest_checkpoint = get_latest_checkpoint(model_dir)
    if latest_checkpoint:
        model.load_state_dict(torch.load(latest_checkpoint))
        print(f"Loaded model from {latest_checkpoint}")
    
    return model

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

def get_model_by_name(device: torch.device, directory: str|None=None, name: str|None=None) -> CropperViT:
    """
    Load a model whose filename starts with the given name from the specified directory and move it to the specified device.
    """
    result = CropperViT().to(device)

    if directory and name:
        for file in os.listdir(directory):
            if file.startswith(name):
                model_path = os.path.join(directory, file)
                break
        else:
            raise ValueError(f"No model starting with {name} found in {directory}")

        result.load_state_dict(torch.load(model_path))
    
    return result

def get_labels(directory: str) -> dict[int, list[int]]:
    """
    Load labels from a JSON file in the specified directory.
    """
    labels_file = os.path.join(directory, 'labels.json')
    if os.path.exists(labels_file):
        with open(labels_file, 'r') as f:
            labels = json.load(f)
    else:
        labels = {}
    return labels

def save_labels(directory: str, labels: dict[int, list[int]]):
    """
    Save labels to a JSON file in the specified directory.
    """
    labels_file = os.path.join(directory, 'labels.json')
    with open(labels_file, 'w') as f:
        json.dump(labels, f, indent=4)

def log_print(message):
    current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{current_time}] {message}")
