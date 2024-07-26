import logging
import os
import time
import torch
from torchvision.models.detection import fasterrcnn_resnet50_fpn, FasterRCNN
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
import torchvision.transforms as transforms
from PIL import Image


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

    Args:
    base_model (str | None): Name of the base model if continuing training, or None if starting from scratch.
    samples (int): Number of samples used in training.
    epochs (int): Number of epochs the model was trained for.

    Returns:
    str: A unique model name string.
    """
    result = f"{int(time.time())}"
    if base_model:
        result += f"_b={base_model}"
    
    result += f"_s={samples}_e={epochs}"
    
    return result