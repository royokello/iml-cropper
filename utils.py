import logging
import os
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

def load_model(model_dir: str) -> FasterRCNN:
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

def predict(model: FasterRCNN, image_path: str) -> dict[str, int]:
    """
    Predicts the x, y, d values for cropping.
    """
    model.eval()
    
    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor()
    ])
    
    image = Image.open(image_path).convert('RGB')
    image = transform(image).unsqueeze(0)  # Add batch dimension

    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    image = image.to(device)
    model = model.to(device)

    with torch.no_grad():
        outputs = model(image)

    # Assuming the model returns a list of dictionaries and we're interested in the first prediction
    output = outputs[0]
    
    # Extract the first bounding box (if multiple, you can handle accordingly)
    if 'boxes' in output and len(output['boxes']) > 0:
        box = output['boxes'][0].cpu().numpy()
        x, y, x2, y2 = box
        d = max(x2 - x, y2 - y)  # Assuming d is the largest side of the bounding box
        return {'x': int(x), 'y': int(y), 'd': int(d)}
    else:
        return {'x': 0, 'y': 0, 'd': 0}