import os
import torch
from torchvision.models.detection import fasterrcnn_resnet50_fpn, FasterRCNN

def get_latest_checkpoint(model_dir) -> str | None:
    model_files = [f for f in os.listdir(model_dir) if f.endswith('.pth')]
    if not model_files:
        return None
    latest_model_file = max(model_files, key=lambda x: int(x.split('.')[0]))
    return os.path.join(model_dir, latest_model_file)

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