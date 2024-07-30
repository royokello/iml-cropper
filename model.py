from transformers import ViTModel, ViTConfig
import torch.nn as nn
import torch.nn.functional as F

class CropperViT(nn.Module):
    def __init__(self, num_coords=4):
        super().__init__()
        self.vit = ViTModel.from_pretrained('google/vit-base-patch16-224')
        self.regressor = nn.Linear(self.vit.config.hidden_size, num_coords)

    def forward(self, pixel_values):
        outputs = self.vit(pixel_values=pixel_values)
        crop_coords = self.regressor(outputs.last_hidden_state[:, 0])
        return crop_coords