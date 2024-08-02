from transformers import ViTModel, ViTFeatureExtractor
import torch
import torch.nn as nn

class CropPredictor(nn.Module):
    def __init__(self):
        super(CropPredictor, self).__init__()
        # Load a pretrained Vision Transformer
        self.feature_extractor = ViTFeatureExtractor.from_pretrained('google/vit-base-patch16-224')
        self.vit = ViTModel.from_pretrained('google/vit-base-patch16-224')
        
        # Add our own regressor layers
        self.regressor = nn.Sequential(
            nn.Linear(self.vit.config.hidden_size, 256),
            nn.ReLU(),
            nn.Linear(256, 4)
        )

    def forward(self, x):
        # Extract features with ViT
        # x = self.feature_extractor(x, return_tensors="pt")['pixel_values']
        x = self.vit(x).last_hidden_state[:, 0, :]
        x = self.regressor(x)
        x = torch.sigmoid(x)
        return x
