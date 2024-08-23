import torch
import torch.nn as nn

class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding):
        super(ConvBlock, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
    
    def forward(self, x):
        return self.relu(self.bn(self.conv(x)))

class CropperNet(nn.Module):
    def __init__(self):
        super(CropperNet, self).__init__()
        
        # Backbone for feature extraction
        self.backbone = nn.Sequential(
            ConvBlock(3, 32, kernel_size=3, stride=2, padding=1),
            ConvBlock(32, 64, kernel_size=3, stride=2, padding=1),
            ConvBlock(64, 128, kernel_size=3, stride=2, padding=1),
            ConvBlock(128, 256, kernel_size=3, stride=2, padding=1),
            ConvBlock(256, 512, kernel_size=3, stride=2, padding=1),
        )
        
        # Head for bounding box regression
        self.bbox_head = nn.Sequential(
            nn.Conv2d(512, 256, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 4, kernel_size=1, stride=1, padding=0)  # 4 outputs: x1, y1, x2, y2 for bbox
        )

        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        # Feature extraction
        x = self.backbone(x)
        
        # Bounding box regression
        bbox = self.bbox_head(x)
        
        # Assuming single output per image (global pooling to reduce spatial dimensions)
        bbox = nn.AdaptiveAvgPool2d((1, 1))(bbox)  # Reduce to (batch_size, 4, 1, 1)
        bbox = bbox.view(bbox.size(0), -1)  # Reshape to (batch_size, 4)
        
        # Apply sigmoid to constrain output between 0 and 1
        bbox = self.sigmoid(bbox)
        
        return bbox
