# src/video_model.py
import torch
import torch.nn as nn
from torchvision.models import resnet18
from torchvision.models.video import r3d_18

class VideoModel(nn.Module):
    def __init__(self, num_classes, model_type='2d'):
        super().__init__()
        self.model_type = model_type

        if model_type == '2d':
            base = resnet18(pretrained=True)
            self.backbone = nn.Sequential(*list(base.children())[:-1])
            self.classifier = nn.Linear(512, num_classes)
        elif model_type == '3d':
            base = r3d_18(pretrained=True)
            self.backbone = nn.Sequential(*list(base.children())[:-1])
            self.classifier = nn.Linear(512, num_classes)
        else:
            raise ValueError("Invalid model type")

    def forward(self, x):
        with torch.no_grad():
            feats = self.backbone(x)
        feats = feats.view(feats.size(0), -1)
        out = self.classifier(feats)
        return out

