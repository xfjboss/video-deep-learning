# src/video_model.py
import torch
import torch.nn as nn
from torchvision.models import resnet18
from torchvision.models.video import r3d_18


class VideoModel(nn.Module):
    def __init__(self, num_classes, model_type='2d'):
        super(VideoModel, self).__init__()
        self.model_type = model_type

        if model_type == '2d':
            backbone = resnet18(pretrained=True)
            self.feature_extractor = nn.Sequential(*list(backbone.children())[:-1])  # Remove FC
            self.classifier = nn.Linear(512, num_classes)
        elif model_type == '3d':
            model3d = r3d_18(pretrained=True)
            self.feature_extractor = nn.Sequential(*list(model3d.children())[:-1])
            self.classifier = nn.Linear(512, num_classes)
        else:
            raise ValueError("Invalid model_type: choose '2d' or '3d'")

    def forward(self, x):
        with torch.no_grad():
            features = self.feature_extractor(x)
        features = features.view(features.size(0), -1)
        out = self.classifier(features)
        return out
