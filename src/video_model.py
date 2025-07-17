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
            self.backbone = nn.Sequential(*list(base.children())[:-1])  # 去掉fc
            self.classifier = nn.Linear(512, num_classes)
        elif model_type == '3d':
            base = r3d_18(pretrained=True)
            self.backbone = nn.Sequential(*list(base.children())[:-1])
            self.classifier = nn.Linear(512, num_classes)
        else:
            raise ValueError("Invalid model type")

    def forward(self, x):
        if self.model_type == '2d':
            # 输入 x: [B, T, C, H, W] → 平均时间帧后 [B, C, H, W]
            x = x.mean(dim=1)
        elif self.model_type == '3d':
            # 输入 x: [B, T, C, H, W] → 转为 [B, C, T, H, W]
            x = x.permute(0, 2, 1, 3, 4)
        
        feats = self.backbone(x)           # [B, 512, 1, 1] or [B, 512, 1, 1, 1]
        feats = feats.view(feats.size(0), -1)  # 展平
        out = self.classifier(feats)       # [B, num_classes]
        return out
