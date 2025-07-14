import torch
import torch.nn as nn
from torchvision.models import resnet18

class VideoModel(nn.Module):
    def __init__(self, num_classes):
        super(VideoModel, self).__init__()
        backbone = resnet18(pretrained=True)
        self.feature_extractor = nn.Sequential(*list(backbone.children())[:-1])  # 去掉分类层
        self.classifier = nn.Linear(512, num_classes)  # ResNet-18 最后输出512维特征

    def forward(self, x):
        with torch.no_grad():  # 冻结特征提取部分
            features = self.feature_extractor(x)
            features = features.view(features.size(0), -1)
        out = self.classifier(features)
        return out
