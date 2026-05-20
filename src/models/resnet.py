import torch.nn as nn
import torchvision.models as tv_models


def get_resnet18(num_classes: int = 200, pretrained: bool = True) -> nn.Module:
    weights = tv_models.ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
    model = tv_models.resnet18(weights=weights)
    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features, num_classes)
    return model
