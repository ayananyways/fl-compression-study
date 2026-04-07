import torch.nn as nn
import torchvision.models as tv_models


def get_resnet18(num_classes: int = 200) -> nn.Module:
    model = tv_models.resnet18(weights=None)
    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features, num_classes)
    return model
