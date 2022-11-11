import torch.nn as nn
from torchvision import models


def create_model(num_classes):
    """Create Model.

    Args:
        num_classes (int): Number of classes

    Returns:
        AlexNet: AlexNet
    """
    model = models.alexnet(weights="AlexNet_Weights.DEFAULT")

    for param in model.parameters():
        param.requires_grad = False

    model.classifier[6] = nn.Linear(4096, num_classes)
    model.classifier.add_module("7", nn.LogSoftmax(dim=1))
    print("Done Creating Model")
    return model
