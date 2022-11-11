import os

import torch.nn as nn
from torchvision import datasets, transforms

import model


def get_dataset_paths():
    """Get Dataset Path.

    Returns:
        Tuple: Train/Val/Test Dataset
    """
    current_path = os.path.dirname(os.path.abspath(__file__))
    train_directory = os.path.join(current_path, "data/train")
    valid_directory = os.path.join(current_path, "data/valid")
    test_directory = os.path.join(current_path, "data/test")

    return train_directory, valid_directory, test_directory


def get_transforms():
    """Get Img Transformations.

    Returns:
        dict: Dictionary containing Img Transformations.
    """
    image_transforms = {
        "train": transforms.Compose(
            [
                transforms.RandomResizedCrop(size=256, scale=(0.8, 1.0)),
                transforms.RandomRotation(degrees=15),
                transforms.RandomHorizontalFlip(),
                transforms.CenterCrop(size=224),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ]
        ),
        "valid": transforms.Compose(
            [
                transforms.Resize(size=256),
                transforms.CenterCrop(size=224),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ]
        ),
        "test": transforms.Compose(
            [
                transforms.Resize(size=256),
                transforms.CenterCrop(size=224),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ]
        ),
    }

    return image_transforms


def get_dataset(ds_path, transformation):
    """Get Dataset.

    Args:
        ds_path (str): Path to dataset
        transformation (transforms.Compose): Pytorch Img Transformation

    Returns:
        datasets.ImageFolder: Dataset
    """
    return datasets.ImageFolder(root=ds_path, transform=transformation)


def get_pretrained_model(num_classes, pretrained=True):
    return model.create_model(num_classes)


def show_model_info(model):
    print(model)
