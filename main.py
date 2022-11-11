import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

import inference
import training
import utils

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

hyper_params = {
    "epochs": 5,
    "batchsize": 64,
    "lossfunc": nn.CrossEntropyLoss(),
}


def main():

    # Prepare Training and Validation Dataloader
    train_directory, valid_directory, test_directory = utils.get_dataset_paths()
    image_transforms = utils.get_transforms()

    train_data = utils.get_dataset(train_directory, image_transforms["train"])
    valid_data = utils.get_dataset(valid_directory, image_transforms["valid"])

    train_data_loader = DataLoader(
        train_data, batch_size=hyper_params["batchsize"], shuffle=True
    )
    valid_data_loader = DataLoader(
        valid_data, batch_size=hyper_params["batchsize"], shuffle=True
    )

    # Prepare Label Index Dictionary
    idx_to_class = {v: k for k, v in train_data.class_to_idx.items()}
    num_classes = len(idx_to_class)

    model = utils.get_pretrained_model(num_classes, pretrained=True)
    model = model.to(device)

    trained_model = training.train_model(
        model,
        hyper_params["epochs"],
        optim.Adam(model.parameters()),
        hyper_params["lossfunc"],
        train_data_loader,
        valid_data_loader,
        device,
    )

    inference.predict(
        trained_model, image_transforms["test"], device, idx_to_class, test_directory
    )


if __name__ == "__main__":
    main()
