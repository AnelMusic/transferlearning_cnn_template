import torch


def train_model(model, epochs, optimizer, criterion, train_loader, val_loader, device):
    """_summary_

    Args:
        model (AlexNet): Model
        epochs (int): Number Epochs
        optimizer (_type_): Optimizer
        criterion (nn.CrossEntropyLoss()): Cost Function
        train_loader (DataLoader): Train DataLoader
        val_loader (DataLoader): Validation DataLoader
        device (str): Hardware Device
    """
    for epoch in range(epochs):
        # Loss and Accuracy per the epoch
        train_loss = 0.0
        train_acc = 0.0
        valid_loss = 0.0
        valid_acc = 0.0

        for batch_idx, (data, labels) in enumerate(train_loader):
            data = data.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, labels)
            loss.backward()
            optimizer.step()

            ret, predictions = torch.max(output.data, 1)

            correct_counts = predictions.eq(labels.data.view_as(predictions))
            acc = torch.mean(correct_counts.type(torch.FloatTensor))

            # Log Acc and Loss
            train_acc += acc.item()
            train_loss += loss.item()

        # Set to evaluation mode
        model.eval()

        # Validation loop
        for batch_idx, (inputs, labels) in enumerate(val_loader):
            inputs = inputs.to(device)
            labels = labels.to(device)

            with torch.no_grad():
                outputs = model(inputs)

            loss = criterion(outputs, labels)
            ret, predictions = torch.max(outputs.data, 1)

            correct_counts = predictions.eq(labels.data.view_as(predictions))
            acc = torch.mean(correct_counts.type(torch.FloatTensor))
            # Log Acc and Loss
            valid_loss += loss.item()
            valid_acc += acc.item()

        print(
            "Epoch : {:03d}, Training: Loss: {:.4f}, Accuracy: {:.4f}%, Validation : Loss : {:.4f}, Accuracy: {:.4f}%".format(
                epoch + 1,
                train_loss / len(train_loader),
                train_acc / len(train_loader),
                valid_loss / len(val_loader),
                valid_loss / len(val_loader),
            )
        )

        model.train()

    print("Done Training Model")
    model.eval()
    return model
