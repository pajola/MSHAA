import torch
import torch.nn as nn
import torch.nn.functional as F

from tqdm import tqdm


class CNN(nn.Module):
    """
    Convolutional Neural Network
    """
    def __init__(self, num_classes, num_layers, neurons):
        """
        Initialize the Convolutional Neural Network
        :param num_classes: number of classes
        :param num_layers: number of layers
        :param neurons: number of neurons
        """
        super(CNN, self).__init__()
        self.num_layers = num_layers
        for i in range(num_layers):
            if i == 0:
                self.conv1 = nn.Conv1d(1, 32, 16, 10)
                self.batchnorm1 = nn.BatchNorm1d(32)
                self.maxpool1 = nn.MaxPool1d(4)
            elif i < 4:
                setattr(self, f'conv{i+1}', nn.Conv1d(32, 32, 3, 1))
                setattr(self, f'batchnorm{i+1}', nn.BatchNorm1d(32))
                setattr(self, f'maxpool{i+1}', nn.MaxPool1d(2))
            else:
                setattr(self, f'conv{i+1}', nn.Conv1d(32, 32, 2, 2))
                setattr(self, f'batchnorm{i+1}', nn.BatchNorm1d(32))

        self.fc1 = nn.Linear(32, neurons)
        self.fc2 = nn.Linear(neurons, num_classes)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        """
        Forward pass of the model
        :param x: input tensor
        :return: output tensor
        """
        for i in range(self.num_layers):
            x = getattr(self, f'conv{i+1}')(x)
            x = self.relu(x)
            x = getattr(self, f'batchnorm{i+1}')(x)
            if i < 4:
                x = getattr(self, f'maxpool{i+1}')(x)
        x = F.avg_pool1d(x, x.shape[-1])
        x = x.permute(0, 2, 1)
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        return x


# Train Classification Model
def train(model, optimizer, dataloader_train, num_epochs, device):
    """
    Train the Speech Classification Models
    :param model: CNN model
    :param optimizer: optimizer technique
    :param dataloader_train: dataloader for training
    :param num_epochs: number of epochs
    :param device: device to use
    :return: loss_train, acc_train: list of loss and accuracy for each epoch
    """
    criterion = torch.nn.CrossEntropyLoss()
    loss_train, acc_train = [], []
    for epoch in range(num_epochs):
        model.train()
        total_acc_train, total_count_train, n_train_batches, total_loss_train = 0, 0, 0, 0
        # train each batch
        for batch, labels in tqdm(dataloader_train, desc=f"Epoch {epoch + 1}", leave=True):

            batch = batch.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            # predict train sample
            logits = model(batch).squeeze()
            # compute loss
            loss = criterion(logits, labels)
            total_loss_train += loss.sum()
            # backpropagation
            loss.backward()
            optimizer.step()
            # compute accuracy of train batch
            total_acc_train += (logits.argmax(1) == labels).sum().item()
            total_count_train += labels.size(0)
            n_train_batches += 1
        # get train values for the epoch
        avg_loss_train = total_loss_train / len(dataloader_train.dataset)
        loss_train.append(avg_loss_train.item())
        accuracy_train = total_acc_train / total_count_train
        acc_train.append(accuracy_train)
        # Print progress
        if (epoch + 1) % 5 == 0:
            print(f"epoch: {epoch + 1} -> Accuracy: {100 * accuracy_train:.2f}%, Loss: {avg_loss_train:.8f}")
    return loss_train, acc_train
