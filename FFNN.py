import torch
from torch import nn
from tqdm import tqdm
import torch.nn.functional as F


class LinearReLu(nn.Module):
    """
    Linear + ReLU Layer
    """
    def __init__(self, input_dim, output_dim):
        """
        Initialize the Linear ReLU Layer
        :param input_dim: number of input features
        :param output_dim: number of output features
        """
        super().__init__()
        self.linear = nn.Linear(input_dim, output_dim)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        """
        Forward pass of the model
        :param x: input tensor
        :return: output tensor
        """
        return self.relu(self.linear(x))


class FeedforwardNetwork(nn.Module):
    """
    Feedforward Neural Network
    """
    def __init__(self, input_dim, num_classes, hidden_layers=None, act_fun=F.relu):
        """
        Initialize the Feedforward Neural Network
        :param input_dim: number of features
        :param num_classes: number of classes
        :param hidden_layers: list of hidden layers
        :param act_fun: activation function
        """
        super().__init__()
        if hidden_layers is None:
            hidden_layers = []
        self.layers = nn.ModuleList()
        self.af = act_fun

        if len(hidden_layers) == 0:
            self.layers = self.layers.append(nn.Linear(input_dim, num_classes))
        else:
            for layer_idx in range(len(hidden_layers)):
                if layer_idx == 0:  # first layer, from input to hidden
                    self.layers = self.layers.append(LinearReLu(input_dim, hidden_layers[layer_idx]))
                else:  # hidden layers, depending on the input
                    self.layers = self.layers.append(LinearReLu(hidden_layers[layer_idx - 1], hidden_layers[layer_idx]))
            self.layers = self.layers.append(nn.Linear(hidden_layers[-1], num_classes))  # final output layer

        for m in self.children():  # initialize weight
            if isinstance(m, LinearReLu) or isinstance(m, nn.Linear):
                nn.init.normal_(self.m.weight, mean=0.0, std=0.1)

    def forward(self, x):
        """
        Forward pass of the model
        :param x: input tensor
        :return: output tensor
        """
        for layer in self.layers[:-1]:
            x = layer(x)
        return self.layers[-1](x)


# Train Classification Model
def train(model, optimizer, dataloader_train, num_epochs):
    """
    Train the Feedforward Neural Network
    :param model: FFNN model
    :param optimizer: optimizer technique
    :param dataloader_train: dataloader for training
    :param num_epochs: number of epochs
    :return: loss_train, acc_train: list of loss and accuracy for each epoch
    """
    criterion = torch.nn.CrossEntropyLoss()
    loss_train, acc_train = [], []
    for epoch in range(num_epochs):
        model.train()
        total_acc_train, total_count_train, n_train_batches, total_loss_train = 0, 0, 0, 0
        # train each batch
        for batch, labels in tqdm(dataloader_train, desc=f"Epoch {epoch + 1}", leave=True):
            optimizer.zero_grad()
            # predict train sample
            logits = model(batch)
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


def get_accuracy(model, X_test, y_test):
    """
    Get the accuracy of the model
    :param model: FFNN model
    :param X_test: input test samples
    :param y_test: input test labels
    :return: accuracy: accuracy of the model
    """
    model.eval()
    logits = model(X_test)
    accuracy = (logits.argmax(1) == y_test).sum() / len(y_test)
    return accuracy.item()
