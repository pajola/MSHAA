import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from tqdm import tqdm


# Code taken and adapted from:
#   https://github.com/andreasveit/densenet-pytorch/blob/master/densenet.py
#   https://github.com/andreasveit/densenet-pytorch/blob/master/train.py


class BasicBlock(nn.Module):
    def __init__(self, in_planes, out_planes, dropRate=0.0):
        super(BasicBlock, self).__init__()
        self.bn1 = nn.BatchNorm2d(in_planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=1,
                               padding=1, bias=False)
        self.droprate = dropRate

    def forward(self, x):
        out = self.conv1(self.relu(self.bn1(x)))
        if self.droprate > 0:
            out = F.dropout(out, p=self.droprate, training=self.training)
        return torch.cat([x, out], 1)


class BottleneckBlock(nn.Module):
    def __init__(self, in_planes, out_planes, dropRate=0.0):
        super(BottleneckBlock, self).__init__()
        inter_planes = out_planes * 4
        self.bn1 = nn.BatchNorm2d(in_planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv2d(in_planes, inter_planes, kernel_size=1, stride=1,
                               padding=0, bias=False)
        self.bn2 = nn.BatchNorm2d(inter_planes)
        self.conv2 = nn.Conv2d(inter_planes, out_planes, kernel_size=3, stride=1,
                               padding=1, bias=False)
        self.droprate = dropRate

    def forward(self, x):
        out = self.conv1(self.relu(self.bn1(x)))
        if self.droprate > 0:
            out = F.dropout(out, p=self.droprate, inplace=False, training=self.training)
        out = self.conv2(self.relu(self.bn2(out)))
        if self.droprate > 0:
            out = F.dropout(out, p=self.droprate, inplace=False, training=self.training)
        return torch.cat([x, out], 1)


class TransitionBlock(nn.Module):
    def __init__(self, in_planes, out_planes, dense_idx, dropRate=0.0):
        super(TransitionBlock, self).__init__()
        self.bn1 = nn.BatchNorm2d(in_planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=1,
                               padding=0, bias=False)
        self.droprate = dropRate
        if dense_idx == 2:
            self.out_kernel = 4
        elif dense_idx == 3:
            self.out_kernel = 2
        else:
            self.out_kernel = 1

    def forward(self, x):
        out = self.conv1(self.relu(self.bn1(x)))
        if self.droprate > 0:
            out = F.dropout(out, p=self.droprate, inplace=False, training=self.training)
        return F.avg_pool2d(out, self.out_kernel)


class DenseBlock(nn.Module):
    def __init__(self, nb_layers, in_planes, growth_rate, block, dropRate=0.0):
        super(DenseBlock, self).__init__()
        self.layer = self._make_layer(block, in_planes, growth_rate, nb_layers, dropRate)

    def _make_layer(self, block, in_planes, growth_rate, nb_layers, dropRate):
        layers = []
        for i in range(nb_layers):
            layers.append(block(in_planes + i * growth_rate, growth_rate, dropRate))
        return nn.Sequential(*layers)

    def forward(self, x):
        return self.layer(x)


class DenseNet_MLP(nn.Module):
    """
    Modification of classical DenseNet model
    """
    def __init__(self, num_classes, num_dense, linear_dim, depth=50, growth_rate=12, reduction=0.5, bottleneck=True, dropRate=0.0):
        """
        Initialize the DenseNet_MLP model
        :param num_classes: number of classes
        :param num_dense: number of dense blocks
        :param linear_dim: dimension of the linear layer
        :param depth: depth of the model
        :param growth_rate: growth rate of the model
        :param reduction: reduction rate
        :param bottleneck: use bottleneck block
        :param dropRate: dropout rate
        """
        super().__init__()
        self.num_dense = num_dense
        in_planes = 2 * growth_rate
        n = (depth - 4) / 3
        if bottleneck is True:
            n = n/2
            block = BottleneckBlock
        else:
            block = BasicBlock
        n = int(n)
        # 1st conv before any dense block
        self.conv1 = nn.Conv2d(3, in_planes, kernel_size=3, stride=1, padding=1, bias=False)
        for dense_idx in range(1, num_dense):  # add dense blocks and transition blocks
            setattr(self, "block%d" % dense_idx, DenseBlock(n, in_planes, growth_rate, block, dropRate))
            in_planes = int(in_planes + n * growth_rate)
            if num_dense >= 4 and dense_idx < 3:  # add specific transition blocks if num_dense >= 4
                setattr(self, "trans%d" % dense_idx,
                        TransitionBlock(in_planes, int(math.floor(in_planes * reduction)), 3,
                                        dropRate=dropRate)
                        )
            else:  # add transition block
                setattr(self, "trans%d" % dense_idx,
                        TransitionBlock(in_planes, int(math.floor(in_planes * reduction)), num_dense,
                                        dropRate=dropRate)
                        )
            in_planes = int(math.floor(in_planes * reduction))
        # add last dense block without transition
        setattr(self, "block%d" % num_dense, DenseBlock(n, in_planes, growth_rate, block, dropRate))
        in_planes = int(in_planes + n * growth_rate)
        # global average pooling and classifier
        self.bn1 = nn.BatchNorm2d(in_planes)
        self.relu = nn.ReLU(inplace=True)
        # linear layer
        self.linear = nn.Linear(in_planes, linear_dim)
        self.fc = nn.Linear(linear_dim, num_classes)
        self.in_planes = in_planes

        for m in self.modules():  # initialize weights
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def forward(self, x):
        """
        Forward pass of the model
        """
        out = self.conv1(x)
        for dense_idx in range(1, self.num_dense):
            out = getattr(self, "trans%d" % dense_idx)(getattr(self, "block%d" % dense_idx)(out))
        out = getattr(self, "block%d" % self.num_dense)(out)
        out = self.relu(self.bn1(out))
        out = F.avg_pool2d(out, 8)
        out = out.view(-1, self.in_planes)
        out = self.linear(out)
        return self.fc(out)


def train(model, optimizer, dataloader_train, num_epochs):
    """
    Train the DenseNet
    :param model: model
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
            total_loss_train += loss.sum().item()
            # backpropagation
            loss.backward()
            optimizer.step()
            # compute accuracy of train batch
            total_acc_train += (logits.argmax(1) == labels).sum().item()
            total_count_train += labels.size(0)
            n_train_batches += 1
        # get train values for the epoch
        avg_loss_train = total_loss_train / len(dataloader_train.dataset)
        loss_train.append(avg_loss_train)
        accuracy_train = total_acc_train / total_count_train
        acc_train.append(accuracy_train)
        # Print progress
        if (epoch + 1) % 5 == 0:
            print(f"Epoch: {epoch + 1} -> Accuracy: {100 * accuracy_train:.2f}%, Loss: {avg_loss_train:.8f}")
    return loss_train, acc_train
