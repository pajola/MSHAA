import torch
import numpy as np

from torch import nn
from tqdm import tqdm


class CVAE(nn.Module):
    """
    Conditional Variational Autoencoder
    """
    def __init__(self, input_dim, class_dim, ENC_LAYERS, latent_dim, DEC_LAYERS, act_fun, last_layer_act_fun):
        """
        Initialize the Conditional Variational Autoencoder
        :param input_dim: dimension of the input
        :param class_dim: number of classes
        :param ENC_LAYERS: list of encoder layers dimensions
        :param latent_dim: dimension of the latent space
        :param DEC_LAYERS: list of decoder layers dimensions
        :param act_fun: activation function
        :param last_layer_act_fun: activation function for the last layer
        """
        super().__init__()
        self.E_layers = nn.ModuleList()
        self.D_layers = nn.ModuleList()
        self.latent_dim = latent_dim
        self.af = act_fun
        self.last_af = last_layer_act_fun

        # encoder
        for layer_idx in range(len(ENC_LAYERS)):
            if layer_idx == 0:
                self.E_layers = self.E_layers.append(nn.Linear(input_dim + class_dim, ENC_LAYERS[layer_idx]))
            else:
                self.E_layers = self.E_layers.append(nn.Linear(ENC_LAYERS[layer_idx - 1], ENC_LAYERS[layer_idx]))
        self.linear_mean = nn.Linear(ENC_LAYERS[-1], latent_dim)
        self.linear_var = nn.Linear(ENC_LAYERS[-1], latent_dim)

        # decoder
        for layer_idx in range(len(DEC_LAYERS)):
            if layer_idx == 0:
                self.D_layers = self.D_layers.append(nn.Linear(latent_dim + class_dim, DEC_LAYERS[layer_idx]))
            else:
                self.D_layers = self.D_layers.append(nn.Linear(DEC_LAYERS[layer_idx - 1], DEC_LAYERS[layer_idx]))
        self.final_fc = nn.Linear(DEC_LAYERS[-1], input_dim)

    def conditioning_label(self, x, y):
        """
        Concatenate the condition to the input
        :param x: input image
        :param y: image label
        :return: Concatenated tensor
        """
        return torch.cat((x, y), dim=1)

    def sampling(self, z_mean, z_var):
        """
        Sample from the latent space
        :param z_mean: tensor of the mean
        :param z_var: tensor of the variance
        :return: sampled tensor
        """
        device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        epsilon = torch.randn_like(z_var).to(device)
        z = z_mean + torch.exp(z_var / 2) * epsilon
        return z

    def encode(self, x):
        """
        Encode the input
        :param x: input tensor
        :return: encoded tensor
        """
        for layer in self.E_layers:
            x = self.af(layer(x))
        return x

    def decode(self, z):
        """
        Decode the input
        :param z: input tensor
        :return: decoded tensor
        """
        for layer in self.D_layers:
            z = self.af(layer(z))
        out = self.final_fc(z)
        return self.last_af(out)

    def forward(self, x, y):
        """
        Forward pass of the model
        :param x: input image
        :param y: label of the image
        :return: tensor of the output
        """
        x = self.conditioning_label(x, y)
        x = self.encode(x)
        self.z_mean = self.linear_mean(x)
        self.z_var = self.linear_var(x)
        self.z = self.sampling(self.z_mean, self.z_var)
        z = self.conditioning_label(self.z, y)
        return self.decode(z)


class Flatten(nn.Module):
    """
    Flatten the tensor
    """
    def forward(self, x):
        return x.view(x.size(0), -1)


class UnFlatten(nn.Module):
    """
    Unflatten the tensor
    """
    def __init__(self, size):
        """
        Initialize the Unflatten layer
        :param size: size of the tensor
        """
        super().__init__()
        self.size = size

    def forward(self, x):
        return x.view(x.size(0), self.size, 8, 8)


class UnFlatten1d(nn.Module):
    """
    Unflatten the tensor
    """
    def __init__(self, channels, size):
        """
        Initialize the Unflatten layer
        :param channels: number of channels
        :param size: size of the tensor
        """
        super().__init__()
        self.channels = channels
        self.size = size

    def forward(self, x):
        return x.view(x.size(0), self.channels, self.size)


class ConvCVAE(nn.Module):
    """
    Convolutional Conditional Variational Autoencoder
    """
    def __init__(self, input_dim, class_dim, ENC_LAYERS, latent_dim, DEC_LAYERS, act_fun, last_layer_act_fun):
        """
        Initialize the Convolutional Conditional Variational Autoencoder
        :param input_dim: dimension of the input
        :param class_dim: number of classes
        :param ENC_LAYERS: list of encoder layers dimensions
        :param latent_dim: dimension of the latent space
        :param DEC_LAYERS: list of decoder layers dimensions
        :param act_fun: activation function
        :param last_layer_act_fun: activation function for the last layer
        """
        super().__init__()
        self.E_layers = nn.ModuleList()
        self.D_layers = nn.ModuleList()
        self.latent_dim = latent_dim
        self.af = act_fun
        self.last_af = last_layer_act_fun

        # encoder
        for layer_idx in range(len(ENC_LAYERS) - 2):
            if layer_idx == 0:
                self.E_layers = self.E_layers.append(
                    nn.Conv2d(
                        4,  # 3 channels for the image and 1 channels for the label
                        ENC_LAYERS[layer_idx],
                        kernel_size=6,
                        stride=2,
                        padding=1,
                        bias=False
                    )
                )
                self.E_layers = self.E_layers.append(nn.BatchNorm2d(ENC_LAYERS[layer_idx]))
                self.E_layers = self.E_layers.append(self.af)
            else:
                self.E_layers = self.E_layers.append(
                    nn.Conv2d(
                        ENC_LAYERS[layer_idx - 1],
                        ENC_LAYERS[layer_idx],
                        kernel_size=3,
                        stride=2,
                        padding=1,
                        bias=False
                    )
                )
                self.E_layers = self.E_layers.append(nn.BatchNorm2d(ENC_LAYERS[layer_idx]))
                self.E_layers = self.E_layers.append(self.af)
        self.E_layers = self.E_layers.append(Flatten())
        self.E_layers = self.E_layers.append(nn.Linear(ENC_LAYERS[-2], ENC_LAYERS[-1]))
        self.E_layers = self.E_layers.append(self.af)

        self.linear_mean = nn.Linear(ENC_LAYERS[-1], latent_dim)
        self.linear_var = nn.Linear(ENC_LAYERS[-1], latent_dim)

        # decoder
        self.D_layers = self.D_layers.append(nn.Linear(latent_dim + class_dim, DEC_LAYERS[0]))
        self.D_layers = self.D_layers.append(nn.Linear(DEC_LAYERS[0], DEC_LAYERS[1]))
        self.D_layers = self.D_layers.append(self.af)
        self.D_layers = self.D_layers.append(UnFlatten(size=DEC_LAYERS[2]))
        for layer_idx in range(len(DEC_LAYERS) - 2):
            if layer_idx == len(DEC_LAYERS) - 3:
                self.D_layers = self.D_layers.append(
                    nn.ConvTranspose2d(
                        DEC_LAYERS[layer_idx + 2],
                        3,
                        kernel_size=6,
                        stride=2,
                        padding=1,
                        bias=False
                    )
                )
            else:
                self.D_layers = self.D_layers.append(
                    nn.ConvTranspose2d(
                        DEC_LAYERS[layer_idx + 2],
                        DEC_LAYERS[layer_idx + 3],
                        kernel_size=3,
                        stride=2,
                        padding=1,
                        bias=False
                    )
                )
                self.D_layers = self.D_layers.append(nn.BatchNorm2d(DEC_LAYERS[layer_idx + 3]))
                self.D_layers = self.D_layers.append(self.af)


    def conditioning_label_input(self, x, y):
        """
        Concatenate the condition to the input
        :param x: input image
        :param y: image label
        :return: Concatenated tensor
        """
        device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        y = torch.argmax(y, dim=1).reshape((y.shape[0], 1, 1, 1))
        y = (torch.ones(x.shape).to(device) * y)[:, 0:1, :, :]
        return torch.cat((x, y), dim=1)

    def conditioning_label_latent(self, x, y):
        """
        Concatenate the condition to the latent space
        :param x: input image
        :param y: image label
        :return: Concatenated tensor
        """
        return torch.cat((x, y), dim=1)

    def sampling(self, z_mean, z_var):
        """
        Sample from the latent space
        :param z_mean: tensor of the mean
        :param z_var: tensor of the variance
        :return: sampled tensor
        """
        device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        epsilon = torch.randn_like(z_var).to(device)
        z = z_mean + torch.exp(z_var / 2) * epsilon
        return z

    def encode(self, x):
        """
        Encode the input
        :param x: input tensor
        :return: encoded tensor
        """
        for layer in self.E_layers:
            x = layer(x)
        return x

    def decode(self, z):
        """
        Decode the input
        :param z: input tensor
        :return: decoded tensor
        """
        for layer in self.D_layers:
            z = layer(z)
        return self.last_af(z)

    def forward(self, x, y):
        """
        Forward pass of the model
        :param x: input image
        :param y: label of the image
        :return: tensor of the output
        """
        x = self.conditioning_label_input(x, y)
        x = self.encode(x)
        self.z_mean = self.linear_mean(x)
        self.z_var = self.linear_var(x)
        self.z = self.sampling(self.z_mean, self.z_var)
        z = self.conditioning_label_latent(self.z, y)
        return self.decode(z)


class AudioCVAE(nn.Module):
    """
    Conditional Variational Autoencoder
    """
    def __init__(self, input_dim, class_dim, ENC_LAYERS, latent_dim, DEC_LAYERS, act_fun, last_layer_act_fun):
        """
        Initialize the Conditional Variational Autoencoder
        :param input_dim: dimension of the input
        :param class_dim: number of classes
        :param ENC_LAYERS: list of encoder layers dimensions
        :param latent_dim: dimension of the latent space
        :param DEC_LAYERS: list of decoder layers dimensions
        :param act_fun: activation function
        :param last_layer_act_fun: activation function for the last layer
        """
        super().__init__()
        self.E_layers = nn.ModuleList()
        self.D_layers = nn.ModuleList()
        self.latent_dim = latent_dim
        self.af = act_fun
        self.last_af = last_layer_act_fun
        self.flatten_size = 0

        # encoder
        for layer_idx in range(len(ENC_LAYERS)):
            if layer_idx == 0:
                self.E_layers = self.E_layers.append(
                    nn.Conv1d(
                        1,
                        ENC_LAYERS[layer_idx],
                        kernel_size=40,
                        stride=10,
                        padding=0,
                        bias=False
                    )
                )
                self.flatten_size = (input_dim - 40) // 10 + 1
                self.E_layers = self.E_layers.append(nn.BatchNorm1d(ENC_LAYERS[layer_idx]))
                self.E_layers = self.E_layers.append(self.af)
            else:
                self.E_layers = self.E_layers.append(
                    nn.Conv1d(
                        ENC_LAYERS[layer_idx - 1],
                        ENC_LAYERS[layer_idx],
                        kernel_size=16,
                        stride=6,
                        padding=0,
                        bias=False
                    )
                )
                self.flatten_size = (self.flatten_size - 16) // 6 + 1
                self.E_layers = self.E_layers.append(nn.BatchNorm1d(ENC_LAYERS[layer_idx]))
                self.E_layers = self.E_layers.append(self.af)
        self.E_layers = self.E_layers.append(Flatten())
        self.E_layers = self.E_layers.append(nn.Linear(self.flatten_size * ENC_LAYERS[-1], 64))
        self.E_layers = self.E_layers.append(self.af)

        self.linear_mean = nn.Linear(64, latent_dim)
        self.linear_var = nn.Linear(64, latent_dim)

        # decoder
        self.D_layers = self.D_layers.append(nn.Linear(latent_dim + class_dim, 64))
        self.D_layers = self.D_layers.append(nn.Linear(64, self.flatten_size * DEC_LAYERS[0]))
        self.D_layers = self.D_layers.append(self.af)
        self.D_layers = self.D_layers.append(UnFlatten1d(channels=DEC_LAYERS[0], size=self.flatten_size))
        for layer_idx in range(len(DEC_LAYERS)):
            out_padding = 3 if layer_idx == 0 else 0
            if layer_idx == len(DEC_LAYERS) - 1:
                self.D_layers = self.D_layers.append(
                    nn.ConvTranspose1d(
                        DEC_LAYERS[layer_idx],
                        1,
                        kernel_size=40,
                        stride=10,
                        padding=15,
                        bias=False
                    )
                )
            else:
                self.D_layers = self.D_layers.append(
                    nn.ConvTranspose1d(
                        DEC_LAYERS[layer_idx],
                        DEC_LAYERS[layer_idx + 1],
                        kernel_size=16,
                        stride=6,
                        padding=0,
                        bias=False,
                        output_padding=out_padding
                    )
                )
                self.D_layers = self.D_layers.append(nn.BatchNorm1d(DEC_LAYERS[layer_idx+1]))
                self.D_layers = self.D_layers.append(self.af)

    def conditioning_label_latent(self, x, y, dim):
        """
        Concatenate the condition to the latent space
        :param x: input image
        :param y: image label
        :param dim: dimension to concatenate
        :return: Concatenated tensor
        """
        return torch.cat((x, y), dim)

    def sampling(self, z_mean, z_var):
        """
        Sample from the latent space
        :param z_mean: tensor of the mean
        :param z_var: tensor of the variance
        :return: sampled tensor
        """
        device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        epsilon = torch.randn_like(z_var).to(device)
        z = z_mean + torch.exp(z_var / 2) * epsilon
        return z

    def encode(self, x):
        """
        Encode the input
        :param x: input tensor
        :return: encoded tensor
        """
        for layer in self.E_layers:
            x = layer(x)
        return x

    def decode(self, z):
        """
        Decode the input
        :param z: input tensor
        :return: decoded tensor
        """
        for layer in self.D_layers:
            z = layer(z)
        return self.last_af(z)

    def forward(self, x, y):
        """
        Forward pass of the model
        :param x: input image
        :param y: label of the image
        :return: tensor of the output
        """
        x = self.conditioning_label_latent(x, y.unsqueeze(1), 2)
        x = self.encode(x)
        self.z_mean = self.linear_mean(x)
        self.z_var = self.linear_var(x)
        self.z = self.sampling(self.z_mean, self.z_var)
        z = self.conditioning_label_latent(self.z, y, 1)
        return self.decode(z)


def one_hot(labels, num_classes):
    """
    Convert the labels to one-hot encoding
    :param labels: tensor of one-dimensional labels
    :param num_classes: number of classes
    :return: one-hot encoded tensor of labels
    """
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    targets = torch.zeros(labels.size(0), num_classes)
    for i, label in enumerate(labels):
        targets[i, label] = 1
    return targets.to(device)


def KL_LOSS(model):
    """
    Compute the Kullback-Libeler Divergence
    :param model: VAE model
    :return: computed loss
    """
    loss = -0.5 * torch.sum(1 + model.z_var - torch.square(model.z_mean) - torch.exp(model.z_var), dim=1)
    loss = torch.mean(loss)
    return loss


def compute_cost(model, trained_models, metric, labels, func):
    """
    Compute our cost function
    :param model: VAE model
    :param trained_models: dictionary of trained models - keys: 'layers_neurons', values: trained models
    :param metric: dictionary of activation norms - keys: 'layer_neurons', values: l0 activation norm or energy
    :param labels: list of labels
    :param func: function to compute the cost
    :return: computed cost
    """
    cost = 0
    # Find the maximum activation norm
    max_norm = max(metric.values())
    criterion = torch.nn.CrossEntropyLoss()
    class_counts = np.bincount(labels.cpu().detach().numpy())
    batch, _ = generate_samples(model, class_counts)
    # Compute the cost
    for name, trained_model in trained_models.items():
        trained_model.eval()
        with torch.no_grad():
            if isinstance(model, AudioCVAE):
                logits = trained_model(batch).squeeze()
            else:
                logits = trained_model(batch)
            loss = criterion(logits, labels) / len(labels)
            if func == 'linear':
                # linear formulation
                cost += ((max_norm - metric[name]) / (max_norm - min(metric.values()))) * loss
            if func == 'exponential':
                # exponential formulation
                cost += (np.exp(2 * ((max_norm - metric[name]) / (max_norm - min(metric.values())))) - 1) * loss
    return cost/len(trained_models)  # normalize the cost over the number of surrogate models


def train_vae(model, optimizer, dataloader, models, metric, func, epochs=5):
    """
    Train the Conditional Variational Autoencoder
    :param model: VAE model
    :param optimizer: optimizer
    :param dataloader: dataloader of the training dataset
    :param models: dictionary of trained models - keys: 'layers_neurons', values: trained models
    :param metric: dictionary of activation norms - keys: 'layer_neurons', values: l0 or energy or latency or genError
    :param func: function to compute the cost
    :param epochs: number of epochs
    :return: loss_train, loss_train_rec, loss_train_kl, train_cost: list of the losses per epoch during training
    """
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    loss_train_rec, loss_train_kl, train_cost, loss_train = [], [], [], []

    # choose the reconstruction loss function
    if isinstance(model, ConvCVAE) or isinstance(model, AudioCVAE):
        REC_LOSS = nn.MSELoss()
    else:
        REC_LOSS = nn.BCELoss()

    for epoch in range(epochs):
        model.train()
        n_train_batches, rec_train, kl_train, cost_train, total_loss_train = 0, 0, 0, 0, 0
        for batch, labels in tqdm(dataloader, desc=f"Epoch {epoch + 1}", leave=True):
            # reset the gradient for all parameters
            optimizer.zero_grad()
            # run the model
            batch.to(device)
            if isinstance(model, AudioCVAE):
                rec_sample = model(batch, one_hot(labels, 35))
            else:
                rec_sample = model(batch, one_hot(labels, 10))
            # compute the loss for this sample
            rec_loss = REC_LOSS(rec_sample, batch) / len(batch)
            kl_loss = KL_LOSS(model)
            cost = compute_cost(model, models, metric, labels, func)
            # compute the loss function
            total_loss = (1e-4 * kl_loss + rec_loss - cost) ** 2
            # accumulate the loss for this epoch
            total_loss_train += total_loss
            rec_train += rec_loss
            kl_train += kl_loss
            cost_train += cost
            # compute the gradients and update weights
            total_loss.backward()
            optimizer.step()
            n_train_batches += 1
        # compute losses for this epoch
        avg_loss_train = total_loss_train / n_train_batches
        avg_rec_loss_train = rec_train / n_train_batches
        avg_kl_loss_train = kl_train / n_train_batches
        avg_cost_train = cost_train / n_train_batches
        # store losses
        loss_train.append(avg_loss_train.item())
        loss_train_rec.append(avg_rec_loss_train.item())
        loss_train_kl.append(avg_kl_loss_train.item())
        train_cost.append(avg_cost_train)
        # print progress
        if (epoch+1) % 1 == 0:
            print(f"epoch: {epoch+1} -> Loss: {avg_loss_train:.8f}", end=' ----- ')
            print(f"Rec Loss: {avg_rec_loss_train:.8f}", end=' ----- ')
            print(f"Effective KL Loss: {avg_kl_loss_train:.8f}", end=' ----- ')
            print(f"Cost Function: {avg_cost_train:.8f}", end='\n')
    return loss_train, loss_train_rec, loss_train_kl, train_cost


def generate_samples(model, num_samples_per_label):
    """
    Generate samples from the Conditional Variational Autoencoder
    :param model: VAE model
    :param num_samples_per_label: list of number of samples to generate per label
    :return: generated_samples, generated_labels: tensors of generated samples and labels
    """
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    generated_samples = []
    generated_labels = []

    for i in range(len(num_samples_per_label)):
        # Obtain a list of labels
        labels = [i] * num_samples_per_label[i]
        # Convert the labels to one-hot encoding
        labels = torch.Tensor(labels).type(torch.int).to(device)
        if isinstance(model, AudioCVAE):
            labels_oh = one_hot(labels, 35)
        else:
            labels_oh = one_hot(labels, 10)
        # Sample a vector in the latent space
        z = torch.randn(len(labels_oh), model.latent_dim).to(device)
        # Concatenate the condition and the latent vector
        z = torch.cat((z, labels_oh), dim=1)
        # Pass through the decoder to generate the samples
        generated_samples.append(model.decode(z))
        generated_labels.append(labels)
    # Uniformize the tensors
    generated_labels = torch.cat(generated_labels)
    generated_samples = torch.cat(generated_samples)
    return generated_samples, generated_labels
