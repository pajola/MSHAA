import torch
import torch.nn.functional as F
import numpy as np
import os
import glob
from timeit import default_timer as timer
import pickle

import support
import FFNN
import DenseNet
import generative_VAE
import SpeechClassification


def train_classification_models(name_dataset, X_train, y_train, hp, seed):
    """
    Function to train the classification models
    :param name_dataset: name of the dataset
    :param X_train: Samples for training
    :param y_train: Labels for training
    :param hp: Set of hyperparameters:
        - num_layer: number of hidden layers
        - neurons: number of neurons in each hidden layer
        - input_dim: number of features
        - num_class: number of classes
        - act_fun: activation function
        - lr: learning rate
        - num_epochs: number of epochs
    :param seed: random seed
    :return: dictionary of trained models with keys: 'layers_neurons'
    """
    # Set GPU environment
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # Obtain the dataloader
    dataloader_train = support.get_dataloader(X_train, y_train, device)

    # Start Training Process
    models = {}

    if name_dataset == 'MNIST':
        # create the grid of hidden layers of FFNN
        param_grid = []
        for neurons in hp['neurons']:
            for i in range(1, hp['num_layer'] + 1):
                param_grid.append([neurons] * i)
        # Create & Train FFNN models
        for hidden_layers in param_grid:
            print("\n-------------------------------------\n")
            print("Training FFNN")
            print("Learning Rate: ", hp['lr'])
            print("Hidden Layers: ", hidden_layers)
            # Train Classification Models
            ffnn = FFNN.FeedforwardNetwork(
                input_dim=hp['input_dim'],
                num_classes=hp['num_class'],
                hidden_layers=hidden_layers,
                act_fun=hp['act_fun']
            ).to(device)
            optimizer = torch.optim.Adam(ffnn.parameters(), lr=hp['lr'], weight_decay=1e-4)
            start = timer()
            _, _ = FFNN.train(ffnn, optimizer, dataloader_train, hp['num_epochs'])
            end = timer()
            print(f"Training time in second: {end - start}")
            # Save Trained Models
            model_name = str(len(hidden_layers)) + '_' + str(hidden_layers[0])
            models[model_name] = ffnn
    elif name_dataset == 'CIFAR10':
        depth = 50
        growth_rate = 12
        for num_dense in hp['num_dense']:
            for neurons in hp['neurons']:
                print("\n-------------------------------------\n")
                print("Training DenseNet - depth", depth, "& growth rate", growth_rate)
                print("Learning Rate: ", hp['lr'])
                print("Number of Dense Blocks: ", num_dense)
                print("Neurons: ", neurons)
                # Train Classification Models
                densenet = DenseNet.DenseNet_MLP(
                    num_classes=hp['num_class'],
                    num_dense=num_dense,
                    linear_dim=neurons,
                    depth=50,
                    growth_rate=12
                ).to(device)
                optimizer = torch.optim.Adam(densenet.parameters(), lr=hp['lr'], weight_decay=1e-4)
                start = timer()
                _, _ = DenseNet.train(densenet, optimizer, dataloader_train, hp['num_epochs'])
                end = timer()
                print(f"Training time in second: {end - start}")
                # Save Trained Models
                model_name = str(num_dense) + '_' + str(neurons)
                models[model_name] = densenet
    return models


def train_audio_models(dataloader_train, hp, seed):
    """
    Function to train the audio classification models
    :param dataloader_train: dataloader for training set
    :param hp: hyperparameters for the models
    :param seed: random seed
    :return:
    """
    # Set GPU environment
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # Start Training Process
    models = {}

    for num_layer in hp['num_layer']:
        for neurons in hp['neurons']:
            print("\n-------------------------------------\n")
            print("Training CNN")
            print("Learning Rate: ", hp['lr'])
            print("Number of CNN Layers: ", num_layer)
            print("Neurons: ", neurons)
            # Train Classification Models CNN
            cnn = SpeechClassification.CNN(
                num_classes=hp['num_class'],
                num_layers=num_layer,
                neurons=neurons
            ).to(device)
            optimizer = torch.optim.Adam(cnn.parameters(), lr=hp['lr'], weight_decay=1e-4)
            start = timer()
            _, _ = SpeechClassification.train(cnn, optimizer, dataloader_train, hp['num_epochs'], device)
            end = timer()
            print(f"Training time in second: {end - start}")
            # Save Trained Models
            model_name = str(num_layer) + '_' + str(neurons)
            models[model_name] = cnn
    return models


def train_vae(name_dataset, X, y, hp, models, metric, func):
    """
    Function to train the Conditional Variational Autoencoder
    :param name_dataset: name of the dataset
    :param X: Training samples
    :param y: Training labels
    :param hp: Hyperparameters for the CVAE
        - input_dim: number of features
        - num_class: number of classes
        - ENC_LAYERS: Encoder layers
        - latent_dim: latent dimension
        - DEC_LAYERS: Decoder layers
        - act_func: activation function
        - last_layer_act_fun: activation function for the last layer
        - lr: learning rate
        - num_epochs: number of epochs of training
    :param models: dictionary of trained models
    :param metric: imported dictionary of metric cost of surrogate models
    :param func: cost function used to train the CVAE
    :return: Best CVAE (minimum loss); hyperparameters; losses (train loss, rec loss, kl loss, train cost)
    """
    # Set GPU environment
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    np.random.seed(2090302)
    torch.manual_seed(2090302)
    torch.cuda.manual_seed(2090302)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    # Prepare the variables
    best_vae = None
    max_cost = -1
    best_hp = None
    best_losses = None
    print("\n-------------------------------------\n")
    print("Training VAE ...")
    # Get the dataloader
    dataloader_train = support.get_dataloader(X, y, device)
    # Grid Search for best CVAE
    for layer_idx in range(len(hp['ENC_LAYERS'])):
        for latent_idx in range(len(hp['latent_dim'])):
            for lr_idx in range(len(hp['lr'])):
                print("\n-------------------------------------\n")
                print("Training VAE")
                print("Encoder Layers: ", hp['ENC_LAYERS'][layer_idx])
                print("Latent Dimension: ", hp['latent_dim'][latent_idx])
                print("Learning Rate: ", hp['lr'][lr_idx])
                # Generate the CVAE model
                if name_dataset == 'MNIST':
                    vae = generative_VAE.CVAE(
                        hp['input_dim'],
                        hp['num_class'],
                        hp['ENC_LAYERS'][layer_idx],
                        hp['latent_dim'][latent_idx],
                        hp['DEC_LAYERS'][layer_idx],
                        hp['act_func'],
                        hp['last_layer_act_fun']
                    ).to(device)
                elif name_dataset == 'CIFAR10':
                    vae = generative_VAE.ConvCVAE(
                        hp['input_dim'],
                        hp['num_class'],
                        hp['ENC_LAYERS'][layer_idx],
                        hp['latent_dim'][latent_idx],
                        hp['DEC_LAYERS'][layer_idx],
                        hp['act_func'],
                        hp['last_layer_act_fun']
                    ).to(device)
                elif name_dataset == 'SpeechCommands':
                    vae = generative_VAE.AudioCVAE(
                        hp['input_dim'],
                        hp['num_class'],
                        hp['ENC_LAYERS'][layer_idx],
                        hp['latent_dim'][latent_idx],
                        hp['DEC_LAYERS'][layer_idx],
                        hp['act_func'],
                        hp['last_layer_act_fun']
                    ).to(device)

                optimizer = torch.optim.Adam(vae.parameters(), lr=hp['lr'][lr_idx])
                # Train the CVAE
                print("\n")
                start = timer()
                loss_train, loss_train_rec, loss_train_kl, train_cost = generative_VAE.train_vae(
                    vae, optimizer, dataloader_train, models, metric, func, hp['num_epochs']
                )
                end = timer()
                print(f"Training time in second: {end - start}")
                # Select the CVAE with the greatest cost
                if train_cost[-1] > max_cost:
                    max_cost = train_cost[-1]
                    best_vae = vae
                    best_hp = {'input_dim': hp['input_dim'],
                               'num_class': hp['num_class'],
                               'ENC_LAYERS': hp['ENC_LAYERS'][layer_idx],
                               'DEC_LAYERS': hp['DEC_LAYERS'][layer_idx],
                               'latent_dim': hp['latent_dim'][latent_idx],
                               'act_func': hp['act_func'],
                               'last_layer_act_fun': hp['last_layer_act_fun']
                               }
                    best_losses = {'loss_train': loss_train,
                                   'loss_train_rec': loss_train_rec,
                                   'loss_train_kl': loss_train_kl,
                                   'train_cost': train_cost}
    return best_vae, best_hp, best_losses


def save_models(model, train_type):
    """
    Function to save the trained models
    :param model: pair of 'model_name': model[0] and  trained model: model[1]
    :param train_type: portion of path to save the model
    :return: None
    """
    print("\nSaving Models ...\n")
    path = './trained_models/' + train_type + '/' + str(model[0] + '.pt')
    torch.save(model[1].state_dict(), path)


def load_models(path, name_dataset):
    """
    Function to load the trained FFNN models
    :param path: paths to the trained models
    :param name_dataset: name of the dataset
    :return: dictionary of trained models with keys: 'layers_neurons'
    """
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    models = {}
    model = None
    # Set the input dimension and number of classes based on the dataset
    if name_dataset == 'MNIST':
        input_dim = 784
        num_class = 10
        # retrieve the trained models based on their names
        saved_ffnn_model_names = [os.path.basename(x) for x in glob.glob(path + '/*.pt')]
        for saved_ffnn in saved_ffnn_model_names:
            ffnn_name = saved_ffnn.replace('.pt', '')
            # based on the model name, retrieve the number of hidden layers
            num_neurons = ffnn_name.split('_')[1]
            hidden_layers = [int(num_neurons)] * int(ffnn_name.split('_')[0])
            # Generate the FFNN model
            model = FFNN.FeedforwardNetwork(
                input_dim,
                num_class,
                hidden_layers,
                F.relu
            ).to(device)
            # Load the weights of the FFNN model
            model.load_state_dict(torch.load(path + saved_ffnn))
            models[ffnn_name] = model
    elif name_dataset == 'CIFAR10':
        num_class = 10
        # retrieve the trained models based on their names
        saved_densenet_model_names = [os.path.basename(x) for x in glob.glob(path + '/*.pt')]
        for saved_densenet in saved_densenet_model_names:
            densenet_name = saved_densenet.replace('.pt', '')
            # based on the model name, retrieve the depth and growth rate
            neurons = int(densenet_name.split('_')[1])
            num_dense = int(densenet_name.split('_')[0])
            # Generate the DenseNet model
            model = DenseNet.DenseNet_MLP(
                num_class,
                num_dense,
                neurons
            ).to(device)
            # Load the weights of the DenseNet model
            model.load_state_dict(torch.load(path + saved_densenet))
            models[densenet_name] = model
    elif name_dataset == 'SpeechCommands':
        num_class = 35
        # retrieve the trained models based on their names
        saved_model_names = [os.path.basename(x) for x in glob.glob(path + '/*.pt')]
        # retrieve the trained CNN named with 0, and GRU models named with 1
        for saved_model in saved_model_names:
            model_name = saved_model.replace('.pt', '')
            # Load CNN model
            num_layer = int(model_name.split('_')[0])
            neurons = int(model_name.split('_')[1])
            model = SpeechClassification.CNN(
                num_class,
                num_layer,
                neurons
            ).to(device)
            # Load the weights of the CNN model
            model.load_state_dict(torch.load(path + saved_model))
            models[model_name] = model
    return models


def load_vae(path_vae, path_hp, name_dataset):
    """
    Function to load the trained CVAE model
    :param path_vae: location of the trained CVAE model
    :param path_hp: location of the hyperparameters of the trained CVAE model
    :param name_dataset: name of the dataset
    :return: retrieved VAE model
    """
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    print("\nLoad VAE ...\n")
    # Provide the architecture of the VAE
    with open(path_hp, 'rb') as f:
        hp = pickle.load(f)
    # Generate the VAE model
    vae = None
    if name_dataset == 'MNIST':
        vae = generative_VAE.CVAE(hp['input_dim'],
                                  hp['num_class'],
                                  hp['ENC_LAYERS'],
                                  hp['latent_dim'],
                                  hp['DEC_LAYERS'],
                                  hp['act_func'],
                                  hp['last_layer_act_fun']
                                  ).to(device)
    elif name_dataset == 'CIFAR10':
        vae = generative_VAE.ConvCVAE(hp['input_dim'],
                                      hp['num_class'],
                                      hp['ENC_LAYERS'],
                                      hp['latent_dim'],
                                      hp['DEC_LAYERS'],
                                      hp['act_func'],
                                      hp['last_layer_act_fun']
                                      ).to(device)
    elif name_dataset == 'SpeechCommands':
        vae = generative_VAE.AudioCVAE(hp['input_dim'],
                                       hp['num_class'],
                                       hp['ENC_LAYERS'],
                                       hp['latent_dim'],
                                       hp['DEC_LAYERS'],
                                       hp['act_func'],
                                       hp['last_layer_act_fun']
                                       ).to(device)
    # Load the weights of the VAE model
    vae.load_state_dict(torch.load(path_vae))
    return vae
