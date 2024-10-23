import torch
import numpy as np
import torch.nn.functional as F

from support import load_activations, select_from_dict, get_audio_dataset
from model_handler import load_models


def classification_training_setup(name_dataset, lr):
    """
    Load the Train Dataset and Define Hyperparameters for the models
    :param name_dataset: name of the dataset
    :param lr: learning rate
    :return: X_tr, y_tr, hp: Train Dataset Points, Train Dataset Labels and Hyperparameters
    """
    # Load the Train Dataset
    path = './datasets/' + name_dataset + '/train_dataset.npy'
    print("\nLoading Train Dataset ...\n")
    with open(path, 'rb') as f:
        X_tr = np.load(f)
        y_tr = np.load(f)
    # Define hyperparameters for the models
    hp = None
    if name_dataset == 'MNIST':
        # hyperparameters for FFNN
        hp = {
            'num_layer': 10,
            'neurons': [32, 64, 128],
            'input_dim': X_tr.shape[1],
            'num_class': len(np.unique(y_tr)),
            'act_fun': F.relu,
            'lr': lr,
            'num_epochs': 10
        }
    elif name_dataset == 'CIFAR10':
        # hyperparameters for DenseNet
        hp = {
            'num_dense': [2, 3, 4, 5, 6, 7, 8],
            'neurons': [128, 256, 512],
            'input_dim': X_tr.shape[1],
            'num_class': len(np.unique(y_tr)),
            'lr': lr,
            'num_epochs': 15
        }
    return X_tr, y_tr, hp


def audio_training_setup(lr):
    """
    Load the Train Dataset and Define Hyperparameters for the models
    :param lr: learning rate
    :return: dataloader_hp: Train Dataset Points, Train Dataset Labels and Hyperparameters
    """
    # Load the Train Dataset
    dataloader = get_audio_dataset('training')
    labels = ['backward', 'bed', 'bird', 'cat', 'dog', 'down', 'eight', 'five', 'follow', 'forward', 'four', 'go',
              'happy', 'house', 'learn', 'left', 'marvin', 'nine', 'no', 'off', 'on', 'one', 'right', 'seven', 'sheila',
              'six', 'stop', 'three', 'tree', 'two', 'up', 'visual', 'wow', 'yes', 'zero']
    # hyperparameters for CNN
    hp = {
        'num_class': len(np.unique(labels)),
        'num_layer': [2, 4, 6, 8],
        'neurons': [128, 256],
        'lr': lr,
        'num_epochs': 10
    }
    return dataloader, hp


def vae_training_setup(name_dataset, lr, Lambda, surrogate_type, VAE_metric):
    """
    Load the Validation Dataset (which will be used for training the CVAE), Used Models and respective CHijack Metric
    :param name_dataset: name of the dataset
    :param lr: learning rate
    :param Lambda: dictionary of selected layers and neurons
    :param surrogate_type: clean or surrogate
    :param VAE_metric: Hijack Metric used to train the CVAE (l0, energy, latency, genError)
    :return: X_val, y_val, hp, used_models, metric
    """
    # Load the Validation Dataset (which will be used for training the CVAE)
    path_dataset = './datasets/' + name_dataset + '/validation_clean_dataset.npy'
    print("\nLoading Validation Dataset ...\n")
    with open(path_dataset, 'rb') as f:
        X_val = np.load(f)
        y_val = np.load(f)
    # Load the Used Models & Used Hijack Metric
    print("\nLoading Models & Hijack Metric:", VAE_metric, "...\n")
    if surrogate_type == 'clean':
        path_models = './trained_models/' + name_dataset + '/' + str(lr) + '/MLaaS/'
        metric = load_activations('./hijack_metrics/' + name_dataset + '/' + 'mlaas_' + str(lr) + '_' + VAE_metric + '.pkl')
    else:
        path_models = './trained_models/' + name_dataset + '/' + str(lr) + '/Surrogate/'
        metric = load_activations('./hijack_metrics/' + name_dataset + '/' + 'surrogate_' + str(lr) + '_' + VAE_metric + '.pkl')
    # Load the models and select the ones that are in the Lambda set
    used_models = load_models(path_models, name_dataset)
    used_models = select_from_dict(used_models, Lambda)
    metric = select_from_dict(metric, Lambda)
    # Define the hyperparameters for the CVAE
    hp = None
    if name_dataset == 'MNIST':
        hp = {
            'input_dim': X_val.shape[1],
            'num_class': 10,
            'ENC_LAYERS': [[512, 64], [128, 128], [1024, 32]],
            'latent_dim': [2],
            'DEC_LAYERS': [[64, 512], [128, 128], [32, 1024]],
            'act_func': F.relu,
            'last_layer_act_fun': F.sigmoid,
            'lr': [0.001, 0.005],
            'num_epochs': 10
        }
    elif name_dataset == 'CIFAR10':
        hp = {
            'input_dim': X_val.shape[1],
            'num_class': 10,
            'ENC_LAYERS': [[32, 64, 4096, 256]],
            'latent_dim': [128],
            'DEC_LAYERS': [[256, 4096, 64, 32]],
            'act_func': torch.nn.ReLU(),
            'last_layer_act_fun': F.sigmoid,
            'lr': [0.001, 0.005],
            'num_epochs': 10
        }
    elif name_dataset == 'SpeechCommands':
        hp = {
            'input_dim': 16000,
            'num_class': 35,
            'ENC_LAYERS': [[32, 16, 8]],  # output dimensions of the Conv1d layers
            'latent_dim': [8, 16],
            'DEC_LAYERS': [[8, 16, 32]],  # input dimensions of the TransposeConv1d layers
            'act_func': torch.nn.ReLU(),
            'last_layer_act_fun': F.sigmoid,
            'lr': [0.001, 0.005],
            'num_epochs': 10
        }
    return X_val, y_val, hp, used_models, metric
