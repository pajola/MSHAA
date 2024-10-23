import numpy as np
import torch
import pickle
import os
import torchvision.datasets as datasets
import torchaudio.datasets as audio_datasets
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, TensorDataset
from torchvision import transforms
from sklearn.model_selection import train_test_split


def MNIST_preprocess(img):
    """
    Preprocess MNIST image
    :param img: input image
    :return: img: preprocessed image
    """
    Trns = transforms.ToTensor()
    img = Trns(img)
    C, H, W = img.shape
    img = img.reshape((C * H * W,))
    return img


def CIFAR10_preprocess(img):
    """
    Preprocess CIFAR10 image. Adapted from: 'https://github.com/andreasveit/densenet-pytorch/blob/master/train.py'
    :param img: input image
    :return: img: preprocessed image
    """
    normalize = transforms.Normalize(
        mean=[x / 255.0 for x in [125.3, 123.0, 113.9]],
        std=[x / 255.0 for x in [63.0, 62.1, 66.7]]
    )
    Trns = transforms.ToTensor()
    img = Trns(img)
    img = normalize(img)
    return img


def get_data_from_loader(dataloader):
    """"
    Get the data from the dataloader
    :param dataloader: input dataloader
    :return: X, y: numpy arrays, X: features, y: labels
    """
    X = []
    y = []
    for idx, (sample, label) in enumerate(dataloader):
        for i in range(len(sample)):
            X.append(torch.Tensor.numpy(sample)[i])
            y.append(torch.Tensor.numpy(label)[i])
    X = np.array(X)
    y = np.array(y)
    return X, y


def get_original_dataset(dataset):
    """
    Get the original dataset depending on the task
    :param dataset: name of dataset used
    :return: X, y: numpy arrays, X: features, y: labels
    """
    np.random.seed(2090302)
    torch.manual_seed(2090302)
    X, y = None, None
    if dataset == 'MNIST':
        # Download MNIST dataset, preprocess and normalize the data.
        print("\n Getting MNIST Dataset ... \n")
        MNIST = datasets.MNIST(root='./data', train=True, download=True, transform=MNIST_preprocess)
        dataloader = DataLoader(MNIST, shuffle=False)
        X, y = get_data_from_loader(dataloader)
    elif dataset == 'CIFAR10':
        # Download CIFAR10 dataset, preprocess and normalize the data.
        print("\n Getting CIFAR10 Dataset ... \n")
        CIFAR10 = datasets.CIFAR10(root='./data', train=True, download=True, transform=CIFAR10_preprocess)
        dataloader = DataLoader(CIFAR10, shuffle=False)
        X, y = get_data_from_loader(dataloader)
    return X, y


def pad_sequence(batch):
    """
    Pad the sequence of tensors with zeros, adapted from:
    https://pytorch.org/tutorials/intermediate/speech_command_recognition_with_torchaudio.html
    :param batch: input batch
    :return: padded batch
    """
    batch = [item.t() for item in batch]
    batch = torch.nn.utils.rnn.pad_sequence(batch, batch_first=True, padding_value=0.)
    return batch.permute(0, 2, 1)


def collate_fn(batch):
    """
    Preprocess the batch of tensors, adapted from:
    https://pytorch.org/tutorials/intermediate/speech_command_recognition_with_torchaudio.html
    :param batch: input batch
    :return: tensors, targets
    """
    # A data tuple has the form:
    # waveform, sample_rate, label, speaker_id, utterance_number
    labels = ['backward', 'bed', 'bird', 'cat', 'dog', 'down', 'eight', 'five', 'follow', 'forward', 'four', 'go',
              'happy', 'house', 'learn', 'left', 'marvin', 'nine', 'no', 'off', 'on', 'one', 'right', 'seven', 'sheila',
              'six', 'stop', 'three', 'tree', 'two', 'up', 'visual', 'wow', 'yes', 'zero']

    tensors, targets = [], []
    # Gather in lists, and encode labels as indices
    for waveform, _, label, *_ in batch:
        tensors += [waveform]
        targets += [torch.tensor(labels.index(label))]

    # Group the list of tensors into a batched tensor
    tensors = pad_sequence(tensors)
    targets = torch.stack(targets)

    return tensors, targets


def get_audio_dataset(subset):
    """
    Get the audio dataset, adapted from:
    https://pytorch.org/tutorials/intermediate/speech_command_recognition_with_torchaudio.html
    :param subset: name of the subset
    :return: data_loader: dataloader for the audio dataset
    """
    class SubsetSC(audio_datasets.SPEECHCOMMANDS):
        def __init__(self, subset: str = None):
            super().__init__("./data", download=True)

            def load_list(filename):
                filepath = os.path.join(self._path, filename)
                with open(filepath) as fileobj:
                    return [os.path.normpath(os.path.join(self._path, line.strip())) for line in fileobj]

            if subset == "validation":
                self._walker = load_list("validation_list.txt")
            elif subset == "testing":
                self._walker = load_list("testing_list.txt")
            elif subset == "training":
                excludes = load_list("validation_list.txt") + load_list("testing_list.txt")
                excludes = set(excludes)
                self._walker = [w for w in self._walker if w not in excludes]

    dataset = SubsetSC(subset)
    print(subset, len(dataset))
    data_loader = torch.utils.data.DataLoader(
        dataset,
        shuffle=True,
        collate_fn=collate_fn,
        batch_size=256
    )
    return data_loader


def split_dataset(X, y):
    """
    Split the dataset into: Train: 50%; Validation: 40%; Test: 10%
    :param X: Samples
    :param y: Labels
    :return: X_tr, y_tr, X_tr2, y_tr2, X_val, y_val, X_test, y_test
    """
    seed = 2090302
    # Split the dataset into two sets: 50% and 50%
    X_tr, X_tv, y_tr, y_tv = train_test_split(X, y, test_size=0.5, random_state=seed, stratify=y)
    # Split the 50% set into two parts of 80% and 20%
    X_val, X_test, y_val, y_test = train_test_split(X_tv, y_tv, test_size=0.2, random_state=seed, stratify=y_tv)
    return X_tr, y_tr, X_val, y_val, X_test, y_test


def save_dataset(X, y, portion_name, dataset_name):
    """
    Save the dataset
    :param X: Samples
    :param y: Labels
    :param portion_name: Name of the portion of dataset
    :param dataset_name: Name of the dataset
    :return: None
    """
    print("\n Saving " + portion_name + " ... \n")
    path = './datasets/' + dataset_name + '/' + portion_name + '.npy'
    with open(path, 'wb') as f:
        np.save(f, X)
        np.save(f, y)
    print("Saving Complete\n")


def get_dataloader(X, y, device, batch_size=64):
    """
    Get the Pytorch Dataloader
    :param X: numpy list of samples
    :param y: numpy list of labels
    :param device: cuda or cpu
    :param batch_size: batch dimension
    :return: Pytorch Dataloader
    """
    X_tensor = torch.from_numpy(X).type(torch.FloatTensor).to(device)
    y_tensor = torch.from_numpy(y).to(device)
    data_set = TensorDataset(X_tensor, y_tensor)
    dataloader = DataLoader(data_set, batch_size=batch_size)
    return dataloader


def select_from_dict(dt, Lambda):
    """
    Select the activations that are in the Lambda set
    :param dt: dictionary of keys: 'layer_neuron' or 'growthRate_depth'
    :param Lambda: dictionary of selected measures m1 and m2
    :return: dict_refined: dictionary of selected values
    """
    dict_refined = {}
    for key, value in dt.items():
        layer = int(key.split('_')[0])
        neuron = int(key.split('_')[1])
        if layer <= Lambda['m2'] and neuron in Lambda['m1']:
            dict_refined[key] = value
    return dict_refined


def plot_generated_digits(generated_samples, generated_labels, num_class, path, vae_name, surrogate_type, name_dataset):
    """
    Plot the generated digits
    :param generated_samples: generated samples
    :param generated_labels: generated labels
    :param num_class: number of classes
    :param path: path to save the figure
    :param vae_name: name of the VAE
    :param surrogate_type: clean or surrogate
    :param name_dataset: name of the dataset MNIST or CIFAR10
    :return: None
    """
    X_class = []
    labels = np.unique(generated_labels)
    # Build a figure 10x10
    fig = plt.figure(figsize=(30, 30))
    columns = 10
    rows = num_class
    CIFAR10_dict = {0: 'airplane', 1: 'automobile', 2: 'bird', 3: 'cat', 4: 'deer', 5: 'dog', 6: 'frog',
                    7: 'horse', 8: 'ship', 9: 'truck'}
    for lab in labels:
        class_idx = [i for i, label in enumerate(generated_labels) if label == lab]
        X_class.append(generated_samples[class_idx])
    for i in range(num_class):  # each row represent one of the 10 classes
        for j in range(10):  # 10 samples per class
            if i < len(X_class) and j < len(X_class[i]):  # if there are samples in the class
                # Populate the figure
                if name_dataset == 'MNIST':
                    title = "label:" + str(labels[i])
                    img = X_class[i][j].reshape(28, 28)
                else:
                    title = "label:" + CIFAR10_dict[int(labels[i])]
                    img = np.transpose(X_class[i][j], (1, 2, 0))
                ax = fig.add_subplot(rows, columns, i * 10 + j + 1)
                ax.set_title(title, pad=0)
                fig.tight_layout()
                plt.imshow(img, cmap='gray')
                ax.set_xticks([])
                ax.set_yticks([])
    # Save the figure
    plt.savefig(path + vae_name + '_' + surrogate_type + '_digits.png')
    plt.close(fig)


def load_activations(path):
    """
    Load the activations
    :param path: path to the activations
    :return: loaded_cost: loaded activations
    """
    with open(path, 'rb') as f:
        loaded_cost = pickle.load(f)
    return loaded_cost
