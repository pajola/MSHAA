import torch
import numpy as np
import pickle
import time
from collections import defaultdict

from support import get_dataloader, get_audio_dataset
from model_handler import load_models
from ASICSim import analyse_data_energy, remove_hooks


def get_hijack_metric(name_dataset, lr):
    """
    Compute the cost of the models in terms of l0 activation norm and energy
    :param name_dataset: name of the dataset
    :param lr: learning rate
    :return: None
    """
    # Paths for loading the dataset and the models
    path_surr_models = './trained_models/' + name_dataset + '/' + str(lr) + '/Surrogate/'
    path_clean_models = './trained_models/' + name_dataset + '/' + str(lr) + '/MLaaS/'
    # Paths for saving the results
    path_results_mlaas = './hijack_metrics/' + name_dataset + '/mlaas_' + str(lr) + '_'
    path_results_surrogate = './hijack_metrics/' + name_dataset + '/surrogate_' + str(lr) + '_'

    print("\nLoading Validation Dataset ...\n")
    path_dataset = './datasets/' + name_dataset + '/validation_clean_dataset.npy'
    with open(path_dataset, 'rb') as f:
        X_val = np.load(f)
        y_val = np.load(f)
    # get dataloader for the validation dataset
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    val_dataloader = get_dataloader(X_val, y_val, device, batch_size=256)

    # Load the clean models
    clean_models = load_models(path_clean_models, name_dataset)
    # Load the surrogate models
    surrogate_models = load_models(path_surr_models, name_dataset)

    # Compute the hijack metrics for each clean model
    compute_hm(
        clean_models,
        val_dataloader,
        path_results_mlaas
    )
    # Compute the hijack metrics for each surrogate model
    compute_hm(
        surrogate_models,
        val_dataloader,
        path_results_surrogate
    )


def compute_hm(model_dict, val_dataloader, path_results):
    """
    Compute the activation norm and energy for the hijack models
    :param model_dict: dictionary of models - keys: model_name, values: models
    :param val_dataloader: dataloader of the validation dataset
    :param path_results: path to save the results
    :return: None
    """
    l_0 = {}
    energy = {}
    latency = {}
    genError = {}
    for name, model in model_dict.items():
        print("\nGetting Activation Norm for ", name)
        l_0[name] = get_activation_norm(model, val_dataloader)
        print("\nGetting Energy Estimation for ", name)
        energy[name] = analyse_data_energy(model, val_dataloader)
        print("\nGetting Latency & Generalization Error Estimation for ", name)
        latency[name] = analyse_latency(model, val_dataloader)
        genError[name] = analyse_generalization_error(model, val_dataloader)
    print("\nSaving Activation Norm ...")
    with open(path_results + 'l0.pkl', 'wb') as f:
        pickle.dump(l_0, f)
    with open(path_results + 'energy.pkl', 'wb') as f:
        pickle.dump(energy, f)
    with open(path_results + 'latency.pkl', 'wb') as f:
        pickle.dump(latency, f)
    with open(path_results + 'genError.pkl', 'wb') as f:
        pickle.dump(genError, f)


def analyse_latency(model, dataloader):
    """
    Analyse the latency of the model, as the time required for processing the dataset
    :param model: model
    :param dataloader: dataloader
    :return: latency
    """
    model.eval()
    with torch.no_grad():
        start = time.time()
        for idx, (batch, label) in enumerate(dataloader):
            _ = model(batch)
        end = time.time()
    latency = end - start
    return latency


def analyse_generalization_error(model, dataloader):
    """
    Analyse the generalization error of the model
    :param model: model
    :param dataloader: dataloader
    :return: generalization error
    """
    total_loss, total_count = 0, 0
    model.eval()
    with torch.no_grad():
        for idx, (batch, labels) in enumerate(dataloader):
            # Compute loss
            criterion = torch.nn.CrossEntropyLoss()
            logits = model(batch)
            if len(logits.shape) == 3:
                logits = logits.squeeze()
            total_loss += criterion(logits, labels).sum()
            total_count += labels.size(0)
    avg_loss = total_loss / total_count
    return avg_loss.item()


def get_activation_norm(model, dataloader):
    """
    Get the average l0 activation norm of the model
    :param model: model to evaluate
    :param dataloader: dataloader of the dataset
    :return: l0_exact: average activation norm normalized by the number of batches
    """
    stats = LayersActivationMeter()
    hooks = []
    leaf_nodes = [module for module in model.modules()
                  if len(list(module.children())) == 0]

    def hook_fn(name):
        def register_stats_hook(model, input, output):
            stats.register_output_stats(name, output)
        return register_stats_hook

    # add hooks for each layer
    ids = defaultdict(int)
    for i, module in enumerate(leaf_nodes):
        module_name = str(module).split('(')[0]
        hook = module.register_forward_hook(hook_fn(f'{module_name}-{ids[module_name]}'))
        ids[module_name] += 1
        hooks.append(hook)
    # forward samples to evaluate activations
    model.eval()
    with torch.no_grad():
        for idx, (batch, label) in enumerate(dataloader):
            # record the activation of each layer for each batch
            _ = model(batch)
        # compute the average activation for batch for each layer
        stats.avg_fired()
        print(stats.fired_exact)
    # remove hooks
    remove_hooks(hooks)
    # get total activation norm
    l0_exact = 0
    for key in stats.fired_exact.keys():
        if 'ReLU' in key:
            l0_exact += stats.fired_exact[key]
    print(l0_exact)
    return l0_exact


class LayersActivationMeter:
    """
    Class to compute the activation norm of each layer
    """
    def __init__(self):
        """
        Initialize the LayersActivationMeter
        """
        self.fired_exact = defaultdict(list)
        self.size = 0
        self.sigma = 1e-10

    def register_output_stats(self, name, output):
        """
        Register the output of the layer
        :param name: name of the layer
        :param output: output of the layer
        :return:
        """
        if isinstance(output, tuple):
            output = output[0]  # ignore the hidden state, when processing GRU
        fired_exact = (output.norm(p=0, dim=1)).mean()
        self.fired_exact[name].append(fired_exact.item())
        self.size += 1

    def avg_fired(self):
        """
        Compute the average l0 activation of each layer
        """
        for key in self.fired_exact.keys():
            self.fired_exact[key] = np.mean(self.fired_exact[key])
