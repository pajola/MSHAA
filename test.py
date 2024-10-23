import torch
import numpy as np
import pandas as pd  # version 1.5.3

from support import get_dataloader
from get_hijack_metrics import get_activation_norm, analyse_latency
from ASICSim import analyse_data_energy


def get_results_baseline(model_dict, X_tr, y_tr, X_val, y_val, X_test, y_test):
    """
    Function to get the results of the baseline model
    :param model_dict: dictionary of trained models - keys: model_name, values: models
    :param X_tr: training samples
    :param y_tr: training labels
    :param X_val: clean validation samples
    :param y_val: clean validation labels
    :param X_test: test samples
    :param y_test: test labels
    :return: dataframe of results
    """
    # Set GPU environment
    np.random.seed(2090302)
    torch.manual_seed(2090302)
    torch.cuda.manual_seed(2090302)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    results = {'Layers': [],
               'Neurons': [],
               'Train Loss': [],
               'Validation Loss': [],
               'Test Loss': [],
               'Test Accuracy': [],
               'l0': [],
               'Energy (pJ)': [],
               'Latency (s)': []}

    print("\nTesting Baseline Model")
    for name, model in model_dict.items():
        print("\nTesting Model: ", name)
        # Get the loss and accuracy for the model
        tr_loss, tr_acc = get_loss_and_accuracy(model, X_tr, y_tr, device)
        val_loss, val_acc = get_loss_and_accuracy(model, X_val, y_val, device)
        test_loss, test_acc = get_loss_and_accuracy(model, X_test, y_test, device)
        dataloader_test = get_dataloader(X_test, y_test, device, batch_size=256)
        # Build the results dictionary
        print(name)
        results['Layers'].append(int(name.split('_')[0]))
        results['Neurons'].append(int(name.split('_')[1]))
        results['Train Loss'].append(tr_loss)
        results['Validation Loss'].append(val_loss)
        results['Test Loss'].append(test_loss)
        results['Test Accuracy'].append(test_acc)
        results['l0'].append(get_activation_norm(model, dataloader_test))
        results['Energy (pJ)'].append(analyse_data_energy(model, dataloader_test))
        results['Latency (s)'].append(analyse_latency(model, dataloader_test))

    results = pd.DataFrame(results)
    results = results.sort_values(by=['Neurons', 'Validation Loss'], ascending=True)
    results.reset_index(inplace=True)
    results.drop("index", axis=1, inplace=True)
    return results


def get_results_attack(models, X_p_val, y_p_val, results_baseline):
    """
    Function to get the results of the models
    :param models: dictionary of FFNN models - keys: 'layers_neurons', values: FFNN models
    :param X_p_val: poison validation samples
    :param y_p_val: poison validation labels
    :param results_baseline: dataframe of baseline results
    :return: dataframe of results
    """
    # Set GPU environment
    np.random.seed(2090302)
    torch.manual_seed(2090302)
    torch.cuda.manual_seed(2090302)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    results_baseline = results_baseline.copy()
    results_baseline['Validation Poison Loss'] = [0] * len(results_baseline['Validation Loss'])

    for name, model in models.items():
        print("\nTesting Model: ", name)
        # Get the loss and accuracy for each model
        val_p_loss, val_p_acc = get_loss_and_accuracy(model, X_p_val, y_p_val, device)
        if len(name.split('_')) == 2:
            index = results_baseline.index[(results_baseline['Layers'] == int(name.split('_')[0])) &
                                           (results_baseline['Neurons'] == int(name.split('_')[1]))].tolist()[0]
            # Ensure the model is in the baseline results
            if not (results_baseline['Layers'][index] == int(name.split('_')[0]) and
                    results_baseline['Neurons'][index] == int(name.split('_')[1])):
                valueNotFound_error = "Error: Model not found in the baseline results"
                raise ValueError(valueNotFound_error)
        else:
            index = results_baseline.index[(((results_baseline['lr']).astype(str) == str(name.split('_')[0])) &
                                           (results_baseline['Layers'] == int(name.split('_')[1])) &
                                           (results_baseline['Neurons'] == int(name.split('_')[2])))].tolist()[0]
            # Ensure the model is in the baseline results
            if not (((results_baseline['lr'][index]).astype(str) == str(name.split('_')[0])) and
                    (results_baseline['Layers'][index] == int(name.split('_')[1])) and
                    (results_baseline['Neurons'][index] == int(name.split('_')[2]))):
                valueNotFound_error = "Error: Model not found in the baseline results"
                raise ValueError(valueNotFound_error)

        # Complete the results dictionary
        results_baseline.at[index, 'Validation Poison Loss'] = val_p_loss

    results = pd.DataFrame(results_baseline)
    results = results.sort_values(by=['Neurons', 'Validation Poison Loss'], ascending=True)
    results.reset_index(inplace=True)
    results.drop("index", axis=1, inplace=True)
    return results


def get_results_baseline_audio(model_dict, train_dataset, X_val, y_val, X_test, y_test):
    """
    Function to get the results of the baseline model for the audio dataset
    :param model_dict: dictionary of trained models - keys: model_name, values: models
    :param train_dataset: training dataset
    :param X_val: clean validation samples
    :param y_val: clean validation labels
    :param X_test: test samples
    :param y_test: test labels
    :return: dataframe of results
    """

    # Set GPU environment
    np.random.seed(2090302)
    torch.manual_seed(2090302)
    torch.cuda.manual_seed(2090302)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    results = {'Layers': [],
               'Neurons': [],
               'Train Loss': [],
               'Validation Loss': [],
               'Test Loss': [],
               'Test Accuracy': [],
               'l0': [],
               'Energy (pJ)': [],
               'Latency (s)': []}

    print("\nTesting Baseline Model")
    for name, model in model_dict.items():
        print("\nTesting Model: ", name)
        # Get the loss and accuracy for the model
        tr_loss, tr_acc = get_loss_and_accuracy_audio(model, train_dataset, device)
        val_dataset = get_dataloader(X_val, y_val, device, batch_size=256)
        test_dataset = get_dataloader(X_test, y_test, device, batch_size=256)
        val_loss, val_acc = get_loss_and_accuracy_audio(model,val_dataset, device)
        test_loss, test_acc = get_loss_and_accuracy_audio(model, test_dataset, device)
        # Build the results dictionary
        print(name)
        results['Layers'].append(int(name.split('_')[0]))
        results['Neurons'].append(int(name.split('_')[1]))
        results['Train Loss'].append(tr_loss)
        results['Validation Loss'].append(val_loss)
        results['Test Loss'].append(test_loss)
        results['Test Accuracy'].append(test_acc)
        results['l0'].append(get_activation_norm(model, test_dataset))
        results['Energy (pJ)'].append(analyse_data_energy(model, test_dataset))
        results['Latency (s)'].append(analyse_latency(model, test_dataset))

    results = pd.DataFrame(results)
    results = results.sort_values(by=['Neurons', 'Validation Loss'], ascending=True)
    results.reset_index(inplace=True)
    results.drop("index", axis=1, inplace=True)
    return results


def get_loss_and_accuracy(model, X, y, device):
    """
    Get the loss and accuracy of the model
    :param model: Classification model
    :param X: input samples
    :param y: input labels
    :param device: cuda or cpu
    :return: loss: loss of the model, accuracy: accuracy of the model
    """
    dataloader = get_dataloader(X, y, device, batch_size=256)
    total_loss, total_acc, total_count = 0, 0, 0
    # Evaluate model
    model.eval()
    with torch.no_grad():
        for idx, (batch, labels) in enumerate(dataloader):
            # Compute loss and accuracy
            criterion = torch.nn.CrossEntropyLoss()
            logits = model(batch)
            if len(logits.shape) == 3:
                logits = logits.squeeze()
            total_loss += criterion(logits, labels).sum()
            total_acc += (logits.argmax(1) == labels).sum()
            total_count += labels.size(0)
    accuracy = total_acc / total_count
    avg_loss = total_loss / total_count
    print(f"Loss: {avg_loss:.8f}, Accuracy: {accuracy:.2f}%")
    return avg_loss.item(), accuracy.item()


def get_loss_and_accuracy_audio(model, dataloader, device):
    """
    Get the loss and accuracy of the audio model
    :param model: Classification model
    :param dataloader: input dataset
    :param device: cuda or cpu
    :return: loss: loss of the model, accuracy: accuracy of the model
    """
    total_loss, total_acc, total_count = 0, 0, 0
    # Evaluate model
    model.eval()
    with torch.no_grad():
        for idx, (batch, labels) in enumerate(dataloader):
            batch = batch.to(device)
            labels = labels.to(device)
            # Compute loss and accuracy
            criterion = torch.nn.CrossEntropyLoss()
            logits = model(batch).squeeze()
            total_loss += criterion(logits, labels).sum()
            total_acc += (logits.argmax(1) == labels).sum()
            total_count += labels.size(0)
    accuracy = total_acc / total_count
    avg_loss = total_loss / total_count
    print(f"Loss: {avg_loss:.8f}, Accuracy: {accuracy:.2f}%")
    return avg_loss.item(), accuracy.item()