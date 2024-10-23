import numpy as np
import pandas as pd  # version 1.5.3
import pickle

from model_handler import load_models
from support import plot_generated_digits, get_audio_dataset
from test import get_results_attack, get_results_baseline, get_results_baseline_audio


def test_baseline(name_dataset, lr):
    """
    Function to test the baseline models
    :param name_dataset: name of the dataset
    :param lr: learning rate
    :return: None
    """
    # Paths for loading the dataset and the models
    path_models = './trained_models/' + name_dataset + '/' + str(lr) + '/MLaaS/'
    result_path = './results/' + name_dataset + '/baseline' + '/baseline_results_models_' + str(lr) + '.pkl'
    # Load the models
    print("\nLoading Models ...\n")
    models = load_models(path_models, name_dataset)

    print("\nTesting Baseline: \n",
          "  Dataset:", name_dataset, "\n"
          "  Learning Rate:", lr, "\n")

    if name_dataset == 'SpeechCommands':
        train_dataloader = get_audio_dataset('training')  # not imported as .npy to avoid memory issues
        path_test = './datasets/' + name_dataset + '/test_dataset.npy'
        path_val_c = './datasets/' + name_dataset + '/validation_clean_dataset.npy'
        print("\nLoading Datasets ...\n")
        with open(path_test, 'rb') as f:
            X_test = np.load(f)
            y_test = np.load(f)
        with open(path_val_c, 'rb') as f:
            X_val = np.load(f)
            y_val = np.load(f)
        # Get the results
        results = get_results_baseline_audio(models, train_dataloader, X_val, y_val, X_test, y_test)
    else:
        path_test = './datasets/' + name_dataset + '/test_dataset.npy'
        path_train = './datasets/' + name_dataset + '/train_dataset.npy'
        path_val_c = './datasets/' + name_dataset + '/validation_clean_dataset.npy'
        print("\nLoading Datasets ...\n")
        with open(path_test, 'rb') as f:
            X_test = np.load(f)
            y_test = np.load(f)
        with open(path_train, 'rb') as f:
            X_tr = np.load(f)
            y_tr = np.load(f)
        with open(path_val_c, 'rb') as f:
            X_val = np.load(f)
            y_val = np.load(f)
        # Get the results
        results = get_results_baseline(models, X_tr, y_tr, X_val, y_val, X_test, y_test)

    print("\nSaving Results ...")
    results.to_pickle(result_path)
    print("\nResults: \n", results)


def test_setup(name_dataset, lr, Lambda, surrogate_type, pr, VAE_metric, VAE_cost_function):
    """
    Function to test the models
    :param name_dataset: name of the dataset
    :param lr: learning rate
    :param Lambda: dictionary of selected layers and neurons
    :param surrogate_type: clean or surrogate
    :param pr: poison rate
    :param VAE_metric: cost metric used to train the CVAE
    :param VAE_cost_function: cost function used to train the CVAE
    :return: None
    """
    path_val_p = './datasets/' + name_dataset + '/validation_poison_dataset_' + str(lr) + '.npy'
    path_models = './trained_models/' + name_dataset + '/' + str(lr) + '/MLaaS/'
    path_baseline = './results/' + name_dataset + '/baseline' + '/baseline_results_models_' + str(lr) + '.pkl'
    print("\nLoading Datasets ...\n")
    with open(path_val_p, 'rb') as f:
        X_val_p = np.load(f)
        y_val_p = np.load(f)
    # Load the models
    print("\nLoading Models ...\n")
    models = load_models(path_models, name_dataset)
    # Load the baseline results
    results_baseline = pd.read_pickle(path_baseline)
    # Get the results
    print("Testing Attack: \n",
          "  Dataset:", name_dataset, "\n",
          "  Poison Rate:", pr, "\n",
          "  Lambda:", Lambda, "\n",
          "  Surrogate Type:", surrogate_type, "\n",
          "  VAE Metric:", VAE_metric, "\n",
          "  VAE Cost Function:", VAE_cost_function, "\n")
    results = get_results_attack(models, X_val_p, y_val_p, results_baseline)
    print("\nSaving Results ...")
    vae_name = str(Lambda['m1']).replace(' ', '')
    result_path = './results/' + name_dataset + '/' + VAE_metric + '_' + VAE_cost_function + '/'
    results.to_pickle(
        result_path + 'results_' + str(lr) + '_VAE_' + vae_name + '_' +
        surrogate_type + '_' + str(pr) + '.pkl'
    )
    print("\nResults: \n", results)

    if pr == 1.0 and lr == 0.001 and name_dataset != 'SpeechCommands':
        plot_generated_digits(
            X_val_p, y_val_p, len(np.unique(y_val_p)), result_path, vae_name, surrogate_type, name_dataset
        )


def test_setup_complete(name_dataset, attacker_lr, victim_lrs,  Lambda, surrogate_type, pr, VAE_metric, VAE_cost_function):
    """
    Function to test the models
    :param name_dataset: name of the dataset
    :param attacker_lr: learning rate of the models known by the adversary for training the CVAE
    :param victim_lrs: learning rates of the victim models
    :param Lambda: dictionary of selected layers and neurons
    :param surrogate_type: clean or surrogate
    :param pr: poison rate
    :param VAE_metric: cost metric used to train the CVAE
    :param VAE_cost_function: cost function used to train the CVAE
    :return: None
    """
    path_val_p = './datasets/' + name_dataset + '/validation_poison_dataset_' + str(attacker_lr) + '.npy'
    print("\nLoading Datasets ...\n")
    with open(path_val_p, 'rb') as f:
        X_val_p = np.load(f)
        y_val_p = np.load(f)
    # Load the models
    print("\nLoading Models ...\n")
    models = {}
    for lr in victim_lrs:
        path_models = './trained_models/' + name_dataset + '/' + str(lr) + '/MLaaS/'
        tested_models = load_models(path_models, name_dataset)
        for name, model in tested_models.items():
            models[str(lr) + '_' + name] = model

    # Get full knowledge baseline results
        path = './results/'+ name_dataset +'/baseline/baseline_results_models_full_knowledge_vae.pkl'
        results_baseline = pd.DataFrame()
        for lr in victim_lrs:
            result_path = str(
                './results/' + name_dataset + '/baseline/baseline_results_models_' + str(lr) + '.pkl')
            results = pd.read_pickle(result_path)
            results['lr'] = lr  # Add 'lr' column
            results_baseline = pd.concat([results_baseline, results], ignore_index=True)
        # Save the results
        with open(path, 'wb') as f:
            pickle.dump(results_baseline, f)

    # Get the results
    print("Testing Attack: \n",
          "  Dataset:", name_dataset, "\n",
          "  Poison Rate:", pr, "\n",
          "  Lambda:", Lambda, "\n",
          "  Surrogate Type:", surrogate_type, "\n",
          "  VAE Metric:", VAE_metric, "\n",
          "  VAE Cost Function:", VAE_cost_function, "\n")

    results = get_results_attack(models, X_val_p, y_val_p, results_baseline)

    print("\nSaving Results ...")
    vae_name = str(Lambda['m1']).replace(' ', '')
    result_path = './results/' + name_dataset + '/' + VAE_metric + '_' + VAE_cost_function + '/'
    results.to_pickle(
        result_path + 'results_' + str(attacker_lr) + '_VAE_' + vae_name + '_' +
        surrogate_type + '_' + str(pr) + '.pkl'
    )
    print("\nResults: \n", results)

    if pr == 1.0 and name_dataset != 'SpeechCommands':
        plot_generated_digits(
            X_val_p, y_val_p, len(np.unique(y_val_p)), result_path, vae_name, surrogate_type, name_dataset
        )
