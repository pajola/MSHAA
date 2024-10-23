import numpy as np
from sklearn.model_selection import train_test_split

from support import save_dataset
from model_handler import load_vae
from generative_VAE import generate_samples


def generate_poison(name_dataset, lr, Lambda, surrogate_type, pr, VAE_metric, VAE_cost_function):
    """
    Generate Poison Validation
    :param name_dataset: name of the dataset
    :param lr: learning rate
    :param Lambda: dictionary of selected layers and neurons
    :param surrogate_type: clean or surrogate
    :param pr: poison rate
    :param VAE_metric: cost metric used to train the CVAE
    :param VAE_cost_function: cost function used to train the CVAE
    :return: None
    """
    validation_path = './datasets/' + name_dataset + '/validation_clean_dataset.npy'
    # Load the Validation Dataset
    print("\nLoading Validation Dataset ...\n")
    with open(validation_path, 'rb') as f:
        X_val = np.load(f)
        y_val = np.load(f)
    # Load the trained CVAE & generate poison validation
    print("\nLoading CVAE ...")
    vae_name = str(Lambda['m1']).replace(' ', '')
    vae_path = name_dataset + '/' + str(lr) + '/' + VAE_metric + '_' + VAE_cost_function
    print("\nGenerating Poison Validation ...")
    vae = load_vae(
        './trained_models/' + vae_path + '/VAE_' + surrogate_type + '_' + vae_name + '.pt',
        './trained_models/' + vae_path + '/VAE_' + surrogate_type + '_' + vae_name + '_hp.pkl',
        name_dataset
    )
    # Get the number of samples per class to substitute with poison samples
    class_counts = np.bincount(y_val) * pr
    class_counts = class_counts.astype(int)
    print("Poison Amount per class: ", class_counts)
    print("Shape of Clean Validation: ", X_val.shape)
    # Generate Poison Validation
    X_poison, y_poison = generate_samples(vae, class_counts)
    # Split the validation set to allow for the addition of the poison samples
    if pr == 1.0:
        X_val_p = X_poison.cpu().detach().numpy()
        y_val_p = (y_poison.cpu().detach().numpy()).astype(np.int64)
    else:
        X_val, _, y_val, _ = train_test_split(X_val, y_val, test_size=pr, random_state=2090302,
                                              stratify=y_val)
        # Add the poison samples to the validation set
        X_val_p = np.concatenate((X_val, X_poison.cpu().detach().numpy()), axis=0)
        y_val_p = np.concatenate((y_val, y_poison.cpu().detach().numpy()), axis=0)
    print("Shape of Poison Validation: ", X_val_p.shape)
    # Save the Poison Validation
    save_dataset(X_val_p, y_val_p, 'validation_poison_dataset_' + str(lr), name_dataset)
