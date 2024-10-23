import numpy as np
import seaborn as sns
from sklearn.manifold import TSNE
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

from support import save_dataset, plot_generated_digits
from model_handler import load_vae
from generative_VAE import generate_samples


def generate_poison_for_tSNE(name_dataset, lr, Lambda, surrogate_type, pr, VAE_metric, VAE_cost_function):
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
    # Change the poison samples labels to 10 -> poison class
    y_poison = y_poison.cpu().detach().numpy() + 10
    # Split the validation set to allow for the addition of the poison samples
    if pr == 1.0:
        X_val_p = X_poison.cpu().detach().numpy()
        y_val_p = y_poison
    else:
        X_val, _, y_val, _ = train_test_split(X_val, y_val, test_size=pr, random_state=2090302,
                                              stratify=y_val)
        # Add the poison samples to the validation set
        X_val_p = np.concatenate((X_val, X_poison.cpu().detach().numpy()), axis=0)
        y_val_p = np.concatenate((y_val, y_poison), axis=0)
    print("Shape of Poison Validation: ", X_val_p.shape)
    print("Poison Labels: ", np.unique(y_val_p))
    # Save the Poison Validation
    save_dataset(X_val_p, y_val_p, 'validation_poison_dataset_for_tSNE_' + VAE_metric + '_' + str(lr), name_dataset)


def tSNE_poison(name_dataset, attacker_lr, VAE_metric, n_components=2):
    """
    tSNE on Poisoned Validation Set
    :param name_dataset: name of the dataset
    :param attacker_lr: knowledge of the Adversary
    :param VAE_metric: metric used to train the CVAE
    :param n_components: number of components for tSNE
    :return:
    """
    # Load the Poison Validation Dataset
    path_val_p = ('./datasets/' + name_dataset + '/validation_poison_dataset_for_tSNE_' + VAE_metric +
                  '_' + str(attacker_lr) + '.npy')
    print("\nLoading Datasets ...\n")
    with open(path_val_p, 'rb') as f:
        X_val_p = np.load(f)
        y_val_p = np.load(f)
    # Plot Poison Validation
    print('Plot Poison Validation ...')
    plot_generated_digits(
        X_val_p, y_val_p, len(np.unique(y_val_p)), './plots/', 'full_knowledge_VAE_' + VAE_metric, 'WB', name_dataset
    )
    # Run tSNE
    print("\nRunning tSNE ...")
    tsne = TSNE(n_components=n_components, random_state=2090302)
    X_tsne = tsne.fit_transform(X_val_p)
    # Plot tSNE
    print("\nPlotting tSNE ...")
    plt.figure(figsize=(15, 12), dpi=300)
    # Costume graph settings
    grey_palette = [(i / 20 - 0.05, i / 20 - 0.05, i / 20 - 0.05) for i in range(10, 20)]
    palette = sns.hls_palette(10) + grey_palette
    markers = ['o', 's', 'D', '^', 'v', '<', '>', 'p', '*', 'h', 'o', 's', 'D', '^', 'v', '<', '>', 'p', '*', 'h']

    scatter = sns.scatterplot(x=X_tsne[:, 0],
                              y=X_tsne[:, 1],
                              hue=y_val_p,
                              style=y_val_p,
                              markers=markers,
                              palette=palette,
                              legend='full')

    handles, labels = scatter.get_legend_handles_labels()
    labels = ['Class ' + label if int(label) < 10 else 'Poison ' + str(int(label) - 10) for label in labels]
    scatter.legend(handles, labels, title='Legend')
    # Save the plot
    plot_path = './plots/' + name_dataset + '_' + VAE_metric + '_tSNE_poison_plot.png'
    plt.savefig(plot_path)
    print(f"Plot saved to {plot_path}")
    plt.close()
