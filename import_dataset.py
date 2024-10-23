import numpy as np

from support import get_original_dataset, get_audio_dataset, split_dataset, save_dataset, get_data_from_loader


def import_dataset(name_dataset):
    """
    Import, Split and Save the  chosen Dataset
    :param name_dataset: name of the dataset
    :return: None
    """
    if name_dataset == 'SpeechCommands':
        print("\nGetting " + name_dataset + " Dataset ... \n")
        dataloader_val = get_audio_dataset('validation')
        dataloader_test = get_audio_dataset('testing')
        X_val, y_val = get_data_from_loader(dataloader_val)
        X_test, y_test = get_data_from_loader(dataloader_test)
        # Save Datasets
        save_dataset(X_val, y_val, 'validation_clean_dataset', name_dataset)
        print("Shape of Validation Dataset: ", X_val.shape)
        save_dataset(X_test, y_test, 'test_dataset', name_dataset)
        print("Shape of Test Dataset: ", X_test.shape)
    else:
        print("\nGetting " + name_dataset + " Dataset ... \n")
        X, y = get_original_dataset(name_dataset)
        print("Shape of Dataset: ", X.shape)
        print("Max value in Dataset: ", np.amax(X))
        print("Min value in Dataset: ", np.amin(y))
        # Splitting dataset: Train: 50%; Validation: 40%; Test: 10%
        X_tr, y_tr, X_val, y_val, X_test, y_test = split_dataset(X, y)
        # Save Datasets
        save_dataset(X_tr, y_tr, 'train_dataset', name_dataset)
        print("Shape of Train Dataset: ", X_tr.shape)
        save_dataset(X_val, y_val, 'validation_clean_dataset', name_dataset)
        print("Shape of Validation Dataset: ", X_val.shape)
        save_dataset(X_test, y_test, 'test_dataset', name_dataset)
        print("Shape of Test Dataset: ", X_test.shape)
