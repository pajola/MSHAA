import torch
import pickle

from import_dataset import import_dataset
from training_setup import classification_training_setup, audio_training_setup, vae_training_setup
from get_hijack_metrics import get_hijack_metric
from generate_poison import generate_poison
from test_setup import test_setup, test_baseline, test_setup_complete
from model_handler import train_classification_models, save_models, train_vae, train_audio_models
from tabulate_results import tabulate_results, tabulate_stats, get_stats
from tSNE_poison import tSNE_poison, generate_poison_for_tSNE


def project_handler(
        attack_step, name_dataset, lrs, surrogate_types, Lambdas, poison_rate, VAE_metrics, VAE_cost_function
):
    """
    Function to handle the project pipeline
    :param attack_step: starting point of the attack
        - 1: Import Dataset
        - 2: Train MLaaS Models
        - 3: Train Surrogate Models
        - 4: Get Surrogate Cost
        - 5: Train CVAE
        - 6: Generate Poison Validation
        - 7: Test
        - 8: All
    :param name_dataset: name of the used dataset
    :param lrs: learning rates for the victim models
    :param surrogate_types: types of surrogate models
        - clean: use the clean models
        - surrogate: use the models trained with different seed
    :param Lambdas: set of hyperparameters of trained models used for training the CVAE
    :param poison_rate: percentage of poison samples to be substituted in the validation set
    :param VAE_metrics: cost metric used to train the CVAE
    :param VAE_cost_function: cost function used to train the CVAE
    :return: None
    """

    # Set GPU environment
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    print(' Device: ' + str(device) + '\n',
          'Attack Step: ' + str(attack_step) + '\n',
          'Dataset: ' + name_dataset + '\n',
          'Learning Rates: ' + str(lrs) + '\n',
          'Surrogate Types: ' + str(surrogate_types) + '\n',
          'Lambdas: ' + str(Lambdas) + '\n',
          'Poison Rate: ' + str(poison_rate) + '\n',
          'VAE Metric: ' + str(VAE_metrics) + '\n',
          'VAE Cost Function: ' + VAE_cost_function + '\n')

    # STEP 1: Import Dataset
    if attack_step in [1]:
        print("\nSTEP 1: IMPORTING DATASET")
        import_dataset(name_dataset)

    for lr in lrs:
        # STEP 2: Train MLaaS Models
        if attack_step in [1, 2]:
            print("\nSTEP 2: TRAINING MODELS")
            if name_dataset == 'SpeechCommands':
                dataloader, hp = audio_training_setup(lr)
                print('training set shape:', len(dataloader.dataset))
                models = train_audio_models(dataloader, hp, 2090302)
            else:
                X_tr, y_tr, hp = classification_training_setup(name_dataset, lr)
                print('training set shape:', X_tr.shape, y_tr.shape)
                models = train_classification_models(name_dataset, X_tr, y_tr, hp, 2090302)
            print("\nSaving Models ...")
            for model in models.items():
                save_models(model, name_dataset + '/' + str(lr) + '/MLaaS/')

        # STEP 3: Train Surrogate Models
        if attack_step in [1, 2, 3]:
            print("\n STEP 3: TRAINING SURROGATE MODELS")
            if name_dataset == 'SpeechCommands':
                dataloader, hp = audio_training_setup(lr)
                print('training set shape:', len(dataloader.dataset))
                models = train_audio_models(dataloader, hp, 42)
            else:
                X_tr, y_tr, hp = classification_training_setup(name_dataset, lr)
                models = train_classification_models(name_dataset, X_tr, y_tr, hp, 42)
            print("\nSaving Models ...")
            for model in models.items():
                save_models(model, name_dataset + '/' + str(lr) + '/Surrogate/')

        # STEP 4: Get Surrogate Cost
        if attack_step in [1, 2, 3, 4]:
            print('\nSTEP 4: GETTING HIJACK METRICS')
            get_hijack_metric(name_dataset, lr)

        # STEP 5: Test Baseline
        if attack_step in [5]:
            print("\nSTEP 5: TEST BASELINE")
            test_baseline(name_dataset, lr)

        # Iterate over the surrogate types and the lambdas to train the CVAE
        for surrogate_type in surrogate_types:
            for VAE_metric in VAE_metrics:
                for Lambda in Lambdas:
                    # STEP 6: Train CVAE
                    if attack_step in [6]:
                        print("\nSTEP 6: TRAINING VAE with Lambda: ", Lambda['m1'], " lr: ", lr,
                              " surrogate type: ", surrogate_type)
                        X_val, y_val, hp, used_models, metric = vae_training_setup(
                            name_dataset, lr, Lambda, surrogate_type, VAE_metric
                        )
                        # Train the CVAE
                        vae, best_hp, best_losses = train_vae(
                            name_dataset, X_val, y_val, hp, used_models, metric, VAE_cost_function
                        )
                        # Save the trained CVAE
                        vae_name = str(Lambda['m1']).replace(' ', '')
                        vae_path = name_dataset + '/' + str(lr) + '/' + VAE_metric + '_' + VAE_cost_function
                        save_models(
                            ('VAE_' + surrogate_type + '_' + vae_name, vae), vae_path
                        )
                        # Save the hyperparameters of the best model
                        with open('./trained_models/' + vae_path +
                                  '/VAE_' + surrogate_type + '_' + vae_name + '_hp.pkl', 'wb') as f:
                            pickle.dump(best_hp, f)
                        # Save the losses of the best model
                        with open('./trained_models/' + vae_path +
                                  '/VAE_' + surrogate_type + '_' + vae_name + '_losses.pkl', 'wb') as f:
                            pickle.dump(best_losses, f)

                    # Iterate over the poison rates - use the same VAE to generate poison validation data
                    for pr in poison_rate:
                        # STEP 7: Generate Poison Validation & Test Attack
                        if attack_step in [7]:
                            print("\nSTEP 7: GENERATING POISON VALIDATION")
                            # lr of attacker knowledge
                            generate_poison(name_dataset, lr, Lambda, surrogate_type, pr, VAE_metric, VAE_cost_function)
                            print("\nTESTING ATTACK")
                            # learning rate of victim model
                            test_setup(name_dataset, lr, Lambda, surrogate_type, pr, VAE_metric, VAE_cost_function)

    # STEP 8: Tabulate Results
    if attack_step in [8]:
        print("\nSTEP 8: TABULATING RESULTS")
        if VAE_metrics == ['l0', 'energy', 'latency', 'genError'] and name_dataset == 'SpeechCommands':
            VAE_metrics = ['l0', 'latency', 'genError']
        for VAE_metric in VAE_metrics:
            tabulate_results(name_dataset, lrs, surrogate_types, poison_rate, VAE_metric, VAE_cost_function)

    # STEP 9: Train Complete HVAE
    if attack_step in [9]:
        print("\nSTEP 9: TRAINING COMPLETE HVAE")
        for surrogate_type in surrogate_types:
            for VAE_metric in VAE_metrics:
                models = {}
                metrics = {}
                X_val, y_val, hp = None, None, None
                for lr in lrs:
                    Lambda = Lambdas[0]
                    X_val, y_val, hp, used_models, metric = vae_training_setup(
                        name_dataset, lr, Lambda, surrogate_type, VAE_metric
                    )
                    for name, model in used_models.items():
                        models[str(lr) + '_' + name] = model
                    for name, m in metric.items():
                        metrics[str(lr) + '_' + name] = m
                # Train the CVAE
                vae, best_hp, best_losses = train_vae(
                    name_dataset, X_val, y_val, hp, models, metrics, VAE_cost_function
                )
                # Save the trained CVAE
                vae_path = name_dataset + '/full_knowledge_vae/' + VAE_metric + '_' + VAE_cost_function
                save_models(
                    ('VAE_' + surrogate_type + '_full_knowledge', vae), vae_path
                )
                # Save the hyperparameters of the best model
                with open('./trained_models/' + vae_path +
                          '/VAE_' + surrogate_type + '_full_knowledge_hp.pkl', 'wb') as f:
                    pickle.dump(best_hp, f)
                # Save the losses of the best model
                with open('./trained_models/' + vae_path +
                          '/VAE_' + surrogate_type + '_full_knowledge_losses.pkl', 'wb') as f:
                    pickle.dump(best_losses, f)

    # STEP 10: Test Complete HVAE Attack
    if attack_step in [10]:
        print("\nSTEP 10: TESTING COMPLETE HVAE")

        # Generate Poison Validation & Test Attack
        for surrogate_type in surrogate_types:
            for VAE_metric in VAE_metrics:
                for pr in poison_rate:
                    generate_poison(name_dataset, 'full_knowledge_vae', {'m1': 'full_knowledge'},
                                    surrogate_type, pr, VAE_metric, VAE_cost_function)
                    print("\nTESTING ATTACK")
                    test_setup_complete(name_dataset, 'full_knowledge_vae', lrs, {'m1': 'full_knowledge'},
                                        surrogate_type, pr, VAE_metric, VAE_cost_function)

    # STEP 11: Tabulate Results Complete HVAE
    if attack_step in [11]:
        # Tabulate Results
        for VAE_metric in VAE_metrics:
            tabulate_results(name_dataset, ['full_knowledge_vae'], surrogate_types, poison_rate,
                             VAE_metric, VAE_cost_function)

    # STEP 12: Final Tabulation of Statistics
    if attack_step in [12]:
        # overall statistics
        tabulate_stats()
        # statistics for each dataset
        get_stats('success_rate')
        get_stats('score')

    # STEP 13: t-SNE on Poisoned Validation Set
    if attack_step in [13]:
        surrogate_type = surrogate_types[0]
        pr = 0.5

        for VAE_metric in VAE_metrics:
            # Generate Poison Validation from the full knowledge HVAE substituting 10% of the validation set
            generate_poison_for_tSNE(name_dataset, 'full_knowledge_vae', {'m1': 'full_knowledge'},
                                     surrogate_type, pr, VAE_metric, VAE_cost_function)
            tSNE_poison(name_dataset, 'full_knowledge_vae', VAE_metric, n_components=2)


