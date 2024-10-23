import argparse

from project_handler import project_handler

attack_step = 1
name_dataset = 'MNIST'
surrogate_types = ['clean', 'surrogate']
lrs_MNIST = [0.0001, 0.0005, 0.001, 0.005, 0.01, 0.05]
Lambdas_MNIST = [  # m1 = neurons / m2 = layers in FFNN
    {'m1': [32, 64, 128], 'm2': 10},
    {'m1': [32], 'm2': 10},
    {'m1': [64], 'm2': 10},
    {'m1': [128], 'm2': 10}
]
lrs_CIFAR = [0.001, 0.005]
Lambdas_CIFAR = [  # m1 = neurons of second to last linear layer / m2 = number of dense blocks
    {'m1': [128, 256, 512], 'm2': 8},
    {'m1': [128], 'm2': 8},
    {'m1': [256], 'm2': 8},
    {'m1': [512], 'm2': 8}
]
lrs_SpeechCommands = [0.001, 0.005]
Lambdas_SpeechCommands = [  # m1 = neurons of second to last linear layer/ m2 = number of convolutional layers
    {'m1': [128, 256], 'm2': 8},
    {'m1': [128], 'm2': 8},
    {'m1': [256], 'm2': 8}
]
poison_rate = [0.1, 0.2, 0.5, 0.8, 1.0]
VAE_metric = ['l0', 'energy', 'latency', 'genError']
VAE_cost_function = 'linear'

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Control the parameters for testing the project.\n'
                                                 'For a correct execution from scratch, first run with attack_step '
                                                 'equal to 1, and then with attack_step equal to 5. ')
    parser.add_argument('--attack_step',
                        type=int,
                        default=attack_step,
                        help=('Attack Step: starting point of the attack pipeline. Options: \n'
                              '    - 1: Import Dataset \n'
                              '    - 2: Train MLaaS Models \n'
                              '    - 3: Train Surrogate Models \n'
                              '    - 4: Get Surrogate Cost \n'
                              '    - 5: Get Baseline Results \n'
                              '    - 6: Train CVAE \n'
                              '    - 7: Generate Poison Validation & Test \n'
                              '    - 8: Tabulate Results \n'
                              '    - 9: Train Complete HVAE \n'
                              '    - 10: Test Complete HVAE Attack \n'
                              '    - 11: Tabulate Results Complete HVAE \n'
                              '    - 12: Final Tabulation of Statistics \n'
                              '    - 13: t-SNE on Poisoned Validation Set \n')
                        )
    parser.add_argument('--name_dataset',
                        type=str,
                        default=name_dataset,
                        help='Name of the dataset')
    parser.add_argument('--surrogate_types',
                        type=str,
                        default=surrogate_types,
                        help='Surrogate Types. Options: '
                             '    - \'clean\' \n'
                             '    - \'surrogate\' \n')
    parser.add_argument('--lrs',
                        type=str,
                        default=None,
                        help='Learning Rates for classification models')
    parser.add_argument('--Lambda',
                        type=int,
                        default=None,
                        help='Lambdas, knowledge of the Adversary, when training the CVAE\n')
    parser.add_argument('--poison_rate',
                        type=str,
                        default=poison_rate,
                        help='Poison Rate')
    parser.add_argument('--VAE_metric',
                        type=str,
                        default=VAE_metric,
                        help='Hijack Metrics: hijack metric used to train the CVAE. Options: \n'
                             '    - \'l0\' \n'
                             '    - \'energy\' \n'
                             '    - \'latency\' \n'
                             '    - \'genError\' \n'
                        )
    parser.add_argument('--VAE_cost_function',
                        type=str,
                        default=VAE_cost_function,
                        help='VAE Cost Function: cost function used to train the CVAE. Options: \n'
                             '    - \'linear\' \n'
                             '    - \'exponential\' \n'
                        )
    args = parser.parse_args()
    # Convert the string to list
    if type(args.lrs) is str:
        args.lrs = [float(i) for i in args.lrs.split(',')]
    if args.lrs is None:
        args.lrs = lrs_MNIST if args.name_dataset == 'MNIST'\
            else lrs_CIFAR if args.name_dataset == 'CIFAR10'\
            else lrs_SpeechCommands
    if type(args.poison_rate) is str:
        args.poison_rate = [float(i) for i in args.poison_rate.split(',')]
    if type(args.surrogate_types) is str:
        args.surrogate_types = [args.surrogate_types]
    if type(args.VAE_metric) is str:
        # Divide string into list if it contains commas
        if ',' in args.VAE_metric:
            args.VAE_metric = [i for i in args.VAE_metric.split(',')]
        else:
            args.VAE_metric = [args.VAE_metric]
    if type(args.Lambda) is int:
        args.Lambda = [Lambdas_MNIST[args.Lambda]] if args.name_dataset == 'MNIST' \
            else [Lambdas_CIFAR[args.Lambda]] if args.name_dataset == 'CIFAR10' \
            else [Lambdas_SpeechCommands[args.Lambda]]
    if args.Lambda is None:
        args.Lambda = Lambdas_MNIST if args.name_dataset == 'MNIST' \
            else Lambdas_CIFAR if args.name_dataset == 'CIFAR10' \
            else Lambdas_SpeechCommands

    project_handler(attack_step=args.attack_step,
                    name_dataset=args.name_dataset,
                    surrogate_types=args.surrogate_types,
                    lrs=args.lrs,
                    Lambdas=args.Lambda,
                    poison_rate=args.poison_rate,
                    VAE_metrics=args.VAE_metric,
                    VAE_cost_function=args.VAE_cost_function)
