# Toward the Model Selection Hijacking Adversarial Attack

This repository contains the PyTorch implementation of the **Model
Selection Hijacking Adversarial Attack**.

## Reprodicubility & Initalization
To allow for reproducibility we include the anaconda environment:
`envoronment.yml`, which imports all the dependencies necessary for running
the project.

```shell
conda env create -f envoronment.yml
```

The created environment is called `thesis_env`, and is activated using the
following:

``` shell
conda activate thesis_env
```

Before running the code, it is also required to run the
`create_directories.sh`, which is a shell script that creates the necessary
directories used for the project.

```shell
conda ./create_directories.sh
```

## Project Steps
As the project requires some time to be completely executed, we divided it
into consequential steps:
1. Import Dataset
2. Train Victim Models
3. Train Surrogate Models
4. Get Surrogate Cost
5. Get Baseline Results
6. Train HVAE
7. Generate Poison Validation & Test
8. Tabulate Results
9. Train Complete HVAE
10. Test Complete HVAE Attack
11. Tabulate Results Complete HVAE
12. Final Tabulation of Statistics
13. t-SNE on Poisoned Validation Set



## Guide
The project can be run from the `main.py` script, which allows to specify:
 - dataset used
 - hyperparameters and adversary knowledge
 - metrics for the attack
 - starting step of the project

Now we will describe how to run the project:

```shell
python3 ./main.py --attack_step 1 --name_dataset MNIST
```

This will start the project at step 1 for the MNIST dataset and use the
default arguments for the other selectable variables.
The program starting from 1 will complete steps: 1, 2, 3, and 4.
Afterward, we will have obtained the trained victim and surrogate models,
alongside obtaining their hijack metrics.

Once these steps are done, the following ones can be run sequentially
and autonomously.