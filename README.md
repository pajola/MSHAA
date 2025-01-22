<div id="top"></div>
<!-- PROJECT LOGO -->
<br />
<div align="center">
  <a href="https://github.com/pajola/MSHAA/tree/main">
    <img src="https://i.postimg.cc/GhxVZcYG/phone.png" alt="Logo" width="150" height="150">
  </a>

  <h1 align="center">Moshi Moshi?</h1>

  <p align="center">A Model Selection Hijacking Adversarial Attack
    <br />
    <a href="https://github.com/pajola/MSHAA/tree/main"><strong>In progress »</strong></a>
    <br />
    <br />
    <a href="https://github.com/pajola/MSHAA/tree/main">Anonymous Authors</a>
  </p>
</div>

<!-- TABLE OF CONTENTS -->
<details>
  <summary><strong>Table of Contents</strong></summary>
  <ol>
    <li>
      <a href="#abstract">Abstract</a>
    </li>
    <li>
      <a href="#usage">Usage</a>
    </li>
    <li>
      <a href="#project">Project Steps</a>
    </li>
    <li>
      <a href="#guide">Guide</a>
    </li>
  </ol>
</details>


<div id="abstract"></div>

## 🧩 Abstract

>Model selection is a fundamental task in Machine Learning~(ML), focusing on selecting the most suitable model from a pool of candidates by evaluating their performance on specific metrics. This process ensures optimal performance, computational efficiency, and adaptability to diverse tasks and environments. Despite its critical role, its security from the perspective of adversarial ML remains unexplored. This risk is heightened in the Machine-Learning-as-a-Service model, where users delegate the training phase and the model selection process to third-party providers, supplying data and training strategies. Therefore, attacks on model selection could harm both the user and the provider, undermining model performance and driving up operational costs. In this work, we present **MOSHI** (**MO**del **S**election **HI**jacking adversarial attack), the first adversarial attack specifically targeting model selection. Our novel approach manipulates model selection data to favor the adversary, even without prior knowledge of the system. Utilizing a framework based on Variational Auto Encoders, we provide evidence that an attacker can induce inefficiencies in ML deployment. We test our attack on diverse computer vision and speech recognition benchmark tasks and different settings, obtaining an average attack success rate of 75.42%. In particular, our attack causes an average 88.30% decrease in generalization capabilities, an 83.33% increase in latency, and an increase of up to 105.85% in energy consumption. These results highlight the significant vulnerabilities in model selection processes and their potential impact on real-world applications.

<p align="right"><a href="#top">(back to top)</a></p>
<div id="usage"></div>

## ⚙️ Usage

To allow for reproducibility we include the anaconda environment:
`environment.yml`, which imports all the dependencies necessary for running
the project.

```shell
conda env create -f environment.yml
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

<p align="right"><a href="#top">(back to top)</a></p>
<div id="project"></div>

## 📃 Project Steps
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

<p align="right"><a href="#top">(back to top)</a></p>
<div id="guide"></div>

## 🎯 Guide
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