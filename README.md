<div align="center">

# AdvGrads

[![Latest Release](https://img.shields.io/github/release/myuito3/AdvGrads.svg?&color=blue)](https://github.com/myuito3/AdvGrads/releases)
[![License](https://img.shields.io/badge/License-Apache_2.0-green.svg)](https://github.com/myuito3/AdvGrads/blob/master/LICENSE)

</div>

## 🌐 About
AdvGrads is an all-in-one tool for comprehensive experimentation with adversarial attacks on image recognition. This repository provides an environment used for research purposes to validate the performance of attack and defense methods in adversarial attacks.

This repository is still under development. For more information, please contact with me (m.yuito3@gmail.com).

## 🔍 Features
AdvGrads is developed on `PyTorch`.

### 💥 Attacks
Currently supported attack methods are as follows:

| Method              | Type                | References          |
| :------------------ | :------------------ | :------------------ |
| DeepFool            | White-box           | 📃[DeepFool: a simple and accurate method to fool deep neural networks](https://arxiv.org/abs/1511.04599) |
| DI-MI-FGSM          | White-box           | 📃[Improving Transferability of Adversarial Examples with Input Diversity](https://arxiv.org/abs/1803.06978) |
| FGSM                | White-box           | 📃[Explaining and Harnessing Adversarial Examples](https://arxiv.org/abs/1412.6572) |
| I-FGSM (BIM)        | White-box           | 📃[Adversarial examples in the physical world](https://arxiv.org/abs/1607.02533) |
| MI-FGSM (MIM)       | White-box           | 📃[Boosting Adversarial Attacks with Momentum](https://arxiv.org/abs/1710.06081) |
| NI-FGSM             | White-box           | 📃[Nesterov Accelerated Gradient and Scale Invariance for Adversarial Attacks](https://arxiv.org/abs/1908.06281) |
| PGD                 | White-box           | 📃[Towards Deep Learning Models Resistant to Adversarial Attacks](https://arxiv.org/abs/1706.06083) |
| PI-FGSM             | White-box           | 📃[Patch-wise Attack for Fooling Deep Neural Network](https://arxiv.org/abs/2007.06765) |
| SI-NI-FGSM          | White-box           | 📃[Nesterov Accelerated Gradient and Scale Invariance for Adversarial Attacks](https://arxiv.org/abs/1908.06281) |
| SignHunter          | Black-box           | 📃[Sign Bits Are All You Need for Black-Box Attacks](https://openreview.net/forum?id=SygW0TEFwH) |
| SimBA               | Black-box           | 📃[Simple Black-box Adversarial Attacks](https://arxiv.org/abs/1905.07121) |
| Square attack       | Black-box           | 📃[Square Attack: a query-efficient black-box adversarial attack via random search](https://arxiv.org/abs/1912.00049) |

### 💠 Defenses
Currently supported defense methods including adversarially trained models are as follows:

| Method              | Type                | References          |
| :------------------ | :------------------ | :------------------ |
| Bit-Red             | Input transform     | 📃[Feature Squeezing: Detecting Adversarial Examples in Deep Neural Networks](https://arxiv.org/abs/1704.01155) |
| JPEG                | Input transform     | 📃[A study of the effect of JPG compression on adversarial images](https://arxiv.org/abs/1608.00853) |
| Randomization       | Input transform     | 📃[Mitigating Adversarial Effects Through Randomization](https://arxiv.org/abs/1711.01991) |
| TRADES              | Adv. training       | 📃[Theoretically Principled Trade-off between Robustness and Accuracy](https://arxiv.org/abs/1901.08573) |

### 🧩 Others
And also, some naturally trained models are supported.

| Source              | Datasets            | References          |
| :------------------ | :------------------ | :------------------ |
| pytorch-playground  | MNIST, CIFAR-10     | 🌐[pytorch-playground](https://github.com/aaron-xichen/pytorch-playground) |

## 💻 Installation

### Create environment
AdvGrads requires `Python >= 3.9`. An example of creating an environment using Python venv:
```bash
py -3.9 -m venv [ENV_NAME]
```

### Dependencies
After creating and activating your virtual environment, you can install necessary libraries via the requirements.txt.

```bash
git clone https://github.com/myuito3/AdvGrads.git
cd AdvGrads/
pip install -r requirements.txt
```

### Installing AdvGrads
Install AdvGrads in editable mode from source code:

```bash
python -m pip install -e .
```

## 🚀 Usage
You can execute the attack immediately using the config files provided in this repository.

```bash
python advgrads_cli/attack.py --load_config configs/mnist_untargeted.yaml
```

### ⚙ Description format of config files
The attack configs are managed by a YAML file. The main fields and variables are described below.

- `data`: _(str, required)_ Specify a dataset for which adversarial examples are to be generated.
- `model`: _(str, required)_ Select a model to be attacked. See [here](https://github.com/myuito3/AdvGrads/blob/main/advgrads/models/__init__.py) for currently supported models.
- `attacks`: _(list, required)_ This field allows you to specify attack methods that you wish to execute in a list format. You can set values including hyperparameters defined for each method. The parameters that can be specified for all methods are as follows:
  - `method`: _(str)_ Attack method. See [here](https://github.com/myuito3/AdvGrads/blob/main/advgrads/adversarial/__init__.py) for currently supported attack methods.
  - `norm`: _(str)_ Norm for adversarial perturbations.
  - `eps`: _(float)_ Maximum norm constraint.
  - `max_iters`: _(int)_ Maximum number of iterations used in iterative methods.
  - `targeted`: _(bool)_ Whether or not to perform targeted attacks which aim to misclassify an adversarial example into a particular class.
- `thirdparty_defense`: _(str, optional)_ Thirdparty defensive method. See [here](https://github.com/myuito3/AdvGrads/blob/main/advgrads/adversarial/__init__.py) for currently supported defensive methods.
