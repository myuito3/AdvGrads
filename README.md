<div align="center">

# AdvGrads

[![Latest Release](https://img.shields.io/github/release/myuito3/AdvGrads.svg?&color=blue)](https://github.com/myuito3/AdvGrads/releases)
[![License](https://img.shields.io/badge/License-Apache_2.0-green.svg)](https://github.com/myuito3/AdvGrads/blob/master/LICENSE)

</div>

## 🌐 About
AdvGrads is an all-in-one tool for comprehensive experimentation with adversarial attacks on image recognition. This repository provides an environment used for research purposes to validate the performance of attack and defense methods in adversarial attacks.

This repository is still under development. For more information, please contact with me (m.yuito3@gmail.com).

## 💻 Installation

### Create environment
AdvGrads requires `Python >= 3.9`. The following is an example of building an environment using conda:

```bash
conda create --name advgrads -y python=3.9
conda activate advgrads
pip install --upgrade pip
```

### Dependencies
Install other packages including PyTorch with CUDA (this repo has been tested with CUDA 11.8).

```bash
pip install torch==2.1.2+cu118 torchvision==0.16.2+cu118 --extra-index-url https://download.pytorch.org/whl/cu118
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

## 🔍 Features
AdvGrads is developed on `PyTorch`.

### 💥 Attacks
Currently supported attack methods are as follows:

| Method                                                        | Type                |
| :------------------------------------------------------------ | :------------------ |
| [DeepFool](https://arxiv.org/abs/1511.04599)                  | White-box           |
| [DI-MI-FGSM](https://arxiv.org/abs/1803.06978)                | White-box           |
| [FGSM](https://arxiv.org/abs/1412.6572)                       | White-box           |
| [I-FGSM (BIM)](https://arxiv.org/abs/1607.02533)              | White-box           |
| [MI-FGSM (MIM)](https://arxiv.org/abs/1710.06081)             | White-box           |
| [NI-FGSM](https://arxiv.org/abs/1908.06281)                   | White-box           |
| [PGD](https://arxiv.org/abs/1706.06083)                       | White-box           |
| [PI-FGSM](https://arxiv.org/abs/2007.06765)                   | White-box           |
| [SI-NI-FGSM](https://arxiv.org/abs/1908.06281)                | White-box           |
| [SignHunter](https://openreview.net/forum?id=SygW0TEFwH)      | Black-box           |
| [SimBA](https://arxiv.org/abs/1905.07121)                     | Black-box           |
| [Square attack](https://arxiv.org/abs/1912.00049)             | Black-box           |

### 💠 Defenses
Currently supported defense methods including adversarially trained models are as follows:

| Method                                                  | Type                |
| :------------------------------------------------------ | :------------------ |
| [Bit-Red](https://arxiv.org/abs/1704.01155)             | Input transform     |
| [JPEG](https://arxiv.org/abs/1608.00853)                | Input transform     |
| [Randomization](https://arxiv.org/abs/1711.01991)       | Input transform     |
| [TRADES](https://arxiv.org/abs/1901.08573)              | Adv. training       |

### 🧩 Others
And also, some naturally trained models are supported.

| Source                                                                    | Datasets            |
| :------------------------------------------------------------------------ | :------------------ |
| [pytorch-playground](https://github.com/aaron-xichen/pytorch-playground)  | MNIST, CIFAR-10     |
