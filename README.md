<div align="center">

# AdvGrads

</div>

## 🌐 About
AdvGrads is an all-in-one tool for comprehensive experimentation with adversarial attacks on image recognition. This provides an environment used for research purposes to validate the performance of attack and defense methods in adversarial attacks.

This repository is still under development. For more information, please contact with me (m.yuito3@gmail.com).

## 🔍 Features
AdvGrads is developed on `PyTorch`.

### 💥 Attacks
Currently supported attack methods are as follows:

| Method              | Type                | References          |
| :------------------ | :------------------ | :------------------ |
| FGSM                | White-box           | 📃[Explaining and Harnessing Adversarial Examples](https://arxiv.org/abs/1412.6572) |
| I-FGSM (BIM)        | White-box           | 📃[Adversarial examples in the physical world](https://arxiv.org/abs/1607.02533) |
| MI-FGSM (MIM)       | White-box           | 📃[Boosting Adversarial Attacks with Momentum](https://arxiv.org/abs/1710.06081) |
| Square attack       | Black-box           | 📃[Square Attack: a query-efficient black-box adversarial attack via random search](https://arxiv.org/abs/1912.00049) |

### 💠 Defenses
Currently supported defense methods including adversarially trained models are as follows:

| Method              | Type                | References          |
| :------------------ | :------------------ | :------------------ |
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
pip install -r requirements.txt
```

### Installing AdvGrads
Install AdvGrads in editable mode from source code:

```bash
git clone https://github.com/myuito3/AdvGrads.git
python -m pip install -e .
```

## 🚀 Usage
The attack configs are managed by a YAML file, which can be used as input to call `attack.py` to execute the attack.

```bash
python advgrads_cli/attack.py --load_config configs/mnist.yaml
```
