# Copyright 2023 Makoto Yuito. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Base class for attack methods."""

from abc import abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List, Literal, Optional, Type

import torch
from torch import Tensor

from advgrads.adversarial.attacks.utils.result_heads import ResultHeadNames
from advgrads.adversarial.defenses.input_transform.base_defense import Defense
from advgrads.configs.base_config import InstantiateConfig
from advgrads.models.base_model import Model


NormType = Literal["l_0", "l_2", "l_inf"]


@dataclass
class AttackConfig(InstantiateConfig):
    """The base configuration class for attack methods."""

    _target: Type = field(default_factory=lambda: Attack)
    """Target class to instantiate."""
    targeted: bool = False
    """Whether or not to perform targeted attacks."""
    min_val: float = 0.0
    """Min value of image used to clip perturbed images."""
    max_val: float = 1.0
    """Max value of image used to clip perturbed images."""
    norm: Optional[NormType] = None
    """Norm bound of adversarial perturbations."""
    eps: float = 0.0
    """Radius of a l_p ball."""
    max_iters: int = 0
    """Max number of iterations to search an adversarial example."""


class Attack:
    """The base class for attack methods.

    Args:
        config: Configuration for attack methods.
        norm_allow_list: List of supported perturbation norms. Each method defines this
            within its own class.
    """

    config: AttackConfig
    norm_allow_list: List[NormType]

    def __init__(self, config: AttackConfig, **kwargs) -> None:
        self.config = config

        if self.eps < 0:
            raise ValueError(f"eps must be greater than or equal to 0, got {self.eps}.")
        if self.max_iters < 0:
            raise ValueError(
                f"max_iters must be greater than or equal to 0, got {self.max_iters}."
            )
        if self.norm not in self.norm_allow_list:
            raise ValueError(f"Method does not support {self.norm} perturbation norm.")

    def __call__(self, *args: Any, **kwargs: Any) -> Dict[ResultHeadNames, Any]:
        return self.get_outputs(*args, **kwargs)

    @property
    def targeted(self) -> bool:
        return self.config.targeted

    @property
    def min_val(self) -> float:
        return self.config.min_val

    @property
    def max_val(self) -> float:
        return self.config.max_val

    @property
    def norm(self) -> str:
        return self.config.norm

    @property
    def eps(self) -> float:
        return self.config.eps

    @property
    def max_iters(self) -> int:
        return self.config.max_iters

    @abstractmethod
    def run_attack(
        self, x: Tensor, y: Tensor, model: Model, **kwargs
    ) -> Dict[ResultHeadNames, Tensor]:
        """Run the attack to search adversarial examples.

        Args:
            x: Images to be searched for adversarial examples.
            y: Ground truth labels of images.
            model: A model under attack.
        """
        raise NotImplementedError

    def get_outputs(
        self,
        x: Tensor,
        y: Tensor,
        model: Model,
        thirdparty_defense: Optional[Defense] = None,
        **kwargs,
    ) -> Dict[ResultHeadNames, Any]:
        """Returns raw attack results processed.

        Args:
            x: Images to be searched for adversarial examples.
            y: Ground truth labels of images.
            model: A model to be attacked.
            thirdparty_defense: Thirdparty defense method instance.
        """
        attack_outputs = self.run_attack(x, y, model, **kwargs)
        self.sanity_check(x, attack_outputs[ResultHeadNames.X_ADV])

        # If a defensive method is defined, the process is performed here. This
        # corresponds to Section 5.2 (GRAY BOX: IMAGE TRANSFORMATIONS AT TEST TIME) in
        # the paper of Guo et al.
        if thirdparty_defense is not None:
            attack_outputs[ResultHeadNames.X_ADV] = thirdparty_defense(
                attack_outputs[ResultHeadNames.X_ADV]
            )

        with torch.no_grad():
            logits = model(attack_outputs[ResultHeadNames.X_ADV])
        preds = torch.argmax(logits, dim=-1)
        cond = (preds == y) if self.targeted else (preds != y)
        attack_outputs[ResultHeadNames.NUM_SUCCEED] = cond.sum()

        if ResultHeadNames.QUERIES in attack_outputs.keys():
            attack_outputs[ResultHeadNames.QUERIES_SUCCEED] = attack_outputs[
                ResultHeadNames.QUERIES
            ][cond]

        for key, value in attack_outputs.items():
            if isinstance(value, Tensor):
                attack_outputs[key] = value.cpu()
        return attack_outputs

    def sanity_check(self, x: Tensor, x_adv: Tensor) -> None:
        """Ensure that the amount of perturbation is properly controlled. This method
        is specifically used to check the amount of perturbation of norm-constrained
        type attack methods.

        Args:
            x: Original images.
            x_adv: Perturbed images.
        """
        if self.eps > 0.0:
            deltas = x_adv - x
            if self.norm == "l_inf":
                real = (
                    deltas.abs().max().half()
                )  # ignore slight differences within the decimal point
                msg = f"Perturbations beyond the l_inf sphere ({real})."
            elif self.norm == "l_2":
                real = torch.norm(deltas.view(x.shape[0], -1), p=2, dim=-1).max()
                msg = f"Perturbations beyond the l_2 sphere ({real})."
            elif self.norm == "l_0":
                raise NotImplementedError

            assert real <= self.eps, msg
