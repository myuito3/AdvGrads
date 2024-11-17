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

"""The implementation of the Iterative Fast Gradient Sign Method (I-FGSM) attack. This
method is also called Basic Iterative Method (BIM).

Paper: Adversarial examples in the physical world
Url: https://arxiv.org/abs/1607.02533
"""

from dataclasses import dataclass, field
from typing import List, Type

import torch
from torch import Tensor

from advgrads.adversarial.attacks.base_attack import AttackConfig, NORM_TYPE
from advgrads.adversarial.attacks.fgsm.fgsm import FgsmAttack
from advgrads.adversarial.attacks.utils.types import AttackOutputs
from advgrads.models.base_model import Model


@dataclass
class IFgsmAttackConfig(AttackConfig):
    """The configuration class for the I-FGSM attack."""

    _target: Type = field(default_factory=lambda: IFgsmAttack)
    """Target class to instantiate."""


class IFgsmAttack(FgsmAttack):
    """The class of the I-FGSM attack.

    Args:
        config: The I-FGSM attack configuration.
        norm_allow_list: List of supported perturbation norms.
    """

    config: IFgsmAttackConfig
    norm_allow_list: List[NORM_TYPE] = ["l_inf"]

    @property
    def alpha(self) -> float:
        return self.eps / self.max_iters

    def run_attack(self, x: Tensor, y: Tensor, model: Model) -> AttackOutputs:
        x_adv = x

        for _ in range(self.max_iters):
            x_adv = x_adv.clone().detach().requires_grad_(True)
            model.zero_grad()

            gradients = self.get_gradients(self.get_loss(x_adv, y, model), x_adv)
            x_adv = x_adv + self.alpha * torch.sign(gradients)
            x_adv = torch.clamp(x_adv, min=self.min_val, max=self.max_val)

        return AttackOutputs(x_adv=x_adv)
