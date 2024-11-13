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

"""The implementation of the Momentum Iterative Fast Gradient Sign Method (MI-FGSM)
attack. This method is also called Momentum Iterative Method (MIM).

Paper: Boosting Adversarial Attacks with Momentum
Url: https://arxiv.org/abs/1710.06081

Original code is referenced from
https://github.com/dongyp13/Non-Targeted-Adversarial-Attacks
"""

from dataclasses import dataclass, field
from typing import Dict, List, Type

import torch
from torch import Tensor

from advgrads.adversarial.attacks.base_attack import AttackConfig, NORM_TYPE
from advgrads.adversarial.attacks.fgsm.i_fgsm import IFgsmAttack
from advgrads.adversarial.attacks.utils.result_heads import ResultHeadNames
from advgrads.models.base_model import Model


@dataclass
class MiFgsmAttackConfig(AttackConfig):
    """The configuration class for the MI-FGSM attack."""

    _target: Type = field(default_factory=lambda: MiFgsmAttack)
    """Target class to instantiate."""
    momentum: float = 1.0
    """Momentum about the model."""


class MiFgsmAttack(IFgsmAttack):
    """The class of the MI-FGSM attack.

    Args:
        config: The MI-FGSM attack configuration.
        norm_allow_list: List of supported perturbation norms.
    """

    config: MiFgsmAttackConfig
    norm_allow_list: List[NORM_TYPE] = ["l_inf"]

    def run_attack(
        self, x: Tensor, y: Tensor, model: Model
    ) -> Dict[ResultHeadNames, Tensor]:
        x_adv = x
        accumulated_grads = torch.zeros_like(x)

        for _ in range(self.max_iters):
            x_adv = x_adv.clone().detach().requires_grad_(True)
            model.zero_grad()

            gradients = self.get_gradients(self.get_loss(x_adv, y, model), x_adv)

            gradients = gradients / torch.mean(
                torch.abs(gradients), dim=(1, 2, 3), keepdims=True
            )
            gradients += self.config.momentum * accumulated_grads
            accumulated_grads = gradients.clone().detach()

            x_adv = x_adv + self.alpha * torch.sign(gradients)
            x_adv = torch.clamp(x_adv, min=self.min_val, max=self.max_val)

        return {ResultHeadNames.X_ADV: x_adv}
