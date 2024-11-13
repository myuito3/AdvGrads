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

"""The implementation of the Fast Gradient Sign Method (FGSM) attack.

Paper: Explaining and Harnessing Adversarial Examples
Url: https://arxiv.org/abs/1412.6572
"""

from dataclasses import dataclass, field
from typing import Callable, Dict, List, Type

import torch
import torch.nn.functional as F
from torch import Tensor

from advgrads.adversarial.attacks.base_attack import Attack, AttackConfig, NORM_TYPE
from advgrads.adversarial.attacks.utils.result_heads import ResultHeadNames
from advgrads.models.base_model import Model


@dataclass
class FgsmAttackConfig(AttackConfig):
    """The configuration class for the FGSM attack."""

    _target: Type = field(default_factory=lambda: FgsmAttack)
    """Target class to instantiate."""


class FgsmAttack(Attack):
    """The class of the FGSM attack.

    Args:
        config: The FGSM attack configuration.
        norm_allow_list: List of supported perturbation norms.
    """

    config: FgsmAttackConfig
    norm_allow_list: List[NORM_TYPE] = ["l_inf"]

    def get_loss(
        self, x: Tensor, y: Tensor, model: Model, loss_fn: Callable = F.cross_entropy
    ) -> Tensor:
        logits = model(x)
        loss = loss_fn(logits, y)
        if self.targeted:
            loss *= -1
        return loss

    def get_gradients(self, loss: Tensor, x: Tensor) -> Tensor:
        return torch.autograd.grad(loss, [x])[0].detach()

    def run_attack(
        self, x: Tensor, y: Tensor, model: Model
    ) -> Dict[ResultHeadNames, Tensor]:
        x_adv = x.clone().detach().requires_grad_(True)
        model.zero_grad()

        gradients = self.get_gradients(self.get_loss(x_adv, y, model), x_adv)
        x_adv = x_adv + self.eps * torch.sign(gradients)
        x_adv = torch.clamp(x_adv, min=self.min_val, max=self.max_val)
        return {ResultHeadNames.X_ADV: x_adv}
