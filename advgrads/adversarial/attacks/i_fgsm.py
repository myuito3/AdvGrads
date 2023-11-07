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

"""Implementation of the I-FGSM attack. This method is also called Basic Iterative
Method (BIM).

Paper: Adversarial examples in the physical world
Url: https://arxiv.org/abs/1607.02533
"""

from dataclasses import dataclass, field
from typing import Dict, Type

import torch
import torch.nn.functional as F
from torch import Tensor

from advgrads.adversarial.attacks.base_attack import Attack, AttackConfig
from advgrads.adversarial.attacks.utils.result_heads import ResultHeadNames
from advgrads.models.base_model import Model


@dataclass
class IFgsmAttackConfig(AttackConfig):
    """The configuration class for I-FGSM attack."""

    _target: Type = field(default_factory=lambda: IFgsmAttack)
    """Target class to instantiate."""


class IFgsmAttack(Attack):
    """The numpy implementation of the I-FGSM attack.

    Args:
        config: The I-FGSM attack configuration.
    """

    config: IFgsmAttackConfig

    def run_attack(
        self, x: Tensor, y: Tensor, model: Model
    ) -> Dict[ResultHeadNames, Tensor]:
        x_adv = x
        alpha = self.eps / self.max_iters

        for _ in range(self.max_iters):
            x_adv = x_adv.clone().detach().requires_grad_(True)

            logits = model(x_adv)
            loss = F.cross_entropy(logits, torch.as_tensor(y, dtype=torch.long))
            model.zero_grad()
            loss.backward()
            gradients_raw = x_adv.grad.data.detach()

            if self.targeted:
                gradients_raw *= -1

            x_adv = x_adv + alpha * gradients_raw.sign()
            x_adv = x_adv.clamp(min=self.min_val, max=self.max_val)

        return {ResultHeadNames.X_ADV: x_adv}
