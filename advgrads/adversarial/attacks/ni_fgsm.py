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

"""Implementation of the NI-FGSM attack.

Paper: Nesterov Accelerated Gradient and Scale Invariance for Adversarial Attacks
Url: https://arxiv.org/abs/1908.06281

Original code is referenced from https://github.com/JHL-HUST/SI-NI-FGSM
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
class NiFgsmAttackConfig(AttackConfig):
    """The configuration class for the NI-FGSM attack."""

    _target: Type = field(default_factory=lambda: NiFgsmAttack)
    """Target class to instantiate."""
    momentum: float = 1.0
    """Momentum about the model."""


class NiFgsmAttack(Attack):
    """The class of the NI-FGSM attack.

    Args:
        config: The NI-FGSM attack configuration.
    """

    config: NiFgsmAttackConfig

    def run_attack(
        self, x: Tensor, y: Tensor, model: Model
    ) -> Dict[ResultHeadNames, Tensor]:
        x_adv = x
        grad = torch.zeros_like(x).detach()
        alpha = self.eps / self.max_iters

        for _ in range(self.max_iters):
            x_adv = x_adv.clone().detach().requires_grad_(True)

            x_nes = x_adv + self.config.momentum * alpha * grad

            logits = model(x_nes)
            loss = F.cross_entropy(logits, torch.as_tensor(y, dtype=torch.long))
            model.zero_grad()
            loss.backward()
            gradients_raw = x_adv.grad.data.detach()

            if self.targeted:
                gradients_raw *= -1

            gradients_raw = gradients_raw / torch.mean(
                torch.abs(gradients_raw), dim=(1, 2, 3), keepdims=True
            )
            gradients_raw = gradients_raw + self.config.momentum * grad

            x_adv = x_adv + alpha * gradients_raw.sign()
            x_adv = x_adv.clamp(min=self.min_val, max=self.max_val)
            grad = gradients_raw.clone().detach()

        return {ResultHeadNames.X_ADV: x_adv}
