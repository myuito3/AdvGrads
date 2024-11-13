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

"""The implementation of the Projected Gradient Descent (PGD) attack.

Paper: Towards Deep Learning Models Resistant to Adversarial Attacks
Url: https://arxiv.org/abs/1706.06083
"""

from dataclasses import dataclass, field
from typing import Dict, List, Type

import torch
import torch.nn.functional as F
from torch import Tensor

from advgrads.adversarial.attacks.base_attack import AttackConfig, NORM_TYPE
from advgrads.adversarial.attacks.fgsm.i_fgsm import IFgsmAttack
from advgrads.adversarial.attacks.utils.result_heads import ResultHeadNames
from advgrads.models.base_model import Model


@dataclass
class PGDAttackConfig(AttackConfig):
    """The configuration class for the PGD attack."""

    _target: Type = field(default_factory=lambda: PGDAttack)
    """Target class to instantiate."""


class PGDAttack(IFgsmAttack):
    """The class of the PGD attack.

    Args:
        config: The PGD attack configuration.
        norm_allow_list: List of supported perturbation norms.
    """

    config: PGDAttackConfig
    norm_allow_list: List[NORM_TYPE] = ["l_inf"]

    def run_attack(
        self, x: Tensor, y: Tensor, model: Model
    ) -> Dict[ResultHeadNames, Tensor]:
        init_deltas = torch.empty_like(x).uniform_(-self.eps, self.eps)
        x_adv = torch.clamp(x + init_deltas, min=self.min_val, max=self.max_val)

        for _ in range(self.max_iters):
            x_adv = x_adv.clone().detach().requires_grad_(True)
            model.zero_grad()

            gradients = self.get_gradients(self.get_loss(x_adv, y, model), x_adv)

            x_adv = x_adv + self.alpha * torch.sign(gradients)
            deltas = torch.clamp(x_adv - x, min=-self.eps, max=self.eps)
            x_adv = torch.clamp(x + deltas, min=self.min_val, max=self.max_val)

        return {ResultHeadNames.X_ADV: x_adv}
