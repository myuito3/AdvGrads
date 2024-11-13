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

"""The implementation of the Patch-wise Iterative Fast Gradient Sign Method (PI-FGSM)
attack.

Paper: Patch-wise Attack for Fooling Deep Neural Network
Url: https://arxiv.org/abs/2007.06765

Original code is referenced from
https://github.com/qilong-zhang/Patch-wise-iterative-attack/tree/master/Pytorch%20version
"""

from dataclasses import dataclass, field
from typing import Dict, List, Type

import torch
import torch.nn.functional as F
from torch import Tensor

from advgrads.adversarial.attacks.base_attack import AttackConfig, NORM_TYPE
from advgrads.adversarial.attacks.fgsm.fgsm import FgsmAttack
from advgrads.adversarial.attacks.utils.result_heads import ResultHeadNames
from advgrads.models.base_model import Model


def project_kern(kern_size: int = 3, channels: int = 3):
    """Generate a special uniform projection kernel."""
    kern = torch.ones((kern_size, kern_size)) / (kern_size**2 - 1)
    kern[kern_size // 2, kern_size // 2] = 0.0
    stack_kern = torch.stack([kern] * channels)[:, None, :, :]
    return stack_kern, kern_size // 2


def project_noise(x: Tensor, stack_kern: Tensor, padding_size: int, groups: int = 3):
    """Convolution using the project kernel."""
    return F.conv2d(x, stack_kern, padding=(padding_size, padding_size), groups=groups)


@dataclass
class PiFgsmAttackConfig(AttackConfig):
    """The configuration class for the PI-FGSM attack."""

    _target: Type = field(default_factory=lambda: PiFgsmAttack)
    """Target class to instantiate."""
    amplification: float = 10.0
    """Parameter to amplifythe step size."""


class PiFgsmAttack(FgsmAttack):
    """The class of the PI-FGSM attack.

    Args:
        config: The PI-FGSM attack configuration.
        norm_allow_list: List of supported perturbation norms.
    """

    config: PiFgsmAttackConfig
    norm_allow_list: List[NORM_TYPE] = ["l_inf"]

    def run_attack(
        self, x: Tensor, y: Tensor, model: Model
    ) -> Dict[ResultHeadNames, Tensor]:
        x_adv = x
        alpha = self.eps / self.max_iters
        alpha_beta = alpha * self.config.amplification

        c = x.shape[1]
        stack_kern, padding_size = project_kern(kern_size=3, channels=c)
        stack_kern = stack_kern.to(x.device)

        amplification = 0.0
        for _ in range(self.max_iters):
            x_adv = x_adv.clone().detach().requires_grad_(True)
            model.zero_grad()

            gradients = self.get_gradients(self.get_loss(x_adv, y, model), x_adv)

            amplification += alpha_beta * torch.sign(gradients)
            cut_noise = torch.clamp(
                abs(amplification) - self.eps, 0.0, 10000.0
            ) * torch.sign(amplification)
            projection = alpha_beta * torch.sign(
                project_noise(cut_noise, stack_kern, padding_size, groups=c)
            )
            amplification += projection

            x_adv = x_adv + alpha_beta * torch.sign(gradients) + projection
            deltas = torch.clamp(x_adv - x, min=-self.eps, max=self.eps)
            x_adv = torch.clamp(x + deltas, min=self.min_val, max=self.max_val)

        return {ResultHeadNames.X_ADV: x_adv}
