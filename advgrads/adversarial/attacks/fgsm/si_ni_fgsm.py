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

"""The implementation of the Scale-Invariant Nesterov Iterative Fast Gradient Sign
Method (SI-NI-FGSM) attack.

Paper: Nesterov Accelerated Gradient and Scale Invariance for Adversarial Attacks
Url: https://arxiv.org/abs/1908.06281

Original code is referenced from https://github.com/JHL-HUST/SI-NI-FGSM
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Type

import torch
import torch.nn.functional as F
from torch import Tensor

from advgrads.adversarial.attacks.base_attack import AttackConfig, NORM_TYPE
from advgrads.adversarial.attacks.fgsm.i_fgsm import IFgsmAttack
from advgrads.adversarial.attacks.utils.result_heads import ResultHeadNames
from advgrads.models.base_model import Model


@dataclass
class SiNiFgsmAttackConfig(AttackConfig):
    """The configuration class for the SI-NI-FGSM attack."""

    _target: Type = field(default_factory=lambda: SiNiFgsmAttack)
    """Target class to instantiate."""
    momentum: float = 1.0
    """Momentum about the model."""
    num_scale_copies: int = 5
    """Number of scale copies."""


class SiNiFgsmAttack(IFgsmAttack):
    """The class of the SI-NI-FGSM attack.

    Args:
        config: The SI-NI-FGSM attack configuration.
        norm_allow_list: List of supported perturbation norms.
    """

    config: SiNiFgsmAttackConfig
    norm_allow_list: List[NORM_TYPE] = ["l_inf"]

    def scale_invariant(
        self,
        x: Tensor,
        y: Tensor,
        model: Model,
        x_to_be_scaled: Optional[Tensor] = None,
    ):
        """Function that returns the mean of the gradients of multiple scale copies of
        the input image by the Scale-Invariant Method (SIM).

        Args:
            x: Images referenced when calculating the gradient.
            y: Labels of images.
            model: Model to be attacked.
            x_to_be_scaled: Images used for scale copies; if None, this will be the
                same as x.
        """
        if x_to_be_scaled is None:
            x_to_be_scaled = x
        gradients = torch.zeros_like(x, device=x.device)

        for i in range(self.config.num_scale_copies):
            x_scaled = 1 / (2**i) * x_to_be_scaled
            gradients += self.get_gradients(self.get_loss(x_scaled, y, model), x)

        return gradients / self.config.num_scale_copies

    def run_attack(
        self, x: Tensor, y: Tensor, model: Model
    ) -> Dict[ResultHeadNames, Tensor]:
        x_adv = x
        accumulated_grads = torch.zeros_like(x)

        for _ in range(self.max_iters):
            x_adv = x_adv.clone().detach().requires_grad_(True)
            model.zero_grad()

            x_nes = x_adv + self.config.momentum * self.alpha * accumulated_grads
            gradients = self.scale_invariant(x_adv, y, model, x_to_be_scaled=x_nes)

            gradients /= torch.mean(torch.abs(gradients), dim=(1, 2, 3), keepdims=True)
            gradients += self.config.momentum * accumulated_grads
            accumulated_grads = gradients.clone().detach()

            x_adv = x_adv + self.alpha * torch.sign(gradients)
            x_adv = torch.clamp(x_adv, min=self.min_val, max=self.max_val)

        return {ResultHeadNames.X_ADV: x_adv}
