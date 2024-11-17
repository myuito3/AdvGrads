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

"""The implementation of the Diverse Inputs Momentum Iterative Fast Gradient Sign
Method (DI-MI-FGSM) attack. This method is referred to as Momentum Diverse Inputs
Iterative Fast Gradient Sign Method (M-DI2-FGSM) in the original paper.

Paper: Improving Transferability of Adversarial Examples with Input Diversity
Url: https://arxiv.org/abs/1803.06978

Original code is referenced from https://github.com/cihangxie/DI-2-FGSM
"""

import random
from dataclasses import dataclass, field
from typing import List, Type

import torch
import torch.nn.functional as F
from torch import Tensor

from advgrads.adversarial.attacks.base_attack import AttackConfig, NORM_TYPE
from advgrads.adversarial.attacks.fgsm.i_fgsm import IFgsmAttack
from advgrads.adversarial.attacks.utils.types import AttackOutputs
from advgrads.models.base_model import Model


@dataclass
class DiMiFgsmAttackConfig(AttackConfig):
    """The configuration class for the DI-MI-FGSM attack."""

    _target: Type = field(default_factory=lambda: DiMiFgsmAttack)
    """Target class to instantiate."""
    max_resolution_ratio: float = 1.104
    """Ratio of the length of one side of the transformed image to one of the original
    image. The default value is calculated w.r.t the ImageNet setting mentioned in the
    original paper (330/299 = 1.1036)."""
    keep_dims: bool = True
    """Whether to keep the original image size."""
    prob: float = 0.5
    """Probability of using diverse inputs."""
    momentum: float = 1.0
    """Momentum about the model."""


class DiMiFgsmAttack(IFgsmAttack):
    """The class of the DI-MI-FGSM attack.

    Args:
        config: The DI-MI-FGSM attack configuration.
        norm_allow_list: List of supported perturbation norms.
    """

    config: DiMiFgsmAttackConfig
    norm_allow_list: List[NORM_TYPE] = ["l_inf"]

    def apply_input_diversity(self, x: Tensor) -> Tensor:
        """Apply diverse input patterns, i.e., random transformations, on the input
        image x.

        Args:
            x: Images to be transformed.
        """
        if torch.rand(1) > self.config.prob:
            return x

        h, w = x.shape[2:]
        h_final = int(h * self.config.max_resolution_ratio)
        w_final = int(w * self.config.max_resolution_ratio)

        # 1. random resize
        h_resize = random.randint(h, h_final - 1)
        w_resize = random.randint(w, w_final - 1)
        x_resize = F.interpolate(x, size=[h_resize, w_resize], mode="nearest")

        # 2. random padding
        h_remain = h_final - h_resize
        w_remain = w_final - w_resize
        pad_top = random.randint(0, h_remain)
        pad_left = random.randint(0, w_remain)
        dim = [pad_left, w_remain - pad_left, pad_top, h_remain - pad_top]
        x_pad = F.pad(x_resize, dim, mode="constant", value=0)

        assert x_pad.shape[2:] == (h_final, w_final)
        if self.config.keep_dims:
            x_pad = F.interpolate(x_pad, size=[h, w], mode="nearest")

        return x_pad

    def run_attack(self, x: Tensor, y: Tensor, model: Model) -> AttackOutputs:
        x_adv = x
        accumulated_grads = torch.zeros_like(x)

        for _ in range(self.max_iters):
            x_adv = x_adv.clone().detach().requires_grad_(True)
            model.zero_grad()

            gradients = self.get_gradients(
                self.get_loss(self.apply_input_diversity(x_adv), y, model), x_adv
            )

            gradients = gradients / torch.mean(
                torch.abs(gradients), dim=(1, 2, 3), keepdims=True
            )
            gradients += self.config.momentum * accumulated_grads
            accumulated_grads = gradients.clone().detach()

            x_adv = x_adv + self.alpha * torch.sign(gradients)
            x_adv = torch.clamp(x_adv, min=self.min_val, max=self.max_val)

        return AttackOutputs(x_adv=x_adv)
