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

"""Implementation of the Randomization defense.

Paper: Mitigating Adversarial Effects Through Randomization
Url: https://arxiv.org/abs/1711.01991
"""

import random
from dataclasses import dataclass, field
from typing import Type

import torch
import torch.nn.functional as F
from torch import Tensor

from advgrads.adversarial.defenses.input_transform.base_defense import (
    Defense,
    DefenseConfig,
)


@dataclass
class RandomizationDefenseConfig(DefenseConfig):
    """The configuration class for the Randomization defense."""

    _target: Type = field(default_factory=lambda: RandomizationDefense)
    """Target class to instantiate."""
    max_resolution_ratio: float = 1.11
    """Ratio of the length of one side of the transformed image to one of the original
    image. The default value is calculated w.r.t the ImageNet setting mentioned in the
    paper (331/299 = 1.107)."""
    keep_dims: bool = True
    """Whether to keep the original image size."""


class RandomizationDefense(Defense):
    """The class of the Randomization defense.

    Args:
        config: The Randomization defense configuration.
    """

    config: RandomizationDefenseConfig

    def run_defense(self, x: Tensor) -> Tensor:
        h, w = x.shape[2:]
        h_final = int(h * self.config.max_resolution_ratio)
        w_final = int(w * self.config.max_resolution_ratio)

        if self.config.keep_dims:
            x_defended = torch.zeros_like(x, device=x.device)
        else:
            x_defended = torch.zeros((*x.shape[:2], h_final, w_final), device=x.device)

        for i_img in range(x.shape[0]):
            x_i = x[i_img : i_img + 1].detach().clone()

            # 1. random resize
            h_resize = random.randint(h, h_final - 1)
            w_resize = random.randint(w, w_final - 1)
            x_i_resize = F.interpolate(x_i, size=[h_resize, w_resize], mode="nearest")

            # 2. random padding
            h_remain = h_final - h_resize
            w_remain = w_final - w_resize
            pad_top = random.randint(0, h_remain)
            pad_left = random.randint(0, w_remain)
            dim = [pad_left, w_remain - pad_left, pad_top, h_remain - pad_top]
            x_i_pad = F.pad(x_i_resize, dim, mode="constant", value=0)

            assert x_i_pad.shape[2:] == (h_final, w_final)
            if self.config.keep_dims:
                x_i_pad = F.interpolate(x_i_pad, size=[h, w], mode="nearest")

            x_defended[i_img] = x_i_pad.squeeze(0)

        return x_defended
