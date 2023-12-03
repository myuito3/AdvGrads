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

"""The implementation of the Bit-Depth Reduction defense.

Paper: Feature Squeezing: Detecting Adversarial Examples in Deep Neural Networks
Url: https://arxiv.org/abs/1704.01155
"""

from dataclasses import dataclass, field
from typing import Type

from torch import Tensor

from advgrads.adversarial.defenses.input_transform.base_defense import (
    Defense,
    DefenseConfig,
)


@dataclass
class BitDepthReductionDefenseConfig(DefenseConfig):
    """The configuration class for the Bit-Depth Reduction defense."""

    _target: Type = field(default_factory=lambda: BitDepthReductionDefense)
    """Target class to instantiate."""
    num_bits: int = 4
    """Number of bits after squeezing."""


class BitDepthReductionDefense(Defense):
    """The class of the Bit-Depth Reduction defense.

    Args:
        config: The Bit-Depth Reduction defense configuration.
    """

    config: BitDepthReductionDefenseConfig

    def run_defense(self, x: Tensor) -> Tensor:
        max_val_squeezed = 2**self.config.num_bits
        x_defended = (x.detach().clone() * max_val_squeezed).int()
        x_defended = x_defended / max_val_squeezed
        return x_defended
