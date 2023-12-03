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

"""The implementation of the JPEG Compression defense.

Paper: A study of the effect of JPG compression on adversarial images
Url: https://arxiv.org/abs/1608.00853
"""

from dataclasses import dataclass, field
from io import BytesIO
from typing import Type

import torch
import torchvision.transforms.functional as F
from PIL import Image
from torch import Tensor

from advgrads.adversarial.defenses.input_transform.base_defense import (
    Defense,
    DefenseConfig,
)


@dataclass
class JpegCompressionDefenseConfig(DefenseConfig):
    """The configuration class for the JPEG Compression defense."""

    _target: Type = field(default_factory=lambda: JpegCompressionDefense)
    """Target class to instantiate."""
    quality: int = 75
    """The compressed image quality."""


class JpegCompressionDefense(Defense):
    """The class of the JPEG Compression defense.

    Args:
        config: The JPEG Compression defense configuration.
    """

    config: JpegCompressionDefenseConfig

    def run_defense(self, x: Tensor) -> Tensor:
        x_defended = torch.zeros_like(x, device=x.device)

        for i_img in range(x.shape[0]):
            x_i_pil = F.to_pil_image(x[i_img].detach().clone().cpu())
            buffer = BytesIO()
            x_i_pil.save(buffer, format="JPEG", quality=self.config.quality)
            x_defended[i_img] = F.to_tensor(Image.open(buffer)).to(x.device)

        return x_defended
