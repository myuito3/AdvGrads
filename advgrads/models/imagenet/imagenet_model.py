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

"""The base model for ImageNet."""

import os
from collections import OrderedDict
from dataclasses import dataclass, field
from typing import Tuple, Type

import torch
import torch.nn as nn
import torchvision.transforms.functional as F
from torch import Tensor
from torchvision import transforms

from advgrads.models.base_model import Model, ModelConfig


@dataclass
class ImagenetModelConfig(ModelConfig):
    """The base configuration class for the ImageNet model."""

    _target: Type = field(default_factory=lambda: ImagenetModel)
    """Target class to instantiate."""
    crop_size: int = 224
    """Size of the image to be cropped, i.e., the size of the input to the model."""
    resize_size: int = 256
    """Size of the image to resize before cropping."""
    mean: Tuple[float, ...] = (0.485, 0.456, 0.406)
    """Mean values per channel used to normalize the ImageNet image."""
    std: Tuple[float, ...] = (0.229, 0.224, 0.225)
    """Std values per channel used to normalize the ImageNet image."""


class ImagenetModel(Model):
    """The base model classifying ImageNet dataset.

    Args:
        config: The base model configuration.
    """

    config: ImagenetModelConfig
    model: nn.Module

    def __init__(self, config: ImagenetModelConfig) -> None:
        super().__init__(config)
        self.mean = list(self.config.mean)
        self.std = list(self.config.std)

    def load(self) -> None:
        if not os.path.exists(self.config.checkpoint_path):
            self.download()

        checkpoint = torch.load(self.config.checkpoint_path, map_location="cpu")
        new_checkpoint = OrderedDict()
        for key, value in checkpoint.items():
            new_key = "model." + key
            new_checkpoint[new_key] = value
        del checkpoint

        self.load_state_dict(new_checkpoint)
        self.eval()

    def forward(self, x_input: Tensor) -> Tensor:
        x_input = F.normalize(x_input, mean=self.mean, std=self.std)
        return self.model(x_input)

    def get_transform(self) -> transforms.Compose:
        transform = transforms.Compose(
            [
                transforms.Resize(self.config.resize_size),
                transforms.CenterCrop(self.config.crop_size),
                transforms.ToTensor(),
            ]
        )
        return transform
