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

"""The VGG model."""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Type

from torchvision.models import vgg16, vgg16_bn

from advgrads.models.imagenet.imagenet_model import ImagenetModel, ImagenetModelConfig


@dataclass
class Vgg16ImagenetModelConfig(ImagenetModelConfig):
    """The configuration class for the VGG-16 model."""

    _target: Type = field(default_factory=lambda: Vgg16ImagenetModel)
    """Target class to instantiate."""
    checkpoint_path: Path = Path("checkpoints/imagenet/vgg/vgg16-397923af.pth")
    """Path to the checkpoint file to be loaded."""
    download_url: Optional[
        str
    ] = "https://download.pytorch.org/models/vgg16-397923af.pth"
    """URL to download the checkpoint file if it is not found."""


class Vgg16ImagenetModel(ImagenetModel):
    """The VGG-16 model classifying ImageNet dataset.

    Args:
        config: The VGG-16 model configuration.
    """

    config: Vgg16ImagenetModelConfig

    def __init__(self, config: Vgg16ImagenetModelConfig) -> None:
        super().__init__(config)
        self.model = vgg16(weights=None)


@dataclass
class Vgg16bnImagenetModelConfig(ImagenetModelConfig):
    """The configuration class for the VGG-16-BN model."""

    _target: Type = field(default_factory=lambda: Vgg16bnImagenetModel)
    """Target class to instantiate."""
    checkpoint_path: Path = Path("checkpoints/imagenet/vgg/vgg16_bn-6c64b313.pth")
    """Path to the checkpoint file to be loaded."""
    download_url: Optional[
        str
    ] = "https://download.pytorch.org/models/vgg16_bn-6c64b313.pth"
    """URL to download the checkpoint file if it is not found."""


class Vgg16bnImagenetModel(ImagenetModel):
    """The VGG-16-BN model classifying ImageNet dataset.

    Args:
        config: The VGG-16-BN model configuration.
    """

    config: Vgg16bnImagenetModelConfig

    def __init__(self, config: Vgg16bnImagenetModelConfig) -> None:
        super().__init__(config)
        self.model = vgg16_bn(weights=None)
