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

"""The ResNet model."""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Type

from torchvision.models import resnet50

from advgrads.models.imagenet.imagenet_model import ImagenetModel, ImagenetModelConfig


@dataclass
class Resnet50ImagenetModelConfig(ImagenetModelConfig):
    """The configuration class for the ResNet-50 model."""

    _target: Type = field(default_factory=lambda: Resnet50ImagenetModel)
    """Target class to instantiate."""
    checkpoint_path: Path = Path("checkpoints/imagenet/resnet/resnet50-0676ba61.pth")
    """Path to the checkpoint file to be loaded."""
    download_url: Optional[
        str
    ] = "https://download.pytorch.org/models/resnet50-0676ba61.pth"
    """URL to download the checkpoint file if it is not found."""


class Resnet50ImagenetModel(ImagenetModel):
    """The ResNet-50 model classifying ImageNet dataset.

    Args:
        config: The ResNet-50 model configuration.
    """

    config: Resnet50ImagenetModelConfig

    def __init__(self, config: Resnet50ImagenetModelConfig) -> None:
        super().__init__(config)
        self.model = resnet50(weights=None)
