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

"""The Inception model."""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Type

from torchvision.models import inception_v3

from advgrads.models.imagenet.imagenet_model import ImagenetModel, ImagenetModelConfig


@dataclass
class InceptionV3ImagenetModelConfig(ImagenetModelConfig):
    """The configuration class for the Inception-v3 model."""

    _target: Type = field(default_factory=lambda: InceptionV3ImagenetModel)
    """Target class to instantiate."""
    checkpoint_path: Path = Path(
        "checkpoints/imagenet/inception/inception_v3_google-0cc3c7bd.pth"
    )
    """Path to the checkpoint file to be loaded."""
    download_url: Optional[
        str
    ] = "https://download.pytorch.org/models/inception_v3_google-0cc3c7bd.pth"
    """URL to download the checkpoint file if it is not found."""
    crop_size: int = 299
    """Size of the image to be cropped, i.e., the size of the input to the model."""
    resize_size: int = 342
    """Size of the image to resize before cropping."""


class InceptionV3ImagenetModel(ImagenetModel):
    """The Inception-v3 model classifying ImageNet dataset.

    Args:
        config: The Inception-v3 model configuration.
    """

    config: InceptionV3ImagenetModelConfig

    def __init__(self, config: InceptionV3ImagenetModelConfig) -> None:
        super().__init__(config)
        self.model = inception_v3(weights=None)
