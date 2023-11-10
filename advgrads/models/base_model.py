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

"""Base model class."""

import os
import requests
from abc import abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Type

import torch
import torch.nn as nn
from torch import Tensor

from advgrads.configs.base_config import InstantiateConfig


@dataclass
class ModelConfig(InstantiateConfig):
    """Configuration for the base model instantiation."""

    _target: Type = field(default_factory=lambda: Model)
    """Target class to instantiate."""
    checkpoint_path: Path = Path()
    """Path to the checkpoint file to be loaded."""
    download_url: Optional[str] = None
    """URL to download the checkpoint file if it is not found."""


class Model(nn.Module):
    """Base model class for PyTorch.

    Args:
        config: The base model configuration.
    """

    config: ModelConfig

    def __init__(self, config: ModelConfig, **kwargs) -> None:
        super().__init__()
        self.config = config

    @abstractmethod
    def forward(self, x_input: Tensor) -> Tensor:
        """Query the model and obtain logits output.

        Args:
            x_input: Images to be input to the model.
        """
        raise NotImplementedError

    def load(self) -> None:
        """The function to load checkpoint."""
        if not os.path.exists(self.config.checkpoint_path):
            self.download()

        checkpoint = torch.load(self.config.checkpoint_path, map_location="cpu")
        self.load_state_dict(checkpoint)
        self.eval()

    def download(self) -> None:
        """The function to download the checkpoint file."""
        assert self.config.download_url is not None

        data = requests.get(self.config.download_url).content
        os.makedirs(os.path.dirname(self.config.checkpoint_path), exist_ok=True)
        with open(self.config.checkpoint_path, mode="wb") as file:
            file.write(data)
