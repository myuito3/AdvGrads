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

from abc import abstractmethod

import torch
import torch.nn as nn
from torch import Tensor


class Model(nn.Module):
    """Base model class for PyTorch."""

    @abstractmethod
    def forward(self, x_input: Tensor) -> Tensor:
        """Query the model and obtain logits output.

        Args:
            x_input: Images to be input to the model.
        """
        raise NotImplementedError

    def load(self, checkpoint_path: str) -> None:
        """The function to load checkpoint.

        Args:
            checkpoint_path: Path to checkpoint file to be read.
        """
        checkpoint = torch.load(checkpoint_path, map_location="cpu")
        self.load_state_dict(checkpoint)
        self.eval()
