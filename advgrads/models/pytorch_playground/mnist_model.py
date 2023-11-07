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

"""This is the model from pytorch-playground.

See their github page (https://github.com/aaron-xichen/pytorch-playground) for the
original code and trained models.
"""

from collections import OrderedDict

import torch.nn as nn
from torch import Tensor

from advgrads.models.base_model import Model


class PtPgMnistModel(Model):
    """The model classifying MNIST dataset."""

    def __init__(self) -> None:
        super().__init__()
        self.input_dims = 784

        n_hiddens = [256, 256]
        current_dims = self.input_dims
        layers = OrderedDict()

        for i, n_hidden in enumerate(n_hiddens):
            layers[f"fc{i+1}"] = nn.Linear(current_dims, n_hidden)
            layers[f"relu{i+1}"] = nn.ReLU()
            layers[f"drop{i+1}"] = nn.Dropout(0.2)
            current_dims = n_hidden
        layers["out"] = nn.Linear(current_dims, 10)

        self.model = nn.Sequential(layers)

    def forward(self, x_input: Tensor) -> Tensor:
        return self.model(x_input.view(x_input.shape[0], -1))
