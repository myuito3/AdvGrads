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

from typing import Any, List

import torch.nn as nn
from torch import Tensor

from advgrads.models.base_model import Model


class PtPgCifar10Model(Model):
    """The model classifying CIFAR-10 dataset."""

    def __init__(self) -> None:
        super().__init__()
        n_channel = 128

        source = [
            n_channel,
            n_channel,
            "M",
            2 * n_channel,
            2 * n_channel,
            "M",
            4 * n_channel,
            4 * n_channel,
            "M",
            (8 * n_channel, 0),
            "M",
        ]
        self.features = make_layers(source, batch_norm=True)
        self.classifier = nn.Sequential(nn.Linear(8 * n_channel, 10))

    def forward(self, x_input: Tensor) -> Tensor:
        features = self.features(x_input)
        logits = self.classifier(features.view(x_input.shape[0], -1))
        return logits


def make_layers(source: List[Any], batch_norm: bool = False) -> nn.Module:
    """Make nn.Sequential layers from source.

    Args:
        source: A list of the layers to be configured.
        batch_norm: Whether to use batch normalization.
    """
    layers = []
    in_channels = 3

    for i, v in enumerate(source):
        if v == "M":
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        else:
            padding = v[1] if isinstance(v, tuple) else 1
            out_channels = v[0] if isinstance(v, tuple) else v
            conv2d = nn.Conv2d(
                in_channels, out_channels, kernel_size=3, padding=padding
            )
            if batch_norm:
                layers += [
                    conv2d,
                    nn.BatchNorm2d(out_channels, affine=False),
                    nn.ReLU(),
                ]
            else:
                layers += [conv2d, nn.ReLU()]
            in_channels = out_channels

    return nn.Sequential(*layers)
