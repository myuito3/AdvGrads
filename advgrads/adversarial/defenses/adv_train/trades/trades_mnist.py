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

"""The TRADES (TRadeoff-inspired Adversarial DEfense via Surrogate-loss minimization)
model.

Paper: Theoretically Principled Trade-off between Robustness and Accuracy
Url: https://arxiv.org/abs/1901.08573

See their github page (https://github.com/yaodongyu/TRADES) for the original code and
trained models.
"""

from collections import OrderedDict

import torch.nn as nn
from torch import Tensor

from advgrads.models.base_model import Model


class TradesMnistModel(Model):
    """The TRADES model classifying MNIST dataset."""

    def __init__(self) -> None:
        super().__init__()
        activ = nn.ReLU(True)

        self.feature_extractor = nn.Sequential(
            OrderedDict(
                [
                    ("conv1", nn.Conv2d(1, 32, 3)),
                    ("relu1", activ),
                    ("conv2", nn.Conv2d(32, 32, 3)),
                    ("relu2", activ),
                    ("maxpool1", nn.MaxPool2d(2, 2)),
                    ("conv3", nn.Conv2d(32, 64, 3)),
                    ("relu3", activ),
                    ("conv4", nn.Conv2d(64, 64, 3)),
                    ("relu4", activ),
                    ("maxpool2", nn.MaxPool2d(2, 2)),
                ]
            )
        )

        self.classifier = nn.Sequential(
            OrderedDict(
                [
                    ("fc1", nn.Linear(64 * 4 * 4, 200)),
                    ("relu1", activ),
                    ("drop", nn.Dropout(0.5)),
                    ("fc2", nn.Linear(200, 200)),
                    ("relu2", activ),
                    ("fc3", nn.Linear(200, 10)),
                ]
            )
        )

        for m in self.modules():
            if isinstance(m, (nn.Conv2d)):
                nn.init.kaiming_normal_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        nn.init.constant_(self.classifier.fc3.weight, 0)
        nn.init.constant_(self.classifier.fc3.bias, 0)

    def forward(self, x_input: Tensor) -> Tensor:
        features = self.feature_extractor(x_input)
        logits = self.classifier(features.contiguous().view(-1, 64 * 4 * 4))
        return logits
