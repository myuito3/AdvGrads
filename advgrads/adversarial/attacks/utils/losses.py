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

"""Collection of loss functions."""

from abc import abstractmethod

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


class Loss(nn.Module):
    """Base class for a loss function.

    Args:
        targeted: Whether or not to perform targeted attacks.
    """

    def __init__(self, targeted: bool = False) -> None:
        super().__init__()
        self.targeted = targeted

    @abstractmethod
    def forward(self, logits: Tensor, y: Tensor) -> Tensor:
        """Calculates loss values.

        Args:
            logits: Logits values outputed by victim model.
            y: Ground truth or target labels.
        """


class CrossEntropyLoss(Loss):
    """Implementation of the cross-entropy loss.

    Args:
        targeted: Whether or not to perform targeted attacks.
    """

    def forward(self, logits: Tensor, y: Tensor) -> Tensor:
        loss = F.cross_entropy(logits, y, reduction="none")
        if not self.targeted:
            loss = loss * -1
        return loss


class MarginLoss(Loss):
    """Implementation of the margin loss (difference between the correct and 2nd best
    class).

    Args:
        targeted: Whether or not to perform targeted attacks.
    """

    def forward(self, logits: Tensor, y: Tensor) -> Tensor:
        logits = logits.clone().detach()
        u = torch.arange(y.shape[0])

        probs_correct = logits[u, y].clone()
        logits[u, y] = -torch.inf
        probs_second_best = logits.max(dim=-1).values
        loss = probs_correct - probs_second_best

        loss = loss * -1 if self.targeted else loss
        return loss
