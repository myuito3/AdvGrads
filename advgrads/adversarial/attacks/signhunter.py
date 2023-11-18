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

"""Implementation of the SignHunter attack.

Paper: Sign Bits Are All You Need for Black-Box Attacks
Url: https://openreview.net/forum?id=SygW0TEFwH
"""

from dataclasses import dataclass, field
from typing import Dict, List, Type

import numpy as np
import torch
import torch.nn as nn
from torch import Tensor

from advgrads.adversarial.attacks.base_attack import Attack, AttackConfig, NormType
from advgrads.adversarial.attacks.utils.losses import MarginLoss
from advgrads.adversarial.attacks.utils.result_heads import ResultHeadNames
from advgrads.models.base_model import Model


@dataclass
class SignHunterAttackConfig(AttackConfig):
    """The configuration class for the SignHunter attack."""

    _target: Type = field(default_factory=lambda: SignHunterAttack)
    """Target class to instantiate."""


class SignHunterAttack(Attack):
    """The class of the SignHunter attack.

    Args:
        config: The SignHunter attack configuration.
        norm_allow_list: List of supported perturbation norms.
    """

    config: SignHunterAttackConfig
    norm_allow_list: List[NormType] = ["l_inf"]

    def __init__(self, config: SignHunterAttackConfig) -> None:
        super().__init__(config)

        self.loss = (
            nn.CrossEntropyLoss(reduction="none")
            if self.targeted
            else MarginLoss(targeted=self.targeted)
        )
        self.margin = MarginLoss(self.targeted)

    @torch.no_grad()
    def run_attack(
        self, x: Tensor, y: Tensor, model: Model
    ) -> Dict[ResultHeadNames, Tensor]:
        n_dim = np.prod(x.shape[1:])
        n_queries = torch.zeros((x.shape[0]), dtype=torch.int16).to(x.device)

        deltas = torch.ones((x.shape[0], n_dim), device=x.device)
        x_adv = torch.clamp(
            x + self.eps * deltas.view(*x.shape), self.min_val, self.max_val
        )
        logits = model(x_adv)
        loss_min = self.loss(logits, y)
        margin_min = self.margin(logits, y)
        n_queries += 1

        i = 0
        h = 0

        for _ in range(self.max_iters - 1):
            idx_to_fool = torch.atleast_1d((margin_min > 0.0).nonzero().squeeze())
            if len(idx_to_fool) == 0:
                break

            # Extract the images in which the adversarial sample should be found.
            x_curr = x[idx_to_fool]
            x_adv_curr = x_adv[idx_to_fool]
            deltas_curr = deltas[idx_to_fool]
            y_curr = y[idx_to_fool]
            margin_min_curr = margin_min[idx_to_fool]
            loss_min_curr = loss_min[idx_to_fool]

            # Generate candidates for new adversarial examples
            chunk_len = np.ceil(n_dim / (2**h)).astype(int)
            istart = i * chunk_len
            iend = min(n_dim, (i + 1) * chunk_len)

            deltas_new = deltas_curr.clone()
            deltas_new[:, istart:iend] *= -1.0
            x_new = torch.clamp(
                x_curr + self.eps * deltas_new.view(*x_curr.shape),
                self.min_val,
                self.max_val,
            )
            logits = model(x_new)
            loss = self.loss(logits, y_curr)
            margin = self.margin(logits, y_curr)

            # Update current loss values and adversarial examples
            idx_improved = loss < loss_min_curr
            loss_min[idx_to_fool] = idx_improved * loss + ~idx_improved * loss_min_curr
            margin_min[idx_to_fool] = (
                idx_improved * margin + ~idx_improved * margin_min_curr
            )

            idx_improved = torch.reshape(idx_improved, [-1, *[1] * len(x.shape[:-1])])
            x_adv[idx_to_fool] = idx_improved * x_new + ~idx_improved * x_adv_curr
            idx_improved = torch.reshape(
                idx_improved.squeeze(), [-1, *[1] * len(deltas.shape[:-1])]
            )
            deltas[idx_to_fool] = (
                idx_improved * deltas_new + ~idx_improved * deltas_curr
            )
            n_queries[idx_to_fool] += 1

            # Update i and h for next iteration.
            i += 1
            if i == 2**h or iend == n_dim:
                h += 1
                i = 0
                if h == np.ceil(np.log2(n_dim)).astype(int) + 1:
                    h = 0

        return {ResultHeadNames.X_ADV: x_adv, ResultHeadNames.QUERIES: n_queries}
