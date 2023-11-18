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

"""The implementation of the Square attack.

Paper: Square Attack: a query-efficient black-box adversarial attack via random search
Url: https://arxiv.org/abs/1912.00049

Original code is referenced from https://github.com/max-andr/square-attack
"""

import math
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
class SquareAttackConfig(AttackConfig):
    """The configuration class for the Square attack."""

    _target: Type = field(default_factory=lambda: SquareAttack)
    """Target class to instantiate."""
    p_init: float = 0.05
    """Percentage of elements of x to be modified."""


class SquareAttack(Attack):
    """The class of the Square attack.

    Args:
        config: The Square attack configuration.
        norm_allow_list: List of supported perturbation norms.
    """

    config: SquareAttackConfig
    norm_allow_list: List[NormType] = ["l_inf"]

    def __init__(self, config: SquareAttackConfig) -> None:
        super().__init__(config)

        self.loss = (
            nn.CrossEntropyLoss(reduction="none")
            if self.targeted
            else MarginLoss(targeted=self.targeted)
        )
        self.margin = MarginLoss(self.targeted)

    def p_selection(self, it: int) -> float:
        """Piece-wise constant schedule for p (the fraction of pixels changed on every
        iteration).

        Args:
            it: Current iteration.
        """
        p_init = self.config.p_init
        it = int(it / self.max_iters * 10000)

        if 10 < it <= 50:
            p = p_init / 2
        elif 50 < it <= 200:
            p = p_init / 4
        elif 200 < it <= 500:
            p = p_init / 8
        elif 500 < it <= 1000:
            p = p_init / 16
        elif 1000 < it <= 2000:
            p = p_init / 32
        elif 2000 < it <= 4000:
            p = p_init / 64
        elif 4000 < it <= 6000:
            p = p_init / 128
        elif 6000 < it <= 8000:
            p = p_init / 256
        elif 8000 < it <= 10000:
            p = p_init / 512
        else:
            p = p_init

        return p

    def get_new_deltas(self, x: Tensor, x_best: Tensor, i_iter: int) -> Tensor:
        """Returns candidates for new perturbations.

        Args:
            x: Original images.
            x_best: Current best perturbed images.
            i_iter: Current iteration.
        """
        c, h, w = x.shape[1:]
        deltas = x_best - x

        p = self.p_selection(i_iter)
        for i_img in range(x.shape[0]):
            s = int(round(math.sqrt(p * h * w)))
            s = min(
                max(s, 1), h - 1
            )  # at least c x 1 x 1 window is taken and at most c x h-1 x h-1
            center_h = np.random.randint(0, h - s)
            center_w = np.random.randint(0, w - s)

            def looking_window(arr: Tensor) -> Tensor:
                return arr[i_img, :, center_h : center_h + s, center_w : center_w + s]

            x_curr_window = looking_window(x)
            x_best_curr_window = looking_window(x_best)
            # prevent trying out a delta if it doesn't change x_curr
            # (e.g. an overlapping patch)
            x_new_window = torch.clamp(
                x_curr_window + looking_window(deltas), self.min_val, self.max_val
            )
            while (
                torch.sum(torch.abs(x_new_window - x_best_curr_window) < 10**-7)
                == c * s * s
            ):
                deltas[
                    i_img, :, center_h : center_h + s, center_w : center_w + s
                ] = torch.from_numpy(
                    np.random.choice([-self.eps, self.eps], size=[c, 1, 1])
                )
                x_new_window = torch.clamp(
                    x_curr_window + looking_window(deltas), self.min_val, self.max_val
                )

        return deltas

    @torch.no_grad()
    def run_attack(
        self, x: Tensor, y: Tensor, model: Model
    ) -> Dict[ResultHeadNames, Tensor]:
        c, h, w = x.shape[1:]
        n_queries = torch.zeros((x.shape[0]), dtype=torch.int16).to(x.device)

        init_delta = torch.from_numpy(
            np.random.choice([-self.eps, self.eps], size=[x.shape[0], c, 1, w])
        ).to(x)
        x_best = torch.clamp(x + init_delta, self.min_val, self.max_val)
        logits = model(x_best)
        loss_min = self.loss(logits, y)
        margin_min = self.margin(logits, y)
        n_queries += 1

        for i_iter in range(self.max_iters - 1):
            idx_to_fool = torch.atleast_1d((margin_min > 0.0).nonzero().squeeze())
            if len(idx_to_fool) == 0:
                break

            # Extract the images in which the adversarial sample should be found.
            x_curr = x[idx_to_fool]
            x_best_curr = x_best[idx_to_fool]
            y_curr = y[idx_to_fool]
            margin_min_curr = margin_min[idx_to_fool]
            loss_min_curr = loss_min[idx_to_fool]

            # Generate candidates for new adversarial examples.
            deltas = self.get_new_deltas(x_curr, x_best_curr, i_iter)
            x_new = torch.clamp(x_curr + deltas, self.min_val, self.max_val)
            logits = model(x_new)
            loss = self.loss(logits, y_curr)
            margin = self.margin(logits, y_curr)

            # Update current loss values and adversarial examples.
            idx_improved = loss < loss_min_curr
            loss_min[idx_to_fool] = idx_improved * loss + ~idx_improved * loss_min_curr
            margin_min[idx_to_fool] = (
                idx_improved * margin + ~idx_improved * margin_min_curr
            )

            idx_improved = torch.reshape(idx_improved, [-1, *[1] * len(x.shape[:-1])])
            x_best[idx_to_fool] = idx_improved * x_new + ~idx_improved * x_best_curr
            n_queries[idx_to_fool] += 1

        return {ResultHeadNames.X_ADV: x_best, ResultHeadNames.QUERIES: n_queries}
